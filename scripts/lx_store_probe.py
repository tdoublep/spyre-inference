#!/usr/bin/env python
"""Does an indirect store into a [kv, head, pages, block] V cache work on device?

The head-major V layout that keeps the page tile LX-resident puts the token axis
innermost, so the slot-major store view `kv_slot_views` builds no longer exists.
Ordering the cache [kv, head, pages, block] makes the flat index
(h*D + d) * (P*B) + (p*B + off), whose trailing term is exactly the slot index --
so a [kv*head, pages*block] view should take the whole scatter as one
`index_copy_` along dim 1.

`slot_major_kv_layout` exists because the *default* device layout spreads the
slot index across two device dims and the indirect store then writes the wrong
rows (torch-spyre#3705). This checks whether the dim-1 store hits the same class
of bug, and with which layout it comes out correct.

Usage:
  python scripts/lx_store_probe.py [--layout default|flat]
"""

from __future__ import annotations

import argparse

P, B, KV, D = 8, 128, 8, 128  # pages, block_size, kv heads, head size
DTYPE = None  # set in main once torch is imported


def build_layout(kind: str):
    import torch
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick

    if kind == "default":
        return None
    eps = get_elem_in_stick(torch.float16)
    sticks = (B + eps - 1) // eps
    print(f"elems_per_stick(fp16)={eps}, sticks over block_size={sticks}")
    # Row-major over [kv, head, pages, sticks, eps]: the slot axis (pages, block)
    # stays a flat contiguous run so the dim-1 scatter addresses it directly.
    return SpyreTensorLayout(
        device_size=[KV, D, P, sticks, eps],
        stride_map=[D * P * B, P * B, B, eps, 1],
        device_dtype=get_device_dtype(torch.float16),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="default", choices=["default", "flat"])
    ap.add_argument("--tokens", type=int, default=1, help="tokens stored per call")
    args = ap.parse_args()

    import torch

    from spyre_inference.custom_ops import utils as custom_op_utils
    from spyre_inference.custom_ops.utils import convert

    custom_op_utils.register()
    device = torch.device("spyre")
    torch.manual_seed(0)

    T = args.tokens
    # Slots chosen in distinct pages so a page-collapsing layout bug shows up.
    slots_cpu = torch.tensor(
        [(t % P) * B + (t * 7 + 3) % B for t in range(T)], dtype=torch.int32
    )
    value_cpu = torch.randn(T, KV, D, dtype=torch.float16)

    # CPU reference: the store we want, written the obvious way.
    ref = torch.zeros(KV, D, P, B, dtype=torch.float16)
    for t in range(T):
        p, off = divmod(int(slots_cpu[t]), B)
        ref[:, :, p, off] = value_cpu[t]

    layout = build_layout(args.layout)
    v_host = torch.zeros(KV, D, P, B, dtype=torch.float16)
    v_pages = (
        v_host.to(device, device_layout=layout) if layout is not None else convert(v_host, device=device)
    )
    slots = convert(slots_cpu, device=device)
    value = convert(value_cpu, device=device)

    # Built outside the graph: inductor cannot lower a store through a view of a
    # Spyre-layout tensor created inside one.
    v_flat = v_pages.view(KV * D, P * B)

    def store(value, v_flat, slots):
        # [T, kv, head] -> [kv*head, T]; contiguous for T == 1.
        src = value.permute(1, 2, 0).reshape(KV * D, T)
        v_flat.index_copy_(1, slots, src)
        return v_flat

    compiled = torch.compile(store, dynamic=False)
    compiled(value, v_flat, slots)

    got = v_pages.cpu().float()
    exp = ref.float()
    diff = (got - exp).abs().max().item()
    nz_got = int((got != 0).sum())
    nz_exp = int((exp != 0).sum())
    print(f"layout={args.layout} tokens={T}")
    print(f"nonzeros: got {nz_got}, expected {nz_exp}")
    print(f"max abs diff {diff:.4g} -> {'OK' if diff == 0 else 'WRONG'}")
    if diff != 0 and nz_got == nz_exp:
        # Same count in the wrong place is the torch-spyre#3705 signature.
        print("  (right number of writes, wrong locations -> misaddressed store)")


if __name__ == "__main__":
    main()
