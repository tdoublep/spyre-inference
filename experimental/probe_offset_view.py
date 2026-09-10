"""Does a compiled region read a nonzero-offset slot slice correctly?

The KV-tiled decode kernels compute the wrong answer (max_abs 0.145 against a
reference whose absmax is 0.118), both identically, and they stay wrong at full
context with no masking -- so the fault is in how the tile reads K/V, not in the
mask. The suspicion is torch-spyre#3770, which _page_attn_kernel already
documents as "a compiled region reads a view from offset 0, ignoring
storage_offset". KV tiling slices the slot axis at lo=kv_tile, which is exactly
such a view; if tile 1 re-reads tile 0 the kernel does half the KV work, which
would explain the apparent speedup as an artifact.

Fills one page with a distinct value per slot and returns the second 64-slot tile
alone -- no reduction and no stack, both of which the backend rejects here. If the
compiled result equals the FIRST tile, storage_offset was ignored.
"""

import torch

from spyre_inference.custom_ops.utils import convert
from spyre_inference.custom_ops.utils import register as register_convert_op
from spyre_inference.v1.attention.backends.spyre_attn import slot_major_kv_layout

DEV = torch.device("spyre")
BLOCK, KV, D, TILE = 128, 8, 128, 64


def second_tile(cache, idx):
    """The kernels' own access pattern: gather a page, slice the slot axis, permute."""
    page = cache.index_select(0, idx)          # [1, BLOCK, KV, D]
    t = page[:, TILE : 2 * TILE]               # nonzero storage_offset
    return t.permute(0, 2, 1, 3).clone()       # [1, KV, TILE, D]


def first_tile(cache, idx):
    page = cache.index_select(0, idx)
    t = page[:, 0:TILE]                        # offset 0
    return t.permute(0, 2, 1, 3).clone()


def main():
    register_convert_op()
    torch.ones(2, 2, dtype=torch.float16).to(DEV)

    dtype = torch.float16
    num_pages = 2
    host = torch.zeros(num_pages, BLOCK, KV, D, dtype=dtype)
    for s in range(BLOCK):
        host[0, s] = (s + 1) / 256.0
    layout = slot_major_kv_layout(num_pages * BLOCK, KV, D, dtype)
    cache = host.to(DEV, device_layout=layout)
    idx = convert(torch.tensor([0], dtype=torch.int32), device=DEV)

    want_second = host[0, TILE : 2 * TILE].permute(1, 0, 2).unsqueeze(0).float()
    want_first = host[0, 0:TILE].permute(1, 0, 2).unsqueeze(0).float()

    got_second = torch.compile(second_tile, dynamic=False)(cache, idx).cpu().float()
    got_first = torch.compile(first_tile, dynamic=False)(cache, idx).cpu().float()

    d_second = (got_second - want_second).abs().max().item()
    d_first = (got_first - want_first).abs().max().item()
    d_cross = (got_second - want_first).abs().max().item()

    print("tile0: |got - want_tile0| = %.6f" % d_first)
    print("tile1: |got - want_tile1| = %.6f" % d_second)
    print("tile1: |got - want_tile0| = %.6f   <- 0 here means it read tile 0" % d_cross)
    print("first slot values: want_tile1[0,0,0,0]=%.4f got=%.4f"
          % (want_second[0, 0, 0, 0].item(), got_second[0, 0, 0, 0].item()))
    print()
    if d_cross < 1e-6 and d_second > 1e-3:
        print("BUG CONFIRMED: the offset slice returned tile 0 (torch-spyre#3770).")
        print("Slicing the slot axis is therefore unusable for KV tiling; a tile has")
        print("to be gathered, not sliced.")
    elif d_second < 1e-3:
        print("Offset slices read correctly -- the KV-tiling bug is elsewhere.")
    else:
        print("tile1 matches neither tile: some other mis-read (diff %.6f)." % d_second)


if __name__ == "__main__":
    main()
