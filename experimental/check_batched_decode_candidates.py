"""Correctness check for the candidate batched decode kernels.

The timing harness fills the KV cache with zeros, which makes every variant emit
zeros -- fine for data-independent timing, useless for correctness. This fills
K/V with random values and a real per-block mask, then compares each candidate
against num_seqs sequential _page_attn_kernel calls, which is the shipped path.

Run with LAYOUT_SOLVER=greedy so #4347 is active, as in the timing harness.
"""

import argparse

import torch

from spyre_inference.custom_ops.utils import convert
from spyre_inference.custom_ops.utils import register as register_convert_op
from spyre_inference.v1.attention.backends.spyre_attn import (
    INT32_ELEMS_PER_STICK,
    _batched_decode_kernel,
    _page_attn_kernel,
    _stick_aligned_len,
    slot_major_kv_layout,
)

from microbench_batched_decode_v2 import (
    _batched_ktile_kernel,
    _batched_masklist_kernel,
    _chunked_gather_kernel,
    _chunked_ktile_kernel,
    _gather_shared_kernel,
    _merged_cat_kernel,
    _merged_sk_kernel,
    _qgroup_loop_kernel,
)

DEV = torch.device("spyre")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-seqs", type=int, default=4)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--num-queries-per-kv", type=int, default=4)
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--staging-rows", type=int, default=513)
    p.add_argument("--context-len", type=int, default=0,
                   help="real KV length; 0 means all blocks fully valid")
    p.add_argument("--chunk", type=int, default=2)
    p.add_argument("--kv-tile", type=int, default=32)
    a = p.parse_args()

    register_convert_op()
    nkv, qpkv, hs, bs = a.num_kv_heads, a.num_queries_per_kv, a.head_size, a.block_size
    ns, nb = a.num_seqs, a.num_blocks
    nh = nkv * qpkv
    dtype = torch.float16
    scale = hs**-0.5
    num_pages = ns * nb + 1
    ctx = a.context_len or nb * bs

    torch.manual_seed(0)
    k_host = torch.randn(num_pages, bs, nkv, hs).to(dtype) * 0.5
    v_host = torch.randn(num_pages, bs, nkv, hs).to(dtype) * 0.5
    layout = slot_major_kv_layout(num_pages * bs, nkv, hs, dtype)
    k_pages = k_host.to(DEV, device_layout=layout)
    v_pages = v_host.to(DEV, device_layout=layout)

    q_host = torch.randn(a.staging_rows, nh, hs).to(dtype)
    query = convert(q_host, device=DEV)

    # Mask the tail of the last block so the check exercises masking, not just
    # a uniform attention over every slot.
    neg = torch.finfo(dtype).min
    mask_host = []
    for b in range(nb):
        t = torch.zeros(1, bs, dtype=dtype)
        lo = b * bs
        for j in range(bs):
            if lo + j >= ctx:
                t[0, j] = neg
        mask_host.append(t)
    mask_tiles = [convert(t, device=DEV) for t in mask_host]
    # Same mask, broadcast across seqs and KV heads, for the batched shapes.
    mask_list_4d = [
        convert(mask_host[b].reshape(1, 1, 1, bs).expand(ns, nkv, 1, bs).contiguous(), device=DEV)
        for b in range(nb)
    ]
    mask_by_block = convert(
        torch.stack([
            mask_host[b].reshape(1, 1, bs).expand(ns * nkv, 1, bs).contiguous()
            for b in range(nb)
        ]),
        device=DEV,
    )

    row_idx, page_tables = [], []
    for s in range(ns):
        t = torch.zeros(_stick_aligned_len(1), dtype=torch.int32)
        t[0] = s
        row_idx.append(convert(t, device=DEV))
        pt = torch.zeros(nb, INT32_ELEMS_PER_STICK, dtype=torch.int32)
        for b in range(nb):
            pt[b, 0] = s * nb + b
        page_tables.append(convert(pt, device=DEV))

    bid = torch.zeros(nb, _stick_aligned_len(ns), dtype=torch.int32)
    for b in range(nb):
        for s in range(ns):
            bid[b, s] = s * nb + b
    block_ids = convert(bid, device=DEV)

    chunk_pages = []
    if nb % a.chunk == 0:
        for c in range(nb // a.chunk):
            ids = [s * nb + c * a.chunk + j for j in range(a.chunk) for s in range(ns)]
            chunk_pages.append(convert(torch.tensor(ids, dtype=torch.int32), device=DEV))

    ref = torch.cat([
        _page_attn_kernel(query, row_idx[s], k_pages, v_pages, page_tables[s],
                          mask_tiles, scale, nb, 1, nh, nkv, hs).cpu()
        for s in range(ns)
    ], dim=0).float()

    # Compiled, not eager: the KV-tiled forms do not run eager at all ("no
    # mechanism to resolve stick incompatibility"), and compiled is the path that
    # was benchmarked. The reference above stays eager, so this compares an eager
    # per-sequence reference against a compiled batched candidate.
    def c(fn):
        return torch.compile(fn, dynamic=False)

    cands = {
        "batched": lambda: _batched_decode_kernel(
            query, None, k_pages, v_pages, block_ids, mask_by_block, scale,
            ns, nb, nkv, qpkv, bs, hs),
        "batched_masklist": lambda: _batched_masklist_kernel(
            query, k_pages, v_pages, block_ids, mask_list_4d, scale,
            ns, nb, nkv, qpkv, bs, hs),
        "qgroup_loop": lambda: _qgroup_loop_kernel(
            query, k_pages, v_pages, block_ids, mask_list_4d, scale,
            ns, nb, nkv, qpkv, bs, hs),
        "gather_shared": lambda: _gather_shared_kernel(
            query, k_pages, v_pages, block_ids, mask_tiles, scale,
            ns, nb, nkv, qpkv, bs, hs),
        "merged_sk": lambda: _merged_sk_kernel(
            query, k_pages, v_pages, block_ids, mask_tiles, scale,
            ns, nb, nkv, qpkv, bs, hs),
        "merged_cat": lambda: _merged_cat_kernel(
            query, k_pages, v_pages, page_tables, mask_tiles, scale,
            ns, nb, nkv, qpkv, bs, hs),
        "batched_ktile": lambda: c(_batched_ktile_kernel)(
            query, k_pages, v_pages, block_ids, mask_list_4d, scale,
            ns, nb, a.kv_tile, nkv, qpkv, bs, hs),
        "chunked_gather": lambda: _chunked_gather_kernel(
            query, k_pages, v_pages, chunk_pages, mask_tiles, scale,
            ns, nb, a.chunk, nkv, qpkv, bs, hs),
        # The design under test.
        "chunked_ktile": lambda: c(_chunked_ktile_kernel)(
            query, k_pages, v_pages, chunk_pages, mask_list_4d, scale,
            ns, nb, a.chunk, a.kv_tile, nkv, qpkv, bs, hs),
    }

    print("ref (per_seq) shape %s  absmax %.4f  ctx=%d" % (tuple(ref.shape), ref.abs().max(), ctx))
    worst = {}
    for name, fn in cands.items():
        try:
            got = fn().cpu().float()
        except Exception as exc:
            print("%-18s FAILED %s: %s" % (name, type(exc).__name__, str(exc)[:200]))
            continue
        if got.shape != ref.shape:
            print("%-18s SHAPE MISMATCH %s vs %s" % (name, tuple(got.shape), tuple(ref.shape)))
            continue
        adiff = (got - ref).abs()
        denom = ref.abs().clamp(min=1e-3)
        rel = (adiff / denom).max().item()
        worst[name] = (adiff.max().item(), rel)
        print("%-18s max_abs %.5f  max_rel %.5f  %s"
              % (name, adiff.max().item(), rel, "OK" if adiff.max().item() < 2e-2 else "CHECK"))
    return worst


if __name__ == "__main__":
    main()
