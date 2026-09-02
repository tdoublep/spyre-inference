#!/usr/bin/env python
"""Is the transpose what earns a gathered KV page its LX allocation?

torch-spyre#4055 reports that a gathered page is never LX-resident. Its
reproducer consumes the page *directly* as matmul Input2 -- reduction dim already
innermost-but-one, no transpose -- which is the V lane of paged attention. This
runs that kernel unchanged as one arm, and against it the same gather feeding the
same matmul through ``.transpose(-2,-1)``, which is the K lane (Q @ K^T).

Everything else is held fixed: identical page tensor, identical indirect
index_select, identical dtype and batch dims. If only the transposed arm gets
``allocation={'lx'`` then #4055 is specifically about the non-transposed operand
and says nothing about K.

Counting is #4055's own method (grep the inductor debug log), not this repo's
probe hook, so the two instrumentations cross-check each other.

Usage:
  python scripts/lx_transpose_ab.py --lane v-direct
  python scripts/lx_transpose_ab.py --lane k-transposed
"""

from __future__ import annotations

import argparse

NUM_KV_HEADS = 8
LQ = 512
BLK = 128
D = 128
NUM_PAGES = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True, choices=["v-direct", "k-transposed"])
    args = ap.parse_args()

    import torch

    dev = torch.device("spyre")
    torch.manual_seed(0)

    # The page tensor is byte-identical between arms: [pages, KV, BLK, D].
    pages = torch.randn(NUM_PAGES, NUM_KV_HEADS, BLK, D, dtype=torch.float16).to(dev)
    page_index_table = torch.zeros(NUM_PAGES, 32, dtype=torch.int32).to(dev)

    if args.lane == "v-direct":
        # #4055 verbatim: [1, KV, LQ, BLK] @ [1, KV, BLK, D], page not transposed.
        other = torch.randn(1, NUM_KV_HEADS, LQ, BLK, dtype=torch.float16).to(dev)

        def kernel(other, pages, page_index_table):
            page_idx = page_index_table[0, 0:1]
            page = pages.index_select(0, page_idx)
            return torch.matmul(other, page)
    else:
        # K lane: [1, KV, LQ, D] @ [1, KV, BLK, D]^T -> the page carries the transpose.
        other = torch.randn(1, NUM_KV_HEADS, LQ, D, dtype=torch.float16).to(dev)

        def kernel(other, pages, page_index_table):
            page_idx = page_index_table[0, 0:1]
            page = pages.index_select(0, page_idx)
            return torch.matmul(other, page.transpose(-2, -1))

    compiled = torch.compile(kernel, dynamic=False)
    out = compiled(other, pages, page_index_table)
    print(f"LANE={args.lane} out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
