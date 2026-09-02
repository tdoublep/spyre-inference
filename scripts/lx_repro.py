#!/usr/bin/env python
"""Minimal reproducer: where does paged-attention K/V page data live?

Compiles the *shipped* attention kernel (`_create_compilable_page_attn`) over
Granite-3.3-8B decode shapes, with the KV cache allocated exactly as
`TorchSpyreModelRunner.initialize_kv_cache_tensors` does, and prints
torch-spyre's own LX decision for every buffer.

Decisions come from torch-spyre's `spyre.inductor.scratchpad.allocator` logger
(the `lx_pinning: <buf> (<op>) -> <reason>` lines). A bare `-> lx` means LX was
GRANTED; anything else is the verbatim refusal reason. Nothing here is inferred
from absence and there is no monkeypatching, so the output is torch-spyre's
account of itself.

What to look at: the page data path per block per lane is
`restickify -> index -> clone -> batched_matmul`. Every buffer on that path that
is refused LX is page-sized data materialized in HBM. The goal is that a page is
read out of HBM once, into LX, and consumed there.

Usage:
  python scripts/lx_repro.py                       # production cache layout
  python scripts/lx_repro.py --layout default      # contrast: default layout
  python scripts/lx_repro.py --blocks 5 --verbose  # 576-token context, full table
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import tempfile
from collections import Counter

# torch-spyre only emits its `lx_pinning` decisions when its own inductor logging
# is on, and it reads these at import time -- so set them before importing torch.
os.environ.setdefault("SPYRE_INDUCTOR_LOG", "1")
os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "DEBUG")
# The allocator only runs on a real compile. A warm inductor cache short-circuits
# it and the run reports no decisions at all, so force a cold cache per run.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR", tempfile.mkdtemp(prefix="lx_repro_inductor_")
)

# Granite 3.3 8B, one decode step.
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 128
TOTAL_PAGES = 64
QUERY_ROWS = 8  # batch query buffer; the kernel gathers this sequence's row(s) from it

_LINE = re.compile(r"lx_pinning: (\S+) \(([^)]*)\) → (.*)")
_PAGE_PATH_OPS = {"restickify", "index", "clone", "batched_matmul"}


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.rows: list[tuple[str, str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        m = _LINE.search(record.getMessage())
        if m:
            self.rows.append((m.group(1), m.group(2), m.group(3)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=4, help="active KV pages (context/128)")
    ap.add_argument("--layout", default="slot-major", choices=["slot-major", "default"],
                    help="slot-major matches production; default shows the contrast")
    ap.add_argument("--verbose", action="store_true", help="print every buffer, not just the page path")
    args = ap.parse_args()

    import torch

    from spyre_inference.custom_ops import utils as custom_op_utils
    from spyre_inference.custom_ops.utils import convert
    from spyre_inference.v1.attention.backends.spyre_attn import (
        INT32_ELEMS_PER_STICK,
        _create_compilable_page_attn,
        slot_major_kv_layout,
    )

    custom_op_utils.register()
    device = torch.device("spyre")
    torch.manual_seed(0)
    dtype = torch.float16
    nb = args.blocks

    def make_cache() -> torch.Tensor:
        host = torch.randn(TOTAL_PAGES, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=dtype)
        if args.layout == "default":
            return convert(host, device=device)
        # Exactly what initialize_kv_cache_tensors does.
        layout = slot_major_kv_layout(
            TOTAL_PAGES * BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype
        )
        return host.to(device, device_layout=layout)

    k_pages, v_pages = make_cache(), make_cache()
    # Whole batch query buffer, as the real kernel receives it: the gather must
    # select a strict subset, or selecting the entire source faults the device
    # (RAS ComputeHardwareError 0x7b1b, torch-spyre#4033).
    query = convert(torch.randn(QUERY_ROWS, NUM_HEADS, HEAD_SIZE, dtype=dtype), device=device)
    row_index = convert(torch.zeros(INT32_ELEMS_PER_STICK, dtype=torch.int32), device=device)
    table = torch.zeros(nb, INT32_ELEMS_PER_STICK, dtype=torch.int32)
    table[:, 0] = torch.arange(nb, dtype=torch.int32)
    page_index_table = convert(table, device=device)
    mask_tiles = [convert(torch.zeros(1, BLOCK_SIZE, dtype=dtype), device=device) for _ in range(nb)]

    collector = _Collect()
    alloc_log = logging.getLogger("spyre.inductor.scratchpad.allocator")
    alloc_log.addHandler(collector)
    alloc_log.setLevel(logging.DEBUG)
    # Keep the (very chatty) debug stream out of the console; we only want the table.
    alloc_log.propagate = False

    kernel = _create_compilable_page_attn(nb, 1, NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE)
    out = torch.compile(kernel, dynamic=False)(
        query, row_index, k_pages, v_pages, page_index_table, mask_tiles, HEAD_SIZE**-0.5
    )
    out.cpu()

    print(f"\ncache layout: {args.layout}   active pages: {nb}   "
          f"context: {nb * BLOCK_SIZE} tokens   out: {tuple(out.shape)}")

    granted = [r for r in collector.rows if r[2] == "lx"]
    refused = [r for r in collector.rows if r[2] != "lx"]
    print(f"buffers: {len(collector.rows)} decided, {len(granted)} granted LX, {len(refused)} refused\n")

    rows = collector.rows if args.verbose else [r for r in collector.rows if r[1] in _PAGE_PATH_OPS]
    title = "all buffers" if args.verbose else "page-data path only (restickify/index/clone/batched_matmul)"
    print(f"--- {title} ---")
    for buf, op, reason in sorted(rows, key=lambda r: int(re.sub(r"\D", "", r[0]) or 0)):
        verdict = "LX " if reason == "lx" else "HBM"
        detail = "" if reason == "lx" else f"  {reason[:110]}"
        print(f"  {verdict}  {buf:<8} {op:<16}{detail}")

    print("\n--- refusal reasons, page-data path, by count ---")
    for reason, n in Counter(
        r[2].split(":")[0] for r in refused if r[1] in _PAGE_PATH_OPS
    ).most_common():
        print(f"  {n:3d}  {reason}")

    hbm_pages = sum(1 for r in refused if r[1] in _PAGE_PATH_OPS)
    print(f"\nverdict: {hbm_pages} page-path buffers in HBM "
          f"({nb} blocks x 2 lanes = {nb * 2} pages per layer per step)")


if __name__ == "__main__":
    main()
