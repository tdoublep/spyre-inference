"""Cost of the KV page gather as a function of index width.

Six full decode kernels varied mask access, matmul batch dims, M, the K/V
broadcast, op count and work division without any of them tracking runtime; the
one property that separated the fast variants from the slow ones was the width
of the page index_select. This isolates that op.

In the slot-major layout a page is a contiguous run of block_size slots, so a
1-wide index_select can lower to a contiguous slice while a wider one is a gather
over disjoint runs. If that is the whole story, cost per call jumps between width
1 and 2 and then grows slowly.

Also prices the slot-granularity gather that would let (S,KV) be merged without a
copy: same bytes, but block_size*KV*S indices instead of S.

    LAYOUT_SOLVER=greedy python experimental/microbench_page_gather.py
"""

import argparse
import statistics
import time

import torch

from spyre_inference.custom_ops.utils import convert
from spyre_inference.custom_ops.utils import register as register_convert_op
from spyre_inference.v1.attention.backends.spyre_attn import slot_major_kv_layout

DEV = torch.device("spyre")


def _gather_pages(cache, idx):
    return cache.index_select(0, idx)


def _gather_pages_permuted(cache, idx):
    # What the kernels actually feed the matmul: gather then token->head permute.
    return cache.index_select(0, idx).permute(0, 2, 1, 3).clone()


def _gather_slots(flat, idx, g, block_size, head_size):
    return flat.index_select(0, idx).reshape(g, block_size, head_size)


def wall_ms(fn, iters):
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.spyre.synchronize()
        out.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(out), min(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-seqs", type=int, default=4)
    p.add_argument("--num-blocks", type=int, default=16)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--widths", default="1,2,4,8,16,32,64",
                   help="the gather penalty looked flat in width, so push it far "
                        "enough to find where bytes start to dominate")
    a = p.parse_args()

    register_convert_op()
    # A plain .to() creates the RuntimeContext; the device_layout form below
    # requires one to exist already and errors out otherwise.
    torch.ones(2, 2, dtype=torch.float16).to(DEV)

    nkv, hs, bs = a.num_kv_heads, a.head_size, a.block_size
    dtype = torch.float16
    num_pages = a.num_seqs * a.num_blocks + 1

    layout = slot_major_kv_layout(num_pages * bs, nkv, hs, dtype)
    cache = torch.zeros(num_pages, bs, nkv, hs, dtype=dtype).to(DEV, device_layout=layout)
    flat = cache.view(num_pages * bs * nkv, hs)
    print("cache: %d pages x %d slots, %.2f MB; one page = %.0f KB\n"
          % (num_pages, bs, num_pages * bs * nkv * hs * 2 / 1e6, bs * nkv * hs * 2 / 1e3))

    gather_c = torch.compile(_gather_pages, dynamic=False)
    gather_perm_c = torch.compile(_gather_pages_permuted, dynamic=False)
    slots_c = torch.compile(_gather_slots, dynamic=False)

    widths = [int(w) for w in a.widths.split(",") if w.strip()]
    print("%-28s %6s %-9s %9s %10s %12s"
          % ("op", "width", "layout", "wall med", "wall min", "MB/call"))

    for w in widths:
        # "spread" is what a real block table looks like: each index its own
        # contiguous run. "adjacent" asks whether the penalty is about having
        # more than one index at all, or about the run not being contiguous --
        # a width-1 gather is already a contiguous block_size-slot run, so if
        # adjacent pages are cheap the lever is contiguity, not index count.
        variants = {
            "spread": [i * a.num_blocks for i in range(w)],
            "adjacent": list(range(w)),
        }
        mb = w * bs * nkv * hs * 2 / 1e6
        for tag, pages_list in variants.items():
            if w == 1 and tag == "adjacent":
                continue
            idx = convert(torch.tensor(pages_list, dtype=torch.int32), device=DEV)
            for label, fn in (
                ("index_select(page)", lambda idx=idx: gather_c(cache, idx)),
                ("index_select+permute+clone", lambda idx=idx: gather_perm_c(cache, idx)),
            ):
                try:
                    med, mn = wall_ms(fn, a.iters)
                    print("%-28s %6d %-9s %9.3f %10.3f %12.3f"
                          % (label, w, tag, med, mn, mb))
                except Exception as exc:
                    print("%-28s %6d %-9s FAILED %s: %s"
                          % (label, w, tag, type(exc).__name__, str(exc)[:120]))

    # Slot-granularity gather for the same bytes as width=num_seqs, but landing
    # already in (S,KV)-major order so the (S,KV) merge needs no copy.
    ns = a.num_seqs
    g = ns * nkv
    base = (torch.arange(bs, dtype=torch.int32).view(1, bs) * nkv
            + torch.arange(nkv, dtype=torch.int32).view(nkv, 1))
    pages = torch.tensor([i * a.num_blocks for i in range(ns)], dtype=torch.int32)
    sidx = (pages.view(ns, 1, 1) * (bs * nkv) + base.view(1, nkv, bs)).reshape(-1).contiguous()
    sidx_dev = convert(sidx, device=DEV)
    mb = ns * bs * nkv * hs * 2 / 1e6
    try:
        med, mn = wall_ms(lambda: slots_c(flat, sidx_dev, g, bs, hs), a.iters)
        print("%-28s %6d %11.3f %10.3f %12.3f"
              % ("index_select(slot)", sidx.numel(), med, mn, mb))
    except Exception as exc:
        print("%-28s        FAILED %s: %s"
              % ("index_select(slot)", type(exc).__name__, str(exc)[:200]))


if __name__ == "__main__":
    main()
