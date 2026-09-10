"""Microbenchmark: per-sequence vs batched vs unrolled decode attention.

Isolates why SPYRE_BATCHED_DECODE=1 loses to the per-sequence path even though
it cuts kernel launches 4:1. Drives the real kernels from spyre_attn, so it
tracks the production code rather than a paraphrase.

Variants
  per_seq   num_seqs sequential calls to _page_attn_kernel (what production does
            with SPYRE_BATCHED_DECODE=0)
  batched   one call to _batched_decode_kernel (SPYRE_BATCHED_DECODE=1)
  unrolled  the "dumbest thing": one compiled graph containing the num_seqs
            per-sequence bodies. One launch, per-sequence shapes/schedules.
            Separates launch overhead from schedule quality: if unrolled ~=
            per_seq, launch overhead is not the win; if unrolled << batched,
            the batched kernel's shapes are what cost, not the batching idea.

Defaults reproduce granite-3.3-8b decode at batch 4 / 2048 KV / block 128.

Run (needs a Spyre device; the RPM env must be sourced):
    source ~/spyre-libs/env.sh
    uv run --no-sync python experimental/microbench_batched_decode.py

Useful knobs:
    --num-seqs 4 --num-blocks 16 --block-size 128
    --iters 20 --variants per_seq,batched,unrolled
    --show-warnings          print each lossy work-division message

STATUS
------
Run on tpa-spyre-dev-2, granite shapes, 2026-09-10:

    variant   dev ms/call  kernels  wall med ms  lossy  warmup s
    per_seq         2.343        4        3.129      0      30.6
    batched         7.693        1        8.096     32      37.5
    unrolled        2.267        2        2.905      0     123.0

This reproduces the production regression:

    metric                    harness    production   delta
    batched, device/layer     7.693 ms     7.581 ms   +1.5%
    batched KV bandwidth      4.4 GB/s     4.4 GB/s   exact
    per_seq, device/layer     2.343 ms     3.054 ms   -23%
    batched / per_seq            3.28x        2.48x   same direction

per_seq is optimistic by ~23% because the harness skips the metadata handling
and output scatter the production per-call path also does. The batched
pathology itself lands within 1.5%.

Getting here required matching two things that a naive harness gets wrong, and
both mattered enormously -- with either wrong, per_seq collapsed to 1.7 GB/s
and the ranking INVERTED (batched looked 1.7x faster):

  1. k/v_pages must be allocated the way initialize_kv_cache_tensors does:
     host zeros then .to(device, device_layout=slot_major_kv_layout(...)).
     A plain convert() leaves the default tiled layout, which spreads the slot
     index across two device dims (torch-spyre#3705) and makes the in-graph
     page gather slow for every variant.
  2. `query` must be the production staging width (max_num_batched_tokens + 1
     = 513 rows), not num_seqs. The batched kernel takes a query[:num_seqs]
     prefix of it and the per-sequence kernel gathers out of it.

Findings so far:
  * The batched kernel is the only variant that trips work-division
    fallbacks: 32 x `lossy work-division ... output:d4=absent`, matching the
    count in the full vLLM run exactly.
  * Launch overhead cannot explain the regression. Host residual
    (wall - device) is 0.786 ms over 4 launches for per_seq vs 0.638 ms over
    1 launch for unrolled, so collapsing 4 launches to 1 saves 0.148 ms --
    3% of the batched kernel's +5.350 ms/call device penalty.
  * `unrolled` is the fastest variant: one graph holding the num_seqs
    per-sequence bodies gets a single dispatch AND keeps the per-sequence
    schedule the planner handles well (0 fallbacks). It beats the purpose-built
    batched kernel by 3.4x. So the problem is the batched kernel's shapes, not
    the idea of batching. Cost of that approach: num_seqs is baked into the
    graph, so it needs one compiled variant per seq bucket.

Open: why the unrolled variant reports 2 device kernels for 4 sequence bodies.
"""

import argparse
import logging
import statistics
import time

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

DEV = torch.device("spyre")


def _unrolled_decode_kernel(
    query,
    query_row_indices,
    k_pages,
    v_pages,
    page_index_tables,
    mask_tiles,
    scale,
    num_seqs,
    num_blocks,
    num_heads,
    num_kv_heads,
    head_size,
):
    """One graph, num_seqs per-sequence bodies. Static ints so Dynamo unrolls."""
    outs = []
    for s in range(num_seqs):
        outs.append(
            _page_attn_kernel(
                query,
                query_row_indices[s],
                k_pages,
                v_pages,
                page_index_tables[s],
                mask_tiles,
                scale,
                num_blocks,
                1,
                num_heads,
                num_kv_heads,
                head_size,
            )
        )
    return torch.cat(outs, dim=0)


class LossyCounter(logging.Handler):
    """Counts torch-spyre work-division fallbacks emitted while compiling."""

    PATTERNS = ("lossy work-division", "RetileWarning", "re-tiling")

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.hits = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if any(p in msg for p in self.PATTERNS):
            self.hits.append(msg)

    def reset(self):
        self.hits = []


def build_inputs(a):
    """Device tensors matching each kernel's documented shape contract."""
    # convert() routes through torch.ops.vllm.spyre_convert, which only exists
    # once this runs; the runner normally does it during platform setup.
    register_convert_op()

    num_heads = a.num_kv_heads * a.num_queries_per_kv
    dtype = torch.float16
    # One distinct page per (seq, block) so no sequence shares KV with another.
    num_pages = a.num_seqs * a.num_blocks + 1

    torch.manual_seed(0)
    # Allocated exactly as TorchSpyreModelRunner.initialize_kv_cache_tensors does:
    # host zeros then .to(device, device_layout=slot-major). The default tiled
    # layout spreads the slot index across two device dims (torch-spyre#3705),
    # which changes what the in-graph page gather costs -- so a plain convert()
    # here does NOT measure the same thing production does.
    layout = slot_major_kv_layout(
        num_pages * a.block_size, a.num_kv_heads, a.head_size, dtype
    )
    k_pages = torch.zeros(
        num_pages, a.block_size, a.num_kv_heads, a.head_size, dtype=dtype
    ).to(DEV, device_layout=layout)
    v_pages = torch.zeros(
        num_pages, a.block_size, a.num_kv_heads, a.head_size, dtype=dtype
    ).to(DEV, device_layout=layout)

    # Decode: one query row per sequence in rows 0..num_seqs-1, but the buffer is
    # the production staging buffer width (max_num_batched_tokens + 1), not
    # num_seqs: the batched kernel takes a query[:num_seqs] prefix of it and the
    # per-sequence kernel gathers out of it, and both cost differently against a
    # 513-row source than against a snug one.
    query = convert(
        torch.randn(a.staging_rows, num_heads, a.head_size).to(dtype), device=DEV
    )
    out_buf = convert(
        torch.zeros(a.staging_rows, num_heads, a.head_size).to(dtype), device=DEV
    )

    # --- per-sequence / unrolled inputs -------------------------------------
    # query_row_index: stick-aligned int32, first padded_query_len entries are
    # this sequence's absolute query rows (padded_query_len == 1 for decode).
    idx_len = _stick_aligned_len(1)
    row_idx = []
    for s in range(a.num_seqs):
        t = torch.zeros(idx_len, dtype=torch.int32)
        t[0] = s
        row_idx.append(convert(t, device=DEV))

    # page_index_table: [num_blocks, INT32_ELEMS_PER_STICK], page index at col 0.
    page_tables = []
    for s in range(a.num_seqs):
        t = torch.zeros(a.num_blocks, INT32_ELEMS_PER_STICK, dtype=torch.int32)
        for b in range(a.num_blocks):
            t[b, 0] = s * a.num_blocks + b
        page_tables.append(convert(t, device=DEV))

    # mask_tiles: [padded_query_len, block_size] additive, zeros = nothing masked.
    mask_tiles = [
        convert(torch.zeros(1, a.block_size, dtype=dtype), device=DEV)
        for _ in range(a.num_blocks)
    ]

    # --- batched inputs ------------------------------------------------------
    # block_ids: [num_blocks, stick-padded num_seqs], row i = block i's page per seq.
    bid = torch.zeros(a.num_blocks, _stick_aligned_len(a.num_seqs), dtype=torch.int32)
    for b in range(a.num_blocks):
        for s in range(a.num_seqs):
            bid[b, s] = s * a.num_blocks + b
    block_ids = convert(bid, device=DEV)

    # mask_by_block: [num_blocks, num_seqs * num_kv_heads, 1, block_size],
    # pre-broadcast across KV heads by the production builder.
    mask_by_block = convert(
        torch.zeros(
            a.num_blocks, a.num_seqs * a.num_kv_heads, 1, a.block_size, dtype=dtype
        ),
        device=DEV,
    )

    return dict(
        query=query,
        out_buf=out_buf,
        k_pages=k_pages,
        v_pages=v_pages,
        row_idx=row_idx,
        page_tables=page_tables,
        mask_tiles=mask_tiles,
        block_ids=block_ids,
        mask_by_block=mask_by_block,
        num_heads=num_heads,
    )


def make_callables(a, t):
    """One zero-arg callable per variant, each compiled with dynamic=False."""
    scale = a.head_size**-0.5
    nh, nkv, hs = t["num_heads"], a.num_kv_heads, a.head_size

    per_seq_c = torch.compile(_page_attn_kernel, dynamic=False)
    batched_c = torch.compile(_batched_decode_kernel, dynamic=False)
    unrolled_c = torch.compile(_unrolled_decode_kernel, dynamic=False)

    def per_seq():
        outs = []
        for s in range(a.num_seqs):
            outs.append(
                per_seq_c(
                    t["query"], t["row_idx"][s], t["k_pages"], t["v_pages"],
                    t["page_tables"][s], t["mask_tiles"], scale,
                    a.num_blocks, 1, nh, nkv, hs,
                )
            )
        return outs[-1]

    def batched():
        return batched_c(
            t["query"], None, t["k_pages"], t["v_pages"], t["block_ids"],
            t["mask_by_block"], scale, a.num_seqs, a.num_blocks, nkv,
            a.num_queries_per_kv, a.block_size, hs,
        )

    def unrolled():
        return unrolled_c(
            t["query"], t["row_idx"], t["k_pages"], t["v_pages"], t["page_tables"],
            t["mask_tiles"], scale, a.num_seqs, a.num_blocks, nh, nkv, hs,
        )

    return {"per_seq": per_seq, "batched": batched, "unrolled": unrolled}


def device_kernel_ms(fn, iters):
    """Sum of device kernel durations per call, via the torch profiler."""
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "PrivateUse1"):
        acts.append(ProfilerActivity.PrivateUse1)
    with profile(activities=acts, record_shapes=False) as prof:
        for _ in range(iters):
            fn()
        torch.spyre.synchronize()
    total, n = 0.0, 0
    for e in prof.events():
        # Device kernels only; CPU ops carry the same names in some builds.
        if getattr(e, "device_type", None) is not None and "spyre_kernel" in str(e.key):
            total += e.self_device_time_total or 0.0
            n += 1
    return total / 1000.0 / iters, n // max(iters, 1)


def wall_ms(fn, iters):
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.spyre.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-seqs", type=int, default=4)
    p.add_argument("--num-blocks", type=int, default=16, help="2048 KV / block 128")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--num-queries-per-kv", type=int, default=4, help="32 heads / 8 kv")
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=40, help="only scales the report")
    p.add_argument("--staging-rows", type=int, default=513,
                   help="production is max_num_batched_tokens + 1 (512 + 1)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--variants", default="per_seq,batched,unrolled")
    p.add_argument("--show-warnings", action="store_true")
    a = p.parse_args()

    kv_mb = a.num_blocks * a.block_size * a.num_kv_heads * a.head_size * 2 * 2 / 1e6
    print("shapes: num_seqs=%d num_blocks=%d block_size=%d kv_heads=%d "
          "queries_per_kv=%d head_size=%d" %
          (a.num_seqs, a.num_blocks, a.block_size, a.num_kv_heads,
           a.num_queries_per_kv, a.head_size))
    print("KV per layer: %.2f MB/seq, %.2f MB for %d seqs\n"
          % (kv_mb, kv_mb * a.num_seqs, a.num_seqs))

    counter = LossyCounter()
    logging.getLogger().addHandler(counter)
    logging.getLogger().setLevel(logging.DEBUG)

    t = build_inputs(a)
    fns = make_callables(a, t)

    rows = []
    for name in [v.strip() for v in a.variants.split(",") if v.strip()]:
        fn = fns[name]
        counter.reset()
        t0 = time.perf_counter()
        for _ in range(a.warmup):
            fn()
        torch.spyre.synchronize()
        compile_s = time.perf_counter() - t0
        lossy = len(counter.hits)
        if a.show_warnings:
            for h in counter.hits[:40]:
                print("   [%s] %s" % (name, h[:150]))

        dev_ms, nkern = device_kernel_ms(fn, a.iters)
        w = wall_ms(fn, a.iters)
        rows.append((name, dev_ms, nkern, statistics.median(w), min(w), lossy, compile_s))

    print("%-9s %11s %8s %11s %10s %7s %9s"
          % ("variant", "dev ms/call", "kernels", "wall med ms", "wall min", "lossy", "warmup s"))
    for name, dev, nk, wmed, wmin, lossy, cs in rows:
        print("%-9s %11.3f %8d %11.3f %10.3f %7d %9.1f"
              % (name, dev, nk, wmed, wmin, lossy, cs))

    base = next((r for r in rows if r[0] == "per_seq"), None)
    if base:
        print("\nrelative to per_seq (device time):")
        for name, dev, *_ in rows:
            print("  %-9s %6.2fx   -> %8.1f ms/step across %d layers"
                  % (name, dev / base[1], dev * a.layers, a.layers))
        print("\neffective KV bandwidth:")
        for name, dev, *_ in rows:
            print("  %-9s %6.1f GB/s" % (name, kv_mb * a.num_seqs / 1e3 / (dev / 1e3)))

    print("\nreference (granite-3.3-8b, batch 4, from the profiled runs):")
    print("  per_seq : 763.4 us x 4 seqs = 3.05 ms/layer -> 122.1 ms/step, 11.0 GB/s")
    print("  batched : 7581.1 us x 1     = 7.58 ms/layer -> 303.2 ms/step,  4.4 GB/s")


if __name__ == "__main__":
    main()
