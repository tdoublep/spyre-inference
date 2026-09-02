#!/usr/bin/env python
"""Probe whether the K and V page tiles stay LX-resident in paged attention.

Two modes:

  standalone  compile the real ``_create_compilable_page_attn`` kernel on device
              over synthetic Granite-shaped tensors, in isolation. Seconds per
              run, exact K/V attribution, and where kernel variants are A/B'd.

  bench       run a real ``vllm bench latency`` in-process (engine-core
              multiprocessing disabled so the hook sees the worker's compiles)
              and classify every paged-attention graph it compiles.

Both print one row per page tile: which lane (K or V), whether the allocator
granted LX, and if not, the verbatim reason. See scripts/lx_probe_hook.py for
why the reason string is the important output.

Usage:
  python scripts/lx_residency_probe.py --variant baseline
  python scripts/lx_residency_probe.py --variant sink-v
  python scripts/lx_residency_probe.py --mode bench
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = dict(
    num_heads=32,
    num_kv_heads=8,
    head_size=128,
    block_size=128,
    num_blocks=4,
    padded_query_len=1,
    total_pages=64,
)

BENCH_ARGV = [
    "vllm", "bench", "latency",
    "--model", "ibm-granite/granite-3.3-8b-instruct",
    "--input-len", "64", "--output-len", "512", "--batch-size", "1",
    "--num-iters-warmup", "2", "--num-iters", "1", "--max-model-len", "128",
    "-cc.compile_sizes=[1,64]",
]


# --------------------------------------------------------------------------
# Kernel variants
# --------------------------------------------------------------------------


def build_kernel(variant: str, cfg: dict):
    """Return the kernel under test.

    ``baseline`` delegates to the shipped factory so the control never drifts
    from production. Every other variant is a deliberate edit of that loop.
    """
    from spyre_inference.v1.attention.backends.spyre_attn import (
        _create_compilable_page_attn,
    )

    if variant == "baseline":
        return _create_compilable_page_attn(
            cfg["num_blocks"],
            cfg["padded_query_len"],
            cfg["num_heads"],
            cfg["num_kv_heads"],
            cfg["head_size"],
        )
    if variant == "sink-v":
        return _sink_v_kernel(cfg)
    if variant == "v-transposed":
        return _v_transposed_kernel(cfg)
    if variant == "fold-gqa":
        return _fold_gqa_kernel(cfg)
    if variant == "kv-head-major":
        return _kv_head_major_kernel(cfg)
    if variant == "mixed-layout":
        return _mixed_layout_kernel(cfg)
    raise SystemExit(f"unknown variant {variant!r}")


def _sink_v_kernel(cfg: dict):
    """Baseline with the V gather sunk to just before the P@V matmul.

    Identical arithmetic; the only change is that the V page tile is produced
    after the softmax chain instead of before it, cutting its live range from
    the whole iteration to a single op. Tests the capacity hypothesis: if the
    baseline's V spill reason is "no room on scratchpad", this is the fix.
    """
    import torch

    num_blocks = cfg["num_blocks"]
    padded_query_len = cfg["padded_query_len"]
    num_heads = cfg["num_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_size = cfg["head_size"]
    num_queries_per_kv = num_heads // num_kv_heads

    def kernel(query, query_row_index, k_pages, v_pages, page_index_table, mask_tiles, scale):
        q_rows = query.index_select(0, query_row_index[:padded_query_len])
        q = (
            q_rows.unsqueeze(0)
            .transpose(1, 2)
            .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
        )
        tile_max = tile_sum = tile_output = None
        for i in range(num_blocks):
            page_idx = page_index_table[i, 0:1]
            k_page = k_pages.index_select(0, page_idx)
            k_page_4d = k_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)
            mask_tile = mask_tiles[i]

            scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
            scores = scores + mask_tile
            scores_max = torch.amax(scores, dim=-1, keepdim=True)

            v_page = v_pages.index_select(0, page_idx)
            v_page_4d = v_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)

            if i == 0:
                tile_max = scores_max
                tile_probs = torch.exp(scores - tile_max)
                tile_output = torch.matmul(tile_probs, v_page_4d)
                tile_sum = tile_probs.sum(dim=-1, keepdim=True)
            else:
                new_max = torch.maximum(tile_max, scores_max)
                rescale = torch.exp(tile_max - new_max)
                tile_output = tile_output * rescale
                tile_sum = tile_sum * rescale
                tile_probs = torch.exp(scores - new_max)
                tile_output += torch.matmul(tile_probs, v_page_4d)
                tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
                tile_max = new_max
        attn = tile_output / tile_sum
        attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
        return attn.reshape(padded_query_len, num_heads, head_size)

    return kernel


def _v_transposed_kernel(cfg: dict):
    """Baseline, but V enters its matmul through the same transpose K does.

    K is consumed as ``permute(1,0,2) ... .transpose(-2,-1)`` and its clone lands
    in LX; V is consumed as a bare ``permute(1,0,2)`` and its clone does not.
    Here V is permuted to [kv, 1, head, block] and handed to matmul transposed,
    so both operands reach their matmul by the same route. Arithmetic is
    unchanged: (P @ V) with V read as the transpose of its transpose.
    """
    import torch

    num_blocks = cfg["num_blocks"]
    padded_query_len = cfg["padded_query_len"]
    num_heads = cfg["num_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_size = cfg["head_size"]
    num_queries_per_kv = num_heads // num_kv_heads

    def kernel(query, query_row_index, k_pages, v_pages, page_index_table, mask_tiles, scale):
        q_rows = query.index_select(0, query_row_index[:padded_query_len])
        q = (
            q_rows.unsqueeze(0)
            .transpose(1, 2)
            .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
        )
        tile_max = tile_sum = tile_output = None
        for i in range(num_blocks):
            page_idx = page_index_table[i, 0:1]
            k_page = k_pages.index_select(0, page_idx)
            v_page = v_pages.index_select(0, page_idx)
            k_page_4d = k_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)
            # [block, kv, head] -> [kv, head, block]; the matmul transposes it back.
            v_page_t = v_page.squeeze(0).permute(1, 2, 0).unsqueeze(1)
            mask_tile = mask_tiles[i]

            scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
            scores = scores + mask_tile
            scores_max = torch.amax(scores, dim=-1, keepdim=True)

            if i == 0:
                tile_max = scores_max
                tile_probs = torch.exp(scores - tile_max)
                tile_output = torch.matmul(tile_probs, v_page_t.transpose(-2, -1))
                tile_sum = tile_probs.sum(dim=-1, keepdim=True)
            else:
                new_max = torch.maximum(tile_max, scores_max)
                rescale = torch.exp(tile_max - new_max)
                tile_output = tile_output * rescale
                tile_sum = tile_sum * rescale
                tile_probs = torch.exp(scores - new_max)
                tile_output += torch.matmul(tile_probs, v_page_t.transpose(-2, -1))
                tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
                tile_max = new_max
        attn = tile_output / tile_sum
        attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
        return attn.reshape(padded_query_len, num_heads, head_size)

    return kernel


def _fold_gqa_kernel(cfg: dict):
    """Baseline with the GQA broadcast folded into the query axis.

    The baseline reshapes q to [kv, num_queries_per_kv, q, head] and unsqueezes
    each page to [kv, 1, block, head], so both matmul operands broadcast over
    num_queries_per_kv -- and the broadcast is *materialized*: a 256 KB gather
    output becomes a 1 MB clone. Here q is folded to
    [kv, 1, num_queries_per_kv * q, head] instead, so the page tiles are already
    the right rank and nothing broadcasts. Same arithmetic, same output memory
    order (the folded axes are contiguous in the same order), 4x less to pin.
    """
    import torch

    num_blocks = cfg["num_blocks"]
    padded_query_len = cfg["padded_query_len"]
    num_heads = cfg["num_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_size = cfg["head_size"]
    num_queries_per_kv = num_heads // num_kv_heads
    folded_q = num_queries_per_kv * padded_query_len

    def kernel(query, query_row_index, k_pages, v_pages, page_index_table, mask_tiles, scale):
        q_rows = query.index_select(0, query_row_index[:padded_query_len])
        q = (
            q_rows.unsqueeze(0)
            .transpose(1, 2)
            .reshape(num_kv_heads, 1, folded_q, head_size)
        )
        tile_max = tile_sum = tile_output = None
        for i in range(num_blocks):
            page_idx = page_index_table[i, 0:1]
            k_page = k_pages.index_select(0, page_idx)
            v_page = v_pages.index_select(0, page_idx)
            k_page_4d = k_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)
            v_page_4d = v_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)

            # [q, block] masks the folded axis by broadcast when q == 1; otherwise
            # tile it, outer-major, to match the (per_kv, q) fold order.
            mask_tile = mask_tiles[i]
            if padded_query_len != 1:
                mask_tile = mask_tile.repeat(num_queries_per_kv, 1)

            scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
            scores = scores + mask_tile
            scores_max = torch.amax(scores, dim=-1, keepdim=True)

            if i == 0:
                tile_max = scores_max
                tile_probs = torch.exp(scores - tile_max)
                tile_output = torch.matmul(tile_probs, v_page_4d)
                tile_sum = tile_probs.sum(dim=-1, keepdim=True)
            else:
                new_max = torch.maximum(tile_max, scores_max)
                rescale = torch.exp(tile_max - new_max)
                tile_output = tile_output * rescale
                tile_sum = tile_sum * rescale
                tile_probs = torch.exp(scores - new_max)
                tile_output += torch.matmul(tile_probs, v_page_4d)
                tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
                tile_max = new_max
        attn = tile_output / tile_sum
        attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
        return attn.reshape(padded_query_len, num_heads, head_size)

    return kernel


def _kv_head_major_kernel(cfg: dict):
    """Baseline over a head-major KV cache: [pages, kv, head, block].

    ``v-transposed`` failed because inductor canonicalizes views *between* the
    gather and the matmul. This changes the gather itself: with the cache stored
    head-major, each page arrives as [kv, head, block] and needs no permute at
    all -- K feeds its matmul directly, V feeds its own through one transpose.
    Requires the caller to pass caches in this layout (the probe does; production
    would need ``get_kv_cache_shape`` and the store path to match).
    """
    import torch

    num_blocks = cfg["num_blocks"]
    padded_query_len = cfg["padded_query_len"]
    num_heads = cfg["num_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_size = cfg["head_size"]
    num_queries_per_kv = num_heads // num_kv_heads

    def kernel(query, query_row_index, k_pages, v_pages, page_index_table, mask_tiles, scale):
        q_rows = query.index_select(0, query_row_index[:padded_query_len])
        q = (
            q_rows.unsqueeze(0)
            .transpose(1, 2)
            .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
        )
        tile_max = tile_sum = tile_output = None
        for i in range(num_blocks):
            page_idx = page_index_table[i, 0:1]
            # [1, kv, head, block] -> [kv, 1, head, block]; no permute.
            k_page_4d = k_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
            v_page_4d = v_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
            mask_tile = mask_tiles[i]

            scores = torch.matmul(q, k_page_4d) * scale
            scores = scores + mask_tile
            scores_max = torch.amax(scores, dim=-1, keepdim=True)

            if i == 0:
                tile_max = scores_max
                tile_probs = torch.exp(scores - tile_max)
                tile_output = torch.matmul(tile_probs, v_page_4d.transpose(-2, -1))
                tile_sum = tile_probs.sum(dim=-1, keepdim=True)
            else:
                new_max = torch.maximum(tile_max, scores_max)
                rescale = torch.exp(tile_max - new_max)
                tile_output = tile_output * rescale
                tile_sum = tile_sum * rescale
                tile_probs = torch.exp(scores - new_max)
                tile_output += torch.matmul(tile_probs, v_page_4d.transpose(-2, -1))
                tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
                tile_max = new_max
        attn = tile_output / tile_sum
        attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
        return attn.reshape(padded_query_len, num_heads, head_size)

    return kernel


def _mixed_layout_kernel(cfg: dict):
    """K token-major, V head-major -- so *both* operands are the transposed one.

    ``baseline`` and ``kv-head-major`` are mirror images: whichever operand
    reaches its matmul through ``.transpose(-2,-1)`` gets the core division its
    consuming matmul wants and lands in LX, and the operand consumed directly
    does not. K is transposed under a token-major cache, V under a head-major
    one -- and K and V are separate tensors in ``SpyrePagedKVCache``, so each can
    have the layout that makes it the transposed operand.

    k_pages: [pages, block, kv, head]   (unchanged)
    v_pages: [pages, kv, head, block]   (transposed at store time)
    """
    import torch

    num_blocks = cfg["num_blocks"]
    padded_query_len = cfg["padded_query_len"]
    num_heads = cfg["num_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_size = cfg["head_size"]
    num_queries_per_kv = num_heads // num_kv_heads

    def kernel(query, query_row_index, k_pages, v_pages, page_index_table, mask_tiles, scale):
        q_rows = query.index_select(0, query_row_index[:padded_query_len])
        q = (
            q_rows.unsqueeze(0)
            .transpose(1, 2)
            .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
        )
        tile_max = tile_sum = tile_output = None
        for i in range(num_blocks):
            page_idx = page_index_table[i, 0:1]
            k_page = k_pages.index_select(0, page_idx)
            k_page_4d = k_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)
            # Already head-major: [1, kv, head, block] -> [kv, 1, head, block].
            v_page_4d = v_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
            mask_tile = mask_tiles[i]

            scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
            scores = scores + mask_tile
            scores_max = torch.amax(scores, dim=-1, keepdim=True)

            if i == 0:
                tile_max = scores_max
                tile_probs = torch.exp(scores - tile_max)
                tile_output = torch.matmul(tile_probs, v_page_4d.transpose(-2, -1))
                tile_sum = tile_probs.sum(dim=-1, keepdim=True)
            else:
                new_max = torch.maximum(tile_max, scores_max)
                rescale = torch.exp(tile_max - new_max)
                tile_output = tile_output * rescale
                tile_sum = tile_sum * rescale
                tile_probs = torch.exp(scores - new_max)
                tile_output += torch.matmul(tile_probs, v_page_4d.transpose(-2, -1))
                tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
                tile_max = new_max
        attn = tile_output / tile_sum
        attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
        return attn.reshape(padded_query_len, num_heads, head_size)

    return kernel


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def run_standalone(args, cfg: dict, out_path: str) -> None:
    import torch

    import lx_probe_hook
    from spyre_inference.custom_ops import utils as custom_op_utils
    from spyre_inference.custom_ops.utils import convert
    from spyre_inference.v1.attention.backends.spyre_attn import INT32_ELEMS_PER_STICK

    custom_op_utils.register()
    apply_config_overrides(args)
    lx_probe_hook.install(out_path)

    dtype = torch.float16
    torch.manual_seed(0)
    device = torch.device("spyre")

    nb, bs = cfg["num_blocks"], cfg["block_size"]
    nkv, hs, nh = cfg["num_kv_heads"], cfg["head_size"], cfg["num_heads"]
    pq = cfg["padded_query_len"]

    query_cpu = torch.randn(pq, nh, hs, dtype=dtype)
    row_index_cpu = torch.zeros(INT32_ELEMS_PER_STICK, dtype=torch.int32)
    k_cpu = torch.randn(cfg["total_pages"], bs, nkv, hs, dtype=dtype)
    v_cpu = torch.randn(cfg["total_pages"], bs, nkv, hs, dtype=dtype)
    table = torch.zeros(nb, INT32_ELEMS_PER_STICK, dtype=torch.int32)
    table[:, 0] = torch.arange(nb, dtype=torch.int32)
    mask_cpu = [torch.zeros(pq, bs, dtype=dtype) for _ in range(nb)]

    query = convert(query_cpu, device=device)
    row_index = convert(row_index_cpu, device=device)
    # [pages, block, kv, head] -> [pages, kv, head, block] for whichever cache the
    # variant wants head-major; the reference below keeps the token-major copies
    # so the comparison stays honest.
    head_major = {
        "kv-head-major": ("k", "v"),
        "mixed-layout": ("v",),
    }.get(args.variant, ())
    k_dev_cpu = k_cpu.permute(0, 2, 3, 1).contiguous() if "k" in head_major else k_cpu
    v_dev_cpu = v_cpu.permute(0, 2, 3, 1).contiguous() if "v" in head_major else v_cpu
    # Algebra check, CPU vs CPU: does this variant compute the same function as
    # the shipped kernel? Kept separate from the device comparison below, which
    # is confounded by a harness fidelity bug that mismatches on baseline too.
    ref_cpu = build_kernel("baseline", cfg)(
        query_cpu, row_index_cpu, k_cpu, v_cpu, table, mask_cpu, hs**-0.5
    ).float()
    var_cpu = build_kernel(args.variant, cfg)(
        query_cpu, row_index_cpu, k_dev_cpu, v_dev_cpu, table, mask_cpu, hs**-0.5
    ).float()
    a_diff = (var_cpu - ref_cpu).abs().max().item()
    a_scale = ref_cpu.abs().max().item() or 1.0
    print(f"algebra (CPU vs CPU baseline): max abs diff {a_diff:.4g} "
          f"(rel {a_diff / a_scale:.2g}) -> "
          f"{'OK' if a_diff / a_scale < 1e-3 else 'DIFFERENT FUNCTION'}")

    k_pages = convert(k_dev_cpu, device=device)
    v_pages = convert(v_dev_cpu, device=device)
    page_index_table = convert(table, device=device)
    mask_tiles = [convert(m, device=device) for m in mask_cpu]

    kernel = build_kernel(args.variant, cfg)
    compiled = torch.compile(kernel, dynamic=False)
    out = compiled(
        query, row_index, k_pages, v_pages, page_index_table, mask_tiles, hs**-0.5
    )
    print(f"kernel ran, output {tuple(out.shape)} on {out.device}")

    # A variant that changes the numbers is not a fix. Compare against the
    # shipped kernel run eagerly on CPU over the same inputs.
    ref_kernel = build_kernel("baseline", cfg)
    ref = ref_kernel(
        query_cpu, row_index_cpu, k_cpu, v_cpu, table, mask_cpu, hs**-0.5
    )
    got = out.cpu().float()
    diff = (got - ref.float()).abs().max().item()
    scale_ref = ref.float().abs().max().item() or 1.0
    ok = diff / scale_ref < 2e-2
    print(f"vs eager CPU baseline: max abs diff {diff:.4g} "
          f"(rel {diff / scale_ref:.2g}) -> {'OK' if ok else 'MISMATCH'}\n")

    lx_probe_hook.write_report()
    report(out_path, args)


def run_bench(args, out_path: str) -> None:
    import torch  # noqa: F401  (backend autoload must happen before torch_spyre)

    import lx_probe_hook

    apply_config_overrides(args)
    lx_probe_hook.install(out_path)

    # The hook patches this process, so the engine core must not be a subprocess.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = list(BENCH_ARGV)
    try:
        vllm_main()
    except SystemExit:
        pass
    lx_probe_hook.write_report()
    report(out_path, args)


def apply_config_overrides(args) -> None:
    from torch_spyre._inductor import config as tsconfig

    if args.allow_all_lx:
        tsconfig.allow_all_ops_in_lx_planning = True
    if args.no_relayout:
        tsconfig.lx_planner_relayout = False
    if args.co_opt:
        # Joint work-division + LX solver. The direct lever when a buffer loses
        # LX to a producer/consumer core-division disagreement.
        tsconfig.co_optimizing_lx_planning = True
    if args.layout_solver:
        tsconfig.layout_solver = args.layout_solver


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def report(out_path: str, args) -> None:
    with open(out_path) as f:
        payload = json.load(f)

    graphs = [g for g in payload["graphs"] if g.get("lanes")]
    errored = [g for g in payload["graphs"] if g.get("error")]
    for g in errored:
        print(f"!! instrumentation error: {g['error']}")

    if not graphs:
        print("No graph with a classifiable K/V lane was seen.")
        print("Graphs recorded:", len(payload["graphs"]))
        if not payload["graphs"]:
            print("Zero graphs means LX planning never ran -- almost always a warm\n"
                  "inductor cache. Drop --warm-cache, or clear TORCHINDUCTOR_CACHE_DIR.")
        for g in payload["graphs"][:4]:
            print("  op kinds:", ", ".join(g.get("op_kinds", [])[:24]))
        return

    print(f"=== {len(graphs)} paged-attention graph(s) ===")
    for gi, g in enumerate(graphs):
        by_name = {b["name"]: b for b in g["buffers"]}
        print(f"\n--- graph {gi}: {g['num_ops']} ops, kv inputs {g.get('kv_roots')} ---")
        if g.get("lane_conflict"):
            print("  !! lane classification is ambiguous; treat the rows below with care")
        for lane in ("K", "V"):
            info = g["lanes"].get(lane)
            if not info:
                print(f"  {lane}: no lane identified")
                continue
            print(f"  {lane} lane   root input(s): {', '.join(info['roots'])}")
            verdicts = Counter()
            reasons = Counter()
            for chain in info["chains"]:
                tile = by_name.get(chain["operand"], {})
                v = tile.get("verdict", "?")
                if tile.get("demoted"):
                    v = "hbm(demoted)"
                verdicts[v] += 1
                reasons[tile.get("demoted") or tile.get("reason") or "-"] += 1
            total = info["num_tiles"]
            summary = ", ".join(f"{n}/{total} {k}" for k, n in verdicts.most_common())
            print(f"    matmul operand tiles: {summary}")
            for r, n in reasons.most_common():
                if r != "-":
                    print(f"      x{n}  {r}")
            # First chain in full: shows every intermediate between gather and matmul.
            chain = info["chains"][0]["chain"]
            print("    chain (first tile):")
            for name in chain:
                b = by_name.get(name, {})
                mark = "LX " if b.get("verdict") == "lx" else "HBM"
                if b.get("demoted"):
                    mark = "DEM"
                size = b.get("size_bytes")
                size_s = f"{size / 1024:8.1f} KB" if size else "        ? KB"
                life = b.get("lifetime") or []
                span = f"t={life[0]}-{life[-1]}" if life else "t=?"
                why = b.get("demoted") or b.get("reason") or ""
                shape = b.get("shape")
                shape_s = "x".join(str(x) for x in shape) if shape else "?"
                print(
                    f"      {mark} {name:>10} {b.get('op','?'):<16}{size_s} {span:>10}"
                    f"  [{shape_s}]  {why}"
                )
                for d in b.get("divisions", []):
                    dims = d.get("work_slice_dims")
                    dims_s = (
                        "x".join(f"dim{dd}/{ff}" for dd, ff in dims) if dims else "none"
                    )
                    print(
                        f"           {d.get('role','?'):<5} by {d.get('op','?'):<8}"
                        f" cores={dims_s}"
                    )

    verdict_lines(graphs)


def verdict_lines(graphs) -> None:
    print("\n=== verdict ===")
    agg = {lane: Counter() for lane in ("K", "V")}
    reasons = {lane: Counter() for lane in ("K", "V")}
    for g in graphs:
        by_name = {b["name"]: b for b in g["buffers"]}
        for lane in ("K", "V"):
            info = g["lanes"].get(lane)
            if not info:
                continue
            for chain in info["chains"]:
                b = by_name.get(chain["operand"], {})
                resident = b.get("verdict") == "lx" and not b.get("demoted")
                agg[lane]["lx" if resident else "hbm"] += 1
                if not resident:
                    reasons[lane][b.get("demoted") or b.get("reason") or "?"] += 1
    for lane in ("K", "V"):
        c = agg[lane]
        total = sum(c.values())
        if not total:
            print(f"{lane}: no tiles classified")
            continue
        print(f"{lane}: {c['lx']}/{total} page tiles LX-resident")
        for r, n in reasons[lane].most_common():
            kind = "CAPACITY" if r.startswith("no room on scratchpad") else "DECLARED"
            print(f"   [{kind}] x{n}  {r}")
    print(
        "\nCAPACITY  -> shorten the tile's live range or shrink the working set"
        "\n            (try --variant sink-v)"
        "\nDECLARED  -> structural veto in lowering/layout; the reason names the gate"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["standalone", "bench"], default="standalone")
    p.add_argument(
        "--variant",
        default="baseline",
        help="baseline | sink-v | v-transposed | fold-gqa | kv-head-major "
        "| mixed-layout (the one that gets both lanes resident)",
    )
    p.add_argument("--num-blocks", type=int, default=DEFAULTS["num_blocks"])
    p.add_argument("--block-size", type=int, default=DEFAULTS["block_size"])
    p.add_argument("--padded-query-len", type=int, default=DEFAULTS["padded_query_len"])
    p.add_argument("--num-kv-heads", type=int, default=DEFAULTS["num_kv_heads"])
    p.add_argument("--num-heads", type=int, default=DEFAULTS["num_heads"])
    p.add_argument("--head-size", type=int, default=DEFAULTS["head_size"])
    p.add_argument("--allow-all-lx", action="store_true")
    p.add_argument("--no-relayout", action="store_true")
    p.add_argument("--co-opt", action="store_true",
                   help="joint work-division + LX solver (co_optimizing_lx_planning)")
    p.add_argument("--layout-solver", default=None)
    p.add_argument("--warm-cache", action="store_true",
                   help="do not force a cold inductor cache (LX planning may be skipped)")
    p.add_argument("--json", default=None, help="where to write the raw report")
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    for k in ("num_blocks", "block_size", "padded_query_len", "num_kv_heads",
              "num_heads", "head_size"):
        cfg[k] = getattr(args, k)

    out_path = args.json or os.path.join(
        os.environ.get("CLAUDE_JOB_DIR", "/tmp"), f"lx_residency_{args.variant}.json"
    )

    # A warm inductor cache skips LX planning entirely, so the probe would see
    # no decisions at all. Every run compiles from scratch unless asked not to.
    if not args.warm_cache:
        cache_dir = tempfile.mkdtemp(prefix="lx_probe_inductor_")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        os.environ["SPYRE_INDUCTOR_SDSC_CACHE"] = "0"

    print(f"mode={args.mode} variant={args.variant} report={out_path}")

    if args.mode == "standalone":
        run_standalone(args, cfg, out_path)
    else:
        run_bench(args, out_path)


if __name__ == "__main__":
    main()
