#!/usr/bin/env python3
"""Collate the micro-benchmark legs into the per-context-length comparison.

Usage: report.py [results_dir]

WARNING on baselines. The `main` and `off` legs use --kv-layout plain, i.e. the default
tiled device layout. Production does NOT allocate KV pages that way -- see
spyre_model_runner.py:1039, which uses slot_major_kv_layout for the unfolded cache. The
tiled layout makes the per-page gather cost the whole tensor, so it is ~10x slower than
the real baseline and its cost grows with num_blocks. A speedup quoted against `main`
here is inflated by roughly that factor. The production-faithful baseline is the
slot_major_devfill leg in results_devfill; see ATTN_MICROBENCH_FINDINGS.md.
"""

import glob
import sys

import pandas as pd

LEGS = ["main", "off", "on", "on-nogate", "on-32core"]
RESULTS = sys.argv[1] if len(sys.argv) > 1 else ".claude-scratch/mb/results"


def load(leg, span):
    hits = sorted(glob.glob(f"{RESULTS}/{leg}_{span}/**/*_final.csv", recursive=True))
    if not hits:
        return None
    df = pd.read_csv(hits[-1], sep="\t")
    df["leg"] = leg
    df["span_name"] = span
    return df


def main():
    for span in ["online_softmax", "forward"]:
        frames = [f for f in (load(leg, span) for leg in LEGS) if f is not None]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df["us"] = df["ms"] * 1000.0
        df["us_per_page"] = df["us"] / df["num_kv_blocks_iterated"]

        print(f"\n{'=' * 100}\nSPAN: {span}\n{'=' * 100}")
        if "main" in set(df["leg"]) and "plain" in set(df["kv_layout"].astype(str)):
            print(
                "\nNOTE: main/off are the `plain` (tiled) layout, ~10x slower than the\n"
                "slot-major layout production uses. Percentages below are against that\n"
                "inflated baseline -- use results_devfill for the real one."
            )

        bad = df[(~df["allclose_pass"]) | (~df["fallback_clean"]) | df["error"].notna()]
        if len(bad):
            print("\n!! rows failing correctness / fallback / with an error -- do not read these:")
            print(
                bad[
                    [
                        "leg",
                        "capture_name",
                        "allclose_pass",
                        "max_abs_diff",
                        "num_outliers",
                        "fallback_clean",
                        "error",
                    ]
                ].to_string(index=False)
            )

        us = df.pivot_table(index="max_seq_len", columns="leg", values="us")
        pages = df.groupby("max_seq_len")["num_kv_blocks_iterated"].first()
        mem = df.pivot_table(index="max_seq_len", columns="leg", values="memory_share_pct")

        order = [c for c in LEGS if c in us.columns]
        us, mem = us[order], mem[order]

        out = pd.DataFrame({"pages": pages})
        for leg in order:
            out[f"{leg}_us"] = us[leg].round(1)
        base = "main" if "main" in order else order[0]
        for leg in order:
            if leg == base:
                continue
            out[f"{leg}_vs_{base}_%"] = ((us[leg] / us[base] - 1) * 100).round(1)
        print("\ndevice us attributed to the span, and change vs the baseline:")
        print(out.to_string())

        perpage = pd.DataFrame({"pages": pages})
        for leg in order:
            perpage[f"{leg}"] = (us[leg] / pages).round(2)
        print("\ndevice us per KV page iterated (the README's normalisation):")
        print(perpage.to_string())

        print("\nmemory_share_pct (memcpy/memset/restickify share of span device time):")
        print(mem.round(1).to_string())

        if base in us.columns and "off" in us.columns:
            spread = ((us["off"] / us[base] - 1) * 100).abs()
            print(
                f"\nbaseline error bar (off vs {base}): max |delta| = {spread.max():.2f}% "
                f"across shapes (e2e A/B baseline spread was 0.43%)"
            )


if __name__ == "__main__":
    main()
