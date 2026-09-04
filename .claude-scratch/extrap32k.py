#!/usr/bin/env python3
"""Extrapolate the attention share of decode latency to 32k context.

Inputs are all measured, from two places:

  * the micro-benchmark (.claude-scratch/ATTN_MICROBENCH_FINDINGS.md): device us in the
    online_softmax span, per KV page iterated, at the saturated end of the sweep. Both
    the production slot-major layout and the folded LX layout are linear in pages with
    no knee from 4 to 64 pages.
  * the end-to-end A/B (commit fd72d44): 18.03 s baseline mean vs 16.597 s with the
    folded layout at 3200 in / 64 out / bs 1, and ~5.1 tok/s generation throughput in
    the on-leg log, i.e. ~196 ms per decode step.

The micro-benchmark overstates absolute cost -- it profiles one layer in isolation with
AIUPTI overhead and no KV store. Rather than pretend otherwise, the overstatement is
calibrated out with a single multiplicative factor: the e2e A/B measured a 22.4 ms/step
saving where the micro-benchmark predicts 40.5 ms/step, so the factor is their ratio.
Both the uncalibrated and calibrated numbers are printed, and the truth is bracketed by
them -- the calibrated one is the better estimate, the uncalibrated one the upper bound.

Load-bearing assumption: attention cost stays linear in pages out to 256, which is 4x
beyond the measured range. Everything else per step (weight streaming above all) is
taken as context-independent, which is what makes the share grow.
"""

# --- measured: micro-benchmark, us per KV page per layer, saturated ---------------
US_PER_PAGE_SLOT_MAJOR = 66.5   # production baseline layout, 64 pages
US_PER_PAGE_FOLDED = 26.2       # folded LX layout, 64 pages

# --- measured: micro-benchmark at the e2e A/B's own context, us per layer --------
US_LAYER_SLOT_MAJOR_25P = 1679.6
US_LAYER_FOLDED_25P = 666.7

# --- model / hardware ------------------------------------------------------------
LAYERS = 40
BLOCK = 128
KV_HEADS = 8
HEAD_SIZE = 128
FP16 = 2
WEIGHTS_GB = 16.34
BW_GB_S = 204.8

# --- measured: end to end --------------------------------------------------------
STEP_MS_AT_3200 = 196.0         # from ~5.1 tok/s generation throughput, folded leg
E2E_SAVING_MS_PER_STEP = (18.03 - 16.597) * 1000 / 64


def attn_ms(pages, us_per_page, cal):
    return pages * us_per_page * LAYERS / 1000.0 * cal


def main():
    predicted = (US_LAYER_SLOT_MAJOR_25P - US_LAYER_FOLDED_25P) * LAYERS / 1000.0
    cal = E2E_SAVING_MS_PER_STEP / predicted
    print(f"e2e measured saving   : {E2E_SAVING_MS_PER_STEP:.1f} ms/step")
    print(f"microbench predicts   : {predicted:.1f} ms/step at 25 pages")
    print(f"=> calibration factor : {cal:.3f}\n")

    for label, c in (("uncalibrated (upper bound)", 1.0), ("calibrated (best estimate)", cal)):
        print(f"--- {label} ---")
        # Anchor the context-independent remainder at 3200 tokens / 25 pages.
        folded_3200 = US_LAYER_FOLDED_25P * LAYERS / 1000.0 * c
        other = STEP_MS_AT_3200 - folded_3200
        print(f"  attention at 3200 (25 pages), folded : {folded_3200:6.1f} ms  "
              f"({100 * folded_3200 / STEP_MS_AT_3200:.1f}% of a {STEP_MS_AT_3200:.0f} ms step)")
        print(f"  context-independent remainder        : {other:6.1f} ms\n")

        for ctx in (3200, 8192, 16384, 32768):
            pages = ctx / BLOCK
            a_fold = attn_ms(pages, US_PER_PAGE_FOLDED, c)
            a_slot = attn_ms(pages, US_PER_PAGE_SLOT_MAJOR, c)
            s_fold, s_slot = other + a_fold, other + a_slot
            print(f"  ctx {ctx:>6} ({pages:5.0f} pages): "
                  f"attn {a_fold:6.1f} ms = {100 * a_fold / s_fold:4.1f}% of {s_fold:6.1f} ms step"
                  f"   | unfolded {a_slot:6.1f} ms = {100 * a_slot / s_slot:4.1f}% of {s_slot:6.1f} ms"
                  f"   | layout win {100 * (1 - s_fold / s_slot):4.1f}%")
        print()

    # How far above its own bandwidth floor does attention sit? This is what says whether
    # there is headroom left inside attention at long context, or whether it is done.
    print("--- attention vs its own memory-bandwidth floor ---")
    w_floor = WEIGHTS_GB * 1000 / BW_GB_S
    print(f"  weight stream floor : {w_floor:.1f} ms/step ({WEIGHTS_GB} GB at {BW_GB_S} GB/s)")
    for ctx in (3200, 32768):
        kv_bytes = 2 * ctx * LAYERS * KV_HEADS * HEAD_SIZE * FP16
        kv_floor = kv_bytes / (BW_GB_S * 1e9) * 1000
        a_fold = attn_ms(ctx / BLOCK, US_PER_PAGE_FOLDED, cal)
        print(f"  ctx {ctx:>6}: KV re-read {kv_bytes / 1e9:5.2f} GB -> {kv_floor:5.1f} ms floor; "
              f"folded attention {a_fold:6.1f} ms = {a_fold / kv_floor:4.1f}x the floor")


if __name__ == "__main__":
    main()
