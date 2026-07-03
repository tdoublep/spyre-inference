# Roadmap

Free-form strategic memory owned by the Orchestrator. Update every round
before choosing the next task. The framework reads this file back into
your next prompt verbatim — format however you like, but keep entries
scannable.

Conventions:
- **Major** — 1-3-round structural changes (e.g. "Fuse RMSNorm and residual
  in `custom_ops/rms_norm.py`", "Increase KV-cache alignment bucket to 512").
- **Minor** — 1-round tweaks / bug fixes / gates.
- Status: `todo` / `in_progress` / `done` / `parked` (bug — direction still
  fine) / `abandoned` (direction is wrong here — needs a mechanism-level
  reason, not just "didn't help").
- One-line *why* for each item (which bottleneck it targets).

## Bottleneck model (single-prompt decode, `--num-prompts 1`)

Decode steps have `max_query_len=1`, `num_seqs=1`, `num_blocks_needed=1`
(post-M7). Post-r8 the remaining hot-path costs are:

- **[M3a target] CPU staging buffer in `_online_softmax_attention`.**
  `output_cpu = torch.zeros_like(output, device="cpu")` at
  `spyre_attn.py:895` allocates a fresh CPU staging tensor every layer
  per step. Result flows: Spyre-side `result` → CPU-side `result_cpu`
  (reshape/transpose there — Spyre transpose+contiguous is broken) →
  writes into `output_cpu[q_start:q_end]` → `output.copy_(convert(
  output_cpu, ...))` bulk H2D at end. **For num_seqs=1 (bench case)**
  the staging buffer holds one row (`q_start=0`, `q_end=1` == `query_len`
  for decode), so `output_cpu` is literally a copy of `result_cpu`. The
  staging + memcpy + push adds one CPU alloc + one CPU memcpy per layer
  per step — 26 layers × 120 steps = 3,120 redundant allocations. Direct
  H2D of `result_cpu[0, :query_len]` into `output` skips both.
- **[M6c candidate] QKV D→H at `custom_ops/linear.py:128`.** QKV output
  pulled to CPU because downstream `.split()` on strided Spyre views is
  broken. Structural change; requires rewriting the split or accepting
  a torch-spyre limitation.
- **[Minor] RMSNorm `torch.full(x.shape, ...)`** — size-matched broadcast
  crutch; mechanism-level probe on Spyre scalar/0-d broadcast support.
- **[Stale comment] spyre_attn.py:429** still says "rounded up to
  KV_LENGTH_ALIGNMENT (256)"; alignment is 512 since r7. Cosmetic.

## Meta-pattern observations

Successful rounds have all fit one of two templates:
1. **Cache-what's-redundant-across-N-invocations** (M2, M5, M6a).
2. **Reduce compile-bucket variants materialized during a run** (M4, M7).

M3a fits template 1: `output_cpu` is allocated fresh 3,120× per bench
for identical purpose. Same pattern as M2/M6a — the fix is to short-
circuit the staging when the general-varlen scaffolding isn't needed.

## Measurement (fixed r3, sharpened r4-r7)

Warmup + 120-token bench. Cross-session drift is real (~10-25% per
triplet through the machine day). Same-session A/B (two triplets each,
second-triplet ratio as primary signal per r5-r7 lessons) is the
reliable verdict. Judge accepts "primary median ≥ 0.7315 baseline OR
same-session A/B ≥ 1.02×" alternative-pathway rule.

## Learnings from previous rounds

- **r1 (M1, done):** metadata-builder Q=1 guard.
- **r2 (M1', done):** dedicated Q=1 decode kernel with single global
  softmax. Decode path fallback-free.
- **r3 (M0, done):** bench methodology fix. Baseline 0.7315 tok/s.
- **r4 (M2, done):** cache mask-tile H2D across layers within a step.
  Same-session +9.8%.
- **r5 (M5, done):** precompute per-seq scalars + page_indices. +5-10%.
- **r6 (M6a, done):** cache RMSNorm weight on Spyre. Clean, drift-
  dominated signal.
- **r7 (M4, done):** `KV_LENGTH_ALIGNMENT` 256 → 512. Same-session
  +17.9% (second-triplet), +5.9% (first-triplet).
- **r8 (M7, done):** platform `block_size >= 128`. Same-session +15.2%
  (second-triplet). Every decode step now hits the `num_blocks == 1`
  fast path.
- **Meta-learning:** compile-bucket count reduction is a real end-to-
  end win invisible to per-op timing but shows up in wall-clock.

## Major

- **[done] M0: Bench methodology.** Warmup + 120-token measured.
- **[done] M1: Q=1 metadata guard.**
- **[done] M1': Q=1 decode kernel with single global softmax.**
- **[done] M2: Cache mask tiles across layers per step.** +9.8%.
- **[done] M5: Precompute per-seq scalars in metadata builder.** +5-10%.
- **[done] M6a: Cache RMSNorm weight on Spyre.**
- **[done] M4: `KV_LENGTH_ALIGNMENT` 256 → 512.** +17.9% (T2).
- **[done] M7: Force `block_size >= 128`.** +15.2% (T2).

- **[done] M3a (r9): Skip CPU staging buffer in `_online_softmax_attention`
  when num_seqs == 1.** Guarded the `torch.zeros_like(output, device="cpu")`
  allocation, the `output_cpu[q_start:q_end] = ...` scatter, and the
  trailing bulk-H2D behind `if num_seqs > 1:` blocks. Added a
  `num_seqs == 1` branch that pushes `result_cpu[0, :query_len, :, :]`
  directly into `output` via one H2D per attention layer per step.
  Multi-seq path unchanged. Also bundled the stale comment fix at
  spyre_attn.py:429 (`KV_LENGTH_ALIGNMENT (256)` → `(512)`).
  Same-session A/B second-triplet ratio (primary signal):
  M3a/r8-end = 0.8985/0.7679 = **1.170× (+17.0%)**. Primary bench
  median on HEAD 0.8130 tok/s = 1.11× the 0.7315 methodology floor.

- **[todo] M3: Full CPU-staging-buffer removal.** Broader than M3a —
  would eliminate the staging even for num_seqs > 1 by using an on-
  Spyre write primitive. Blocked on `torch.ops.spyre.overwrite`
  reliability probe. M3a captures the bench-case win without the
  mechanism risk.
- **[todo] M6b: Sweep remaining custom_ops.** Low expected value.
- **[todo] M6c: QKV `.split()` on Spyre.** Structural, requires torch-
  spyre op-support probe.

## Minor

- **[done] Fix stale comment at spyre_attn.py:429** — bundled into
  r9/M3a. `KV_LENGTH_ALIGNMENT (256)` → `(512)`.
- **[todo] Prefill `reshape_and_cache` Python-unrolled loop** —
  bench doesn't spend meaningful time in prefill.
- **[todo] RMSNorm `torch.full(x.shape, ...)` epsilon buffer** —
  size-matched broadcast crutch. Mechanism-level probe needed.

## Done

- M0 (r3): bench warmup + 120-token measured. New baseline 0.7315 tok/s.
- M1 (r1): metadata-builder Q=1 guard.
- M1' (r2): Q=1 decode kernel (single global softmax, no per-block max).
- M2 (r4): mask-tile H2D caching across L layers per step (+9.8%).
- M5 (r5): metadata-builder precompute for CPU-tensor scalars (+5-10%).
- M6a (r6): RMSNorm weight caching on Spyre (mechanically clean; drift-
  dominated signal).
- M4 (r7): `KV_LENGTH_ALIGNMENT` 256 → 512. Same-session +17.9%
  (second-triplet), +5.9% (first-triplet).
- M7 (r8): platform-level `block_size >= 128` bump. Same-session
  second-triplet +15.2%; primary bench median 0.9670 tok/s.
- M3a (r9): skip CPU staging buffer in `_online_softmax_attention`
  when num_seqs == 1. Same-session second-triplet +17.0%; primary
  bench median 0.8130 tok/s. Also fixed stale KV_LENGTH_ALIGNMENT
  (256) → (512) comment.

## Parked

(none yet)

## Abandoned

(none yet)
