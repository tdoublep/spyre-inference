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

Decode steps have `max_query_len=1`. After r1-r7, the following are handled:
query padding, per-block argmax fallback, per-layer mask-tile H2D, per-layer
scalar `.item()` sync, per-forward RMSNorm weight H2D, measurement noise,
and coarser KV alignment (M4's +17.9% surprise showed compile-bucket
churn was a hidden cost).

M4's success reframed the bottleneck model: **compile-bucket churn from
shape-varying kernel specializations is a hidden per-step cost** that's
invisible to per-op analysis but shows up in end-to-end tok/s. Any change
that reduces the number of distinct `(num_blocks, padded_query_len,
aligned_max_seq_len, ...)` combinations materialized across a run is a
candidate.

Remaining candidates:

- **[M7 target] `block_size` = 128** (currently 64, min stick alignment).
  Bench has `MAX_MODEL_LEN_CAP=128`, so with block_size=128 every decode
  step uses exactly ONE KV page — `num_blocks_needed = 1` always. The
  decode kernel has an explicit `num_blocks == 1` fast path
  (spyre_attn.py:308-320) that skips `_indirect_matmul_mock` dispatch
  and the concat/cat overhead. With block_size=64 the current bench
  crosses into `num_blocks == 2` at kv_len > 64 — 90% of the run's
  120 decode steps. Bumping block_size to 128 (a) collapses two decode
  kernel compile buckets to one, (b) keeps all decode steps on the
  fast path, (c) halves the number of KV pages allocated (each is
  larger but total memory unchanged). Same class of intervention as
  M4 — one hyperparameter change that reduces kernel variant count.
- **RMSNorm `torch.full(x.shape, ...)` epsilon buffer** — size-matched
  broadcast crutch; scalar or 0-d tensor may work. Mechanism-level probe.
- **QKV D→H at custom_ops/linear.py:128.** Structural change (M6c).
- **`output_cpu = zeros_like(output, device="cpu")` staging round-trip**
  (M3). Requires torch-spyre op-support probe.

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
  +17.9% (second-triplet), +5.9% (first-triplet). Halved distinct
  compile-bucket shapes materialized during 120-token bench.
- **Meta-learning:** compile-bucket count reduction is a real end-to-
  end win because each new bucket triggers a Spyre-side re-specialization
  that isn't visible in per-op timing but shows up in wall-clock.

## Major

- **[done] M0: Bench methodology.** Warmup + 120-token measured.
- **[done] M1: Q=1 metadata guard.**
- **[done] M1': Q=1 decode kernel with single global softmax.**
- **[done] M2: Cache mask tiles across layers per step.** +9.8%.
- **[done] M5: Precompute per-seq scalars in metadata builder.** +5-10%.
- **[done] M6a: Cache RMSNorm weight on Spyre.**
- **[done] M4: `KV_LENGTH_ALIGNMENT` 256 → 512.** +17.9% (T2).

- **[in_progress] M7: Force `block_size = 128` in
  `TorchSpyrePlatform.check_and_update_config`.** *Collapses decode-
  kernel compile buckets from {1, 2} to {1} for the whole 120-token
  bench.* One-line change in `platform.py:194-204`: after the
  "round up to multiple of 64" logic, add a Spyre-specific
  `cache_config.block_size = max(cache_config.block_size, 128)`
  (or set an explicit default of 128). With
  `MAX_MODEL_LEN_CAP=128` in the platform, every sequence fits in
  exactly one 128-token block, so `num_blocks_needed = 1` for the
  entire decode. That routes every step through the `num_blocks == 1`
  fast path in `_create_compilable_page_attn_decode` (spyre_attn.py:308)
  which skips the `_indirect_matmul_mock` dispatch + concat overhead.

- **[todo] M3: Eliminate the CPU staging buffer in attention output.**
  Blocked on mechanism-level probing of `torch.ops.spyre.overwrite`.
- **[todo] M6b: Sweep remaining custom_ops.** Low expected value.
- **[todo] M6c: QKV `.split()` on Spyre.** Structural, requires torch-
  spyre op support probe.

## Minor

- **[todo] Fix stale comment at spyre_attn.py:429** — still says
  "rounded up to KV_LENGTH_ALIGNMENT (256)" after r7 bump to 512.
  Cosmetic, noted by r7 judge. Bundle into a future round if convenient.
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
  (second-triplet), +5.9% (first-triplet). Halved distinct compile-
  bucket shapes materialized across a 120-token bench.

## Parked

(none yet)

## Abandoned

(none yet)
