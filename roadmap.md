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
(post-M7). Post-r9 the remaining hot-path costs are:

- **[M8 target] RMSNorm shape-matched epsilon buffer.** `forward_spyre`
  (rms_norm.py:160-166) allocates `torch.full(x.shape, variance_epsilon,
  dtype=torch.float16, device=x.device)` — a full `[batch, hidden_size]`
  tensor filled with a single scalar constant, just so `variance +
  variance_epsilon_t` "matches shape" before `rsqrt`. Downstream this
  computes `hidden_size` copies of the same rsqrt value per batch row
  (redundant), and the intermediate tensor is 2× larger than necessary.
  Replacing with `variance + variance_epsilon` (Python float, or 0-d
  tensor) computes only `batch` rsqrt values before broadcasting.
  Same numerical result, less work per RMSNorm call. 52 calls/step ×
  120 steps = 6,240 RMSNorm invocations per bench, each currently doing
  hidden_size=2048 redundant rsqrts.
- **[M6c candidate] QKV D→H at `custom_ops/linear.py:128`.** QKV output
  pulled to CPU. Structural: requires either splitting on Spyre with
  contiguous materialization (comment says strided scatter breaks V) or
  keeping QKV on Spyre and reshaping Q there. Higher risk.
- **[M3 full] Multi-seq staging removal.** Not exercised by bench.
- **RMSNorm `torch.full(...)`** — see M8 above.

## Meta-pattern observations

Successful rounds have fit two templates:
1. **Cache-what's-redundant-across-N-invocations** (M2, M5, M6a, M3a).
2. **Reduce compile-bucket variants materialized during a run** (M4, M7).

M8 fits a new template: **shape-shrink of an intermediate tensor** so
that a downstream elementwise op does less work. Not "caching redundant
work" or "compile bucket reduction", but "eliminate FLOPs we didn't
need in the first place". Small tensor, small op, but 6,240 invocations.

## Measurement (fixed r3, sharpened r4-r9)

Warmup + 120-token bench. Cross-session drift is real (~10-25% per
triplet through the machine day). Same-session A/B (two triplets each,
second-triplet ratio as primary signal) is the reliable verdict. Judge
accepts "primary median ≥ 0.7315 baseline OR same-session A/B ≥ 1.02×"
alternative-pathway rule.

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
- **r7 (M4, done):** `KV_LENGTH_ALIGNMENT` 256 → 512. +17.9% (T2).
- **r8 (M7, done):** platform `block_size >= 128`. +15.2% (T2).
- **r9 (M3a, done):** skip CPU staging for `num_seqs == 1`. +17.0% (T2).
  Bundled comment fix.
- **Meta-learning:** compile-bucket count reduction and per-invocation
  redundant work are both real end-to-end wins invisible to per-op
  timing but shows up in wall-clock.

## Major

- **[done] M0: Bench methodology.** Warmup + 120-token measured.
- **[done] M1: Q=1 metadata guard.**
- **[done] M1': Q=1 decode kernel with single global softmax.**
- **[done] M2: Cache mask tiles across layers per step.** +9.8%.
- **[done] M5: Precompute per-seq scalars in metadata builder.** +5-10%.
- **[done] M6a: Cache RMSNorm weight on Spyre.**
- **[done] M4: `KV_LENGTH_ALIGNMENT` 256 → 512.** +17.9% (T2).
- **[done] M7: Force `block_size >= 128`.** +15.2% (T2).
- **[done] M3a: Skip CPU staging when num_seqs==1.** +17.0% (T2).

- **[in_progress] M8: Replace RMSNorm shape-matched epsilon buffer with a
  scalar broadcast.** *Removes 6,240 shape-matched-tensor allocations +
  6,240 hidden_size-fold redundant rsqrts per bench.* In
  `SpyreRMSNorm.forward_spyre` (rms_norm.py:136-173), replace:

  ```
  variance_epsilon_t = torch.full(
      x.shape, variance_epsilon, dtype=torch.float16, device=x.device
  )
  variance = x.pow(2).mean(dim=-1, keepdim=True)
  x = x * torch.rsqrt(variance + variance_epsilon_t)
  ```

  with:

  ```
  variance = x.pow(2).mean(dim=-1, keepdim=True)
  x = x * torch.rsqrt(variance + variance_epsilon)
  ```

  Where `variance_epsilon` is the Python float already passed in as a
  parameter. `variance` has shape `[batch, 1]`, adding a scalar
  broadcasts trivially, `rsqrt` now runs on a `[batch, 1]` tensor
  (batch reciprocal-sqrts instead of `batch × hidden_size`), and the
  final `x * rsqrt(...)` broadcasts `[batch, 1]` back to
  `[batch, hidden_size]` — same numerical result as the old code.

  Risks: the `torch.full` was a workaround per the module docstring
  ("Creates epsilon tensor via `torch.full()`" is listed as one of the
  key differences from upstream). If Spyre lacks scalar broadcast in
  add or `.rsqrt` on `[batch, 1]` triggers a fallback, pytest will
  catch it (numerical) or the FallbackWarning gate will (silent CPU
  routing). Revert if either fires.

- **[todo] M3: Full CPU-staging-buffer removal.**
- **[todo] M6b: Sweep remaining custom_ops.**
- **[todo] M6c: QKV `.split()` on Spyre.**

## Minor

- **[done] Fix stale comment at spyre_attn.py:429** — bundled into r9/M3a.
- **[todo] Prefill `reshape_and_cache` Python-unrolled loop.**

## Done

- M0 (r3): bench warmup + 120-token measured. New baseline 0.7315 tok/s.
- M1 (r1): metadata-builder Q=1 guard.
- M1' (r2): Q=1 decode kernel (single global softmax, no per-block max).
- M2 (r4): mask-tile H2D caching across L layers per step (+9.8%).
- M5 (r5): metadata-builder precompute for CPU-tensor scalars (+5-10%).
- M6a (r6): RMSNorm weight caching on Spyre.
- M4 (r7): `KV_LENGTH_ALIGNMENT` 256 → 512 (+17.9% T2).
- M7 (r8): platform `block_size >= 128` (+15.2% T2).
- M3a (r9): skip CPU staging when num_seqs==1 (+17.0% T2). Bundled
  stale-comment fix.

## Parked

(none yet)

## Abandoned

(none yet)
