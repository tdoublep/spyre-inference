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

Decode steps have `max_query_len=1`. After r1-r5, query padding, per-block
argmax fallback, per-layer mask-tile H2D, per-layer scalar `.item()` sync,
and measurement noise are all gone. Remaining hot-path costs, ranked by
expected EV under the "cache-what's-redundant-across-decode-steps" heuristic:

- **[M6a target] Per-forward-call RMSNorm weight H2D.** In
  `custom_ops/rms_norm.py:201`, every RMSNorm forward transfers
  `self.weight.data` from CPU to Spyre via `convert()`. The weight is a
  frozen model Parameter — the transfer produces an identical Spyre-side
  tensor every call. A 26-layer model with 2 RMSNorms per layer (pre-attn,
  pre-mlp) executes 52 RMSNorm forwards per decode step; over a 120-token
  bench that's **6,240 redundant weight H2D transfers per bench**. Same
  cache-once-reuse-forever pattern that landed M2 (+9.8%). Weight is a
  Parameter so it's a valid class-attribute lifetime; not per-step
  metadata like M2, but per-layer-instance lifetime.
- **[Potential M6b] Same pattern for other model-weight custom ops** —
  need to audit `custom_ops/rotary_embedding.py`, `custom_ops/silu_and_mul.py`,
  `custom_ops/vocab_parallel_embedding.py`, `custom_ops/parallel_lm_head.py`
  for parameters that are re-transferred to Spyre every forward. Defer
  audit until M6a lands (want the M6a lift measurement clean before
  bundling further custom-op work).
- **`output_cpu = zeros_like(output, device="cpu")` staging round-trip**
  in `_online_softmax_attention` (spyre_attn.py:895). Per-layer per-step
  D→H+H→D round-trip on a small tensor. Code comment at 885-894 warns
  prior scattering attempts corrupted data — needs mechanism-level probe
  first, so M3 stays parked as risky.
- Compile bucket tiers at every 256-token `kv_len` boundary — post-M0
  warmup, negligible.
- Per-token `reshape_and_cache` unrolls a Python loop of length `num_tokens`;
  fine for decode (num_tokens=1), wasteful in prefill.

## Measurement (fixed r3, sharpened r4-r5)

Warmup + 120-token bench. CoV 9-19% cross-session but the shared host has
strong **session-warmup drift** (~10-25% acceleration per triplet through
the day). Same-session A/B comparison is the reliable signal — the
implementer must checkout the *previous round's plugin state*, run
three bench triplets, restore HEAD, and run three more. r5 judge's data:
- M5 cold triplet 1 (session start): median 0.7135
- r4-end mid-session triplet: median 0.8226
- M5 warm triplet 2 (session late): median 0.9017
Same monotonic drift, so raw first-triplet numbers under-report warm code.
r5 same-session A/B: 1.096× (+9.6%) confirmed the improvement.

For strict-floor criteria that survive cold-triplet warmup, either
require a **run-of-warmups** (throw away the first triplet, use the
second) or lean on the same-session A/B ratio as primary signal.

## Learnings from previous rounds

- **r1 (M1, done):** metadata-builder Q=1 guard. Correctness green.
- **r2 (M1', done):** dedicated Q=1 decode kernel with single global
  softmax. Decode path fallback-free.
- **r3 (M0, done):** bench methodology fix. New reproducible baseline
  0.7315 tok/s.
- **r4 (M2, done):** cache mask-tile H2D across layers within a step.
  Same-session +9.8%. Established the "cache-across-L-layers" pattern.
- **r5 (M5, done):** precompute per-seq scalars + page_indices in
  metadata builder. Same-session +5-10%. Extended the cache pattern to
  CPU-tensor `.item()` GIL round-trips.
- **General pattern:** every high-EV win so far has been *removing work
  that runs L layers × N steps times for identical result*. RMSNorm
  weight H2D fits that exact template (52 forwards/step × 120 steps =
  6,240 redundant transfers).

## Major

- **[done] M0: Fix bench measurement signal.** Warmup + 120-token bench.
- **[done] M1: Decode-only fast path (query_len==1).**
- **[done] M1': Q=1 decode kernel with single global softmax.**
- **[done] M2: Cache mask tiles across layers within a decode step.**
  +9.8% same-session.
- **[done] M5: Precompute per-seq scalars and page_indices in the metadata
  builder.** +5-10% same-session.

- **[in_progress] M6a: Cache RMSNorm weight on Spyre across forward calls.**
  *Removes ~6,240 redundant per-step weight H2D transfers per bench.*
  In `SpyreRMSNorm._forward_spyre_impl` (custom_ops/rms_norm.py:167),
  the `convert(self.weight.data, self._target_device, self._target_dtype)`
  at line 201 runs every forward call. Weight is a frozen Parameter, so
  the Spyre-side copy is invariant across calls. Store a Spyre-cached
  version as an instance attribute (populated lazily on first forward
  or eagerly in `__init__`); every subsequent forward reads the cached
  tensor. Include a defensive check that the underlying Parameter
  hasn't been re-assigned (compare `id(self.weight.data)`).

- **[todo] M3: Eliminate the CPU staging buffer in attention output.**
  *Removes one Spyre → CPU → Spyre round-trip per attention layer.*
  Blocked on mechanism-level probing of `torch.ops.spyre.overwrite` per
  spyre_attn.py:885-894. Higher risk than M2/M5/M6a.

- **[todo] M4: Try `KV_LENGTH_ALIGNMENT=512`.** *Fewer compile buckets.*
  One-line probe. Lower priority now that warmup absorbs compile cost.

- **[todo] M6b: Sweep remaining custom_ops for weight/parameter re-transfer
  patterns.** Candidates: `rotary_embedding.py` (cos/sin caches),
  `silu_and_mul.py`, `vocab_parallel_embedding.py`,
  `parallel_lm_head.py`. Do this after M6a to avoid bundling.

- **[todo] M6c: Attention `SpyreQKVParallelLinear` D→H at line 128 of
  `custom_ops/linear.py`.** QKV output is pulled to CPU because
  "downstream `.split()` cannot handle strided views on Spyre". If we
  could do the split on Spyre (or avoid the strided view), the QKV
  output could stay on device. Investigate — likely a bigger structural
  change than the M2/M5/M6a class.

## Minor

(none yet)

## Done

- M0 (r3): bench warmup + 120-token measured. New baseline 0.7315 tok/s.
- M1 (r1): metadata-builder Q=1 guard.
- M1' (r2): Q=1 decode kernel (single global softmax, no per-block max).
- M2 (r4): mask-tile H2D caching across L layers per step (+9.8%).
- M5 (r5): metadata-builder precompute for CPU-tensor scalars (+5-10%).

## Parked

(none yet)

## Abandoned

(none yet)
