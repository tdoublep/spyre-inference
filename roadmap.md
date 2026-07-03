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

Decode steps have `max_query_len=1`. After r1-r6, query padding, per-block
argmax fallback, per-layer mask-tile H2D, per-layer scalar `.item()` sync,
per-forward RMSNorm weight H2D, and measurement noise are all handled.
Remaining hot-path costs:

- **[M6b/M6c candidates]** RMSNorm `torch.full(x.shape, variance_epsilon, ...)`
  at rms_norm.py:160 allocates a full x-shaped fp16 buffer every forward
  call. This is a shape-matched broadcast crutch (comment says "instead of
  scalar"). If Spyre supports 0-d or scalar broadcast for that add, a
  size-1 tensor or plain scalar would eliminate the allocation. Needs
  mechanism-level probe of Spyre add-broadcast semantics.
- **QKV D→H at custom_ops/linear.py:128.** `SpyreQKVParallelLinear.forward`
  always pulls the merged QKV result to CPU so downstream `.split()` and
  `.view()` can run on CPU (Spyre `.split()` on strided views is broken).
  Every attention layer per decode step → D→H per layer per step.
  Downstream attention `_online_softmax_attention` then re-transfers Q
  back to Spyre (`q_dev = convert(q, ...)` at spyre_attn.py:932). Two
  crossings that could plausibly become zero if the split ran on Spyre.
  Structural change (M6c).
- **`output_cpu = zeros_like(output, device="cpu")` staging round-trip**
  in `_online_softmax_attention` (spyre_attn.py:895). Per-layer per-step
  D→H+H→D round-trip on a small tensor. Code comment at 885-894 warns
  prior scattering attempts corrupted data — needs mechanism-level probe.
  Marked M3, higher risk than the "cache-across-L-layers" class.
- Compile bucket tiers at every 256-token `kv_len` boundary — post-M0
  warmup, negligible on the timed run. **M4 probes the compile-tier
  dimension by bumping alignment 256→512.** Cheap one-line probe.
- Per-token `reshape_and_cache` unrolls a Python loop of length `num_tokens`;
  fine for decode (num_tokens=1), wasteful in prefill. Prefill runs once
  in warmup + once in the timed run's prompt processing — small relative
  weight vs 120 decode steps.

## Diminishing returns

The M2/M5/M6a series has landed the obvious "redundant work across L
layers × N steps" targets. Each successive round has smaller signal:
- M2: +9.8% same-session (mask tiles: ~6,240 H2D avoided, ~5-15KB each)
- M5: +5-10% same-session (CPU scalar syncs: ~15,600 GIL round-trips)
- M6a: drift-dominated (RMSNorm weights: ~6,240 H2D, but tensors small
  ~4KB, and shared-host drift envelope covers the win)

Remaining levers are either (a) risky mechanism-level probes (M3,
attention output staging; M6c, QKV split-on-Spyre), (b) cheap probes
of unlikely value (M4, KV alignment), or (c) audit-and-sweep passes
(M6b, other custom_ops).

## Measurement (fixed r3, sharpened r4-r6)

Warmup + 120-token bench. Cross-session drift is real (~10-25% per
triplet through the machine day). Same-session A/B is the reliable
signal. For strict-floor criteria that survive cold-triplet warmup:
either use a run-of-warmups (throw away the first triplet) or lean on
same-session A/B ratio as primary signal. r6 judge accepted the
"primary median ≥ baseline OR A/B ≥ 1.02×" alternative-pathway rule.

## Learnings from previous rounds

- **r1 (M1, done):** metadata-builder Q=1 guard. Correctness green.
- **r2 (M1', done):** dedicated Q=1 decode kernel with single global
  softmax. Decode path fallback-free.
- **r3 (M0, done):** bench methodology fix. New reproducible baseline
  0.7315 tok/s.
- **r4 (M2, done):** cache mask-tile H2D across layers within a step.
  Same-session +9.8%. Established the "cache-across-L-layers" pattern.
- **r5 (M5, done):** precompute per-seq scalars + page_indices in
  metadata builder. Same-session +5-10%.
- **r6 (M6a, done):** cache RMSNorm weight on Spyre across forward
  calls. Primary median 0.8477 tok/s (1.159× baseline); same-session
  A/B drift-dominated but the change is mechanistically clean.

## Major

- **[done] M0: Fix bench measurement signal.** Warmup + 120-token bench.
- **[done] M1: Decode-only fast path (query_len==1).**
- **[done] M1': Q=1 decode kernel with single global softmax.**
- **[done] M2: Cache mask tiles across layers within a decode step.**
  +9.8% same-session.
- **[done] M5: Precompute per-seq scalars and page_indices in the metadata
  builder.** +5-10% same-session.
- **[done] M6a: Cache RMSNorm weight on Spyre across forward calls.**

- **[in_progress] M4: Try `KV_LENGTH_ALIGNMENT=512`.** *Cheap probe of a
  dimension we haven't measured.* One-line change:
  `spyre_attn.py:48` `KV_LENGTH_ALIGNMENT = 256` → `512`. Halves
  compile-bucket count over kv_len (bench spans kv_len ~8 → 128, one
  compile tier at 256 currently, would still be one tier at 512).
  Also changes the mask-tile dimension along the KV axis (256 → 512
  elements padded) — that's actually a *cost* increase per mask tile,
  offset only by having fewer distinct tile shapes. For num_seqs=1,
  the tiles are already per-seq per-num_blocks — the alignment mostly
  affects `aligned_max_seq_len` used inside `_build_attention_mask`
  which builds the CPU-side mask. Net effect unknown; that's why it
  needs empirical probing. If it regresses, revert and mark abandoned.

- **[todo] M3: Eliminate the CPU staging buffer in attention output.**
  Blocked on mechanism-level probing of `torch.ops.spyre.overwrite`.
- **[todo] M6b: Sweep remaining custom_ops for weight/parameter re-transfer
  patterns.** Audit rotary/silu/embedding/lm_head. Rotary is CPU-native
  (no target). Silu has no weights. Embedding runs once/step. LM head
  runs once/step with weight already on Spyre. **Low expected value
  from this sweep on TP=1 single-seq bench.**
- **[todo] M6c: QKV `.split()` on Spyre.** Structural change to avoid
  D→H in `SpyreQKVParallelLinear.forward`. Requires torch-spyre op
  support for strided splits, or a rewrite to do the split before the
  matmul (splitting weights, not activations).

## Minor

- **[todo] Prefill `reshape_and_cache` Python-unrolled loop** —
  bench doesn't spend meaningful time in prefill.
- **[todo] RMSNorm `torch.full(x.shape, ...)` epsilon buffer** —
  size-matched broadcast crutch; scalar or 0-d tensor may work if
  Spyre semantics allow. Mechanism-level probe needed.

## Done

- M0 (r3): bench warmup + 120-token measured. New baseline 0.7315 tok/s.
- M1 (r1): metadata-builder Q=1 guard.
- M1' (r2): Q=1 decode kernel (single global softmax, no per-block max).
- M2 (r4): mask-tile H2D caching across L layers per step (+9.8%).
- M5 (r5): metadata-builder precompute for CPU-tensor scalars (+5-10%).
- M6a (r6): RMSNorm weight caching on Spyre (mechanically clean; drift-
  dominated signal).

## Parked

(none yet)

## Abandoned

(none yet)
