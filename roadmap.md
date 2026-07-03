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

Decode steps have `max_query_len=1`. After r1–r4, query padding, per-block
argmax fallback, per-layer mask-tile H2D, and measurement noise are gone.
Remaining hot-path costs, ranked by expected EV:

- **[M5 target] Per-layer per-block CPU→Python sync in the seq loop of
  `_online_softmax_attention`.** For each of L layers per decode step,
  `_online_softmax_attention` executes:
  - `int(query_start_loc[seq_idx].item())` (spyre_attn.py:879)
  - `int(query_start_loc[seq_idx + 1].item())` (line 880)
  - `int(seq_lens[seq_idx].item())` (line 882)
  - `[int(block_table[seq_idx, i]) for i in range(num_blocks_needed)]` (line 908)
  Each `.item()` reads a CPU tensor header and drops the GIL. Per decode
  step this is `L × (3 + num_blocks_needed) × num_seqs` calls. For a 26-layer
  model, num_seqs=1, num_blocks≈2, that's ~130 `.item()` calls per step, or
  ~15,600 per 120-token bench. All redundant across layers within a step.
  The metadata builder already runs once per step and already does
  `int(seq_lens[s].item())` at line 578 when building mask tiles — trivial
  to also compute and stash `page_indices`, `q_starts`, `q_ends`, `kv_lens`
  as plain Python lists.
- `output_cpu = zeros_like(output, device="cpu")` staging round-trip in
  `_online_softmax_attention` (`spyre_attn.py:874`). Per-token results
  bounce Spyre → CPU → Spyre. Per-step overhead, not per-layer, but real.
  Currently blocked by the "Spyre slicing corrupts memory" comment
  (lines 864-873) — needs mechanism-level work to unlock (M3).
- Compile bucket tiers at every 256-token `kv_len` boundary. After M0
  warmup this is negligible on the timed run.
- Per-token `reshape_and_cache` unrolls a Python loop of length `num_tokens`;
  fine for decode (num_tokens=num_seqs=1) but wasteful in prefill.

## Measurement (fixed r3)

Warmup + 120-token bench. CoV ~9-19% across 3 runs (varies by session
load). Same-session A/B comparison (implementer applies own baseline and
change in one session) is the most reliable signal — cross-session drift
on the shared host is real and can swing ~4× on individual runs. r4
implementer proved this by doing an in-session A/B that showed M2 = 1.098×
r3-end (+9.8%), while cross-session numbers on M2 alone spanned 0.663 →
1.0135 tok/s. Future rounds should prefer same-session A/B for verdicts.

## Learnings from previous rounds

- **r1 (M1, done):** metadata-builder Q=1 guard. Correctness green.
- **r2 (M1', done):** dedicated Q=1 decode kernel with single global
  softmax. Decode path fallback-free.
- **r3 (M0, done):** bench methodology fix. New reproducible baseline
  0.7315 tok/s (r3 judge's three-run median).
- **r4 (M2, done):** cache mask-tile H2D across layers within a step.
  Same-session speedup +9.8% vs r3-end; judge measured 0.8193 median
  (1.12× baseline). The "cache-what's-redundant-across-L-layers" pattern
  is a proven high-EV lever — M5 applies the same pattern to CPU→Python
  syncs.

## Major

- **[done] M0: Fix bench measurement signal.** Warmup + 120-token bench.

- **[done] M1: Decode-only fast path (query_len==1).**

- **[done] M1': Q=1 decode kernel with single global softmax.**

- **[done] M2: Cache mask tiles across layers within a decode step.**
  Same-session +9.8% vs r3-end.

- **[in_progress] M5: Precompute per-seq scalars and page_indices in the
  metadata builder.** *Applies the M2 pattern to CPU-tensor `.item()`
  round-trips.* Move the four `.item()`/int() call sites in
  `_online_softmax_attention` (spyre_attn.py:879-908) into
  `SpyreAttentionMetadataBuilder.build` so the per-seq scalars are
  computed once per step and read as plain Python ints per layer.
  Concretely, add three new fields to `SpyreAttentionMetadata`:
    * `query_starts: list[int]` — `query_start_loc.tolist()[:-1]`
    * `query_ends: list[int]` — `query_start_loc.tolist()[1:]`
    * `kv_lens_list: list[int]` — `seq_lens.tolist()`
    * `page_indices_per_seq: list[list[int]]` — one list per sequence,
      length `ceil(kv_len_s / block_size)`, built by
      `block_table[s, :n].tolist()`.
  Then in `_online_softmax_attention`, replace the four `.item()` calls
  with plain list indexing. The builder already computes `kv_len_s` for
  mask tiling — reuse that loop.

- **[todo] M3: Eliminate the CPU staging buffer in attention output.**
  *Removes one Spyre → CPU → Spyre round-trip per attention layer.* Only
  viable on the Q=1 decode path where `q_end - q_start == 1` for the
  single sequence. Investigate whether `torch.ops.spyre.overwrite` on a
  single row along dim=0 is dependable. Higher-risk: comment at
  spyre_attn.py:864-873 says prior attempts at Spyre dim=0 scattering
  silently corrupted data.

- **[todo] M4: Try `KV_LENGTH_ALIGNMENT=512`.** *Fewer compile buckets.*
  One-line probe. Lowest priority now that warmup absorbs compile cost.

- **[todo] M6: Custom-ops sweep.** After attention hot-path work saturates,
  the remaining per-decode-step time lives in the linear layers
  (`custom_ops/linear.py`), RMSNorm, rotary, silu_and_mul, and vocab
  embeddings. Look for redundant convert()/CPU-round-trips there.
  Deferred until attention path is drained.

## Minor

(none yet)

## Done

- M0 (r3): bench warmup + 120-token measured. New baseline 0.7315 tok/s.
- M1 (r1): metadata-builder Q=1 guard.
- M1' (r2): Q=1 decode kernel (single global softmax, no per-block max).
- M2 (r4): mask-tile H2D caching across L layers per step (+9.8% same-session).

## Parked

(none yet)

## Abandoned

(none yet)
