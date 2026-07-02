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

Decode steps have `max_query_len=1`. Current code pads to `QUERY_CHUNK_SIZE=32`
so **every attention step does 32× more query work than needed**. This is the
single biggest visible waste on the hot path (flagged in code TODO at
`spyre_attn.py:51-53`). Other suspects:

- **CPU fallback on `scores.max(dim=-1)` at `spyre_attn.py:252`** — emits
  `aten.argmax.default` FallbackWarning. Runs once per KV block per layer
  per decode step. **This is now the dominant hot-path cost after M1
  shrank query padding.** Before r1, this fallback was amortized over 32
  query rows; after r1's Q=1 metadata guard the ratio is one CPU roundtrip
  per single query row per KV block. Almost certainly the reason M1
  produced no measurable end-to-end gain.
- `output_cpu = zeros_like(output, device="cpu")` staging round-trip in
  `_online_softmax_attention` (`spyre_attn.py:779`). Per-token results
  bounce Spyre → CPU → Spyre.
- `convert(m, device=_target_device)` per mask tile every step
  (`spyre_attn.py:815`) — no reuse across steps.
- Compiled attn variants keyed by `(num_blocks, padded_query_len)`
  (`spyre_attn.py:645-651`) — each new `kv_len` tier triggers a new
  compile. Bigger `KV_LENGTH_ALIGNMENT` = fewer compiles across a full run.
- Per-token `reshape_and_cache` unrolls a Python loop of length `num_tokens`;
  fine for decode (num_tokens=num_seqs=1) but wasteful in prefill.

## Learnings from previous rounds

- **r1 (M1, done):** `SpyreAttentionMetadataBuilder.build` now sets
  `aligned_max_query_len = 1` when `max_query_len == 1`. Static gates all
  green. Correctness suite (265 passed) unchanged. **End-to-end tok/s
  ratio 0.957× (−4.3%)** — within noise, no measurable perf win. Root
  cause per judge: the per-KV-block `argmax` fallback on
  `spyre_attn.py:252` is now paid *per query row* instead of amortized
  over 32 rows, cancelling the theoretical FMA saving. The metadata
  change is nonetheless kept — it's a prerequisite for M1' below and
  correct on its own terms.
- The 10% threshold on a 20-token benchmark is aggressive — first-step
  compile / warmup dominates. Even a real attention-kernel speedup may
  be diluted by non-attention work. Future criteria should either use
  a longer bench or drop to ≥ 1.02× on 20 tokens.

## Major

- **[done] M1: Decode-only fast path (query_len==1).** Metadata-builder
  guard so `aligned_max_query_len=1` when `max_query_len==1`. Merged r1.
  Correctness green, perf-neutral by itself. Kept as prerequisite for M1'.

- **[in_progress] M1': Eliminate per-block CPU `argmax` fallback on the
  Q=1 hot path.** *Targets the newly-dominant fallback cost after M1.*
  Two credible sub-approaches, pick one:
  (a) **Skip online softmax entirely when `padded_query_len==1`.** With
  a single query row, the numerical-stability motivation for online
  softmax (per-block max/normalizer) is much weaker — a single global
  softmax over the concatenated scores works and avoids the per-block
  `.max()`. Compile a separate specialization
  `_create_compilable_page_attn_decode(num_blocks)` that computes
  scores per page, concatenates along the KV axis, does one global
  `softmax`, then computes `output = probs @ V` across pages.
  (b) **Replace `scores.max(dim=-1)` with a fallback-free equivalent on
  Spyre.** E.g. `torch.amax(scores, dim=-1, keepdim=True)` if
  `aten.amax.default` doesn't fall back; or clamp-and-subtract using a
  running max maintained on Spyre. Requires probing which reduction
  primitives torch-spyre supports.
  Prefer (a) — it's a structural change with no dependence on which
  reductions torch-spyre supports today, and the resulting kernel is
  simpler (one softmax, no rescale). Numerics still float16 stable for
  a single query row because the total number of scored KV tokens is
  bounded by `MAX_MODEL_LEN_CAP=128` (see `platform.py:70`), so a
  softmax over ≤ 128 values in float16 is safe with a single global
  max subtraction.

- **[todo] M2: Cache mask tiles across decode steps.** *Removes a per-step
  H2D per KV page.* For steady decode of one sequence, `mask_tiles` only
  grow at the boundary where `kv_len` crosses into a new page. Memoize the
  Spyre-side mask tiles keyed by `(seq_idx, num_blocks_needed,
  aligned_max_seq_len)` in `SpyreAttentionImpl`, invalidating on shape
  change.

- **[todo] M3: Eliminate the CPU staging buffer in attention output.**
  *Removes one Spyre → CPU → Spyre round-trip per attention layer.* If we
  can write attention result directly into `output` (a Spyre tensor) via
  `_overwrite` on a decode-only fast path where `q_start=q_end-1` for a
  single row, the CPU staging in `_online_softmax_attention` is
  unnecessary. Investigate whether `torch.ops.spyre.overwrite` on a single
  row along dim=0 is dependable (it's already used in `_overwrite`).

- **[todo] M4: Try `KV_LENGTH_ALIGNMENT=512`.** *Fewer compile buckets over
  the whole run.* Current 256 alignment means every 256-token increment
  causes an attn recompile. Bumping to 512 halves the compile tiers at the
  cost of 1 extra mask row (masked out) per step. Cheap probe.

## Minor

(none yet)

## Done

- M1 (r1): metadata-builder Q=1 guard. See "Learnings" above.

## Parked

(none yet)

## Abandoned

(none yet)
