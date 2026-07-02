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

## Major

- **[in_progress] M1: Decode-only fast path (query_len==1).** *Targets the
  32× query-pad waste on every decode step.* Skip the pad-to-32,
  transpose-to-4D, and reshape dance when `max_query_len==1`. A dedicated
  attention path with `padded_query_len=1` should produce far fewer FMAs
  per step. Compile a distinct `_get_attn_fn` variant per `num_blocks` at
  `padded_query_len=1`.

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

(none yet)

## Parked

(none yet)

## Abandoned

(none yet)
