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

Decode steps have `max_query_len=1`. After r1/r2/r3, query padding, per-block
argmax fallback, and measurement noise are gone. Remaining hot-path costs:

- **[M2 target] Per-layer mask-tile H2D**: `convert(m, device=_target_device)`
  at `spyre_attn.py:891` runs `num_blocks × num_layers × num_decode_steps`
  times. For a 26-layer model with 120 decode steps and ~2-4 blocks per
  step, that's ~6000-12000 H2D transfers per bench, all of identical CPU
  tensors within one step. All L layers see the *same* `attn_metadata` per
  step and re-convert the *same* CPU mask tiles.
- `output_cpu = zeros_like(output, device="cpu")` staging round-trip in
  `_online_softmax_attention` (`spyre_attn.py:855`). Per-token results
  bounce Spyre → CPU → Spyre.
- Per-token `reshape_and_cache` unrolls a Python loop of length `num_tokens`;
  fine for decode (num_tokens=num_seqs=1) but wasteful in prefill.
- Compile bucket tiers at every 256-token `kv_len` boundary. After M0
  warmup this is negligible on the timed run.

## Measurement (fixed r3)

`examples/offline_inference/torch_spyre_inference.py` now runs a warmup
generation ahead of the timed one and generates 120+ tokens instead of 20.
CoV ~8-11% across 3 sequential process invocations, vs ~40% before.
The r3 judge measured 0.7315 tok/s median on unchanged plugin code, up
from 0.411 pre-methodology — that jump is a measurement-methodology
artifact, not a real speedup. **All future criteria compare against the
new baseline (median 0.7315 tok/s from r3 judge's three primary runs on
plugin state `round-3-end`), not the old 0.411.**

## Learnings from previous rounds

- **r1 (M1, done):** metadata-builder Q=1 guard. Correctness green.
- **r2 (M1', done):** dedicated Q=1 decode kernel with single global
  softmax. Decode path fallback-free.
- **r3 (M0, done):** bench methodology fix — warmup + 120-token measured.
  CoV ~9% (r3 judge) / ~8% (r3 implementer's own probe) vs ~40% baseline.
  New reproducible baseline: 0.7315 tok/s median.
- On a 20-token bench, first-decode-step compile ≈ 30-40% of wall time.
  Post-M0 that's amortized in the warmup pass, so the timed run reflects
  steady-state decode dominated by per-step Python + H2D overhead.

## Major

- **[done] M0: Fix bench measurement signal.** Warmup + 120-token bench
  in the example script. Merged r3. New baseline 0.7315 tok/s.

- **[done] M1: Decode-only fast path (query_len==1).** Metadata-builder
  guard so `aligned_max_query_len=1` when `max_query_len==1`. Merged r1.

- **[done] M1': Q=1 decode kernel with single global softmax.**
  `_create_compilable_page_attn_decode` factory + `_get_attn_fn` dispatch
  at `padded_query_len==1`. Merged r2. Decode-path fallback-free.

- **[in_progress] M2: Cache mask tiles across layers within a decode step.**
  *Removes L-1 redundant mask-tile H2D transfers per step.* All L
  attention layers in one forward pass receive the *same*
  `SpyreAttentionMetadata` object with the *same* CPU-side
  `attention_mask_tiles`. Currently every layer calls
  `convert(m, device=_target_device)` on the same tiles at
  `spyre_attn.py:891`. Lazy-cache the Spyre-side tiles on the
  metadata object: first layer converts and stashes them; later layers
  read the cached Spyre-side list. Since `SpyreAttentionMetadata` is a
  dataclass, adding an `Optional[list[list[Tensor]]]` field defaulting
  to `None` and populating it lazily in `_online_softmax_attention` is
  the minimal-friction fix. The metadata object is discarded at end of
  step, so no cross-step staleness concern.

- **[todo] M3: Eliminate the CPU staging buffer in attention output.**
  *Removes one Spyre → CPU → Spyre round-trip per attention layer.* Only
  viable on the Q=1 decode path where `q_end - q_start == 1` for the
  single sequence. Investigate whether `torch.ops.spyre.overwrite` on a
  single row along dim=0 is dependable.

- **[todo] M4: Try `KV_LENGTH_ALIGNMENT=512`.** *Fewer compile buckets.*
  Bumps compile tiers 256→512 at cost of one extra masked row per step.
  Cheap probe. Lower priority now that M0 warmup absorbs compile cost.

- **[todo] M5: Move the per-page `page_indices = [int(block_table[…])...]`
  CPU sync in `_online_softmax_attention` into the metadata builder.**
  *Removes num_blocks CPU-tensor `.item()` calls per layer per step.*
  Currently line 889 calls `int(block_table[seq_idx, i])` `num_blocks`
  times per layer × L layers × decode steps. `block_table` is CPU, but
  Python-level `int()` conversion still touches the GIL and the tensor
  header per element. The metadata builder already runs once per step
  and could populate `page_indices_per_seq: list[list[int]]` in one
  pass.

## Minor

(none yet)

## Done

- M0 (r3): bench warmup + 120-token measured. New baseline 0.7315 tok/s.
- M1 (r1): metadata-builder Q=1 guard.
- M1' (r2): Q=1 decode kernel (single global softmax, no per-block max).

## Parked

(none yet)

## Abandoned

(none yet)
