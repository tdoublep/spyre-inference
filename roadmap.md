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

Decode steps have `max_query_len=1`. After r1/r2, query padding and
per-block argmax fallback are gone. Remaining hot-path costs:

- `output_cpu = zeros_like(output, device="cpu")` staging round-trip in
  `_online_softmax_attention` (`spyre_attn.py:~845-890`). Per-token results
  bounce Spyre → CPU → Spyre.
- `convert(m, device=_target_device)` per mask tile every step
  (`spyre_attn.py:~915`) — no reuse across steps. Cheap per-tile but
  scales with num_blocks × num_decode_steps × num_layers.
- Compiled attn variants keyed by `(num_blocks, padded_query_len)` — each
  new `kv_len` tier triggers a new compile. First-token compile cost is
  the dominant fraction of a 20-token bench.
- Per-token `reshape_and_cache` unrolls a Python loop of length `num_tokens`;
  fine for decode (num_tokens=num_seqs=1) but wasteful in prefill.

## Measurement problem (open, blocking)

Bench variance is ~40% run-to-run on 20 tokens (spans 44–61s, ~0.33–0.46
tok/s). Both r1 and r2 landed correct architectural changes but couldn't
show a real perf win because a single-shot 20-token bench cannot resolve
sub-40% differences. The judge has now flagged this two rounds in a row
and explicitly said: "Do NOT ask the implementer to iterate further on
the decode kernel unless option (1) [longer bench] or (2) [warmup+measure]
is applied first."

R3 must fix measurement before adding more code paths. Two orthogonal
options both live in `examples/offline_inference/torch_spyre_inference.py`:

- **Longer bench (best signal).** Emit ≥ 128 tokens per prompt. Amortizes
  first-decode-step compile cost across ~6× more steady-state steps.
- **Warmup + measure.** Run one warm-up prompt whose timing is discarded,
  then measure the real prompt(s). The compile-cache is warm for the
  second prompt, so the reported number reflects steady-state decode.

Both are one-file edits in the example script; neither changes plugin
behavior. Ideally do both: warm up once *then* generate many tokens.

## Learnings from previous rounds

- **r1 (M1, done):** metadata-builder Q=1 guard. Correctness green.
  End-to-end tok/s 0.957× (−4.3%), inside noise. The bottleneck moved
  to the per-block argmax fallback, which M1' addressed.
- **r2 (M1', done):** dedicated Q=1 decode kernel with single global
  softmax. `_create_compilable_page_attn_decode` factory at
  `spyre_attn.py:~285`, dispatched from `_get_attn_fn` at
  `spyre_attn.py:~722`. Decode path is fallback-free. Correctness green.
  Primary tok/s 0.980× (−2.0%), still inside noise but closer to baseline
  than r1. The judge explicitly declared the change architecturally
  sound and told us to stop iterating on it until measurement is fixed.
- On a 20-token bench, first-decode-step compile ≈ 30–40% of total
  wall time. Any Q=1-only optimization is fighting against this fixed
  cost. Warmup dodges it entirely; longer bench dilutes it.

## Major

- **[done] M1: Decode-only fast path (query_len==1).** Metadata-builder
  guard so `aligned_max_query_len=1` when `max_query_len==1`. Merged r1.

- **[done] M1': Q=1 decode kernel with single global softmax.**
  `_create_compilable_page_attn_decode` factory + `_get_attn_fn` dispatch
  at `padded_query_len==1`. Merged r2. Decode-path fallback-free.

- **[in_progress] M0: Fix bench measurement signal.** *Prerequisite for
  scoring any further attention change.* Edit
  `examples/offline_inference/torch_spyre_inference.py` to add:
  (a) one warmup generation whose timing is discarded (compile cache
      hot before the measured run), and/or
  (b) a longer measured generation (e.g. per-prompt max_tokens=128 by
      default) so the printed `Time elapsed for <N> generated tokens is
      <T> sec` line reports steady-state decode tok/s.
  Keep the argparse defaults such that the judge's canonical command
  `--num-prompts 1` still works and now yields a low-variance number.
  Do not change command semantics for `--num-prompts N > 1`; only make
  the single-prompt case reproducible.

- **[todo] M2: Cache mask tiles across decode steps.** *Removes a per-step
  H2D per KV page.* For steady decode of one sequence, mask tiles only
  change when `kv_len` crosses a page boundary. Memoize Spyre-side tiles
  in `SpyreAttentionImpl` keyed by `(seq_idx, num_blocks_needed,
  aligned_max_seq_len)`, invalidating on shape change. Target after M0
  lands so we can actually see the effect.

- **[todo] M3: Eliminate the CPU staging buffer in attention output.**
  *Removes one Spyre → CPU → Spyre round-trip per attention layer.* Only
  viable on the Q=1 decode path where `q_end - q_start == 1` for the
  single sequence. Investigate whether `torch.ops.spyre.overwrite` on a
  single row along dim=0 is dependable.

- **[todo] M4: Try `KV_LENGTH_ALIGNMENT=512`.** *Fewer compile buckets.*
  Bumps compile tiers 256→512 at cost of one extra masked row per step.
  Cheap probe; do after M0.

## Minor

(none yet)

## Done

- M1 (r1): metadata-builder Q=1 guard.
- M1' (r2): Q=1 decode kernel (single global softmax, no per-block max).

## Parked

(none yet)

## Abandoned

(none yet)
