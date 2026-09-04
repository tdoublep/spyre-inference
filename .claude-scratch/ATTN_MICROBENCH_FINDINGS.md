# Attention-kernel A/B for the LX-resident KV layout

Answers the hand-off in `HANDOFF_ATTN_MICROBENCH.md`: isolate the attention-kernel win of
`SPYRE_LX_KV_LAYOUT` (branch `lx-kpage-land`), which measured **-7.9% end-to-end** on
granite-3.3-8b at 3200 in / 64 out / bs 1, and decide between two explanations —
(1) a large kernel speedup diluted by weight streaming, or (2) a merely modest kernel gain
with the 8% coming from elsewhere.

**Scope, per instruction:** decode only, `num_reqs=1`, `query_len=1`, contexts 512-8192.
The bucketed decode kernel is *not* measured — `SPYRE_BUCKETED_DECODE` is unset in every
leg, so `_online_softmax_attention` takes the per-sequence path. vLLM caps chunked prefill
at 512, so query lengths above that are unreachable and were not swept.

## Answer

**Neither explanation as framed. The kernel is ~2.5x faster, and that reconciles with the
end-to-end number without needing another source.**

The headline is a factor of **2.2x at 4 pages rising to 2.53x at 25-64 pages**, measured
against the KV layout production actually uses. It is not the 22-27x that a naive baseline
suggests (see the straw-man section — this is the main methodological result of the
session), and it is far more than explanation 2's "<30%".

Attention is also not the negligible 2.3% slice the hand-off's dilution argument assumed:
at 25 pages the production-layout kernel costs ~1.68 ms per layer, i.e. **~67 ms/step**
across 40 layers. Folding it saves ~40 ms/step against a measured e2e win of ~22 ms/step —
the same order, and the residual gap is what a read-path-only micro-benchmark should show.

So the layout is real, finished work. The remaining headroom is **not** inside attention:
torch-spyre #4153 contributes nothing here, and the 8-core cap is already the better
choice, not a ceiling. The next lever is the weight stream (fp8).

## The straw-man baseline (read this before quoting any number)

The micro-benchmark's `--kv-layout plain` is a bare `cache.to(device)`, i.e. the **default
tiled** device layout. Production does not allocate KV pages that way.
`spyre_model_runner.py:1039`:

```python
if fold_kv_heads:
    layout = head_major_kv_layout(...)                              # the LX frame
else:
    layout = slot_major_kv_layout(num_blocks * block_size, ...)      # the real baseline
```

Under the tiled layout the page index is spread across two device dims, which is exactly
the condition that makes `index_select` cost the whole tensor rather than one page. That
inflates the baseline by ~10x:

| pages (ctx) | plain (tiled) us | devfill (slot-major) us | plain / devfill |
|---|---|---|---|
| 4 (512) | 2737.2 | 274.5 | 9.97x |
| 8 (1024) | 5595.3 | 544.4 | 10.28x |
| 16 (2048) | 11140.1 | 1076.8 | 10.34x |
| 25 (3200) | 17826.2 | 1679.6 | 10.61x |
| 32 (4096) | 21346.2 | 2149.9 | 9.93x |
| 64 (8192) | 45550.0 | 4255.3 | 10.70x |

Quoting `plain` as the baseline gives a 22-27x speedup and implies a **686 ms/step** saving
at 25 pages, against ~22 ms/step actually measured e2e — a ~31x impossibility. That
impossibility is what exposed the bad baseline. `slot_major_devfill` removes it.

## Results

`online_softmax` span (the kernel itself), median device us over 10 profiled windows,
`num_blocks=64`, `block_size=128`, collision-free block tables.

| pages (ctx) | devfill (prod baseline) | on (folded) | speedup | devfill us/page | on us/page |
|---|---|---|---|---|---|
| 4 (512) | 274.5 | 124.3 | **2.21x** | 68.6 | 31.1 |
| 8 (1024) | 544.4 | 226.7 | **2.40x** | 68.1 | 28.3 |
| 16 (2048) | 1076.8 | 434.6 | **2.48x** | 67.3 | 27.2 |
| 25 (3200) | 1679.6 | 666.7 | **2.52x** | 67.2 | 26.7 |
| 32 (4096) | 2149.9 | 849.8 | **2.53x** | 67.2 | 26.6 |
| 64 (8192) | 4255.3 | 1679.2 | **2.53x** | 66.5 | 26.2 |

Both legs are linear in pages with no knee; the folded frame simply has a ~2.5x smaller
constant. The speedup rises from 2.21x to ~2.53x and plateaus by 25 pages, so at the
context the e2e A/B used, the win is already saturated.

### Baseline error bars

Two independent baseline pairs, as the hand-off asked for:

- `plain`: `off` vs `main` within 0.71% worst case, <=0.20% on four of six shapes.
- `devfill`: `off` vs `main` within 0.77% worst case (0.00 / -0.77 / -0.06 / +0.13 /
  -0.27 / +0.04 %).

The e2e A/B's own baseline spread was 0.43%. Both pairs are far below the 2.5x effect.

### memory_share_pct does not explain the win

The hand-off expected this column to carry the result: "the memcpy/memset/restickify
share, precisely what the LX layout is supposed to delete". It does not.

| pages | plain | devfill | on (folded) |
|---|---|---|---|
| 4 | 0.8% | 5.3% | 10.5% |
| 25 | 0.5% | 2.5% | 5.0% |
| 64 | 0.5% | 2.2% | 4.0% |

There was never much memory-op time to delete — 2.2-5.3% of the production baseline. The
folded leg's *share* is higher while its absolute total is 2.5x lower. The win is in the
gather and bmm themselves, consistent with the indexed axis reaching device position 0,
not in eliminating relayout traffic.

### The 2.5x transfers to a production-sized cache

`num_blocks` 64 vs 128 at fixed context, which separates "cost of one page" from "cost of
the whole cache". Only the tiled layout depends on cache size:

| pages | plain 64 -> 128 | devfill 64 -> 128 | folded 64 -> 128 |
|---|---|---|---|
| 4 | 2737 -> 7226 (**2.64x**) | 274.5 -> 280.0 (+2.0%) | 124.3 -> 124.7 (+0.3%) |
| 16 | 11140 -> 28935 (**2.60x**) | 1076.8 -> 1075.9 (-0.1%) | 434.6 -> 436.6 (+0.5%) |
| 25 | 17826 -> 46889 (**2.63x**) | 1679.6 -> 1683.8 (+0.2%) | 666.7 -> 668.2 (+0.2%) |
| 64 | 45550 -> 115837 (**2.54x**) | 4255.3 -> 4287.6 (+0.8%) | 1679.2 -> 1675.6 (-0.2%) |

`plain`'s per-page cost tracks `num_blocks` (685 -> 1806 us/page), slightly superlinearly —
the whole-tensor `index_select` signature, and independent confirmation of the mechanism.
Both the production baseline and the folded frame are flat, so the 2.5x ratio is a property
of the layouts and not of the 64-block harness. It carries to production's larger cache.

This also closes the door on `plain` retrospectively: a baseline whose badness *grows* with
cache size would be far worse than 10x at production block counts, which is exactly why
`spyre_model_runner` allocates slot-major.

## Sub-questions the hand-off asked

### torch-spyre #4153 (`SPYRE_LX_RESTICKIFY_RESIDENCY`) contributes nothing

The hand-off states "**the LX win rides on this**" and required it set on every LX leg.
On the read path it makes no measurable difference — `on-nogate` (flag on, gate off):

| pages | on | on-nogate | delta |
|---|---|---|---|
| 4 | 124.3 | 123.6 | -0.6% |
| 8 | 226.7 | 227.6 | +0.4% |
| 16 | 434.6 | 441.9 | +1.7% |
| 25 | 666.7 | 665.4 | -0.2% |
| 32 | 849.8 | 847.6 | -0.3% |
| 64 | 1679.2 | 1680.8 | +0.1% |

All within the baseline error bar. The folded frame alone delivers the entire kernel-level
win. This is consistent with the earlier finding that LX relayout saw zero acceptances in
decode.

**Confirmed end to end** (`ab_gate.sh`, `ab_gate_rev.sh`), which closes the store-path and
prefill gap the read-path benchmark left open, since these legs compile the whole model:

| ordering | gate on | gate off |
|---|---|---|
| pair 1, gate on first | 16.635 s | 16.473 s |
| pair 2, gate off first | 16.464 s | 16.518 s |
| **pooled mean** | **16.550 s** | **16.496 s** |

The pooled gap is 0.33%, inside the 0.43% baseline spread. The reversed pair is what makes
this readable: pair 1 alone looked like a 0.98% *win for disabling* the gate, but in both
pairs the faster leg is whichever ran second, so that gap tracked position, not the gate.
`gate_on` also reproduces the recorded on leg (16.635 vs 16.597 s), so nothing drifted
between sessions.

**Verdict: the LX KV layout does not need #4153.** The hand-off's "the LX win rides on
this" does not hold. The PR may still matter elsewhere, but not for this feature.

Not established: the *mechanism*. `gate_count.sh` was meant to count how many buffers the
local-read proof un-bars, but it emitted no `spyre.inductor.scratchpad.allocator` records
at all under `VLLM_LOGGING_LEVEL=DEBUG` -- not even the WARNING that appears in ordinary
runs -- so its zeros are an instrumentation failure, not evidence. Counting them needs a
mechanism that does not route through vLLM's logging config. Ruled out along the way: a
warm compile cache making all four legs load one binary, which would have made the table
above vacuous. 17k+ files were written under `torchinductor_*/inductor-spyre` during the
legs, so each leg really did compile and the gate really did have the chance to act.

Its second leg also hit a hardware fault -- RAS `0xa35e` "PCIe bus master fence" -- and
died with `EngineDeadError`. All four timing legs are RAS-clean, the fault came after them,
and `aiu-query-devices` shows all 4 AIUs back with full memory, so the results above stand.

### The 8-core cap is an optimisation, not a ceiling

`attn_max_cores`' docstring reads as though `SPYRE_ATTN_MAX_CORES=32` cannot work at
query_len 1 ("a gather can never mirror a split on a value-table data dim", with
`output_units = num_kv_heads * 1 = 8`). Empirically it **lowers and computes correctly** —
it is simply much slower:

| pages | on (8-core cap) | on-32core | ratio |
|---|---|---|---|
| 4 | 124.3 | 772.3 | 6.2x slower |

So the hand-off's fourth interpretation ("`on-32core` much faster than `on`" -> the cap is
the binding constraint, and the next lever is filling more cores) is answered **no**. The
cap is worth ~6x; removing it is a large regression. There is no core-count headroom to
reclaim at batch 1 / query_len 1.

## Harness changes

- **`--kv-layout folded`** added: host-side `permute(0,2,1,3).reshape(nb*h, bsz, d)` with
  `head_major_kv_layout`, row = `page * num_kv_heads + kv_head`, matching the gather
  indices `_mirror_lx_index_tables` builds. Host-side is why this is trivial here; no
  device-side reshape to this frame lowers.
- **`ref_pages`**: the CPU reference inverts the fold, so correctness is checked against
  what the kernel actually read on device rather than the host copy.
- **Startup-probe fix**: the probe built its cache with a hardcoded `plain` regardless of
  `--kv-layout`, so under the flag it would have measured the wrong frame.
- **Mismatch guard**: a half-set `SPYRE_LX_KV_LAYOUT` / `--kv-layout folded` pair now
  aborts instead of quietly measuring the wrong configuration.
- **Collision-free block tables** (`randperm` instead of `randint`): sampling *with
  replacement* left only ~41 of 64 pages distinct at ctx 8192. Correctness was unaffected
  (the reference reads the same aliased table) but the page reuse it created grew with
  context and would have flattered a residency-sensitive layout exactly where the win is
  claimed. Re-measuring `main` with unique pages moved it <0.5% at every shape, so the
  aliasing was not in fact inflating the baseline — but the fix was needed before the
  folded leg could be trusted at long context.
- Baseline legs import `head_major_kv_layout` lazily and read the LX flag via `getattr`,
  since `origin/main` has neither symbol.

## Caveats

- **Absolute us are not production per-layer costs.** Even on the production layout, 40
  layers x 1.68 ms at 25 pages would be 67 ms/step of attention alone. Treat the *ratio*
  as the transferable quantity. The micro-benchmark omits the KV store, runs one layer in
  isolation, uses 64 blocks, and carries AIUPTI profiling overhead.
- **`slot_major_devfill`'s correctness gate is vacuous.** It reports `max_abs_diff` of 0
  and ~1e-5, far tighter than any other leg, consistent with the device cache holding
  mostly zeros because its eager `index_copy_` with an int64 index falls back to CPU (the
  behaviour `do_kv_cache_update` documents). It remains valid as a *layout cost* probe —
  gather/bmm/softmax cost here is data-independent — but it is not a numerics result.
  A production-faithful *and* numerically-populated baseline would need the store path.
- **`on` vs `main`/`off` legs are numerically cross-checked.** `on` reports
  `max_abs_diff` identical to `plain` at all six shapes (0.00293 / 0.0022 / 0.00127 /
  0.000854 / 0.000763 / 0.000549), so the folded frame computes a bit-comparable result.
- One profiled window in `main_online_softmax` at 4096 reported `min=127.5us` against a
  median of 21346us — an AIUPTI trace-buffer truncation. Medians over 10 windows are
  robust to it; `min_ms` is not trustworthy and is not quoted.
- The branch is 3 commits behind `origin/main` @ `8f0936c`, not rebased. The `main` leg is
  that commit, and `main` == `off` confirms the gap is immaterial to this measurement.
- torch-spyre **#4163** was applied to site-packages for every leg (held constant, so it
  cannot bias the A/B). It is not in the pinned rev `e02b78b`.

## Reproducing

```bash
# torch-spyre must be built with USE_SPYRE_PROFILER=1 or the runner aborts by design.
# Built here from the pinned rev and swapped in as _C.so only (the flag affects nothing
# else), with the original kept alongside as _C.so.noprofiler.
.claude-scratch/mb/run_leg.sh <main|off|on|on-nogate|on-32core> <checkout> <config> <span> <outdir>
.claude-scratch/mb/sweep2.sh          # the 5 legs x 2 spans
.claude-scratch/mb/sweep_devfill.sh   # the baseline-validity control
.claude-scratch/mb/sweep_nb128.sh     # the num_blocks control
.claude-scratch/mb/report.py <results_dir>
```
