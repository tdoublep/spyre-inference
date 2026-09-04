# Hand-off: isolate the attention-kernel win of the LX-resident KV layout

## The question

`SPYRE_LX_KV_LAYOUT` (branch `lx-kpage-land`) measured **-7.9% end-to-end latency** on
granite-3.3-8b, 3200 in / 64 out / bs 1. The mechanism is confirmed: the folded decode
attention kernel asks for a **40 KB** HBM spill pool where the legacy one asks for
**22,272 KB**. So the gathered K and V pages really do stay in LX instead of round-tripping
through HBM.

The open question is why that reads as 8% rather than something dramatic. Two candidate
explanations, with very different consequences:

1. **Dilution.** The attention kernel got much faster, but a batch-1 decode step is
   dominated by weight streaming — granite-3.3-8b is 16.34 GB of fp16 weights, a 79.8
   ms/step roofline floor at 204.8 GB/s, and at short context attention was measured at
   only 2.8 ms/step (2.3% of device time). A large multiplier on a small slice is a small
   end-to-end number. If this is the story, the layout is finished work and the next lever
   is elsewhere (fp8 on the weight stream).
2. **The kernel itself is only modestly better.** The spill pool shrank, but the 8-core cap
   (`_LX_ATTN_CORES = 8` of 32, needed because the folded bmm cannot fill 32) and the
   per-block gather give most of it back, and the 8% comes from somewhere else. If this is
   the story, there is headroom left inside attention.

**Goal of this session: settle that with a direct measurement of the attention kernel alone,
main vs this branch, single sequence, across context length.** Use the micro-benchmark from
[PR #642](https://github.com/torch-spyre/spyre-inference/pull/642) (`scripts/microbench/`).

One arithmetic note worth carrying in: the end-to-end win is ~22 ms/step
(1.416 s / 64 tokens), which is *larger than the entire short-context attention time*. That
is consistent with explanation 1 only if attention at 25 pages/step is far more expensive
than attention at 1 page/step. The sweep over context length tests exactly that, so do not
pre-commit to either story.

## State of things

| | |
|---|---|
| Branch under test | `lx-kpage-land` @ `fd72d44`, pushed to remote `tdoublep` |
| Baseline | `origin/main` @ `8f0936c` (the branch is 3 commits behind it — rebase or note it) |
| Micro-benchmark | PR #642, branch `ngl_microbenchs`, fetched locally as `pr642` |
| Flag | `SPYRE_LX_KV_LAYOUT=1`, default off; `SPYRE_ATTN_MAX_CORES` overrides the core cap |
| Prior e2e A/B | `.claude-scratch/ab/` + `ab_report.py`; drivers `ab_lx.sh`, `ab_src.sh` |
| Impl verification | `.claude-scratch/verify_lx_impl.py` (numerics, LX pin count, launch count) |

The e2e numbers, for reference: baseline mean 18.013 s (origin/main 18.006, flag-off 18.056
and 17.978 — 0.43% spread across processes), LX leg 16.597 s.

## Dependencies — read this before running anything

### 1. Pinned RPMs (always)

```bash
source ~/spyre-libs/env.sh
```

A bare `/opt/ibm/spyre` gives an undefined `spyre_comms` symbol. `.claude-scratch/venv.sh`
wraps this plus the shared venv; `./.claude-scratch/venv.sh python3 foo.py` is the safe way
to run anything.

### 2. Never `uv run`

`uv run` re-resolves the pinned `torch-spyre` git rev and silently uninstalls a local build,
and in a worktree it creates a second venv. Use the shared venv at
`/home/senuser/spyre-inference/.venv` with `PYTHONPATH` pointed at the checkout you want to
test — the editable-install finder is appended to `sys.meta_path`, so `sys.path` wins. To run
`uv sync` for unrelated reasons: `uv sync --no-install-package torch-spyre --inexact`.

### 3. torch-spyre must be rebuilt with the profiler — REQUIRED, and it is currently off

This is the one hard blocker. The micro-benchmark's entire signal is device time attributed
to `record_function` spans via AIUPTI; without it the runner aborts (by design — it refuses
to report plausible-looking zeros).

Verified on this box: `_C.so` has **no libaiupti** linkage, and a probe profile of a `matmul`
on device produced **12 events, 0 with device time**. The library itself is present
(`~/spyre-libs/opt/ibm/spyre/runtime/lib/libaiupti.so`) — it is the build that is missing.

The switch is in this repo's `pyproject.toml:103-105`:

```toml
# Disable the torch-spyre build-time profiler. This fixes long model load times on Z, but needs further investigation
[tool.uv.extra-build-variables.torch-spyre]
USE_SPYRE_PROFILER = "0"
```

Flip it to `"1"` and rebuild torch-spyre. Then verify before spending a sweep on it:

```bash
ldd /home/senuser/spyre-inference/.venv/lib/python3.12/site-packages/torch_spyre/_C*.so | grep libaiupti
```

Expect long model load times as a side effect (that is why it is off). Consider a scratch
venv so the shared one keeps the fast-loading build, and **flip the switch back before
committing** — leaving `USE_SPYRE_PROFILER = "1"` in `pyproject.toml` changes everyone's
build.

### 4. torch-spyre #4163 — scatter destination layout compliance

`da15ede` in the local `torch-spyre` checkout at `/home/senuser/torch-spyre`, *"Fix scatter
destination layout compliance check (#4163)"*, touching
`torch_spyre/_inductor/enforce_indirect_access_layout.py`.

**Current state on this box: REVERTED.** The site-packages file is md5
`02e47e1675f26b513222128a84e2a74c`, matching the backup
`_inductor/enforce_indirect_access_layout.py.pre4163`.

Needed for the **folded store** (`_create_folded_cache_store`), i.e. for the end-to-end path.
The micro-benchmark populates the KV cache on the host and never calls the store, so a
read-path-only sweep probably does not need it — but apply it anyway rather than discover
mid-sweep that something does. Extract the commit's diff for the `torch_spyre/` subtree from
that checkout and `patch -p1` it into site-packages
(`/home/senuser/spyre-inference/.venv/lib/python3.12/site-packages`).

To revert, restore from the `.pre4163` backup. **Restore it when you are done** — the venv is
shared across every worktree on this box.

### 5. torch-spyre #4153 — restickify local-read, gated

A draft port lives in site-packages at `torch_spyre/_inductor/scratchpad/allocator.py:197`,
gated on `SPYRE_LX_RESTICKIFY_RESIDENCY=1` (default off), backup `pass_utils.py.pre4153`.

**The LX win rides on this.** It is what lets K's `permute` stay in LX rather than forcing a
relayout through HBM. Every LX leg must set `SPYRE_LX_RESTICKIFY_RESIDENCY=1`; every baseline
leg must leave it unset. To attribute the win between "pages stay in LX" and "the permute
stays in LX", add a leg with `SPYRE_LX_KV_LAYOUT=1` and the gate *off* — a real sub-question,
worth the extra run.

### 6. One process owns the accelerator

Never two Spyre commands at once — no `pytest -n`, no backgrounding one sweep while starting
another. Parallel runs hang, corrupt device state, or corrupt the compile cache. Every leg is
strictly sequential.

## The work: wire the folded layout into the micro-benchmark

`scripts/microbench/spyre_attn_microbench.py` already has a `--kv_layout` axis with `plain` /
`slot_major` / `slot_major_devfill` (CSV column `kv_layout`, argparse choices at ~line 692,
allocation in `build_inputs_from_requests`'s `to_device` at ~line 271). Add a `folded` option.
Four things to know:

1. **The cache allocation is the only real work.** In `build_inputs_from_requests` the host
   cache is built as `[num_blocks, block_size, num_kv_heads, head_size]` and populated token
   by token. The folded frame is `[num_blocks * num_kv_heads, block_size, head_size]` with
   rows ordered `(page, kv_head)`, so:

   ```python
   from spyre_inference.v1.attention.backends.spyre_attn import head_major_kv_layout
   nb, bsz, h, d = cache.shape
   folded = cache.permute(0, 2, 1, 3).contiguous().reshape(nb * h, bsz, d)
   return folded.to(cache_device, device_layout=head_major_kv_layout(nb * h, bsz, d, cache.dtype))
   ```

   This reshape happens **on the host, before transfer**, which is why it is trivial here.
   Do not try to do it device-side: see the pitfalls below.

2. **Everything else comes free.** `_mirror_lx_index_tables` builds the per-block
   `[num_kv_heads, 1]` gather indices inside `impl.forward` from
   `attn_metadata.page_index_table_cpu`, which the real `SpyreAttentionMetadataBuilder`
   already provides. `_head_index_tables` is likewise built in `forward`. The core cap
   applies automatically: `attn_max_cores` caps only when
   `num_kv_heads * aligned_max_query_len < 32`, true for a single-sequence decode (8 × 1) and
   false for prefill.

3. **`SPYRE_LX_KV_LAYOUT` is read at import and at impl construction**
   (`envs.SPYRE_LX_KV_LAYOUT` → `self._lx_kv_layout`, `self._reshape_fn`). It must be in the
   environment before the process starts, so the A/B is one process per leg — the flag cannot
   be a config axis inside one run. Same constraint the README already notes for compiled vs
   eager.

4. **The micro-benchmark does not exercise the KV store.** The comment in `forward` is
   explicit: *"The KV write is not here: attn_layer.py traces it for the layers it splits."*
   The micro-benchmark calls `impl.forward(layer=None, ...)` with a pre-populated cache, so it
   measures gather + bmm + softmax only. That is the right scope for this question — but say
   so in the write-up, and do not read a micro-benchmark delta as an end-to-end prediction.

   Related: PR #642's `--span reshape_and_cache` maps to `spyre_attn::reshape_and_cache`, and
   **no such `record_function` span exists** on `origin/main` or on this branch (only
   `spyre_attn::forward` and `spyre_attn::online_softmax` are decorated). That option will
   trip the startup guard. Use `--span online_softmax` (the default, the kernel itself) and
   `--span forward` (total attention cost).

## Measurement plan

Legs, all `num_reqs=1`, `block_size=128`, `num_blocks=64` pinned constant (the README warns
that changing `num_blocks` between runs changes the footprint and invites
`RAS::FLEXALLOCATOR::OutOfMemory`, and that a failed allocation strands memory and degrades
every later row in that process):

| leg | checkout | env |
|---|---|---|
| `main` | `origin/main` @ `8f0936c` | `kv_layout=plain` |
| `off` | `lx-kpage-land` | flag unset, `kv_layout=plain` |
| `on` | `lx-kpage-land` | `SPYRE_LX_KV_LAYOUT=1 SPYRE_LX_RESTICKIFY_RESIDENCY=1`, `kv_layout=folded` |
| `on-nogate` (optional) | `lx-kpage-land` | `SPYRE_LX_KV_LAYOUT=1`, gate off — attributes #4153's share |
| `on-32core` (optional) | `lx-kpage-land` | as `on` plus `SPYRE_ATTN_MAX_CORES=32` — prices the 8-core cap |

`main` and `off` should agree; that pair is the error bar, exactly as in the e2e A/B where
three baseline legs spread 0.43%. If they disagree by more than that, stop and find out why
before reading the `on` leg.

Shape sweep: the shipped `configs/granite33_8b_bs128_decode.json` already walks
seq_len ∈ {32, 64, 128, 256, 512, 1024, 2048, 3000, 4096, 8192} at query_len 1, which spans
1 → 64 pages and brackets the e2e A/B's 25 pages (3200 tokens ≈ the 3000 entry). Use it
as-is. Add a prefill config only to confirm the core cap is correctly *not* applied there.

```bash
source ~/spyre-libs/env.sh
export SPYRE_ATTN_PROFILING=1 SPYRE_NUM_CPUS=8
# one leg per process, strictly sequential
.venv/bin/python3 scripts/microbench/spyre_attn_microbench.py \
    --config scripts/microbench/configs/granite33_8b_bs128_decode.json \
    --kv_layout folded --span online_softmax
```

Read the results this way:

- **Normalize by `num_kv_blocks_iterated`** before concluding anything about scaling — the
  README is emphatic that raw µs suggests knees that vanish once divided by pages.
- Check `allclose_pass` and `max_abs_diff` on every row. A fast-and-wrong folded config is the
  most likely failure mode, and the folded row ordering is easy to get subtly wrong (a
  `permute` mistake gives plausible numbers for page 0 and garbage after).
- Check `fallback_clean` on every row. A silent CPU fallback would read as a "result".
- `device_time_memory_us` / `memory_share_pct` is the interesting column here: it is the
  memcpy/memset/restickify share, precisely what the LX layout is supposed to delete. If the
  folded leg is faster, this column should say why.
- Watch for `error` set and empty `ms` — that is the OOM cascade, and every subsequent row in
  that process is suspect. Re-run affected shapes in a fresh process.

## What a result means

- **Large kernel speedup that grows with context** (e.g. 2-5× at 25 pages, flat at 1 page) →
  explanation 1. The layout is finished work; write it up, note that end-to-end is
  weight-stream-bound, and point the next session at fp8.
- **Modest kernel speedup (<30%)** → explanation 2, and the 8% end-to-end needs another
  source. Suspect the launch-count changes rather than the residency: check whether the `on`
  leg runs fewer kernels per forward than the baseline, independent of LX.
- **`on-32core` much faster than `on`** → the 8-core cap is the binding constraint, not HBM,
  and the next lever is making the folded bmm fill more cores (batching the GQA groups without
  triggering the page clone that broke the broadcast guard).
- **A null or negative result is a real outcome.** Report it as such rather than hunting for a
  configuration that looks good. The e2e number is measured and reproducible; if the
  kernel-level number does not explain it, that discrepancy is itself the finding.

## Pitfalls already paid for

- **`torch.stack` does not fuse.** The GQA-unrolled kernel's recombination tail costs a second
  kernel launch (~130 µs, ×40 layers) unless it scatters into the output buffer
  (`out.index_copy_(1, head_ids_g, ...)`). `combine="auto"` picks the scatter tail only when
  `store_mode == "copy"`, i.e. batch-1 decode. Batch >1 still pays the stack tail.
- **The folded store has no flat-source form.** Every device-side `view` / `reshape` /
  `flatten` / `clone` / 3-D pairing fails to lower with `InductorError: KeyError: x`, because
  a logical reshape cannot merge axes in a Spyre 4-axis device layout. Only per-kv-head
  `index_copy_` with `key.select(1, h)` works, and its 16 ops fuse into one kernel. Host-side
  reshapes are fine — which is why the micro-benchmark's fill is easy and the real path was
  not.
- **Anything built lazily inside an accessor called from `forward` gets traced into the
  model's compiled graph**, and inductor then compiles host arithmetic for the device
  (`spyre::to_dtype_cpu` on a CPU tensor). Build per-step tensors eagerly, outside the graph.
- **Warmup compile is ~13 min** on the LX leg (vs ~5 min) because the unrolled graphs are ~4×
  larger. Runtime is unaffected. Do not mistake it for a hang.
- **The kernel specializes per `(num_blocks, aligned_max_query_len)`**, so a sweep triggers
  many recompiles. Warmup runs before the profiled windows, so no compile should land inside a
  measured window — but if `max_ms/min_ms` is wild, suspect one did.
- **Do not edit a running bash script.** Bash reads scripts incrementally; an edit shifts the
  byte offsets underneath it.

## Deliverable

A short write-up with the per-context-length table (main / off / on, device µs and µs per page
iterated), the `memory_share_pct` column alongside it, and a one-paragraph verdict on which of
the two explanations the data supports. Commit it under `.claude-scratch/`.

Cleanup before finishing: restore `enforce_indirect_access_layout.py` from `.pre4163` if you
applied it, leave `SPYRE_LX_RESTICKIFY_RESIDENCY` default-off, and revert
`USE_SPYRE_PROFILER` to `"0"` in `pyproject.toml`.
