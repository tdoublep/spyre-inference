# Hand-off: land the LX-resident KV page strategy in `spyre_inference`, then A/B latency

Branch: `lx-residency-kpage-gather` (tdoublep fork). Written 2026-09-03.

## Your goal

1. Implement, in the real codebase, the cache layout + kernel shape that this session
   proved puts **both** the gathered K and V pages in LX.
2. A/B decode latency against `main` on a **long-context** workload.

Everything below the "Verified result" section is instruction; everything in it is
measured. Where I say "unverified", it is genuinely untested — do not promote it.

---

## Environment of record

Every number in this document was produced on exactly this stack. If you are on a
different pod, reproduce the verified result below first, or your baselines will not
match mine.

| component | version |
|---|---|
| `spyre-inference` base | `0c72c0f` (main at branch point; branch `lx-residency-kpage-gather`) |
| `torch-spyre` | **`e02b78ba35f9a1a69a458c3149e9c01d9f4fa6a8`** (the `pyproject.toml` / `uv.lock` pin, installed non-editable) |
| `vllm` | `0.28.0` (git tag pin) |
| `torch` | `2.13.0+cpu` |
| python | `3.12.13` |
| kernel | `5.14.0-570.131.1.el9_6.x86_64` |
| devices | 4 AIU found, PF access; all work used device 0 |

### Spyre runtime libraries -- read this carefully

The RPMs the runtime actually used are **not** the ones installed system-wide. They are
the set pinned in this repo's `spyre-rpms.lock`, unpacked into `~/spyre-libs` by
`scripts/install-pinned-rpms.sh` and selected via `SENTIENT_BASE_INSTALL_DIR`. That is
why every command here starts with `source ~/spyre-libs/env.sh` -- without it you get
the image-baked libraries and `torch_spyre._C` fails to load with an undefined
`spyre_comms` symbol.

Versions used (from `spyre-rpms.lock`, x86_64 prod tree; cross-checked against the
filenames in `~/.cache/spyre-rpms` and the overlay's install stamp, which was written
two minutes after those exact RPMs were downloaded):

| package | version |
|---|---|
| `ibm-deeptools` / `-devel` | `2.0.0-0.main.1+2414.da394b3_325` |
| `ibm-flex` / `-devel` | `2.0.0-0.main.1+529.a5453d3_355` |
| `ibm-senlib-core` / `-dd2` / `-headers` | `2.0.0-0.main.1+261.531e4b5_237` |
| `ibm-spyre-comms` / `-devel` | `1.0.0-0.main.1+135.cf3c000_167` |
| `ibm-aiu-toolbox-e2e` | `2.0.0-0.main.1+28.47d9b91_96` |
| `ibm-libaiupti` | `2.0.0-0.main.1+22.9e597ab_3` |

For contrast, the **system** RPMs on this pod are older and were NOT used:
`ibm-deeptools 2254.3a98611_287`, `ibm-flex 481.bf81df5_295`,
`ibm-senlib-core 237.6e30780_202`, `ibm-spyre-comms 121.3865336_140`. If you hit
different compile errors from mine, check that `env.sh` was sourced before suspecting
anything else.

To rebuild this environment from scratch:

```bash
# needs ARTIFACTORY_TOKEN + ARTIFACTORY_BASE_URL + ARTIFACTORY_RPM_PATH
bash scripts/install-pinned-rpms.sh      # unpacks spyre-rpms.lock into ~/spyre-libs
source ~/spyre-libs/env.sh
uv sync --group dev                      # installs the pinned torch-spyre rev
```

WARNING: both torch-spyre patches below are applied **into `site-packages`**, not to a
source checkout, because `uv run` re-syncs the pinned git rev on every invocation and
would silently revert a local build. For the same reason every command here calls the
venv python directly (`$REPO/.venv/bin/python`) rather than `uv run` -- see
`run_lx_residency.sh`. If you do iterate on a local torch-spyre build instead, use
`uv run --no-sync`.

---

## Verified result (reproduce this first, ~8 min)

```bash
source ~/spyre-libs/env.sh          # bare /opt/ibm/spyre gives an undefined spyre_comms symbol
cd <this worktree>
# apply the torch-spyre patches first -- see Prerequisites
GQA=group K_LAYOUT=head_major V_LAYOUT=head_major \
HEAD_SIZE=128 SENCORES_ATTN=8 SPYRE_LX_RESTICKIFY_RESIDENCY=1 WRITE_KV=0 \
bash .claude-scratch/run_lx_residency.sh
```

Expected:

```
attention: max abs diff vs CPU 1.568e-03  (rel 3.582e-03)
NUMERICS OK
8 gathers pinned LX          (4 K + 4 V; grep 'lx_pinning: buf.* (index) . lx')
restickify proof buf0/buf42/buf112/buf182 -> PASS
HBM roles: tile_output       (only the graph output)
HBM spill pool 66.0 KB       (baseline for comparison: 840 KB)
```

If you get fewer than 8 pinned gathers, stop and diff your env against the line above
before changing anything else.

---

## The strategy, in four parts

### 1. Both caches stored **head-major**, folded on `(page, kv)`

```
K, V:  [num_blocks * num_kv_heads, block_size, head_size]     (was [num_blocks, block_size, num_kv_heads, head_size])
```

Materialised, not viewed, with a rows-outermost device layout so the indexed axis
stays at device position 0. Gather `num_kv_heads` entries per block —
`page * num_kv_heads + kv` — giving `[KV, 1, B, D]`, reshaped to `[KV, B, D]`.

Why this frame and not the obvious one:
- V's gather output **is** the `probs @ V` operand — no permute at all.
- The gather's split lands on **kv**, which is an output axis of `probs @ V`, so the
  consumer can match the gather's per-core view. (A slot-major fold splits the *token*
  axis, which is that matmul's contraction dim and can never be an output split — this
  is why an earlier note wrongly concluded V could never be LX.)
- A token's `head_size` values stay contiguous, so the decode store is a clean row
  scatter. **Do not** be tempted by K stored transposed as
  `[P*KV, head_size, block_size]`: it removes the restickify but has **no working
  decode store at all** (three formulations tried, all fail; the token axis innermost
  puts one element in each 128-byte stick).

### 2. K permuted in the kernel, not in the cache

`k_page.permute(0, 2, 1)` → `[KV, head_size, block_size]` for `q @ Kᵀ`. This is a
restickify, which the pinned build bars unconditionally — hence prerequisite (b).

### 3. GQA query groups unrolled

Replace the single batched matmul over a `[KV, QPK, ...]` query with a Python loop over
`num_queries_per_kv`, each group running its own pair of matmuls against the page as
gathered. Rationale: the batched form makes inductor clone the page out to the group
axis; that clone runs on `kv × group` cores while the page has no group axis, so its
view of the page contracts and the broadcast guard bars LX
(`view covers 8 cores but op runs 32`).

Costs 4× the matmul ops, but they are ops **inside one kernel**, not extra kernels, so
the ~130 µs per-submission cost is not multiplied.

### 4. Core cap of 8, scoped to the attention compile only

With `head_size=128, block_size=64, q_len=1` the attention bmm's only output axis is
`kv=8`, so to reach 32 cores it K-splits the 128-element reduction — and a gather can
never mirror a split on a value-table data dim. Capping cores at 8 removes the K-split
and the views match.

**This must be scoped, or you lose 24 of 32 cores model-wide.** `config.sencores` is
read per compile by `_validate_max_cores()`, verified with a two-graph check: graph
compiled at 8 cores, next graph after restoring 32 cores gets 32.

⚠️ **Subtlety that will bite you:** `torch.compile()` is lazy — the graph compiles on
first *call*, not at decoration. So setting `config.sencores` around the
`torch.compile(...)` call in `_maybe_compile` does nothing. Set it around the
**invocation** of the compiled attention function (`spyre_attn.py:1358`-style call
sites), restoring immediately after. It is an int assignment, so per-forward cost is
nil, and it guarantees the cap is live exactly while the attention graph compiles.

---

## Prerequisites: two torch-spyre patches

Both are in this branch as files. Neither is in the pinned `e02b78b`.

**(a) `#4163` — REQUIRED for the store to be cheap.** `.claude-scratch/` does not carry
it; take it from the local checkout:

```bash
git -C ~/torch-spyre show da15ede -- torch_spyre/ > /tmp/4163.patch
cd <venv>/lib64/python3.12/site-packages && patch -p1 -i /tmp/4163.patch
```

One file, 56 lines, applies cleanly (`e02b78b` is its ancestor). Effect measured:
mutation relayout copies **2 → 0**, HBM pool for a 4 MB cache **4096 → 0 KB**, decode
scatter slope **43.9 → 0.34 µs/MB**. Without it the scatter cost is linear in total
cache size for *both* layouts.

**(b) `#4153`'s restickify local-read proof — REQUIRED for K to reach LX.**
`.claude-scratch/torch_spyre_4153_restickify_proof.patch`, gated on
`SPYRE_LX_RESTICKIFY_RESIDENCY=1`, default off.

**This is my ~160-line port, not the real PR.** It implements only
`_restickify_read_is_core_local` on the cpsat path, dropping the relayout-plan branch,
so it reduces to `per_core_views_equal(producer_write_view, restickify_read_view)`. The
real #4153 also changes the scheduler, codegen and `lx_relayout`. **Before trusting a
latency number, re-confirm against the real stack** — it is draft, default-off, and
sits on main at `a96bb864`, so expect an RPM-compatibility detour (see the
`upgrade-torch-spyre` skill).

**State of the venv as I leave it** (`<venv>/lib64/python3.12/site-packages/torch_spyre/_inductor/`):

| file | state | backup |
|---|---|---|
| `pass_utils.py` | #4153 port applied (adds `per_core_views_equal`) | `pass_utils.py.pre4153` |
| `scratchpad/allocator.py` | #4153 port applied, **gated off** by default | `scratchpad/allocator.py.pre4153` |
| `enforce_indirect_access_layout.py` | **restored to pinned** — #4163 is NOT applied | `enforce_indirect_access_layout.py.pre4163` |

So you must re-apply #4163 yourself; #4153's port is already there but does nothing
unless you export `SPYRE_LX_RESTICKIFY_RESIDENCY=1`.

The venv is **shared with every other worktree on this box**. Restore from the backups
when you are done, and never leave a patch active-by-default.

---

## Correctness prerequisites (independent of residency)

Both were found the hard way. Violate either and the kernel is silently wrong, not
broken loudly.

1. **One index tensor per block, not one table sliced per block.** An index tensor
   reaches the hardware as a real tensor argument, so a per-block slice's nonzero
   storage offset is dropped and *every block gathers block 0's rows*
   (torch-spyre#3770). Alternative fix: pad the table so each row starts on a stick
   boundary — that is exactly why the existing `page_index_table` is
   `[NUM_BLOCKS, INT32_ELEMS_PER_STICK]` and survives. **Your new `(page, kv)` index
   table must do one or the other.**
2. **Take each group's query with `index_select` on the argument**, not `q[:, g]`. A
   compiled region reads a view from offset 0 *and ignores its strides*, so the slice is
   wrong for every `g` including 0, and `.contiguous()` does not help (the clone reads
   the same view). The reproducer uses
   `query.index_select(1, head_ids_g).index_select(0, row_index)`.

---

## Files to change

`spyre_inference/v1/attention/backends/spyre_attn.py`
- `slot_major_kv_layout()` (~148) — add a head-major variant, or generalise.
- `get_kv_cache_shape()` (~1071) — the advertised shape changes. Check what vLLM does
  with it; `initialize_kv_cache_tensors` asserts it matches.
- `_reshape_and_cache_kernel` (~172) and `kv_slot_views` (~1325) — the store's
  destination rows become `(page * num_kv_heads + kv) * block_size + offset_in_block`,
  i.e. `num_kv_heads` rows per token instead of one. `slot_mapping` must be rebuilt
  accordingly (it is built on host, so this is cheap).
- The per-sequence kernel (~288–350) — unroll the group loop, swap the gathers.
- The multi-sequence kernel (~384–430) — has its own gather shape; decide whether to
  port it or leave prefill on the old path (prefill never got K *or* V into LX in any
  configuration, so leaving it alone is defensible).
- The compiled-attention call sites (`self._reshape_fn` is built at ~1159 and called
  at ~1358; the page-attention fn is built via `_maybe_compile` at ~163) — the scoped
  `config.sencores` cap goes around the **call**, not the `torch.compile`.

`spyre_inference/v1/worker/spyre_model_runner.py`
- `initialize_kv_cache_tensors()` (~976–1040) — allocate the new shape and layout.

---

## The A/B

Use the `vllm-bench-latency` skill; it runs the identical benchmark on `main` via a
throwaway worktree. Two notes from prior sessions:
- The skill's docs are outdated — **no env vars and no `-cc.mode` are needed; run the
  plain command.**
- **`git fetch` first.** Local `main` goes stale within hours; A/B against
  `origin/main`, not a stale local ref.

Make it a genuinely long context: `--max-model-len 128` yields exactly **one** page, so
the per-block loop runs once and cannot exercise multi-page behaviour. Use **≥448 input
tokens**, ideally more, so several pages are gathered per step — the LX win scales with
pages per step, so a short prompt will under-report it.

Model: granite-3.3-8b (`head_size=128`, `num_kv_heads=8`, `num_queries_per_kv=4`) is
what all the numbers here assume.

**Never run two Spyre commands concurrently** — one process owns the device; parallel
runs hang or corrupt the compile cache.

### What "success" looks like, and the honest risk

Nothing in this session measured time. The bet is that saving the gathered pages' HBM
round-trip beats running attention on 8 of 32 cores. That bet is genuinely uncertain:
prior work found decode is ~56% weight streaming and only ~29% kernels, and that the
attention pages are small relative to weight traffic. **A neutral or negative A/B is a
real possible outcome** — report it as such rather than hunting for a configuration
that looks good. If it is negative, the most likely lever is removing the need for the
core cap (see below), not tuning the benchmark.

---

## Measurement discipline (learned painfully here)

- **There is no `torch.spyre.synchronize`.** To time device work, queue N launches and
  force a drain. **Do not** sync with `.cpu()` on a slice of a big tensor — `.cpu()` on
  a view materialises the whole base, which scales with cache size and silently
  inflated my scatter slopes ~2×. Difference `2N` against `N` launches so any constant
  probe cost cancels. `.claude-scratch/bench_kv_layout_scaling.py` does this correctly.
- **Compile, and prove it.** Use `fullgraph=True` and assert the SDSC kernel count; an
  eager fallback otherwise looks like a measurement.
- **Always include a positive control.** A flat scaling curve means nothing unless you
  show the benchmark can detect the pathology — the default-stickified gather (G0,
  19.7 µs/MB) is that control.
- **The SDSC is authoritative for residency, not the cost model.** Check
  `lx_pinning: <buf> ... -> lx` in the planner log and `allocate-Tensor*_lx` in
  `sdsc_*.json`; `SPYRE_DUMP_COST` reflects pre-demotion layouts.
- **Fresh `TORCHINDUCTOR_CACHE_DIR` per run** or a warm cache emits no planning logs.
- fp16 noise floor is ~2e-3 absolute (one ulp at magnitude 1). Real failures here were
  rel ≈ 1.0, so there is no ambiguity — but do not read 3e-3 as an error.

---

## Open questions, in the order I would attack them

1. **Can the core cap be removed?** It is the only ugly part. `BLOCK_SIZE=256` removes
   the K-split on its own (measured: bmm takes 32 cores as `((2,4),(3,8))` with no
   K-split) but then the consumer splits two device axes while a gather can only split
   its one entry axis, so the page drops back to HBM. Closing that needs #4153's
   *other* flag, `SPYRE_LX_FUSED_SPLIT_VIEWS` ("decompose one split fused loop into an
   exactly equivalent multi-device-axis PerCoreView"). Untested. If the A/B is hurt by
   the 8-core cap, this is the path.
2. **Prefill.** Untouched, and never had K or V in LX in any configuration.
3. **The store inside the real kernel.** My store measurements were on isolated
   micro-kernels (`bench_kv_store_frames.py`), never end-to-end with the attention
   graph — `WRITE_KV=0` in every residency run.
4. **`torch-spyre#4258`** (`704e419`, collapsed-axis rescue) is **not** on this path.
   It never fired in any run, because an earlier guard always rejected first. Do not
   spend time on it unless (1) above changes that.
5. The 66 KB pool is dominated by the query-gather intermediate
   `query.index_select(1, head_ids)` = `[T, KV, D]` = 16 KB, materialised per group per
   block. Gathering the row *first* would make each 2 KB. Minor, but free.

---

## Artifacts on this branch

| file | what |
|---|---|
| `repro_lx_residency.py` | the standalone reproducer; all knobs documented in its docstring |
| `run_lx_residency.sh` | driver (sources `env.sh`, fresh artifact dir) |
| `bench_kv_layout_scaling.py` | gather/scatter scaling vs cache size, with positive control |
| `bench_kv_store_frames.py` | decode-store cost for the three candidate cache frames |
| `torch_spyre_4153_restickify_proof.patch` | my port of #4153's local-read proof |
| `torch_spyre_4258_collapsed_axis.patch` | #4258, kept for reference; not needed |

Commits, oldest first: `6648fac` (fold + group knobs) → `33be7f1` (restickify barrier
removed) → `5c37334` (K in LX, numerics verified) → `2a5d477` (scoped core cap) →
`ff0e1c4` (V in LX) → `ec40327` (store measurements) → `20ce513` (scaling benchmark) →
`71fd62f` (probe-cost fix, #4163 measured) → this hand-off.
