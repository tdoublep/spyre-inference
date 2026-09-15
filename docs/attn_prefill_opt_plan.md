# Optimising chunked-prefill attention: how to run the experiments

Handoff notes for a fresh attempt. This is deliberately about **method and options**,
not conclusions — measure and decide for yourself.

## Target scenario

Chunked prefill, `query_len=512`, `context_length=4096`, `batch_size=1`, on
granite-3.3-8b's shape (`num_query_heads=32`, `num_kv_heads=8`, `head_size=128`,
`block_size=128`, fp16). Goal: beat `main`'s attention kernel on it.

Prefill and decode are different workloads with different amounts of parallelism.
Measure both; a change that helps one can hurt the other, so gate per-shape rather
than assuming one kernel suits everything.

## Environment — get this right first

**Pin the compiler flags to whatever the deployment uses.** The harness inherits the
compiler's environment and pins nothing itself, so an unpinned flag means you compile
different code than production runs from the same source. In particular:

```bash
export LAYOUT_SOLVER=greedy      # non-negotiable: it changes the generated kernel
export SPYRE_NUM_CPUS=8
```

This is the whole reason for the restart. Set it for **every** microbench run, not
just end-to-end runs.

**Point PYTHONPATH at your own checkout.** The shared venv's editable install of
`spyre_inference` may map to a *different* worktree. Without this you will silently
benchmark someone else's code:

```bash
export PYTHONPATH=<your-worktree>:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/senlib/lib
export SPYRE_ATTN_PROFILING=1    # required, and read at import time
export VLLM_PLUGINS=spyre_inference
```

Use `/home/senuser/spyre-inference/.venv/bin/python3` directly rather than `uv run`,
which re-syncs deps and can revert a local build (`--no-sync` if you must use it).

**One Spyre process at a time.** The device is contested; concurrent runs hang,
corrupt device state, or corrupt the compile cache. No `pytest -n`, no backgrounding
one run while starting another.

## Running the microbenchmark

```bash
python3 scripts/microbench/spyre_attn_microbench.py \
    --config scripts/microbench/configs/<config>.json --no-output
```

Read `scripts/microbench/README.md` in full before trusting a number. The parts that
matter most:

- `--span layer` (the default) is the only span that closes after a device sync, so
  it is the only one whose total cannot silently drop a kernel. The leaf spans
  under-report; use them only to apportion cost *within* a layer.
- Declared shapes are **padded** onto the bucket lattice before the kernel sees them,
  so put `max_model_len` / `max_num_batched_tokens` / `max_num_seqs` and any
  `attn_kv_buckets` pin in the config. The same declared shape lands on different
  kernels under different ladders.
- `ms` is **device time**, median over `--iterations` separate profiled windows. Not
  wall clock, not comparable to GPU numbers.
- Watch these columns every time: `allclose_pass`, `max_abs_diff`, `fallback_clean`,
  `late_compile`, `kernels_attributed` vs `kernels_expected`, `memory_share_pct`.
- The correctness gate runs before timing. A fast-but-wrong kernel will show up as
  `allclose_pass=False` or a `max_abs_diff` far above the others — take an unexplained
  jump in `max_abs_diff` seriously even when the gate passes.

Write a config for the target scenario rather than reusing one whose lattice differs.
`num_blocks` must cover `num_seqs * ceil(max_model_len / block_size)`.

Each run costs minutes (vLLM startup plus Inductor compiles), so prefer one config
with the shapes you need over broad sweeps. Run in the **foreground** with a generous
timeout: output piped through `grep`/`tail` buffers until the process exits, so a
backgrounded run can look silent or stalled when it is fine.

## A/B protocol

Put the change behind an env knob so both arms are the **same build** and the kernel
is the only variable, and make one setting reproduce `main`'s path exactly. Verify
that control first: if it does not match a `main` baseline within noise, the harness
or the wiring is wrong and no other number means anything. The value is read at
import time, so use one process per setting.

Keep a `main` baseline measured in the same session and environment. Do not compare
against numbers from an earlier session.

## Validate against end-to-end

A microbench win is a hypothesis until an end-to-end run agrees. Use the deployment's
own command shape:

```bash
LAYOUT_SOLVER=greedy SPYRE_NUM_CPUS=8 SPYRE_ATTN_KV_BUCKETS=2048 SPYRE_BATCHED_DECODE=1 \
vllm bench latency --model ibm-granite/granite-3.3-8b-instruct \
    --input-len 1984 --output-len 1 --batch-size 4 \
    --num-iters-warmup 2 --num-iters 1 --max-model-len 2048
```

- `--num-iters 1` is a single sample. Raise it, or repeat the run, before believing a
  difference.
- Add `--profile --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=DIR
  --profiler-config.torch_profiler_use_gzip=false` to get a trace. The AIU device
  timeline **truncates** on a whole-request capture (`Exceeded max AIU buffer count`),
  so check that the attention kernels' captured call count matches what the step shapes
  imply before drawing conclusions from device totals.
- In the trace, attention kernels are the fused ones containing `amax`. Distinct kernel
  *names* differing only in trailing hash mean Dynamo specialised more than once for
  what you thought was one shape.
- **Do not assume the step shapes.** Log what the metadata builder actually receives
  (`num_seqs`, `query_lens`, `seq_lens`, `aligned_query_lens`) for a real run and
  benchmark those, rather than the shape you assumed the scheduler produces.
- Watch for `compiled outside warmup` in the log: it means the warmup recorder is not
  tracing the variant dispatch requests, and a request pays an Inductor compile. Note
  it is `warning_once`, so one line can hide many.

## Measured results (head-major round)

All on `q=512, ctx=4096, bs=1`, granite-3.3-8b's shape, `LAYOUT_SOLVER=greedy`,
`SPYRE_NUM_CPUS=8`, torch-spyre `c3d949a`, one process per arm, every arm gated
against the harness's CPU reference. Baselines measured in the same session.

| arm | device | note |
|---|---|---|
| token-major per page (shipped) | 9925 us | |
| token-major, experimental kernel at `base` | 9875 us | control: reproduces the above |
| head-major per page (PR #890) | 10269 us | 1.04x slower than token-major |
| fold `scale` into the query | 9996 us | no win |
| skip the all-zero mask adds | 9892 us | no win |
| both | 9982 us | no win |
| head-major, 2 pages/step | 10903 us | |
| head-major, 4 pages/step | 10227 us | |
| head-major, query tile 128 x 8 pages/step | 14808 us | |

Nothing beat the shipped per-page kernel. Four conclusions worth not re-deriving:

**The elementwise chain is already fused.** Folding the scale into the query and
skipping the provably all-zero mask adds are each worth a full pass over a 4.2 MB
score tile per page on paper, and both measure as noise. Inductor fuses scale, mask
add, `amax` and `exp` into one pass, so removing operands from that chain buys
nothing. Do not re-litigate the score-tile *traffic* without first checking the
fused kernel breakdown.

**Head-major fixes the chunking penalty but does not win.** On the *same* kernel,
4 pages per step costs 14345 us on token-major and 10227 us on head-major -- 1.40x,
because a head-major multi-page gather permutes whole contiguous
`[block_size, head_size]` tiles while token-major transposes within each page. It
still does not get under the per-page baseline. Chunking the KV axis is not the
lever, under either layout.

**Splitting the query axis is worse, even LX-resident.** Query tile 128 x 8 pages
keeps the score tile at 0.25 MB/core against 2 MB of LX and is still 1.44x slower
than the baseline. Residency was not the binding constraint.

**Folding the GQA group axis into the matmul's M does not compile.** Replacing the
4-D `matmul(q[KV, groups, rows, D], k[KV, 1, D, n])`, whose K broadcasts over the
group axis, with a 3-D `bmm` of `[KV, groups*rows, D]` is rejected by torch-spyre's
`batchmatmul` stick validator (`Input2 stick must contain generated_sym`). The 4-D
form with the broadcast is the only shape family the backend accepts here.

### A correctness bug in head-major multi-page gathers

`k_pages[page_ids]` with a `[n, 1]` index on the head-major cache returns wrong
data. Same kernel, same formulation, only the layout differs:

| `chunk4` | max_abs_diff |
|---|---|
| token-major | 0.000168 |
| head-major | 0.291 |

Every head-major arm that gathers more than one page lands at 0.276-0.301 against
the harness's `atol=0.3`, so it slips past the gate rather than failing loudly.
PR #890's own kernels gather a single page with `index_select` and are unaffected,
but any multi-page head-major kernel needs this understood first -- and it means the
chunked timings above are partly timing the wrong work.

## Things to try

Neutral list of the design space. Each is worth measuring; none is known to win.

**KV cache layout.** Two independent levers, easy to conflate:
- the *logical* shape (`get_kv_cache_shape`), e.g. token-major
  `(blocks, block_size, kv_heads, head_size)` vs head-major
  `(blocks, kv_heads, block_size, head_size)`;
- the *device* layout for the same logical shape, via `SpyreTensorLayout`'s `dim_order`
  (the microbench exposes `--kv-layout`, including a `page_major` probe).
Identify what a layout actually becomes by inspecting `device_size` / `stride_map` on
the constructed layout rather than trusting a comment. Note the KV *write* path
(`reshape_and_cache`, `kv_slot_views`) constrains which layouts are usable, so a fast
read layout is not automatically adoptable.

**Softmax structure.** Online softmax over KV chunks (what `main` does per block) vs a
plain single softmax over a gathered range. The trade is running-max/rescale
bookkeeping against the size of the materialised score tensor.

**Chunking axis.** KV length, query length, heads, or combinations. Only the KV axis
forces online-softmax bookkeeping; query and head splits are independent and need
none. `wiki/concepts/attention-on-spyre.md` in the knowledgebase has the `BLOCK_Q` /
`CORES_PER_Q_TILE` co-design table and the KV-read-once invariant — worth reading.

**Score-tile sizing.** Compute the score tensor's bytes for a candidate tiling
(`kv_heads * queries_per_kv * padded_query_len * chunk_len * 2`) and compare against
per-core LX (2 MB × 32 cores). Whether it stays resident is the thing to verify, e.g.
via `memory_share_pct` and the kernel breakdown.

**Gather formulation.** `index_select` with a 1-D index vs advanced indexing with a
2-D `[n, 1]` index (see `batched_decode.py` for the latter and its rationale), and
whether the gathered pages are merged into one KV-token axis or kept as a matmul batch
axis. These differ in how the backend can split work across cores. Some formulations
fail loudly (Inductor `Unsupported`, per-core span limits), some produce wrong numbers
quietly, and some fault the device (`RAS::...ComputeHardwareError`) — so check the
correctness gate and scan for RAS errors on every new formulation, and run each new
one in a fresh process since a fault poisons the stream for everything after it.

**Query-group handling under GQA.** Broadcasting one K tile across the query-group axis
vs unrolling the groups. See PR #783 for prior art on this and on an LX-resident KV
layout — it targets *decode*, which has different constraints, so treat it as a source
of technique rather than of conclusions.

**`block_size`.** 64 and 128 are both legal (multiple of 64 required). It interacts
with chunk width and with per-core buffer limits. Production gets 128 from vLLM's CPU
platform default.

**Use the stack's own attention instead of hand-writing one.** torch-spyre has an SDPA
decomposition registered for `aten._scaled_dot_product_fused_attention_overrideable`
with coarse tiling via `spyre_hint` (see `torch_spyre/_inductor/decompositions.py`),
plus native `torch.ops.spyre.sliding_window_attention` / `spyre.kv_window`. A properly
tiled library kernel may beat a hand-written op chain; the open question is how to
reach it from a *paged* KV cache.

## Ground yourself in the constraints

- `head_size` and `block_size` must be multiples of 64 (128-byte stick / 2 bytes fp16).
- fp16 only.
- A compiled region reads its arguments from storage offset 0 and ignores
  `storage_offset` (torch-spyre#3770) — pass separately-allocated tensors, not slices.
- Attention compiles in its own domain and warmup records every variant the bucketer
  enumerates; a new kernel needs `_record_one` to trace the same key dispatch requests,
  or requests pay compiles.
- `FallbackWarning` means an op silently went to CPU and the numerical path changed:
  `-W "error::torch_spyre.ops.fallbacks.FallbackWarning"` turns it into a traceback.
- Read `site-packages/torch_spyre/ops/{eager,fallbacks}.py` before assuming a gap is
  your bug, and search torch-spyre issues/PRs for the feature area first.

## Operational traps

- `pgrep -f "<pattern>"` matches its **own** shell command line — a liveness check
  built on it always reports "running". Use the PID's `etime`, and sort descending:
  Inductor's forked compile workers inherit the parent's cmdline and skew a `head -1`.
- Backgrounded jobs here can be restarted, truncating a `>` redirect and losing output.
  Prefer foreground runs with long timeouts for anything you need the output of.
- Git identity may be unset. DCO needs the sign-off email to match the GitHub account
  (`gh api user`), and commits need `-s`.
- Use `$CLAUDE_JOB_DIR/tmp` for scratch scripts and logs.
