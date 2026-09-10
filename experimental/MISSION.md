# Mission: make a batched decode attention kernel beat the per-sequence one

## The goal

Design a **batched** decode attention kernel — one that serves all `num_seqs`
sequences of a decode step in a single kernel — that **outperforms the
per-sequence approach**, and demonstrate it in
`experimental/microbench_batched_decode.py`.

The bar to clear is the `per_seq` variant. The `unrolled` variant shows the bar
is not merely reachable but beatable, since a single dispatch at per-sequence
cost already edges it out — so a genuinely batched kernel that shares work
across sequences ought to do better still.

Work in the microbenchmark. Do not change `spyre_inference/` until the
microbenchmark shows a win.

## Why it matters

`SPYRE_BATCHED_DECODE=1` exists for exactly this case and currently makes things
much worse. On granite-3.3-8b at batch 4, input 1984, block 128:

| | attention, per decode step | decode device time per step |
|---|---|---|
| per-sequence (default) | 122.1 ms | 238.2 ms |
| batched kernel | 303.2 ms | 384.7 ms |

That is a **1.61× decode slowdown**, and attention goes from 21% to 51% of
decode. Measured ITL at batch 4 is 293.09 ms/step against 175.90 ms at batch 1;
attention is the only component that scales with batch size, so it is the whole
reason ITL does not stay flat as you batch.

## Current numbers

`tpa-spyre-dev-2`, granite shapes, `LAYOUT_SOLVER=greedy`, 2026-09-10:

| variant | dev ms/call | kernels | wall med ms | host resid | lossy | warmup s |
|---|---|---|---|---|---|---|
| per_seq | 2.343 | 4 | 3.129 | 0.786 | 0 | 30.6 |
| batched | 7.693 | 1 | 8.096 | 0.403 | 32 | 37.5 |
| unrolled | 2.267 | 2 | 2.905 | 0.638 | 0 | 123.0 |

Effective KV bandwidth: per_seq 14.3 GB/s, batched 4.4 GB/s, unrolled
14.8 GB/s. The batched kernel moves the same 33.55 MB of KV per layer as the
other two and takes 3.3× longer doing it.

The harness is faithful to production where it matters: batched device time is
within **1.5%** of the profiled vLLM run (7.693 vs 7.581 ms/layer) and its KV
bandwidth matches exactly at 4.4 GB/s. `per_seq` is optimistic by ~23% (2.343
vs 3.054 ms/layer) because the harness skips the metadata handling and output
scatter production also does per call.

## How to reproduce

### Exact environment the numbers were taken with

Reproducing the *ranking* needs only a working Spyre host. Reproducing the
absolute numbers needs this stack:

| component | value |
|---|---|
| host | `tpa-spyre-dev-2`, 4 AIU devices present, 1 used (TP=1) |
| spyre-inference | `main` @ `0da2a5e` (the base of this branch) |
| torch-spyre | `AdnanHoque/torch-spyre` @ `927c3b83f1e4ebfb46a376f75db905c8229b8fd9` |
| torch-spyre build var | `USE_SPYRE_PROFILER=1` |
| vLLM | `v0.28.0`, built with `VLLM_TARGET_DEVICE=empty` |
| torch | `2.13.0+cpu` |
| `LAYOUT_SOLVER` | `greedy` |
| `SPYRE_NUM_CPUS` | `8` |

Two caveats, because neither matches what this branch checks in:

- **The torch-spyre rev is not the one in `pyproject.toml`.** The committed pin
  is upstream `torch-spyre/torch-spyre` @
  `76978116cb44020b50c8df1a1822d5178629922a`. The runs used the tip of
  torch-spyre PR **4347** ("LX planning: choose work divisions using placement
  cost"), a draft branch on a fork that already contains a merge of torch-spyre
  `main`. That PR rewrites the LX work-division planner — the same machinery
  that emits the fallback warnings discussed below — so it is directly relevant,
  and results may differ on the committed pin.
- **The build had the profiler compiled in** (`USE_SPYRE_PROFILER=1`, versus
  `"0"` on this branch), because the same build served the vLLM profiling runs.
  That inflates wall clock by roughly 30% but not device kernel time, which is
  measured on-device. So `dev ms/call` should carry over; `wall med ms` may
  improve on a `USE_SPYRE_PROFILER=0` build.

To recreate that stack:

```bash
# point the pin at the PR branch tip and build with the profiler enabled
sed -i 's|^torch-spyre = { git = .*|torch-spyre = { git = "https://github.com/AdnanHoque/torch-spyre", rev = "927c3b83f1e4ebfb46a376f75db905c8229b8fd9" }|' pyproject.toml
sed -i 's|^USE_SPYRE_PROFILER = "0"|USE_SPYRE_PROFILER = "1"|' pyproject.toml

source ~/spyre-libs/env.sh
export SEN_COMMON_HEADERS="${SENTIENT_BASE_INSTALL_DIR}/runtime/include"
uv cache clean torch-spyre   # mandatory: the wheel cache keys on the git rev
                             # alone, so a build-var change is otherwise ignored
uv sync --group dev --reinstall-package torch-spyre
```

### Pinned RPMs

The runs did **not** use the host's `/opt/ibm/spyre`. They used the pinned set
from `spyre-rpms.lock`, unpacked into `~/spyre-libs` and activated via
`~/spyre-libs/env.sh`, which the wrapper sources. On this host the installed
`ibm-*` RPMs were *newer* than the lock, so skipping this measures a different
stack.

```bash
bash scripts/install-pinned-rpms.sh --rebuild    # unpack + rebuild torch-spyre
source ~/spyre-libs/env.sh
ldd "$(command -v dxp_standalone)" | grep libdxp  # must resolve under ~/spyre-libs
```

x86_64 pins in effect (lock stamp
`34d30a1e912a86c5eaea8d7735287a311ecbeb087658cc8ea4553bc89190c7bb`):

| package | version |
|---|---|
| ibm-aiu-toolbox-e2e | `2.0.0-0.main.1+28.47d9b91_100` |
| ibm-deeptools | `2.0.0-0.main.1+2429.561f418_334` |
| ibm-deeptools-devel | `2.0.0-0.main.1+2429.561f418_334` |
| ibm-flex | `2.0.0-0.main.1+540.7c58aa7_370` |
| ibm-flex-devel | `2.0.0-0.main.1+540.7c58aa7_370` |
| ibm-senlib-core | `2.0.0-0.main.1+268.c58bbc0_250` |
| ibm-senlib-dd2 | `2.0.0-0.main.1+268.c58bbc0_250` |
| ibm-senlib-headers | `2.0.0-0.main.1+268.c58bbc0_250` |
| ibm-spyre-comms | `1.0.0-0.main.1+144.d3875d2_177` |
| ibm-spyre-comms-devel | `1.0.0-0.main.1+144.d3875d2_177` |
| ibm-libaiupti | `2.0.0-0.main.1+27.e289f7d_12` |

`torch-spyre` must be rebuilt against whichever set is active or it fails at
import with an undefined `flex::`/`senlib::` symbol — hence `--rebuild`.

### Running it

Prerequisites:

- A Spyre host with a free device. The device takes one process at a time —
  never run two Spyre commands concurrently.
- `torch-spyre` built against whichever `ibm-*` libraries are active.
- No model weights or HF cache needed; the harness builds its own tensors.

Run:

```bash
bash experimental/run_microbench_batched_decode.sh
```

Takes **2.5–3.5 minutes** for all three variants (3 min 29 s cold, 2 min 24 s
with a warm inductor cache). It is almost entirely compile: of a 144 s run, 134 s
was the `warmup s` column (per_seq 23 s, batched 30 s, unrolled 82 s) plus ~35 s
of fixed startup. `--iters` barely affects the clock, so iterate with
`--variants batched,<your_new_one>` rather than by trimming iterations.

The wrapper sources `~/spyre-libs/env.sh` when present, sets
`LAYOUT_SOLVER=greedy` and `SPYRE_NUM_CPUS=8` (the environment the numbers above
were taken with), and passes `--no-sync` so `uv` cannot silently swap out a
locally built `torch-spyre`. It echoes the environment it chose, so check that
line matches what you expect before trusting a comparison.

Arguments are forwarded:

```bash
# a subset, to iterate faster -- each variant costs a separate compile
bash experimental/run_microbench_batched_decode.sh --variants per_seq,batched

# the other block-size / trip-count point
bash experimental/run_microbench_batched_decode.sh --num-blocks 8 --block-size 256

# print each work-division fallback message rather than only counting them
bash experimental/run_microbench_batched_decode.sh --show-warnings

# the other layout solver (see Leads)
LAYOUT_SOLVER=cpsat bash experimental/run_microbench_batched_decode.sh
```

Reading the output: `dev ms/call` is the sum of device kernel durations from the
torch profiler, which is the number to optimise. `wall med ms` includes host
time; `host resid` (wall − device) is where launch overhead lives. `lossy`
counts torch-spyre work-division fallbacks emitted while compiling that variant.

`lossy` only sees warnings from an actual compile, so a warm inductor cache
reports 0 even for a variant that does fall back — `batched` showed 32 cold and
0 on a re-run. Treat `lossy = 0` as meaningful only on a cold compile.

**Two fidelity rules.** Break either and the comparison is worthless — with
either wrong, `per_seq` collapsed to 1.7 GB/s and the ranking *inverted*, making
the batched kernel look 1.7× faster than per-sequence:

1. `k/v_pages` must be allocated as `initialize_kv_cache_tensors` does: host
   zeros then `.to(device, device_layout=slot_major_kv_layout(...))`. A plain
   `convert()` leaves the default tiled layout, which spreads the slot index
   across two device dims (torch-spyre#3705) and makes the in-graph page gather
   slow for every variant.
2. `query` must be the production staging width
   (`max_num_batched_tokens + 1` = 513 rows), not `num_seqs`. The batched kernel
   takes a `query[:num_seqs]` prefix of it and the per-sequence kernel gathers
   out of it.

Add a new variant by writing the kernel, compiling it with
`torch.compile(fn, dynamic=False)`, and adding a zero-argument closure to the
dict returned by `make_callables`. It then works with `--variants`.

## Leads — work division is NOT the only suspect

The batched kernel is the only variant that trips work-division fallbacks: 32 ×
`lossy work-division scheduler transport ... output:d4=absent`, matching the
count in the full vLLM run exactly. The mechanism is real — in
`torch_spyre/_inductor/pass_utils.py`, `finalize_work_division_for_scheduler`
warns when a symbol chosen to split work has coefficient 0 in the output's
memory index, i.e. work was divided along an axis absent from the output, which
for a reduction means the contraction axis.

**But that is a correlation, not a demonstrated cause.** Nothing has priced the
fallback. The warning is emitted and the compile continues; no measurement here
attributes any microseconds to it. Do not let 32 loud warnings anchor the
investigation. Other candidates, each testable in this harness:

1. **Layout solver.** Every number above used `LAYOUT_SOLVER=greedy`, because
   that is what the vLLM benchmarks ran with. torch-spyre defaults to `cpsat`.
   The batched kernel's shapes are exactly the kind a greedy solver could
   mishandle. This is one flag and should be tried first — if `cpsat` fixes it,
   the kernel was never the problem.
2. **The page gather pattern.** Batched does one `index_select` of `num_seqs`
   non-adjacent pages; per-sequence does a width-1 gather. Under the slot-major
   layout those pages are far apart, so the wide gather may hit a different
   access path entirely.
3. **The matmul lowering.** Per-sequence uses batch axes `(KV, queries_per_kv)`
   with K broadcast on a size-1 axis and `M=1`. Batched uses `(num_seqs, KV)`
   with `M=queries_per_kv`. Different `lower_bmm` form, different M — either
   could dominate.
4. **The permute.** Batched permutes the gathered 4-D pages `(0,2,1,3)`;
   per-sequence permutes 3-D then unsqueezes. Check whether the 4-D form stays a
   view or materialises.
5. **Block-loop trip count.** The loop runs `num_blocks` times sequentially
   (16 at block 128 / 2048 KV). `--num-blocks 8 --block-size 256` halves it.
6. **The online-softmax chain.** The per-tile `maximum`/`exp`/`mul` rescale runs
   on `(num_seqs, KV, queries_per_kv, block_size)` instead of
   `(KV, queries_per_kv, 1, block_size)`. Worth isolating from the matmuls.

A useful discipline: when a variant is faster, say *which* of these changed, and
confirm with a second variant that changes only that.

## Constraints already discovered

Two reformulations do not compile, so budget for backend limits:

- Slicing per-sequence page indices out of the batched `block_ids` at column
  offset ≠ 0 (`block_ids[i, s:s+1]`) fails with `Unexpected stick expression 1:
  expected Mod(var, 32), a bare variable, 0, or any of those with a constant
  offset`. Per-sequence page indices need their own stick-aligned table with the
  index at column 0, which is what the non-batched path already builds.
- Folding `(num_seqs, KV)` into a single bmm batch axis via
  `.permute(0,2,1,3).reshape(num_seqs*KV, block_size, head_size)` fails with
  `Incompatible host_size and dim_order`. The axis merge the batched kernel's
  own comment warns about ("merging these axes is what materializes the page")
  is not merely expensive, it is rejected.

`unrolled` is also not a drop-in replacement for the batched path: `num_seqs`
bakes into the graph, so it needs one compiled variant per sequence bucket, and
it consumes per-sequence metadata rather than the batched kernel's.

## Open question

`unrolled` reports 2 device kernels for 4 sequence bodies, so something is
fusing across sequences. Understanding what may point at how a real batched
kernel should be shaped.
