# Benchmark: PR #789 + PR #784, Granite 3.3 8B Instruct

Online serving benchmark of the two PRs merged together, run on Spyre against the
pinned libs in `~/spyre-libs/env.sh`.

- **Date:** 2026-09-06
- **Host:** tpa-spyre-dev-4 (4 AIU devices found, TP=1, single device used)

## Commands

Both server and client sourced `~/spyre-libs/env.sh` first, and used
`PYTHONPATH=<worktree>` so the merged worktree code took precedence over the
editable install in the venv.

### Server

```bash
source ~/spyre-libs/env.sh
export PYTHONPATH=/home/senuser/spyre-inference/.claude/worktrees/bench-789-784:$PYTHONPATH
export SPYRE_ATTN_RECORD=1
export SPYRE_NUM_CPUS=8

vllm serve \
    --model ibm-granite/granite-3.3-8b-instruct \
    --max-model-len 8192 \
    --max-num-seqs 32
```

### Client

```bash
source ~/spyre-libs/env.sh

vllm bench serve \
    --backend vllm \
    --model ibm-granite/granite-3.3-8b-instruct \
    --endpoint /v1/completions \
    --dataset-name custom \
    --dataset-path /models/online_benchmarking_data_reordered/aiops_results_2025.11.03_e2ee1b0_correct_order.jsonl \
    --num-prompts 100 \
    --max-concurrency 4 \
    --output-len -1
```

Only `--max-concurrency 4` was run to completion. Runs at concurrency 2 and 1 were
not completed and are not reported here.

## Environment

### Code under test

| Component | Version |
|---|---|
| Merge commit benchmarked | `389da37` |
| Base `main` | `c81a50c` |
| PR #789 `attn-zero-serving-compiles` tip | `f5c9fac` |
| PR #784 `warmup-lm-head-graphs` tip | `75fd29f` |

Both PRs merged cleanly into `c81a50c` with no conflicts, despite both touching
`spyre_inference/v1/worker/spyre_model_runner.py`.

PR #789 commits:

```
f5c9fac attn: drop stale comments, point the staging +1 at torch-spyre#4033
8292241 attn: address review — trim comments, rename dest, move staging helper
cc77650 attn: drop the flag resolvers from the attention cache key
25ec261 attn: drop the unreachable un-fused store variants
b9aacea fix(attn): make recorded attention graphs actually reachable
```

PR #784 commits:

```
75fd29f refactor: say buckets instead of ladder, fix the slice rationale
e3f531e perf(lm_head): compile the projection instead of only warming it
2467ae8 perf(warmup): compile the lm_head projection during warmup
```

### Dependencies

| Package | Version / rev |
|---|---|
| `torch` | 2.13.0+cpu |
| `vllm` | 0.28.0 (`rev=v0.28.0`, built `VLLM_TARGET_DEVICE=empty`) |
| `torch-spyre` | `e02b78ba35f9a1a69a458c3149e9c01d9f4fa6a8` |
| `transformers` | 5.16.1 |
| `spyre_inference` | 0.1.dev359 (editable, overridden by `PYTHONPATH`) |

The installed `torch_spyre` `direct_url.json` was confirmed to match the pinned
rev — not a stale local wheel.

### Pinned libs

`~/spyre-libs/env.sh` sets `SENTIENT_BASE_INSTALL_DIR=/home/senuser/spyre-libs/opt/ibm/spyre`
and re-sources `/etc/profile.d/ibm-aiu-setup.sh`, which prepends the pinned paths
ahead of the system `/opt/ibm/spyre`:

```
/home/senuser/spyre-libs/opt/ibm/spyre/spyre-comms/lib
/home/senuser/spyre-libs/opt/ibm/spyre/runtime/lib
/home/senuser/spyre-libs/opt/ibm/spyre/deeptools/lib
/home/senuser/spyre-libs/opt/ibm/spyre/senlib/lib
/home/senuser/spyre-libs/opt/ibm/spyre/sentinyexec/lib
```

Installed `ibm-*` RPMs:

```
ibm-aiu-toolbox-e2e-2.0.0-0.main.1+28.47d9b91_89.el10.x86_64
ibm-deeptools-2.0.0-0.main.1+2324.4cfac4c_309.el10.x86_64
ibm-deeptools-devel-2.0.0-0.main.1+2324.4cfac4c_309.el10.x86_64
ibm-flex-2.0.0-0.main.1+495.a86bb35_315.el10.x86_64
ibm-flex-devel-2.0.0-0.main.1+495.a86bb35_315.el10.x86_64
ibm-libaiupti-2.0.0-0.main.1+21.bd7054d_0.el10.x86_64
ibm-senlib-core-2.0.0-0.main.1+244.de8291a_214.el10.x86_64
ibm-senlib-dd2-2.0.0-0.main.1+244.de8291a_214.el10.x86_64
ibm-senlib-headers-2.0.0-0.main.1+244.de8291a_214.el10.x86_64
ibm-spyre-comms-1.0.0-0.main.1+124.3523cde_149.el10.x86_64
ibm-spyre-comms-devel-1.0.0-0.main.1+124.3523cde_149.el10.x86_64
```

### Env vars

| Var | Value | Note |
|---|---|---|
| `SPYRE_ATTN_RECORD` | `1` | Set explicitly. `envs.py` defaults this to `0`. |
| `SPYRE_NUM_CPUS` | `8` | Clamped `OMP_NUM_THREADS` 192 → 8, plus OPENBLAS/MKL/NUMEXPR/VECLIB. |
| `HF_HOME` | `/models/huggingface_cache` | Pre-existing; model served from cache. |
| `VLLM_PLUGINS` | `spyre_inference` | Pre-existing. |

### Effective config after platform overrides

| Setting | Value |
|---|---|
| `dtype` | `torch.float16` |
| `max_model_len` | 8192 |
| `max_num_seqs` | 32 |
| `max_num_batched_tokens` | **512** (capped from default by platform) |
| `block_size` | **128** |
| `num_gpu_blocks_override` | 2049 (= 32 seqs × 64 blocks/seq + 1 null) → 262,272 tokens |
| `compilation_config.mode` | `STOCK_TORCH_COMPILE` |
| `compile_sizes` | `[1, 2, 4, 8, 16, 32, 512]` |
| `enable_prefix_caching` | `True` (left enabled) |
| `enable_chunked_prefill` | `True` |
| LM head vocab padding | 49216 → 51200 |

Note on `block_size`: `platform.py` logs `Overriding block_size from 16 to 64`, but
`super().check_and_update_config()` (CpuPlatform) then sets it to 128 at
`cpu.py:148-149` because `user_specified_block_size` is false. The run uses 128.

## Warmup / recorded buckets

| Recorder | Axis | Buckets |
|---|---|---|
| Outer, body | `compile_sizes` | 1, 2, 4, 8, 16, 32, 512 (7) |
| Outer, lm_head (#784) | logits row widths | 1, 2, 4, 8, 16, 32 (6) |
| Inner, attention | KV len | 128, 256, 512, 1024, 2048, 4096, 8192 (7) |
| Inner, attention | query len | 1, 512 (2) |

`SpyreAttnBucketer: 7 kv buckets [128..8192], 2 query buckets [1..512], max num_blocks=64`

Recorded attention variants — the full 7 × 2 cross product, nothing pruned:

```
num_blocks ∈ {1, 2, 4, 8, 16, 32, 64}  ×  padded_query_len ∈ {1, 512}
```

`Attention graph recording complete: 560 graphs in 532.438s` (14 variants × 40 layers).

PR #784's logits projection compile fired:
`Compiling SpyreUnquantizedLMHeadMethod.apply as its own graph`.

### Startup cost

| Phase | Duration |
|---|---|
| Weight load | 1.4 s |
| Body warmup (7 buckets + 6 logits widths) | 141.6 s |
| Attention graph recording (560 graphs) | 532.4 s |
| **Total, process start → `Application startup complete`** | **~13.2 min** |

## Results — `--max-concurrency 4`

```
============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Maximum request concurrency:             4
Benchmark duration (s):                  3359.11
Total input tokens:                      167398
Total generated tokens:                  5414
Request throughput (req/s):              0.03
Output token throughput (tok/s):         1.61
Peak output token throughput (tok/s):    4.00
Peak concurrent requests:                6.00
Total token throughput (tok/s):          51.45
---------------Time to First Token----------------
Mean TTFT (ms):                          11082.64
Median TTFT (ms):                        8606.83
P99 TTFT (ms):                           51299.87
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2251.23
Median TPOT (ms):                        2211.59
P99 TPOT (ms):                           2680.27
---------------Inter-token Latency----------------
Mean ITL (ms):                           2251.59
Median ITL (ms):                         2100.88
P99 ITL (ms):                            4096.15
==================================================
```

Averages per request: 1,674 input tokens, 54 output tokens.

### Server-side observations over the run

Scoped to the 444 log lines of the c4 run (336 engine samples):

| Metric | Value |
|---|---|
| New attention variants compiled in serving path | **0** |
| HTTP 5xx / tracebacks | 0 |
| `Running: 4, Waiting: 0` | 264 / 336 samples (78.6%) |
| Any request waiting | 8 / 336 samples (2.4%) |
| Prefix cache hit rate | 2.7% → 52.7% (cumulative) |
| Peak GPU KV cache usage | 4.1% |

Two `RetileWarning`s were emitted, for `(2, 4096)` and `(4, 4096)` fp16 eager
results. Python dedupes warnings per shape, so these are once per distinct decode
width, not once per occurrence.

### Caveats

- `CustomDataset` shuffles by default, so request order was randomized despite the
  `correct_order` filename. `--disable-shuffle` was not passed.
- Prefix caching was left enabled, so prefill work is lower than a cold run and the
  52.7% hit rate is order-dependent.
- The zero-compile figure covers attention-variant cache misses, which
  `_get_attn_fn` logs on every miss. Dynamo guard-failure recompiles of the body
  graph or the lm_head wrapper are silent by default and were not instrumented
  (`TORCH_LOGS=recompiles` was not set).
