# Performance report — Granite 3.3 8B Instruct on Spyre

Online serving benchmark of PR #789 + PR #784.

- Date: 2026-09-06
- Host: tpa-spyre-dev-4 (4 AIU devices found, TP=1, single device used)
- Model: `ibm-granite/granite-3.3-8b-instruct`, float16

## Commands

Both server and client sourced `~/spyre-libs/env.sh` first, which puts the pinned
libs ahead of the system `/opt/ibm/spyre`, and used `PYTHONPATH=<worktree>` so the
worktree code took precedence over the editable install in the venv.

### Server

```bash
source ~/spyre-libs/env.sh
export PYTHONPATH=/home/senuser/spyre-inference/.claude/worktrees/bench-789-784:$PYTHONPATH
export SPYRE_ATTN_RECORD=1
export SPYRE_NUM_CPUS=8

vllm serve \
    --model ibm-granite/granite-3.3-8b-instruct \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --num-gpu-blocks-override 2049
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

## Environment

### Code under test

| component | version |
|---|---|
| base `main` | `c81a50c` |
| PR #789 `attn-zero-serving-compiles` tip | `f5c9fac` |
| PR #784 `warmup-lm-head-graphs` tip | `75fd29f` |

### Dependencies

| package | version / rev |
|---|---|
| `torch-spyre` | `da15ede83613870e2fc499f363aa0f98359994a9` |
| `vllm` | 0.28.0 (`rev=v0.28.0`, built `VLLM_TARGET_DEVICE=empty`) |
| `torch` | 2.13.0+cpu |
| `transformers` | 5.16.1 |
| `USE_SPYRE_PROFILER` | `0` |

### Pinned libs

`~/spyre-libs/env.sh` sets
`SENTIENT_BASE_INSTALL_DIR=/home/senuser/spyre-libs/opt/ibm/spyre` and re-sources
`/etc/profile.d/ibm-aiu-setup.sh`, prepending the pinned paths:

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

| var | value | note |
|---|---|---|
| `SPYRE_ATTN_RECORD` | `1` | `envs.py` defaults this to `0` |
| `SPYRE_NUM_CPUS` | `8` | clamped `OMP_NUM_THREADS` 192 → 8, plus OPENBLAS/MKL/NUMEXPR/VECLIB |
| `HF_HOME` | `/models/huggingface_cache` | pre-existing; model served from cache |

### Effective config after platform overrides

| setting | value |
|---|---|
| `dtype` | `torch.float16` |
| `max_model_len` | 8192 |
| `max_num_seqs` | 4 |
| `max_num_batched_tokens` | 512 (capped by the platform) |
| `block_size` | 128 |
| `num_gpu_blocks_override` | 2049 → 262,272 tokens of KV cache |
| `compilation_config.mode` | `STOCK_TORCH_COMPILE` |
| `compile_sizes` | `[1, 2, 4, 512]` |
| `enable_prefix_caching` | `True` |
| `enable_chunked_prefill` | `True` |
| LM head vocab padding | 49216 → 51200 |

### Warmup

| phase | duration |
|---|---|
| weight load | 1.5 s |
| body warmup (4 buckets) | 56.9 s |
| attention graph recording (560 graphs) | 316.0 s |

Attention grid: 7 KV buckets `[128..8192]` × 2 query buckets `[1, 512]`
= 14 variants × 40 layers = 560 graphs.

## Results

```
============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Maximum request concurrency:             4
Benchmark duration (s):                  831.84
Total input tokens:                      167398
Total generated tokens:                  4958
Request throughput (req/s):              0.12
Output token throughput (tok/s):         5.96
Peak output token throughput (tok/s):    15.00
Peak concurrent requests:                6.00
Total token throughput (tok/s):          207.20
---------------Time to First Token----------------
Mean TTFT (ms):                          5030.05
Median TTFT (ms):                        3587.85
P99 TTFT (ms):                           25471.14
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          562.82
Median TPOT (ms):                        530.50
P99 TPOT (ms):                           986.79
---------------Inter-token Latency----------------
Mean ITL (ms):                           566.54
Median ITL (ms):                         420.46
P99 ITL (ms):                            2356.17
==================================================
```

Averages per request: 1,674 input tokens, 50 output tokens.

### Server-side over the run

| metric | value |
|---|---|
| attention compiles in the serving path | 0 |
| `FallbackWarning` | 0 |
| HTTP 5xx / tracebacks | 0 |
| prefix cache hit rate | 52.7% (cumulative) |
| peak GPU KV cache usage | 4.9% |
| `Running: 4, Waiting: 0` | 54 / 82 engine samples |
