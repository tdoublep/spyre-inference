# Performance report — Granite 3.3 8B Instruct on Spyre

Online serving benchmark of PR #789 + #784, before and after bumping `torch-spyre`
past the scatter destination layout fix.

**Result: median inter-token latency improved 5.0x, from 2101 ms to 420 ms.**
End-to-end run time for the same 100-prompt workload fell from 56 min to 14 min.

- Date: 2026-09-06
- Host: tpa-spyre-dev-4 (4 AIU devices, TP=1, single device)
- Model: `ibm-granite/granite-3.3-8b-instruct`, float16

## Headline numbers

100 prompts from the aiops dataset, concurrency 4, dataset-driven output lengths,
prefix caching enabled. Identical command and config on both sides; only the
`torch-spyre` build differs.

| metric | before (`e02b78b`) | after (`da15ede`) | change |
|---|---|---|---|
| **median ITL** | 2100.88 ms | **420.46 ms** | **5.0x faster** |
| mean ITL | 2251.59 ms | 566.54 ms | 4.0x |
| P99 ITL | 4096.15 ms | 2356.17 ms | 1.7x |
| median TPOT | 2211.59 ms | 530.50 ms | 4.2x |
| mean TPOT | 2251.23 ms | 562.82 ms | 4.0x |
| P99 TPOT | 2680.27 ms | 986.79 ms | 2.7x |
| median TTFT | 8606.83 ms | 3587.85 ms | 2.4x |
| mean TTFT | 11082.64 ms | 5030.05 ms | 2.2x |
| P99 TTFT | 51299.87 ms | 25471.14 ms | 2.0x |
| output token throughput | 1.61 tok/s | 5.96 tok/s | 3.7x |
| total token throughput | 51.45 tok/s | 207.20 tok/s | 4.0x |
| request throughput | 0.03 req/s | 0.12 req/s | 4.0x |
| benchmark duration | 3359 s | 832 s | 4.0x |
| successful / failed | 100 / 0 | 100 / 0 | — |
| prefix cache hit rate | 52.7% | 52.7% | identical |

Server side, after the bump: **0** attention compiles in the serving path, **0** CPU
fallbacks, **0** 5xx or tracebacks, and `Running: 4` in 54 of 82 engine samples.

## What changed

The pinned `torch-spyre` rev was 6 commits short of
[torch-spyre#4163](https://github.com/torch-spyre/torch-spyre/pull/4163)
("Fix scatter destination layout compliance check"). Three of those six touch the
decode path:

| commit | PR |
|---|---|
| `67142a5ca` | #4212 fix(inductor): factorized matmul inputs in layout propagation and stick compatibility |
| `cec0ce042` | #4234 fix(inductor): use finite fp16 extremes for padding/reduction masks |
| `da15ede83` | #4163 Fix scatter destination layout compliance check |

Six commits moved together, so this measures the bump rather than #4163 in
isolation. Bisecting the six would attribute it precisely.

The bump also removed a KV-cache-size penalty. On the old build the same
1500-token prompt cost 804 ms with a 513-block cache but 2057 ms with a 2049-block
cache; on the new build that same 2049-block cache costs 375 ms. The penalty was
part of the same layout bug.

## Decode cost model

After the fix, decode latency is linear in padded KV length. Measured with the
prompt held at 1500 tokens and only the KV bucket varied, cache size pinned:

| padded KV | blocks | median ITL |
|---|---|---|
| 2048 | 16 | 375.26 ms |
| 2560 | 20 | 419.04 ms |
| 3072 | 24 | 468.24 ms |
| 3584 | 28 | 515.28 ms |
| 4096 | 32 | 557.94 ms |

```
ITL ~= 190 ms + 11.5 ms * num_blocks        R^2 = 0.9995
```

This predicts 374 ms for the benchmark's 1674-token average prompt against a
420 ms measured median, and explains the 2356 ms P99 via the dataset's tail up to
8192 tokens (64 blocks -> ~926 ms).

Attention accounts for the block-dependent term: 184 ms of 375 ms (49%) at 16
blocks, rising to 369 ms of 558 ms (66%) at 32 blocks. Profiler traces put the
device at ~94% busy with the fused attention kernel taking 72-77% of device time,
so decode is device-bound and attention-dominated.

## Remaining optimisation levers

1. **Prefill/decode contention.** With `max_num_batched_tokens` capped at 512, a
   1500-token prompt needs 3 prefill chunks and a 3500-token prompt needs 7, and
   each chunk shares a scheduler step with the decoding requests, whose ITL absorbs
   it. Measured at ~1.8 s of added ITL at 3500-token prompts against ~370 ms of
   actual attention work. Largest lever at long context.
2. **Padding waste in the KV buckets.** Buckets are powers of two from
   `block_size`, so a 1516-token sequence needs 12 blocks but pads to 16 — 25%
   waste, ~46 ms/step at 11.5 ms/block. Worse just above a power of two: a
   1050-token prompt pads 9 -> 16 blocks (44% waste, ~80 ms/step). Tunable today
   via `SPYRE_ATTN_KV_BUCKETS` with no code change; the cost is more warmup graphs.
3. **The ~190 ms load-pattern-fixed term**, now comparable to attention at short
   context and not yet attributed.

## Reproduction

Both runs sourced `~/spyre-libs/env.sh` (pinned libs ahead of system
`/opt/ibm/spyre`) and used `PYTHONPATH=<worktree>`.

### Server

```bash
source ~/spyre-libs/env.sh
export SPYRE_ATTN_RECORD=1        # envs.py defaults this to 0
export SPYRE_NUM_CPUS=8           # clamps OMP_NUM_THREADS 192 -> 8

vllm serve \
    --model ibm-granite/granite-3.3-8b-instruct \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --num-gpu-blocks-override 2049
```

`--max-num-seqs 4` is enough for concurrency 4 and trims `compile_sizes` from 7 to
4 body buckets (~85 s less warmup). `--num-gpu-blocks-override 2049` is required
for comparability: without it, 4 seqs allocate 257 blocks and the prefix cache
collapses. The original run used `--max-num-seqs 32`, which yields the same 2049
blocks.

### Client

```bash
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

### Versions

| component | value |
|---|---|
| branch | merge of PR #789 (`f5c9fac`) + PR #784 (`75fd29f`) onto `main` (`c81a50c`) |
| `torch-spyre` | `da15ede83613870e2fc499f363aa0f98359994a9` (was `e02b78ba...`) |
| `vllm` | 0.28.0 (`rev=v0.28.0`, built `VLLM_TARGET_DEVICE=empty`) |
| `torch` | 2.13.0+cpu |
| `transformers` | 5.16.1 |
| `USE_SPYRE_PROFILER` | `0` for these numbers |

Installed `ibm-*` RPMs: `ibm-deeptools 2.0.0-0.main.1+2324.4cfac4c_309`,
`ibm-flex 2.0.0-0.main.1+495.a86bb35_315`,
`ibm-senlib-core 2.0.0-0.main.1+244.de8291a_214`,
`ibm-spyre-comms 1.0.0-0.main.1+124.3523cde_149`,
`ibm-libaiupti 2.0.0-0.main.1+21.bd7054d_0`,
`ibm-aiu-toolbox-e2e 2.0.0-0.main.1+28.47d9b91_89`.

### Effective config after platform overrides

| setting | value |
|---|---|
| `max_num_batched_tokens` | 512 (capped from default by the platform) |
| `block_size` | 128 (our 64 override is superseded by `CpuPlatform`) |
| `num_gpu_blocks_override` | 2049 -> 262,272 tokens of KV cache |
| `compile_sizes` | `[1, 2, 4, 512]` |
| attention grid | 7 KV buckets x 2 query buckets = 14 variants x 40 layers = 560 graphs |
| warmup | 56.9 s body + 316.0 s attention recording |
| compilation mode | `STOCK_TORCH_COMPILE` |
| prefix caching | enabled |

## Caveats

- Input tokens are identical between runs (167,398) but generated tokens differ
  (5414 -> 4958, -8.4%): #4234 changes fp16 numerics, so sampled tokens diverge and
  EOS lands at different points. Per-token metrics (ITL, TPOT) are unaffected;
  normalising duration for token count gives ~3.7x rather than 4.0x.
- Prefix caching is on, and `CustomDataset` shuffles by default, so the 52.7% hit
  rate is order-dependent. It reproduced exactly across both runs.
- Single run per configuration, so no error bars. The 16-block point reproduced to
  within 0.3% across two independent server configurations (374 vs 375.26 ms).
- `USE_SPYRE_PROFILER=1` breaks `vllm serve` unless
  `--profiler-config '{..., "ignore_frontend": true}'` is passed: the AsyncLLM
  frontend tries to open the single-owner VFIO device the worker already holds, and
  fails after a full warmup. Profiled runs are also not wall-clock comparable, and
  profiled device-kernel durations are inflated roughly 1.8x.

Full investigation notes, including hypotheses tested and discarded, are in
`itl-investigation.md`.
