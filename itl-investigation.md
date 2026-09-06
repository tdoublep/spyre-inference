# Why decode ITL was ~2 s on Granite 3.3 8B

Follow-up to `benchmark-pr789-pr784.md`, which measured median ITL 2101 ms at
concurrency 4 on the merged #789 + #784 branch.

**Headline: most of it was a torch-spyre bug fixed 6 commits after our pin.**
Bumping `e02b78b -> da15ede` took median ITL from 2057 ms to 374 ms at identical
config.

Once fixed, decode cost is **linear** in padded KV: `ITL ~= 190 ms + 11.5 ms *
num_blocks` (R^2 = 0.9995). An apparent cost cliff above 16 blocks turned out to be
an artifact of varying prompt length in the sweep, and is retracted -- see
section 4. The remaining levers are padding waste from the geometric KV buckets
(~25% of blocks at 1500 tokens, more just above a power of two), prefill/decode
contention driven by the 512-token `max_num_batched_tokens` cap, and the ~190 ms
load-pattern-fixed term.

- Date: 2026-09-06
- Host: tpa-spyre-dev-4, pinned libs from `~/spyre-libs/env.sh`
- Code: merge of PR #789 (`f5c9fac`) + PR #784 (`75fd29f`) onto `main` (`c81a50c`)

## 1. Batch size is not the driver

Fixed 1500-token synthetic prompts, 16 output tokens, `--ignore-eos`, 8 prompts.
Server: `max_model_len 2048`, `SPYRE_ATTN_KV_BUCKETS=2048`,
`compile_sizes=[1,2,4,512]`, `SPYRE_ATTN_RECORD=1`, `SPYRE_NUM_CPUS=8`.

| concurrency | median ITL | output tok/s |
|---|---|---|
| 1 | 636 ms | 1.26 |
| 2 | 685 ms | 2.53 |
| 4 | 804 ms | 4.22 |

4x the sequences costs +26% step time, and throughput scales near-linearly.
Batching works; per-sequence work is not the problem.

Also established here: **zero CPU fallbacks** on the decode path, and zero
attention-variant compiles in the serving path (so #789's recorded grid covers
every shape the scheduler produced).

## 2. Cost scales with padded KV length, and with total KV cache size

Concurrency fixed at 4, prompt length varied to land in different KV buckets.
Server: `max_model_len 8192`, `SPYRE_ATTN_KV_BUCKETS=2048,4096,8192`.

| input_len | bucket / blocks | median ITL (`e02b78b`) |
|---|---|---|
| 1500 | 2048 / 16 blk | 2057 ms |
| 3500 | 4096 / 32 blk | 4028 ms |
| 7000 | 8192 / 64 blk | 6063 ms |

Note the 1500 row: **the same 1500-token prompt that measured 804 ms in section 1
measures 2057 ms here.** Identical prompt, identical concurrency, identical
16-block attention kernel. What changed is `num_gpu_blocks_override`, which the
platform pins to `max_num_seqs * ceil(max_model_len / block_size) + 1`:
513 blocks at `max_model_len` 2048 vs 2049 blocks at 8192. A 4x larger KV cache
tensor cost 2.6x ITL even though the kernel gathers the same 16 pages, which
points at the gather's source-tensor footprint rather than the tokens in play.

## 3. The torch-spyre bump

Our pin `e02b78b` (2026-09-02 15:10) was 6 commits short of torch-spyre#4163:

```
67142a5ca  fix(inductor): handle factorized matmul inputs in layout propagation
                          and stick compatibility (#4212)
f12d86b9b  feat(kvc): add get_composite_address accessor (#3587)
837f3c670  fix(hw_diagnostics): pin gh api to GET (#4229)
3919da175  Synchronize docs with implementation (#4195)
cec0ce042  fix(inductor): use finite fp16 extremes for padding/reduction masks (#4234)
da15ede83  Fix scatter destination layout compliance check (#4163)
```

Rebuilt at `da15ede`. Identical server config and bench args either side:

| | `e02b78b` | `da15ede` | change |
|---|---|---|---|
| median ITL | 2057 ms | **374 ms** | 5.5x |
| mean ITL | 2556 ms | 863 ms | 3.0x |
| median TTFT | 12971 ms | 5956 ms | 2.2x |
| output tok/s | 1.08 | 3.14 | 2.9x |

Across the length sweep:

| input_len | blocks | `e02b78b` | `da15ede` | speedup |
|---|---|---|---|---|
| 1500 | 16 | 2057 ms | 374 ms | 5.5x |
| 3500 | 32 | 4028 ms | 2376 ms | 1.7x |
| 7000 | 64 | 6063 ms | 4390 ms | 1.4x |

The bump removes a large mostly-fixed layout cost. It does not fix length scaling.
Six commits moved, so this shows the bump fixes it, not that #4163 alone does;
bisecting those six would pin it exactly, and is cheap now.

## 4. There is no per-block cost cliff (retracted)

The sweep in section 3 appeared to show per-block cost tripling above 16 blocks
(23 -> 74 -> 69 ms per block). That was an artifact: it varied prompt length to
reach different buckets, so prefill chunk count varied with it, and vLLM charges a
shared prefill/decode step to the decoding request's ITL.

Isolated properly -- same 1500-token prompt in every run, forced into different
padded-KV buckets by pinning `max_model_len` equal to a single
`SPYRE_ATTN_KV_BUCKETS` entry, with `--num-gpu-blocks-override 2049` holding cache
size constant (verified: `GPU KV cache size: 262,272 tokens` in all five runs):

| blocks | bucket | median ITL | linear fit | residual |
|---|---|---|---|---|
| 16 | 2048 | 375.26 ms | 374.8 | +0.5 |
| 20 | 2560 | 419.04 ms | 421.0 | -2.0 |
| 24 | 3072 | 468.24 ms | 467.2 | +1.0 |
| 28 | 3584 | 515.28 ms | 513.3 | +2.0 |
| 32 | 4096 | 557.94 ms | 559.5 | -1.6 |

```
ITL ~= 190 ms + 11.5 ms * num_blocks       R^2 = 0.9995
```

Linear to within +/-2 ms across exactly the range that looked like a cliff.
Per-block increments: 10.95, 12.30, 11.76, 10.67 ms.

The artifact is quantified: at 32 blocks the isolated measurement is 558 ms where
the sweep reported 2376 ms, so ~1818 ms was prefill contention (7 chunks per
request at 3500 tokens). The fit extrapolates to ~926 ms at 64 blocks against the
sweep's 4390 ms, i.e. ~3460 ms of contention at 14 chunks per request -- scaling
with chunk count as expected.

Reproducibility: 375.26 ms here vs 374 ms in section 3's A/B, despite different
`max_model_len` and bucket counts. Also note the old build's cache-size penalty
(804 ms at 513 blocks vs 2057 ms at 2049 blocks for the same prompt) is gone --
2049 blocks now gives 375 ms, so that penalty was part of the same layout bug. A
new-build run at 513 blocks would confirm no residual sensitivity; not measured.

## 5. Profiler traces

Traces taken either side of the cliff, same workload shape (4 prompts,
concurrency 4, 8 decode steps), only `input_len` differing.

Requires `USE_SPYRE_PROFILER=1` at build time; with it off the AIUPTI provider is
compiled out and the trace has no device events. Verified via
`ldd _C.so | grep aiupti` (before: not linked; after: linked) and the
`__aiu_profiler__` span appearing in the trace.

Per-call cost, 16 vs 32 blocks:

| op | 16 blk | 32 blk | ratio |
|---|---|---|---|
| attention kernel (`amax/div/exp/index_select`) | 8599 us | 16952 us | 1.97x |
| attention jobplan launch | 4366 us | 8012 us | 1.84x |
| **MLP jobplan launch** | 5986 us | 11878 us | **1.98x** |
| MLP kernel (`mm/silu/rsqrt`) | 4333 us | 4337 us | 1.00x |
| `index_copy/index_select` jobplan | 3643 us | 3696 us | 1.01x |
| `launch::HostCallback` (largest total: 14.7 -> 29.9 s) | 4398 us | 7539 us | 1.71x |

The attention kernel doubling is expected. The MLP kernel staying flat is correct.
The **MLP jobplan launch doubles while its kernel is flat** — a launch cannot
depend on KV length, so those spans measure the host waiting, not launch work.
`launch::HostCallback` count is exactly 1:1 with device kernel count (3335/3335 and
3960/3960) and its total (14668 / 29853 ms) nearly equals device-kernel busy time
(14852 / 30071 ms), so it is the host blocking per kernel.

But the device is ~94% busy in both traces (14852 ms of a 15816 ms window; 30071 of
31924), so the host is not the bottleneck -- it waits because the device is
working. There is only ~6% idle to reclaim, so pipelining submissions would buy
almost nothing. The MLP-launch anomaly is backpressure behind attention, not an
independent problem.

Lane structure, for anyone reading these traces: tid 0 is the device kernel stream
(`spyre_kernel_v1_*`), tid 100/200 are DMA copies, tid 400 memset/release, the
named thread is the host, and the two `Spans` lanes (`__aiu_profiler__`,
`PyTorch Profiler`) are single whole-window wrappers that must be excluded from any
utilization sum.

**Profiled device-kernel durations are inflated ~1.8x.** Section 4's fit gives a
block-dependent term of 11.5 ms/block, i.e. 184 ms at 16 blocks; the trace implies
40 layers x 8.60 ms = 344 ms. So AIUPTI's per-kernel instrumentation roughly
doubles measured kernel time, and trace timings must not be read as absolute
device cost. Honest split from the section 4 fit:

| | block-dependent (attention) | fixed |
|---|---|---|
| 16 blocks | 184 ms (49%) | 190 ms (51%) |
| 32 blocks | 369 ms (66%) | 190 ms (34%) |

The 190 ms is fixed *under this load pattern* and includes the constant prefill
contention held equal across runs, so it is not purely per-step model work.

## Notes for anyone repeating this

- **Profiler build breaks `vllm serve` without `ignore_frontend`.** The AsyncLLM
  frontend constructs its own torch profiler, torch-spyre's hook tries to open the
  VFIO device from that process, and the worker already owns it (single-owner
  device) — `DeviceOpenFail / EBUSY` after a full warmup. Pass
  `--profiler-config '{..., "ignore_frontend": true}'`.
- **Profiling is not wall-clock comparable**, and it also changes the workload
  ratio: profiled runs used 4 prompts (one clean wave) where the sweep used 8
  (two waves), so their absolute ITLs must not be cross-compared with the sweep.
- **Warmup cost is dominated by bucket count, not cache warmth.** 560 graphs took
  532 s; cutting to 80 graphs took 51 s (10x). An identical-config restart with a
  warm cache buys about 1.9x (448 -> 234 s for 240 graphs). Reduce buckets via
  `SPYRE_ATTN_KV_BUCKETS` and `compile_sizes` to iterate quickly — pick buckets the
  run already lands in so the measured path does not change.
- **Killing only the API server orphans `EngineCore`/`Worker`**, which keep
  `/dev/vfio` open and make the next `vllm serve` fail with `DeviceOpenFail`. Kill
  by pattern across the whole tree and verify nothing holds `/dev/vfio`.
- **`RetileWarning` frequency cannot be counted from logs.**
  `torch_spyre/ops/eager.py:97` calls `warnings.simplefilter("once", RetileWarning)`
  at import, which overrides `PYTHONWARNINGS=always`. `FallbackWarning` counts of
  zero are still trustworthy, since "once" would show a first occurrence.
