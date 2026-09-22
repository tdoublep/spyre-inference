# Configuration

## Plugin Setup

To load the plugin, set the `VLLM_PLUGINS` environment variable before running vLLM:

```bash
export VLLM_PLUGINS=spyre_inference,spyre_inference_ops
```

`spyre_inference` activates the platform, and `spyre_inference_ops` registers the OOT
custom ops plus the Spyre Transformers backend (used for `model_impl="transformers"`).

## Usage

You can then use vLLM as usual:

```python
from vllm import LLM

llm = LLM(
    model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    max_model_len=128,
    max_num_seqs=2,
)
```

See the [Examples](../examples/offline_inference/torch_spyre_inference.md) page for more usage patterns.

## Gemma-4: text-only use of a vision checkpoint

Every Gemma-4 repository carries a `vision_config`, so `google/gemma-4-31B` and
`google/gemma-4-26B-A4B` load as `Gemma4ForConditionalGeneration` and build a vision
tower — weights to load and graphs to warm up that a text-only workload never runs.

To use one of those repositories for text only, pin its decoder architecture:

```python
llm = LLM(
    model="google/gemma-4-26B-A4B",
    hf_overrides={"architectures": ["Gemma4ForCausalLM"]},
    tensor_parallel_size=2,
)
```

That is also the configuration the tensor-parallel and compile e2e tests run these
checkpoints under. A repository with no vision tower gets the override by default, so
this is only needed for the multimodal ones.

## Decoder compile buckets

The body pads the packed token count to the next `compile_sizes` bucket, and warmup
dummies every bucket. The lm_head sits outside every body graph and compiles its own, so
it needs the same treatment: it projects one row per *sampled* request, a width that
would otherwise take every value in `1..--max-num-seqs` as requests finish. Those rows
pad onto the same buckets clipped to `--max-num-seqs`, and warmup projects each width, so
no shape reaches the lm_head uncompiled. Pad rows are dropped before sampling.

## Encoder / pooling compile buckets

Spyre compile is on by default (`STOCK_TORCH_COMPILE`, `dynamic=False`). Pass
`--enforce-eager` to disable it.

Pooling has **one** set of compile shapes: `(prompt_length, batch_size)` pairs. Every
sequence is padded to `L` and the batch to `B` before the model runs, so the body sees
exactly `B × L` token rows and attention is a reshape plus one SDPA call.

Both axes default to **powers of two**: lengths from 64 (the Spyre stick) up to
`--max-model-len`, widths from 1 up to `--max-num-seqs`, crossed. A 512-token model at
`--max-num-seqs 4` therefore declares twelve shapes, `{64, 128, 256, 512} × {1, 2, 4}`, so
a short or narrow batch is not padded up to the widest graph.

```bash
vllm serve ibm-granite/granite-embedding-125m-english --runner pooling \
  --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 4096
```

`--max-num-batched-tokens` is a constraint on the set, not a cap on one axis: a shape puts
`B × L` dense rows through the body, so any pair over the budget is dropped. That filters
per pair, so a short length keeps a wider batch than a long one — at a 2048-token budget,
`(64, 32)` survives while `(512, 32)` does not. Pooling defaults it to **2048**; pass the
flag to change it.

Wide batches of *short* sequences therefore need `--max-num-seqs` raised, since the width
ladder stops at it. At the 2048 default, `--max-num-seqs 32` declares eighteen shapes and
makes `(64, 32)` reachable, where the default `--max-num-seqs 4` stops at `(64, 4)`.

To declare a different set, use the same two bucket lists the decoder attention path uses —
`SPYRE_ATTN_QUERY_BUCKETS` for prompt lengths and `SPYRE_ATTN_NUM_SEQS_BUCKETS` for batch
widths:

```bash
SPYRE_ATTN_QUERY_BUCKETS=64,256 \
SPYRE_ATTN_NUM_SEQS_BUCKETS=1,8 \
vllm serve ibm-granite/granite-embedding-125m-english --runner pooling \
  --max-model-len 512 --max-num-seqs 32 --max-num-batched-tokens 16384
```

Here the two lists are **crossed**, as they are for the decoder. Each is clamped to its
limit: entries above `--max-model-len` (lengths) or `--max-num-seqs` (widths) are dropped.
`max_model_len` is appended to the length ladder when missing, because it is an input a
shape has to cover; the width ladder needs no such entry, since `--max-num-seqs` is
lowered to the widest width declared. Lengths are rounded up to a multiple of 64 (the
Spyre stick).

| setting | value |
| --- | --- |
| `--max-model-len` | the largest prompt length; an input, never overridden |
| `--max-num-seqs` | lowered to the widest declared shape, so the gate cannot be overshot |
| `--max-num-batched-tokens` | an input that filters the pairs; defaults to 2048 for pooling and is only ever raised, to `max_model_len` (encoder prefill cannot be chunked) |
| `compile_sizes` | unused for pooling; warmup traces one graph per declared shape |

A prompt longer than `--max-model-len` is rejected by vLLM's own length check.
`PoolingSpyreScheduler` admits only batches a declared shape covers, so no request ever
compiles a new shape mid-serve, and warmup runs one dummy per declared shape.

### Choosing shapes

A batch is padded up to its shape's width, so the shape list is the main throughput and
latency control:

- A batch of 3 requests of 30 tokens against `(512, 8)` still computes `8 × 512 = 4096`
  rows. Add a short length (`SPYRE_ATTN_QUERY_BUCKETS=64`) to avoid paying for 512 columns.
- A single request against `(512, 8)` computes 8 rows' worth. Add `1` to
  `SPYRE_ATTN_NUM_SEQS_BUCKETS` if single-request latency matters — the narrowest covering
  shape wins, so a lone request then lands on `(512, 1)`.
- The lists cross, so `n` lengths × `m` widths is `n × m` graphs to trace at warmup, and
  warmup cost grows with the set. The batch axis dominates: a new width costs noticeably
  more than another length at a width already traced.
  Keep both short.

Masks and pooling always use the real, unpadded lengths.

## Tuning buckets for padding

Bucketing trades warmup time for per-request padding. A request is padded up to the next
bucket on each axis and the padding is masked out, so buckets far above your real shapes
waste compute, while buckets that hug your workload cut that waste but add graphs to
compile at warmup. Attention is recorded as the **product** of its KV-length and
query-length buckets (and, when the batched-decode kernel is enabled, a second KV-length ×
num-sequences product), so extra attention buckets cost multiplicatively — keep those
lists short.

**Decoder body (packed token count).** Override the defaults with `compile_sizes`; the
platform clamps `--max-num-batched-tokens` to the largest entry. A decode-heavy run at
`--max-num-seqs 8` rarely needs the full power-of-two ladder:

```python
from vllm import LLM

llm = LLM(
    model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    max_num_seqs=8,
    max_model_len=2048,
    compilation_config={"compile_sizes": [1, 8, 512]},
)
```

`1` and `8` cover decode steps (one token per running sequence, up to 8); `512` is the
prefill bucket.

**Attention (KV length × query length).** Set the buckets directly as comma-separated
lists. Each is clamped to its limit: entries above `--max-model-len` (KV) or
`--max-num-batched-tokens` (query) are dropped, and the limit is appended if missing, so
every schedulable length keeps a bucket.

```bash
export SPYRE_ATTN_KV_BUCKETS=256,1024,2048    # default: powers of two from block_size
export SPYRE_ATTN_QUERY_BUCKETS=1,512         # 1 = decode; 512 = prefill chunk
```

The default KV buckets are geometric (powers of two) precisely because the recorded set
is a product. If your context never exceeds 2048, dropping the higher powers removes
variants from warmup at no serving cost.

With the batched-decode kernel enabled (`SPYRE_BATCHED_DECODE=1`, the default; the
head-major layout has no batched kernel and ignores it), warmup also records it over the
KV-length × num-sequences grid. `SPYRE_ATTN_NUM_SEQS_BUCKETS`
(default: powers of two from 4 to `--max-num-seqs`) is the extra lever there, and the same
keep-it-short advice applies.

## pyproject.toml Reference

The `pyproject.toml` includes several key build configurations:

### Build Configuration

```toml
[tool.uv]
build-constraint-dependencies = ["torch==2.13.0"]
extra-build-variables = { vllm = { VLLM_TARGET_DEVICE = "empty", CMAKE_ARGS = "--fresh" } }
```

These settings ensure:

- All packages are built with the same PyTorch version (2.13.0)
- vLLM is built with the **empty** backend — no device-specific C kernels. This avoids
  the torch-version coupling of prebuilt CPU wheels and the dependency on `vllm._C`
  (whose CPU-optimized ops we don't need; Spyre provides its own)

### Source Repositories

The plugin pulls dependencies from specific Git repositories:

```toml
[tool.uv.sources]
vllm = { git = "https://github.com/vllm-project/vllm", rev = "..." }
torch-spyre = { git = "https://github.com/torch-spyre/torch-spyre", rev = "..." }
```

This ensures that torch-spyre and vllm are compiled/installed from source, instead of pulling pre-compiled wheels from PyPI.

### PyTorch CPU Index

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

This ensures the CPU flavor of PyTorch is installed, as CUDA support is not required.
