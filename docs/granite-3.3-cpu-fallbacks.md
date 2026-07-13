# CPU Fallbacks in Granite 3.3 Spyre Inference

This report enumerates every place where Granite 3.3 inference currently falls back to
CPU (or accepts a numerical/correctness compromise) due to a `torch-spyre` limitation.
For each case it gives the location in this repo, the root cause, and a minimal PyTorch
snippet showing the functionality we *want* on Spyre but that does not work today.

## Scope

Granite 3.3 uses these layers per forward pass:

```
VocabParallelEmbedding
  → [ RMSNorm → QKVParallelLinear → RotaryEmbedding → Attention → RowParallelLinear
      → RMSNorm → MergedColumnParallelLinear → SiluAndMul → RowParallelLinear ] × N
  → RMSNorm → ParallelLMHead → LogitsProcessor
```

Every custom op in `spyre_inference/custom_ops/` and the attention backend exists to route
around a `torch-spyre` limitation.

## torch-spyre op-support baseline

Confirmed against `torch_spyre/ops/eager.py`, `torch_spyre/ops/fallbacks.py`, and
`torch_spyre/_inductor/lowering.py`:

- `slice` / `narrow` / `select` / `split` — **no native Spyre kernel**.
- `scatter` / `index_select` / `gather` — no kernel; `aten.index_copy.out` is an
  **explicit CPU fallback**.
- `transpose` + `contiguous` — layout-propagation bug on head axes.
- dtype promotion (`.to(float32)`) — dtype-changing op overloads are **explicitly excluded**
  from native registration (`eager.py`: `if "dtype" in op.name(): continue`).
- `aten.embedding` — **registered CPU fallback**.
- `torch.ops.spyre.overwrite` — **deprecated** (`FutureWarning`); no scaling
  symbolic-offset replacement yet (torch-spyre#220 / #1371-3).
- Fallbacks are registration-driven and emit `FallbackWarning` ("... is falling back to cpu").

---

## 1. Rotary Embedding — entire op runs on CPU

**Where:** `spyre_inference/custom_ops/rotary_embedding.py` — `_rotary_cpu_op_func` D2H's
positions/query/key, runs `RotaryEmbedding.forward_static` on CPU, then H2D.

**Root cause:** RoPE gathers `cos_sin_cache` rows by `positions` (`index_select`/`embedding`)
and interleaves head-dim slices. `aten.embedding` is a CPU fallback and `index_select` has
no kernel, so the cos/sin gather cannot run on-device.

```python
import torch
dev = torch.device("spyre")
cos_sin_cache = torch.randn(2048, 64, dtype=torch.float16, device=dev)
positions = torch.arange(32, device=dev)              # int64 positions
# The gather we need on-device but that falls back today:
cos_sin = cos_sin_cache.index_select(0, positions)     # aten::index_select — no Spyre kernel
cos, sin = cos_sin.chunk(2, dim=-1)
# ...then apply rotate_half using head-dim slices (also needs on-device slicing, see #6a)
```

---

## 2. RMSNorm — no float32 promotion (numerical divergence)

**Where:** `spyre_inference/custom_ops/rms_norm.py` — `_forward_spyre_impl` computes variance
in fp16; the class warns "no dtype promotion is performed, expect numerical differences to
upstream vLLM."

**Root cause:** Upstream RMSNorm upcasts to fp32 for the `pow(2).mean()` variance
accumulation. torch-spyre excludes dtype-changing op overloads, so `.to(torch.float32)`
inside the kernel cannot lower. We stay in fp16 and accept the accuracy loss.

```python
import torch
dev = torch.device("spyre")
x = torch.randn(32, 4096, dtype=torch.float16, device=dev)
# Upstream numerics require fp32 accumulation; this upcast has no Spyre lowering:
variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)   # .to(float32) unsupported
x_normed = (x.to(torch.float32) * torch.rsqrt(variance + 1e-6)).to(torch.float16)
```

---

## 3. QKV Projection — forced D2H so `split()` + `view()` + KV scatter run on CPU

**Where:** `spyre_inference/custom_ops/linear.py` — `SpyreQKVParallelLinear.forward` does
`convert(result, device="cpu")` after the linear. Minimal repro:
`tests/test_mlp.py::test_spyre_strided_scatter_source` (marked `xfail(strict=True)`).

**Root cause:** After `qkv.split([q,k,v])`, `v` is a strided view; `Attention.forward` then
does `v.view(-1, num_kv_heads, head_size)` producing a **non-contiguous** tensor used as the
*source* of a scatter into the KV cache. Spyre rejects a non-contiguous scatter source
(`index_copy` is a CPU fallback; strided scatter source corrupts/errors).

```python
import torch
dev = torch.device("spyre")
num_tokens, num_heads, num_kv_heads, head_size = 16, 8, 2, 64
q_size, kv_size = num_heads * head_size, num_kv_heads * head_size
qkv = torch.randn(num_tokens, q_size + 2 * kv_size, dtype=torch.float16, device=dev)
_, _, v = qkv.split([q_size, kv_size, kv_size], dim=-1)   # strided view
v = v.view(-1, num_kv_heads, head_size)                    # non-contiguous 3D
kv_cache = torch.zeros(4, 2, 8, num_kv_heads, head_size, dtype=torch.float16, device=dev)
idx = torch.zeros(num_tokens, dtype=torch.long, device=dev)
off = torch.arange(num_tokens, device=dev) % 8
kv_cache[idx, 1, off] = v            # scatter with strided source — fails on Spyre today
```

---

## 4. SiLU-and-Mul (SwiGLU) — D2H to slice the gate/up halves

**Where:** `spyre_inference/custom_ops/silu_and_mul.py` — `forward_oot` converts `x` to CPU,
slices `x[..., :d]` / `x[..., d:]`, calls `.contiguous()`, then H2D each half.

**Root cause:** (a) slicing a Spyre tensor along the last dim "causes corruption" (no
`aten::slice` kernel), and (b) transferring the resulting non-contiguous slice CPU→Spyre
corrupts data unless made contiguous first.

```python
import torch
import torch.nn.functional as F
dev = torch.device("spyre")
x = torch.randn(32, 8192, dtype=torch.float16, device=dev)   # concatenated [gate | up]
d = x.shape[-1] // 2
gate, up = x[..., :d], x[..., d:]     # last-dim slice of a Spyre tensor — corrupts today
out = F.silu(gate) * up               # want this fully on-device
```

---

## 5. Vocab Embedding (TP>1) — masked-input computation on CPU

**Where:** `spyre_inference/custom_ops/vocab_parallel_embedding.py` — under TP>1,
`get_masked_input_and_mask(convert(input_, "cpu"), ...)` runs on CPU, then H2D.

**Root cause:** `get_masked_input_and_mask` builds boolean masks and combines them; the
embedding gather itself (`aten.embedding`) is a registered CPU fallback. The mask arithmetic
on int indices plus the embedding gather can't stay on-device.

```python
import torch
dev = torch.device("spyre")
weight = torch.randn(4096, 4096, dtype=torch.float16, device=dev)  # per-rank vocab shard
input_ids = torch.randint(0, 32000, (32,), device=dev)
vocab_start, vocab_end = 0, 4096
mask = (input_ids >= vocab_start) & (input_ids < vocab_end)
masked = (input_ids - vocab_start) * mask
out = torch.embedding(weight, masked)     # aten::embedding — CPU fallback today
out = out * mask.unsqueeze(-1)            # want the whole path on-device
```

---

## 6. Attention backend — the largest cluster of fallbacks

`spyre_inference/v1/attention/backends/spyre_attn.py` stages nearly the whole attention op
on CPU. There are four distinct device root causes plus a correctness limitation.

### 6a. q/k/v slicing → sliced on CPU

`forward` does `convert(key/value/query, "cpu")` because "Spyre slicing corrupts memory."
Every `query_cpu[q_start:q_end]`, `key_cpu[:num_actual_tokens]` slice needs CPU.

```python
import torch
dev = torch.device("spyre")
q = torch.randn(64, 8, 64, dtype=torch.float16, device=dev)
q_seq = q[16:48]        # dim-0 slice on Spyre — corrupts; no aten::slice kernel
```

### 6b. KV-cache scatter-write → `torch.ops.spyre.overwrite` deprecated, doesn't scale

`_overwrite` uses `torch.ops.spyre.overwrite` (now `FutureWarning`-deprecated). The standard
replacement (`output[i:j] = x` / `narrow().copy_()`) "silently writes to row 0" on Spyre, and
per-offset `overwrite` compiles one SDSC binary per unique offset (blows the Dynamo cache).
Tracked as torch-spyre#220 / #1371-3.

```python
import torch
dev = torch.device("spyre")
page = torch.zeros(2, 256, 64, dtype=torch.float16, device=dev)  # [kv_heads, block, head]
tok = torch.randn(2, 1, 64, dtype=torch.float16, device=dev)
offset = 37
page.narrow(1, offset, 1).copy_(tok)   # want: symbolic-offset in-place write. Writes row 0 today.
```

### 6c. Result reshape — `transpose(1,2).contiguous()` on head axes is broken

`_online_softmax_attention` pulls the attention result to CPU to do
`result_cpu.transpose(1,2).contiguous()` because "Spyre transpose+contiguous on the head axes
is broken."

```python
import torch
dev = torch.device("spyre")
r = torch.randn(1, 2, 32, 64, dtype=torch.float16, device=dev)  # [1, kv_heads, q, head]
out = r.transpose(1, 2).contiguous()    # layout propagation through transpose fails on Spyre
```

### 6d. Output scatter — staged on a CPU buffer, one bulk H2D at the end

`output_cpu = torch.zeros_like(output, device="cpu")` then `output_cpu[q_start:q_end] = ...`,
because scattering into `output` on dim 0 "has no working primitive" (same overwrite
limitation as 6b).

### 6e. MHA / MQA / multi-sequence (correctness limitation, not a device fallback)

The backend only supports **GQA** (`num_queries_per_kv > 1`) and `num_seqs=1`. MHA/MQA fail
with a Spyre compiler layout-propagation bug:
`"cannot restickify any input layout of y to carry y_var=d2"` (propagate_layouts.py:341).

---

## 7. LM Head — output forced to CPU, weights padded

**Where:** `spyre_inference/custom_ops/parallel_lm_head.py` — the `F.linear` runs on Spyre,
but `forward_oot` does `convert(out, "cpu")` and slices off padding on CPU
(`out_cpu[:, :-padding]`).

**Root causes (two):** (a) the vocab dimension must be padded to a multiple of `64*32`
because "torch-spyre has a limitation with the work division of larger matmuls" — un-padded
shapes fail; (b) un-padding requires a slice, which must run on CPU, and the TP>1 all-gather
needs the logits on CPU anyway.

```python
import torch
import torch.nn.functional as F
dev = torch.device("spyre")
hidden = torch.randn(32, 4096, dtype=torch.float16, device=dev)
weight = torch.randn(32000, 4096, dtype=torch.float16, device=dev)  # vocab NOT multiple of 64*32
logits = F.linear(hidden, weight)   # want: arbitrary vocab size + on-device un-pad slice
logits = logits[:, :32000]          # slice on Spyre also unsupported (see #6a)
```

---

## 8. Logits scaling — `contiguous()` to dodge an in-place compile bug

**Where:** `spyre_inference/custom_ops/logits_processor.py` — forces `logits.contiguous()`
because the downstream in-place `logits *= self.scale` "would trigger a compile issue in
torch-spyre." Granite 3.3 sets this scale via `logits_scaling`, so it is on the hot path.

```python
import torch
dev = torch.device("spyre")
logits = torch.randn(32, 32000, dtype=torch.float16, device=dev).t()[:32]  # non-contiguous
logits *= 1.0 / 6.0    # in-place mul on a non-contiguous Spyre tensor — compile issue today
```

---

## Summary

| # | Layer | Root cause (missing Spyre functionality) | torch-spyre status |
|---|-------|------------------------------------------|--------------------|
| 1 | RotaryEmbedding | `index_select`/`embedding` gather of cos/sin cache | embedding = CPU fallback; index_select = no kernel |
| 2 | RMSNorm | `.to(float32)` for variance accumulation | dtype-changing ops excluded from native reg |
| 3 | QKVParallelLinear | non-contiguous (strided) tensor as scatter source | index_copy = CPU fallback; strided source unsupported |
| 4 | SiluAndMul | last-dim slice + non-contiguous H2D transfer | no `aten::slice` kernel |
| 5 | VocabParallelEmbedding (TP>1) | masked-index build + `embedding` gather | embedding = CPU fallback |
| 6a | Attention (q/k/v) | dim-0 slicing | no `aten::slice` kernel |
| 6b | Attention (KV scatter) | symbolic-offset in-place write | `spyre.overwrite` deprecated, no scaling replacement (#220/#1371-3) |
| 6c | Attention (reshape) | `transpose(...).contiguous()` on head axes | layout-propagation bug |
| 6d | Attention (output) | dim-0 scatter into output | same as 6b |
| 6e | Attention (MHA/MQA, num_seqs>1) | layout propagation through degenerate dims | compiler bug `restickify ... y_var=d2` |
| 7 | ParallelLMHead | arbitrary matmul vocab size + on-device un-pad slice | matmul work-division limit; no slice kernel |
| 8 | LogitsProcessor | in-place mul on non-contiguous tensor | in-place lowering gap |

### Highest-leverage torch-spyre capabilities

The three capabilities that would eliminate most of these fallbacks:

1. **On-device slicing / `narrow` / `select`** — kills 4, 6a, 6c, 7.
2. **A scaling symbolic-offset in-place write / scatter** — kills 6b, 6d, 3.
3. **On-device `embedding` / `index_select`** — kills 1, 5.
