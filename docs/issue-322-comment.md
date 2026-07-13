I did a pass over the full Granite 3.3 forward path (both `custom_ops/` and the attention backend), cross-referenced each fallback against the current `torch-spyre` op-support surface (`ops/eager.py`, `ops/fallbacks.py`, `_inductor/lowering.py`), and mapped the results onto the table above. Below is the root cause for each row plus a **minimal PyTorch snippet** that shows the on-device functionality we want but that does not exist today — these should be usable more or less directly as `xfail` probes or as repros to file against `torch-spyre`.

## torch-spyre op-support baseline

Confirmed against the installed package:

- `slice` / `narrow` / `select` / `split` — **no native Spyre kernel**.
- `scatter` / `index_select` / `gather` — no kernel; `aten.index_copy.out` is an **explicit CPU fallback**.
- `transpose` + `contiguous` — layout-propagation bug on head axes.
- dtype promotion (`.to(float32)`) — dtype-changing overloads are **explicitly excluded** from native registration (`eager.py`: `if "dtype" in op.name(): continue`).
- `aten.embedding` — **registered CPU fallback** (matches [torch-spyre#420](https://github.com/torch-spyre/torch-spyre/issues/420)).
- `torch.ops.spyre.overwrite` — **deprecated** (`FutureWarning`); no scaling symbolic-offset replacement yet.

## Mapping to the table

### `SpyreQKVParallelLinear` — Split (Slicing)

After `qkv.split([q,k,v])`, `v` is a strided view; `Attention.forward` then does `v.view(-1, num_kv_heads, head_size)`, producing a **non-contiguous** tensor used as the *source* of a scatter into the KV cache. Spyre rejects a non-contiguous scatter source (`index_copy` is a CPU fallback). Minimal repro already lives in `tests/test_mlp.py::test_spyre_strided_scatter_source` (`xfail(strict=True)`):

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

### `SpyreParallelLMHead` (all four rows)

- **weights alignment / work division** (tracked by [torch-spyre#1918](https://github.com/torch-spyre/torch-spyre/issues/1918)): the vocab dim must be padded to a multiple of `64*32`, otherwise `F.linear` fails.
- **intermediate (slicing)**: un-padding the result requires a slice, which has no Spyre kernel.
- **input side / output side**: driven by upstream indexing and downstream sampling on CPU.

```python
import torch
import torch.nn.functional as F
dev = torch.device("spyre")
hidden = torch.randn(32, 4096, dtype=torch.float16, device=dev)
weight = torch.randn(32000, 4096, dtype=torch.float16, device=dev)  # vocab NOT a multiple of 64*32
logits = F.linear(hidden, weight)   # want: arbitrary vocab size (torch-spyre#1918)
logits = logits[:, :32000]          # + on-device un-pad slice (no aten::slice kernel)
```

### `SpyreRMSNorm` — proper input placement (resolved via #310)

In addition to the placement issue #310 covers, there is a **numerical** gap worth noting: upstream RMSNorm upcasts to fp32 for the `pow(2).mean()` variance accumulation, but `torch-spyre` excludes dtype-changing overloads, so we stay in fp16 and diverge from upstream numerics.

```python
import torch
dev = torch.device("spyre")
x = torch.randn(32, 4096, dtype=torch.float16, device=dev)
variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)   # .to(float32) has no Spyre lowering
x_normed = (x.to(torch.float32) * torch.rsqrt(variance + 1e-6)).to(torch.float16)
```

### `SpyreRotaryEmbedding` — numerical issues (#45 / [torch-spyre#1668](https://github.com/torch-spyre/torch-spyre/issues/1668))

Beyond the numerical investigation in those issues, the structural blocker is the cos/sin gather: RoPE gathers `cos_sin_cache` rows by `positions` (`index_select`/`embedding`), and `aten.embedding` is a CPU fallback / `index_select` has no kernel.

```python
import torch
dev = torch.device("spyre")
cos_sin_cache = torch.randn(2048, 64, dtype=torch.float16, device=dev)
positions = torch.arange(32, device=dev)
cos_sin = cos_sin_cache.index_select(0, positions)   # aten::index_select — no Spyre kernel
cos, sin = cos_sin.chunk(2, dim=-1)
```

### `SpyreSiluAndMul` — Slicing

Two limitations: (a) slicing the fused `[gate | up]` tensor along the last dim corrupts on Spyre (no `aten::slice` kernel), and (b) transferring the resulting non-contiguous slice CPU→Spyre corrupts unless made contiguous first.

```python
import torch
import torch.nn.functional as F
dev = torch.device("spyre")
x = torch.randn(32, 8192, dtype=torch.float16, device=dev)   # concatenated [gate | up]
d = x.shape[-1] // 2
gate, up = x[..., :d], x[..., d:]     # last-dim slice of a Spyre tensor — corrupts today
out = F.silu(gate) * up               # want fully on-device
```

### `SpyreVocabParallelEmbedding` — implicit fallback ([torch-spyre#420](https://github.com/torch-spyre/torch-spyre/issues/420))

Confirmed: `aten.embedding` is a registered CPU fallback in `torch-spyre`. Under TP>1 we additionally build the masked index / mask on CPU.

```python
import torch
dev = torch.device("spyre")
weight = torch.randn(4096, 4096, dtype=torch.float16, device=dev)   # per-rank vocab shard
input_ids = torch.randint(0, 32000, (32,), device=dev)
out = torch.embedding(weight, input_ids)   # aten::embedding — CPU fallback today (torch-spyre#420)
```

### `_SpyreModelWrapper` — Indexing (Slicing)

Same underlying gap as the LM-head input side: the ModelRunner slices `hidden_states[logits_indices]` on CPU because dim-0 slicing has no Spyre kernel (see the Attention snippet below for the primitive).

### `Attention` — Slicing / Paging

This row is currently "various fallbacks required"; the backend stages nearly the whole op on CPU. Breaking it out, there are **four distinct device root causes** plus a correctness limitation:

**(a) q/k/v slicing** — `query_cpu[q_start:q_end]` etc. need CPU (no `aten::slice` kernel):

```python
import torch
dev = torch.device("spyre")
q = torch.randn(64, 8, 64, dtype=torch.float16, device=dev)
q_seq = q[16:48]        # dim-0 slice on Spyre — corrupts
```

**(b) KV-cache scatter-write** — `torch.ops.spyre.overwrite` is now `FutureWarning`-deprecated; the standard replacement silently writes to row 0, and per-offset `overwrite` compiles one SDSC binary per unique offset (blows the Dynamo cache):

```python
import torch
dev = torch.device("spyre")
page = torch.zeros(2, 256, 64, dtype=torch.float16, device=dev)  # [kv_heads, block, head]
tok = torch.randn(2, 1, 64, dtype=torch.float16, device=dev)
page.narrow(1, 37, 1).copy_(tok)   # want: symbolic-offset in-place write. Writes row 0 today.
```

**(c) result reshape** — `transpose(1,2).contiguous()` on the head axes is broken (layout-propagation bug):

```python
import torch
dev = torch.device("spyre")
r = torch.randn(1, 2, 32, 64, dtype=torch.float16, device=dev)  # [1, kv_heads, q, head]
out = r.transpose(1, 2).contiguous()    # layout propagation through transpose fails
```

**(d) output scatter** — dim-0 scatter into the output buffer, same primitive as (b).

**(e) MHA / MQA / multi-sequence (correctness, not a device fallback)** — the backend only supports GQA (`num_queries_per_kv > 1`) and `num_seqs=1`; MHA/MQA fail with `"cannot restickify any input layout of y to carry y_var=d2"` (propagate_layouts.py:341).

## Not yet in the table

### `SpyreLogitsProcessor` — in-place mul workaround

`custom_ops/logits_processor.py` forces `logits.contiguous()` because the downstream in-place `logits *= self.scale` triggers a compile issue in `torch-spyre`. Granite 3.3 sets this scale via `logits_scaling`, so it is on the hot path.

```python
import torch
dev = torch.device("spyre")
logits = torch.randn(32, 32000, dtype=torch.float16, device=dev).t()[:32]  # non-contiguous
logits *= 1.0 / 6.0    # in-place mul on a non-contiguous Spyre tensor — compile issue today
```

## Highest-leverage capabilities

Three `torch-spyre` capabilities would eliminate the majority of these fallbacks:

1. **On-device slicing / `narrow` / `select`** — kills `SpyreSiluAndMul`, Attention (a)/(c), `SpyreParallelLMHead` intermediate, `_SpyreModelWrapper`.
2. **A scaling symbolic-offset in-place write / scatter** — kills Attention (b)/(d) and the `SpyreQKVParallelLinear` strided-scatter path.
3. **On-device `embedding` / `index_select`** ([torch-spyre#420](https://github.com/torch-spyre/torch-spyre/issues/420)) — kills `SpyreRotaryEmbedding` (cos/sin gather) and `SpyreVocabParallelEmbedding`.
