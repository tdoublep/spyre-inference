# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encoder-only (bidirectional) self-attention for Spyre, without a KV cache.

Selected by ``TorchSpyrePlatform.get_attn_backend_cls`` for ENCODER/ENCODER_ONLY layers.

The runner pads every sequence to ``L`` and the batch to ``B`` before the model runs, so
Q/K/V arrive as exactly ``B * L`` rows with sequence ``s`` at rows ``s*L``. Attention is a
reshape, one SDPA call and a reshape back.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionImpl, AttentionLayer, AttentionType

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionMetadata,
    SpyrePagedKVCache,
    _call_kernel,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    encoder_warmup_shapes,
    pick_encoder_shape,
)

logger = init_logger(__name__)

# Head dim is padded to the Spyre stick (64 fp16 elements) so QKᵀ's reduction dim is
# aligned; Inductor's insert_bmm_padding raises KeyError: 'val' on MiniLM's head_size=32.
ENCODER_SEQ_ALIGNMENT = 64


def _align_up(n: int, align: int = ENCODER_SEQ_ALIGNMENT) -> int:
    return max(align, (n + align - 1) // align * align)


def build_key_pad_mask(
    num_seqs: int,
    aligned_len: int,
    kv_lens: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive key-pad ``[B, 1, 1, L]``: 0 on real keys, a large negative on pad.

    Built on the host: Spyre cannot produce bool from an int32 ``lt``, nor broadcast
    ``where`` into a 2D grid.

    ``finfo.min / 2``, not ``finfo.min`` or ``-inf``: the torch-spyre SDPA decomposition
    does ``amax`` then ``exp(scores - max)``, so a fully masked row yields NaN.
    """
    if num_seqs != len(kv_lens):
        raise ValueError(f"num_seqs={num_seqs} != len(kv_lens)={len(kv_lens)}")
    cpu = torch.device("cpu")
    kv_len = torch.tensor(kv_lens, dtype=torch.int32, device=cpu).unsqueeze(1)
    kv_pos = torch.arange(aligned_len, dtype=torch.int32, device=cpu).unsqueeze(0)
    zeros = torch.zeros((), dtype=dtype, device=cpu)
    neg = torch.tensor(torch.finfo(dtype).min / 2, dtype=dtype, device=cpu)
    row = torch.where(kv_pos < kv_len, zeros, neg)
    return row.view(num_seqs, 1, 1, aligned_len).contiguous()


def _pad_head_dim_to_stick(flat: torch.Tensor, head_size_padded: int) -> torch.Tensor:
    """Pad last dim to a stick. MiniLM ``[T,H,32]`` cannot ``F.pad`` on Spyre."""
    head_size = flat.shape[-1]
    if head_size == head_size_padded:
        return flat
    device = flat.device
    if device.type == "spyre":
        flat = convert(flat, "cpu")
    flat = F.pad(flat, (0, head_size_padded - head_size))
    if device.type == "spyre":
        flat = convert(flat, device)
    return flat


_CompiledFn = Callable[..., torch.Tensor]

_compiled_kernels: dict[Callable[..., torch.Tensor], _CompiledFn] = {}


def _compile_if_spyre(kernel: _CompiledFn, device_type: str) -> _CompiledFn:
    """Compile ``kernel`` once on Spyre. CPU always runs ``kernel``.

    Not gated on ``enforce_eager``: SDPA has no eager Spyre kernel, so the device path
    must be compiled even in an otherwise-eager run.
    """
    if device_type != "spyre":
        return kernel
    compiled = _compiled_kernels.get(kernel)
    if compiled is None:
        compiled = cast(_CompiledFn, torch.compile(kernel, dynamic=False))
        _compiled_kernels[kernel] = compiled
    return compiled


def _sdpa_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """``[B, H, L, Dp]`` bidirectional attention with an additive key-pad mask."""
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scale)


def _sdpa_kernel_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=scale, enable_gqa=True
    )


def _to_grid(flat: torch.Tensor, batch: int, aligned_len: int, head_size_padded: int):
    """``[B*L, H, D]`` → ``[B, H, L, Dp]``.

    The transpose must materialise: a non-zero storage offset reads as offset 0 on Spyre
    (torch-spyre#3770).
    """
    flat = _pad_head_dim_to_stick(flat, head_size_padded)
    num_heads = flat.shape[1]
    return flat.view(batch, aligned_len, num_heads, head_size_padded).transpose(1, 2).contiguous()


def _ensure_encoder_grid(
    attn_metadata: SpyreAttentionMetadata,
    *,
    padded_tokens: int,
    query: torch.Tensor,
    target_device: torch.device,
    shapes: list[tuple[int, int]],
) -> None:
    """Resolve ``(B, L)`` and build the key-pad mask once per step."""
    if attn_metadata.encoder_pack_batch is not None:
        return

    num_seqs = attn_metadata.num_seqs
    # ``query_start_loc``, not ``seq_lens``: this is bidirectional self-attention so
    # query length *is* kv length, and upstream's ``_dummy_run`` leaves ``seq_lens``
    # carrying the whole padded token count rather than the per-sequence length.
    qsl = attn_metadata.query_start_loc.cpu()
    kv_lens = torch.diff(qsl).tolist()[:num_seqs]
    max_len = max(kv_lens, default=0)

    # Pin the grid to the shape the runner padded to: its product is the row count we
    # were handed. Re-deriving it from metadata alone lets the two disagree.
    pair = pick_encoder_shape(
        num_seqs,
        max_len,
        [(length, batch) for length, batch in shapes if length * batch == padded_tokens],
    )
    if pair is None:
        raise ValueError(
            f"No declared encoder shape covers {num_seqs} sequences of up to "
            f"{max_len} tokens in {padded_tokens} rows (shapes={shapes}). "
            "PoolingSpyreScheduler should have prevented this batch; see "
            "SPYRE_WARMUP_PROMPT_LENS / SPYRE_WARMUP_BATCH_SIZES."
        )
    aligned_len, batch = pair

    # Batch-pad sequences get one attendable key rather than none: an all-masked
    # query row NaNs inside SDPA's online softmax. Their output is never read.
    padded_kv_lens = kv_lens + [1] * (batch - num_seqs)
    key_pad = build_key_pad_mask(batch, aligned_len, padded_kv_lens, dtype=query.dtype)
    if target_device.type == "spyre":
        key_pad = convert(key_pad, target_device)
    else:
        key_pad = key_pad.to(target_device)

    attn_metadata.encoder_pack_batch = batch
    attn_metadata.encoder_pack_len = aligned_len
    attn_metadata.encoder_key_pad_mask = key_pad


class SpyreEncoderAttentionImpl(AttentionImpl):
    """Bidirectional encoder self-attention on a dense ``[B, L]`` grid.

    Subclasses ``AttentionImpl``, not ``SpyreAttentionImpl``: inheriting the decoder's
    ``do_kv_cache_update`` would scatter into an unbound cache.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.ENCODER_ONLY,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        del alibi_slopes, sliding_window, logits_soft_cap, kv_sharing_target_layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.attn_type = attn_type

        # get_current_vllm_config() only works at construction time; forward()
        # runs through a custom-op boundary that loses the context.
        cfg = get_current_vllm_config()
        _dtype = cfg.model_config.dtype
        self.model_dtype: torch.dtype = _dtype if isinstance(_dtype, torch.dtype) else torch.float16
        if kv_cache_dtype not in ("auto", str(self.model_dtype).removeprefix("torch.")):
            raise ValueError(
                f"kv_cache_dtype={kv_cache_dtype} does not match the model dtype "
                f"{self.model_dtype} on Spyre; use 'auto'."
            )
        # `== STOCK`, not `!= NONE`: a bare CompilationConfig leaves mode unset.
        self._compile_attn = cfg.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE
        self._shapes = encoder_warmup_shapes()

    def record_graphs(self, *args, **kwargs) -> int:
        """Nothing to page; warmup traces ``forward`` once per declared shape."""
        return 0

    def forward(  # ty: ignore[invalid-method-override]
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: SpyrePagedKVCache,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,  # [num_tokens, num_heads, head_size]
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, kv_cache, output_scale, output_block_scale
        if attn_metadata is None:
            return output

        padded_tokens = query.shape[0]
        target_device = output.device
        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        head_size = query.shape[2]
        head_size_padded = _align_up(head_size)

        _ensure_encoder_grid(
            attn_metadata,
            padded_tokens=padded_tokens,
            query=query,
            target_device=target_device,
            shapes=self._shapes,
        )
        batch = attn_metadata.encoder_pack_batch
        aligned_len = attn_metadata.encoder_pack_len
        key_pad_mask = attn_metadata.encoder_key_pad_mask
        assert batch is not None and aligned_len is not None and key_pad_mask is not None

        if query.device.type != target_device.type:
            query = convert(query, target_device.type)
            key = convert(key, target_device.type)
            value = convert(value, target_device.type)

        q = _to_grid(query, batch, aligned_len, head_size_padded)
        k = _to_grid(key, batch, aligned_len, head_size_padded)
        v = _to_grid(value, batch, aligned_len, head_size_padded)

        kernel = _compile_if_spyre(
            _sdpa_kernel_gqa if num_kv_heads != num_heads else _sdpa_kernel,
            q.device.type,
        )
        attn_out = _call_kernel("encoder SDPA", kernel, q, k, v, key_pad_mask, self.scale)

        # Offset-0 prefix crop, so torch-spyre#3770 cannot bite.
        if attn_out.shape[-1] != head_size:
            if attn_out.device.type == "spyre":
                attn_out = convert(attn_out, "cpu")
            attn_out = attn_out[..., :head_size]
        result = attn_out.transpose(1, 2).reshape(padded_tokens, num_heads, head_size)

        if result.dtype != output.dtype:
            result = convert(result, dtype=output.dtype)

        # MiniLM D=32: flatten to [T, H*D] (384 = 6 sticks) so the write is aligned.
        use_flat_write = target_device.type == "spyre" and head_size % ENCODER_SEQ_ALIGNMENT != 0
        if use_flat_write:
            if result.device.type == "spyre":
                result = convert(result, "cpu")
            src = convert(
                result.reshape(padded_tokens, -1).contiguous(), target_device.type, output.dtype
            )
            output.reshape(padded_tokens, -1).copy_(src)
        else:
            if result.device.type != output.device.type:
                result = convert(result, output.device)
            output.copy_(result.contiguous())

        return output


class SpyreEncoderAttentionBackend(SpyreAttentionBackend):
    """Encoder-only (no KV cache) variant of the Spyre backend."""

    # These layers have no KV cache, but vLLM still hands encoder-only specs a
    # zero-filled slot mapping, so upstream must skip `unified_kv_cache_update` entirely.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_impl_cls() -> type[SpyreEncoderAttentionImpl]:
        return SpyreEncoderAttentionImpl
