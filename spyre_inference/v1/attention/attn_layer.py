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

"""Spyre ``Attention.forward``: the KV write is traced, and decode-only attention with it.

``install()``, called from the attention metadata builder, binds the forward below onto
each eligible layer instance; every other ``Attention`` keeps upstream's forward and its
``unified_kv_cache_update`` op.

The general core stays opaque: its per-sequence Python loop cannot be captured with
``fullgraph=True``. A decode-only batch on the batched decode kernel has no such loop, so
when the step's ``BatchedDecodePlan`` is armed the kernel is traced inline instead.
"""

import types
import weakref
from collections.abc import Iterable
from typing import NamedTuple, cast

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backend import AttentionType

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.batched_decode_plan import BatchedDecodePlan

logger = init_logger(__name__)

# vLLM reserves block 0 as `BlockPool.null_block`, so no sequence is ever given its
# slots. `index_copy_` has no skip index, so they absorb writes with nowhere to go.
_NULL_SLOT = 0


class SlotMapping:
    """This step's slot mapping on device, shared by every split layer."""

    def __init__(self, layers: list[Attention]) -> None:
        self._layers = layers
        self._device: torch.device | None = None
        self.slots: torch.Tensor | None = None

    def _resolve_device(self) -> torch.device | None:
        if self._device is None:
            # `install` runs before bind_kv_cache, so a layer whose cache never arrives
            # still has the empty default and indexing it would raise.
            self._layers = [layer for layer in self._layers if len(layer.kv_cache) > 0]
            if not self._layers:
                return None
            self._device = self._layers[0].kv_cache[0].device
            # Must exist before tracing; see SpyreAttentionImpl.kv_slot_views.
            for layer in self._layers:
                layer.impl.kv_slot_views(layer.kv_cache)  # ty: ignore[possibly-missing-attribute]
        return self._device

    @property
    def device(self) -> torch.device | None:
        """The KV cache's device, or None before ``bind_kv_cache``."""
        return self._resolve_device()

    def publish(self, slot_mapping: torch.Tensor) -> None:
        """Mirror a step's host slot mapping to device for the traced write to read."""
        device = self._resolve_device()
        if device is None:
            return
        self.slots = convert(slot_mapping.clamp(min=_NULL_SLOT), device=device)

    def publish_null(self, num_tokens: int) -> None:
        device = self._resolve_device()
        if device is None:
            return
        self.slots = convert(
            torch.full((num_tokens,), _NULL_SLOT, dtype=torch.int64), device=device
        )


_holders: weakref.WeakSet[SlotMapping] = weakref.WeakSet()


def publish_null_slots(num_tokens: int) -> None:
    """Point every token at the null block ahead of a run that builds no metadata.

    Warmup would otherwise trace a second graph without the KV write, and a dummy run
    after real inference would scatter into whichever step's slots ran last.
    """
    for holder in _holders:
        holder.publish_null(num_tokens)


def _fit_rows(attn: torch.Tensor, rows: int) -> torch.Tensor:
    """Reshape the kernel's ``[padded_num_seqs, ...]`` result to the block's row count.

    The two counts round the same sequence count up on different ladders (the attention
    bucketer's vs ``compile_sizes``), so either can be the larger. Rows past the batch
    are padding the sampler never reads; zero is as good a filler as any. Both branches
    resolve at trace time, so neither reaches the graph.
    """
    num_seqs = attn.shape[0]
    if num_seqs == rows:
        return attn
    if num_seqs > rows:
        return attn[:rows]
    pad = torch.zeros((rows - num_seqs, *attn.shape[1:]), dtype=attn.dtype, device=attn.device)
    return torch.cat([attn, pad], dim=0)


def _spyre_attention_forward(
    self: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = query.dtype
    if output_shape is None:
        output_shape = torch.Size((query.shape[0], self.num_heads * self.head_size_v))
    hidden_size = output_shape[-1]

    query = query.view(-1, self.num_heads, self.head_size)
    if key is not None:
        key = key.view(-1, self.num_kv_heads, self.head_size)
    if value is not None:
        value = value.view(-1, self.num_kv_heads, self.head_size_v)

    dep = None
    slots = cast(SlotMapping, self.spyre_slots).slots
    if slots is not None and key is not None and value is not None:
        # The opaque call reaches its cache through the forward context, so `dep` is what
        # makes "scatter before read" a data dependency. The fused path below needs no
        # such prop: it reads the pages in the same graph that wrote them.
        dep = self.impl.do_kv_cache_update(self, key, value, self.kv_cache, slots)

    # Staged here, not in the impl: the copies are then traced into the block
    # graph, which warmup already compiles at every token bucket.
    staging = getattr(self.impl, "staging_buffers", None)
    buffers = staging(query.device) if staging is not None else None
    rows = query.shape[0]

    plan = cast(BatchedDecodePlan, self.spyre_decode_plan)
    if (
        plan.armed
        and self.spyre_fuse_decode
        and buffers is not None
        and output_dtype == query.dtype
    ):
        # Staging the query is load-bearing here, not shape hygiene: rotary embedding
        # leaves its result a rank-4 buffer ([rows, heads, 2, head_size / 2]) and
        # torch-spyre cannot project a layout from the gather's rank-3 output onto an
        # argument of higher rank. The staging buffer is rank 3, so the gather reads
        # rank 3. The output needs no such buffer -- in-graph the kernel's result is
        # just a value, which is the copy and the program boundary this path saves.
        q_staging, _ = buffers
        q_staging[:rows] = query
        attn = self.impl.fused_batched_decode(plan, q_staging, self.kv_cache)
        return _fit_rows(attn, rows).reshape(-1, hidden_size)

    output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
    output = output.view(-1, self.num_heads, self.head_size_v)

    if buffers is None:
        q_in, out_buf = query, output
    else:
        q_in, out_buf = buffers
        q_in[:rows] = query

    torch.ops.vllm.unified_attention_with_output(
        q_in,  # ty: ignore[invalid-argument-type]
        key,  # ty: ignore[invalid-argument-type]
        value,  # ty: ignore[invalid-argument-type]
        out_buf,  # ty: ignore[invalid-argument-type]
        _encode_layer_name(self.layer_name),  # ty: ignore[invalid-argument-type]
        kv_cache_dummy_dep=dep,  # ty: ignore[invalid-argument-type]
    )
    if buffers is not None:
        output.copy_(out_buf[:rows])
    return output.view(-1, hidden_size)


def _can_split(layer: Attention) -> bool:
    """Only Spyre paged attention, and only where upstream's own prologue is a no-op."""
    return (
        # Encoder-only impls inherit `do_kv_cache_update` from the paged one and would
        # otherwise scatter into an unbound cache.
        layer.attn_type == AttentionType.DECODER
        and hasattr(layer.impl, "do_kv_cache_update")
        and layer.kv_sharing_target_layer_name is None
        and layer.query_quant is None
    )


def _can_fuse_decode(layer: Attention) -> bool:
    """Whether a decode-only step can trace this layer's attention into the block graph.

    ``head_size_v`` must equal ``head_size``: the fused path returns the kernel's result
    as the layer's output, and a narrower value axis reshapes to the wrong row count
    instead of failing.
    """
    return (
        getattr(layer.impl, "supports_fused_decode", False) and layer.head_size == layer.head_size_v
    )


class StepHolders(NamedTuple):
    """The per-step state the patched forward reads from inside the block graph."""

    slots: SlotMapping
    decode_plan: BatchedDecodePlan


def install(layers: Iterable[Attention]) -> StepHolders:
    """Opt eligible layers into the traced KV write; returns their shared per-step state."""
    split = [layer for layer in layers if _can_split(layer)]
    slot_mapping = SlotMapping(split)
    _holders.add(slot_mapping)
    decode_plan = BatchedDecodePlan(
        enabled=envs.SPYRE_BATCHED_DECODE and envs.SPYRE_FUSED_DECODE_ATTN
    )

    fusable = 0
    for layer in split:
        layer.spyre_slots = slot_mapping  # ty: ignore[invalid-assignment]
        layer.spyre_decode_plan = decode_plan  # ty: ignore[invalid-assignment]
        can_fuse = _can_fuse_decode(layer)
        layer.spyre_fuse_decode = can_fuse  # ty: ignore[invalid-assignment]
        fusable += can_fuse
        layer.forward = types.MethodType(  # ty: ignore[invalid-assignment]
            _spyre_attention_forward, layer
        )

    if split:
        logger.info(
            "Scattering the KV cache inside the outer graph for %d attention layers.",
            len(split),
        )
    if decode_plan.enabled and fusable:
        logger.info("Tracing decode-only attention into the outer graph for %d of them.", fusable)
    return StepHolders(slots=slot_mapping, decode_plan=decode_plan)
