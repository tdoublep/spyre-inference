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

"""Spyre ``Attention.forward``: the KV write is traced, the attention core stays opaque.

Upstream has the same split (``forward_includes_kv_cache_update`` plus the
``kv_cache_dummy_dep`` ordering token) but gates it on ``opaque_attention_op()``,
which is all-or-nothing. The online-softmax core must stay opaque: its per-sequence
Python loop cannot be captured with ``fullgraph=True``.
"""

from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.utils.torch_utils import _encode_layer_name

logger = init_logger(__name__)

_original_forward = Attention.forward


class SlotMapping:
    """This step's slot mapping, shared by every split layer so a step publishes once."""

    __slots__ = ("slots",)

    def __init__(self) -> None:
        self.slots: torch.Tensor | None = None


def _spyre_attention_forward(
    self: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # forward is patched on the class, so layers install() never saw (a different kv-cache
    # group, encoder-only attention) must fall back rather than trip over the attribute.
    if not getattr(self, "spyre_kv_write_in_graph", False):
        return _original_forward(self, query, key, value, output_shape, output_dtype)

    if output_dtype is None:
        output_dtype = query.dtype
    if output_shape is None:
        output_shape = torch.Size((query.shape[0], self.num_heads * self.head_size_v))
    output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
    hidden_size = output_shape[-1]

    query = query.view(-1, self.num_heads, self.head_size)
    output = output.view(-1, self.num_heads, self.head_size_v)
    key = key.view(-1, self.num_kv_heads, self.head_size)
    value = value.view(-1, self.num_kv_heads, self.head_size_v)

    # `dep` aliases the mutated pages: passing it to the attention op makes
    # "scatter before read" a real data dependency, which is otherwise invisible
    # because the op reaches its cache through the forward context.
    # No slot mapping: warmup and profile runs have no attention metadata.
    slots = self.spyre_slots.slots
    dep = None
    if slots is not None:
        num_kv_tokens = slots.shape[0]
        dep = self.impl.do_kv_cache_update(
            self,
            key[:num_kv_tokens],
            value[:num_kv_tokens],
            self.kv_cache,
            slots,
        )

    torch.ops.vllm.unified_attention_with_output(
        query,  # ty: ignore[invalid-argument-type]
        key,  # ty: ignore[invalid-argument-type]
        value,  # ty: ignore[invalid-argument-type]
        output,  # ty: ignore[invalid-argument-type]
        _encode_layer_name(self.layer_name),  # ty: ignore[invalid-argument-type]
        kv_cache_dummy_dep=dep,  # ty: ignore[invalid-argument-type]
    )
    return output.view(-1, hidden_size)


def _can_split(layer: Attention) -> bool:
    """Only Spyre paged attention, and only where upstream's own prologue is a no-op."""
    return (
        hasattr(layer.impl, "do_kv_cache_update")
        and layer.kv_sharing_target_layer_name is None
        and layer.query_quant is None
        and not layer.calculate_kv_scales
    )


def install(layers: Iterable[Attention]) -> SlotMapping:
    """Opt eligible layers into the traced KV write; returns their shared slot holder."""
    if Attention.forward is not _spyre_attention_forward:
        Attention.forward = _spyre_attention_forward  # ty: ignore[invalid-assignment]

    slot_mapping = SlotMapping()
    num_split = 0
    for layer in layers:
        split = _can_split(layer)
        layer.spyre_kv_write_in_graph = split
        layer.spyre_slots = slot_mapping
        num_split += split

    if num_split:
        logger.info(
            "Scattering the KV cache inside the outer graph for %d attention layers.",
            num_split,
        )
    return slot_mapping
