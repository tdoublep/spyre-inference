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

"""Narrow the opaque attention boundary so the KV-cache write joins the outer graph.

vLLM hides the whole attention forward behind one opaque custom op. The Spyre
KV write is a plain indirect store, so it can be traced into the compiled
transformer block instead of paying a second compiled-graph launch from inside
the opaque callback. Only the online-softmax core stays opaque — its
per-sequence Python loop cannot be captured with ``fullgraph=True``.

Upstream models this split with ``forward_includes_kv_cache_update`` and the
``kv_cache_dummy_dep`` ordering token, but gates both on
``opaque_attention_op()``, which is all-or-nothing. This override takes the
hybrid: traced KV write, opaque attention core.
"""

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.utils.torch_utils import _encode_layer_name

logger = init_logger(__name__)

_original_forward = Attention.forward


def _spyre_attention_forward(
    self: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not self.spyre_kv_write_in_graph:
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

    # Traced into the caller's graph when the block is compiled. `dep` aliases
    # the mutated pages: handing it to the attention op turns "the scatter must
    # land first" into a real data dependency, which is otherwise invisible
    # because the op reaches its cache through the forward context.
    #
    # No slot mapping means a step with no attention metadata (warmup and
    # profile runs), where there is nothing to scatter.
    slots = self.spyre_slot_mapping
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
        query,
        key,
        value,
        output,
        _encode_layer_name(self.layer_name),
        kv_cache_dummy_dep=dep,
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


def install(model: nn.Module, enabled: bool) -> int:
    """Opt the model's Spyre attention layers into the traced KV write.

    Returns the number of layers opted in. Layers that keep upstream's forward
    (encoder attention, KV sharing, quantised query) are left untouched.
    """
    if Attention.forward is not _spyre_attention_forward:
        Attention.forward = _spyre_attention_forward  # ty: ignore[invalid-assignment]

    num_split = 0
    for module in model.modules():
        if not isinstance(module, Attention):
            continue
        split = enabled and _can_split(module)
        module.spyre_kv_write_in_graph = split
        module.spyre_slot_mapping = None
        num_split += split
    return num_split
