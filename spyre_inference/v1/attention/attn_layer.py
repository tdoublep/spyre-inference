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

"""Spyre ``Attention.forward``: how much of attention lands in the caller's graph.

``install()``, called from the attention metadata builder, binds the forward below onto
each eligible layer instance; every other ``Attention`` keeps upstream's forward and its
``unified_kv_cache_update`` op.

Two shapes of step, two splits:

* **Decode-only** — every sequence contributes one query row, so the KV width is a
  compile-time constant and the whole core traces, giving one graph per decoder block.
  ``SpyreAttentionMetadataBuilder`` publishes the step's plan on ``StepState``; the
  page count is bucketed there so that constant changes rarely. Gated off by default:
  see ``_TRACE_DECODE``.
* **Anything with a prefill** — the core stays behind
  ``unified_attention_with_output``. Its per-sequence loop is shaped by per-sequence
  query lengths, which ``fullgraph=True`` cannot hold.
"""

import os
import types
import weakref
from collections.abc import Iterable
from typing import cast

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backend import AttentionType

from spyre_inference.custom_ops.utils import convert

logger = init_logger(__name__)

# vLLM reserves block 0 as `BlockPool.null_block`, so no sequence is ever given its
# slots. `index_copy_` has no skip index, so they absorb writes with nowhere to go.
_NULL_SLOT = 0

_TRACE_DECODE = os.environ.get("SPYRE_DECODE_TRACED_ATTN", "1") == "1"


def oproj_takes_heads_outer(model: torch.nn.Module) -> int:
    """Mark every Attention whose result may reach its o_proj with heads outermost.

    The traced core stops its epilogue at [num_heads, tokens, head_size] and lets
    o_proj contract each head separately, because folding the head axis into hidden
    inside a graph gives the following matmul a reduction spanning two axes and the
    backend scheduler rejects that (`out_reuse_dim.size() == 1`). Folding it in an
    opaque op instead costs a launch and a relayout per layer, which is more than
    tracing saves.

    So the consumer has to be the sibling ``o_proj`` on the Spyre transposed-weight
    path, whose stored `Wᵀ` is `[num_heads * head_size, out]` — the per-head slices
    the bmm needs are then already its rows. This relies on the model handing
    attention's result straight to ``o_proj``, which is the vLLM idiom; anything that
    reshapes in between has to fold the head axis and keeps the opaque op.

    Returns how many layers were marked.
    """
    from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod

    marked = 0
    for mod in model.modules():
        o_proj = getattr(mod, "o_proj", None)
        if o_proj is None or not isinstance(
            getattr(o_proj, "quant_method", None), SpyreUnquantizedLinearMethod
        ):
            continue
        for child in mod.children():
            impl = getattr(child, "impl", None)
            if not isinstance(child, Attention) or impl is None:
                continue
            if o_proj.weight.shape[0] == impl.num_heads * impl.head_size:
                child.oproj_heads_outer = True
                marked += 1
    return marked


class StepState:
    """This step's device-side attention inputs, shared by every split layer.

    Every field is read while Dynamo traces, so the object itself must outlive the
    trace: it is created once per builder and mutated in place. The container fields
    keep their identity across steps for the same reason — only their tensor elements
    are swapped, which a graph input tolerates.
    """

    def __init__(self, layers: list[Attention]) -> None:
        self._layers = layers
        self._device: torch.device | None = None
        self.slots: torch.Tensor | None = None

        # Whether every split layer can run the traced decode core. Resolved on first
        # use, not in install(): it depends on o_proj wiring that happens once the
        # model's weights are loaded. Tests set it directly.
        self.decode_traceable: bool | None = None

        # This step's decode plan. `decode_pages == 0` means "not a traced decode
        # step" — an int rather than None so the trace guard is a value compare.
        # decode_pages is the (bucketed) page-loop trip count baked into the graph.
        self.decode_pages: int = 0
        self.decode_num_seqs: int = 0
        # Per sequence: [decode_pages * block_size] int32 KV-cache slot indices.
        self.decode_slot_ids: list[torch.Tensor] = []
        # Per sequence: [1, decode_pages * block_size] additive mask.
        self.decode_masks: list[torch.Tensor] = []

    def publish_decode(
        self,
        num_pages: int,
        num_seqs: int,
        slot_ids: list[torch.Tensor],
        masks: list[torch.Tensor],
    ) -> None:
        self.decode_pages = num_pages
        self.decode_num_seqs = num_seqs
        self.decode_slot_ids[:] = slot_ids
        self.decode_masks[:] = masks

    def clear_decode(self) -> None:
        self.decode_pages = 0
        self.decode_num_seqs = 0
        self.decode_slot_ids.clear()
        self.decode_masks.clear()

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
        """Device the split layers' caches live on; None before ``bind_kv_cache``."""
        return self._resolve_device()

    def resolve_decode_traceable(self) -> bool:
        """Whether this step's layers can run the traced decode core.

        Every layer has to qualify, because the builder publishes one plan they all
        read. ALiBi is out: its bias depends on the step's context length, so it is not
        expressible in the step-invariant masks the plan carries. Heads-outer o_proj is
        required rather than preferred — see `oproj_takes_heads_outer` for why the
        alternative costs more than tracing saves.
        """
        if self.decode_traceable is None:
            self.decode_traceable = (
                _TRACE_DECODE
                and bool(self._layers)
                and all(
                    getattr(layer.impl, "alibi_slopes", None) is None
                    and getattr(layer, "oproj_heads_outer", False)
                    for layer in self._layers
                )
            )
            if self.decode_traceable:
                logger.info(
                    "Tracing the attention core into each decoder block for decode steps."
                )
        return self.decode_traceable

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


_holders: weakref.WeakSet[StepState] = weakref.WeakSet()


def publish_null_slots(num_tokens: int) -> None:
    """Point every token at the null block ahead of a run that builds no metadata.

    Warmup would otherwise trace a second graph without the KV write, and a dummy run
    after real inference would scatter into whichever step's slots ran last. The decode
    plan is dropped for the same reason: it describes the last real step's sequences,
    whose count need not match this run's token count.
    """
    for holder in _holders:
        holder.publish_null(num_tokens)
        holder.clear_decode()


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

    state = cast(StepState, self.spyre_step)
    dep = None
    if state.slots is not None and key is not None and value is not None:
        # `dep` makes "scatter before read" a real data dependency, which is otherwise
        # invisible: the opaque op reaches its cache through the forward context, and
        # the traced core reads pages that only alias the scatter's destination.
        dep = self.impl.do_kv_cache_update(self, key, value, self.kv_cache, state.slots)

    if state.decode_pages and dep is not None:
        # Returned unreshaped, as [num_heads, num_tokens, head_size]: this layer's
        # o_proj contracts each head against its own slice of Wᵀ, so the head axis is
        # never folded inside the graph. Only layers whose o_proj can do that reach
        # here — see `oproj_takes_heads_outer`.
        return self.impl.decode_attention(
            query,
            dep,
            state.decode_pages,
            state.decode_num_seqs,
            state.decode_slot_ids,
            state.decode_masks,
        )

    output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
    output = output.view(-1, self.num_heads, self.head_size_v)
    torch.ops.vllm.unified_attention_with_output(
        query,  # ty: ignore[invalid-argument-type]
        key,  # ty: ignore[invalid-argument-type]
        value,  # ty: ignore[invalid-argument-type]
        output,  # ty: ignore[invalid-argument-type]
        _encode_layer_name(self.layer_name),  # ty: ignore[invalid-argument-type]
        kv_cache_dummy_dep=None if dep is None else dep.k_pages,  # ty: ignore[invalid-argument-type]
    )
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
        and not layer.calculate_kv_scales
    )


def install(layers: Iterable[Attention]) -> StepState:
    """Opt eligible layers into the traced KV write; returns their shared step state."""
    split = [layer for layer in layers if _can_split(layer)]
    state = StepState(split)
    _holders.add(state)

    for layer in split:
        layer.spyre_step = state  # ty: ignore[invalid-assignment]
        layer.forward = types.MethodType(  # ty: ignore[invalid-assignment]
            _spyre_attention_forward, layer
        )

    if split:
        logger.info(
            "Scattering the KV cache inside the outer graph for %d attention layers.",
            len(split),
        )
    return state
