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

"""Paged KV-cache attention backend for Spyre using a dense page tensor and online softmax."""

import functools
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar, NamedTuple

import os

import torch

from spyre_inference.custom_ops.utils import convert

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# When set, wraps forward(), _reshape_and_cache(), and _online_softmax_attention()
# in torch.profiler.record_function spans for kineto trace capture.
_ATTN_PROFILING = os.environ.get("SPYRE_ATTN_PROFILING", "0") == "1"


def _record_function(name: str):
    def decorator(fn):
        if not _ATTN_PROFILING:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with torch.profiler.record_function(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


# Force torch.compile(dynamic=False) on the Spyre attention/reshape kernels
# regardless of the vLLM compilation config. Used to evaluate the compiled path
# on Spyre, where CompilationMode.NONE otherwise makes _maybe_compile a no-op.
# Default: off (unset or "0").
_FORCE_COMPILE_ATTN = os.environ.get("SPYRE_FORCE_COMPILE_ATTN", "0") == "1"

# In the traced path, attend to this step's own K/V from the tensors holding it and
# let the cache pages cover only the context, instead of reading back the slots the
# same graph writes. Costs extra tiles; only needed if Inductor stops ordering the
# scatter before the gathers (it does order them today: both address one tensor).
_FUSE_FRESH_TILES = os.environ.get("SPYRE_FUSE_ATTN_FRESH_TILES", "0") == "1"

# TODO: Make these hyperparameters configurable
# KV length alignment: KV tensors are padded to the next multiple of this value.
# Because torch.compile treats shapes as static constants, every distinct kv_len
# triggers a full recompile. Aligning to 256 buckets sequence lengths into tiers
# (256, 512, 768, ...) so only the first request at each tier pays compilation cost,
# rather than recompiling on every decode step.
KV_LENGTH_ALIGNMENT = 256

# Query chunk size for padding - ensures consistent tensor sizes for Spyre compilation.
# TODO: decode sequences in a mixed batch still pad to this; only decode-only
# batches skip it.
QUERY_CHUNK_SIZE = 32

# Elements per stick for int32 (128-byte stick / 4 bytes).
INT32_ELEMS_PER_STICK = 32

# Slot indices of one page, cached per (page, block_size, device). A page always
# covers the same slots, so these never change: built once, reused for the rest
# of the process, and no per-step host-to-device copy.
#
# Each page needs its own 1D tensor. Slicing a row out of a 2D index table
# inside the graph compiles but returns corrupted rows, so the index cannot be
# one table indexed per block.
_slot_row_cache: dict[tuple[int, int, str], torch.Tensor] = {}


def slot_rows(page: int, block_size: int, device) -> torch.Tensor:
    key = (page, block_size, str(device))
    rows = _slot_row_cache.get(key)
    if rows is None:
        rows = convert(
            torch.arange(page * block_size, (page + 1) * block_size, dtype=torch.int32),
            device=device,
        )
        _slot_row_cache[key] = rows
    return rows


class SpyrePagedKVCache(NamedTuple):
    """Per-layer paged KV cache for the Spyre backend.

    Each field is one dense tensor of shape
    [num_blocks, block_size, num_kv_heads, head_size] on the Spyre device,
    matching `SpyreAttentionBackend.get_kv_cache_shape`.

    NamedTuple (not dataclass) because it is a tuple at runtime, so unpacking
    (`k_pages, v_pages = cache`) traces cleanly under Dynamo without relying on
    attribute access on a custom object.

    Allocated by `TorchSpyreModelRunner.initialize_kv_cache_tensors` and
    consumed by `SpyreAttentionImpl.forward`. vLLM's `bind_kv_cache` types
    the relay path as `dict[str, torch.Tensor]`; see the suppression at the
    `bind_kv_cache(...)` call site for why that type-hole is benign.
    """

    k_pages: torch.Tensor
    v_pages: torch.Tensor


def slot_major_kv_layout(num_slots: int, num_kv_heads: int, head_size: int, dtype: torch.dtype):
    """Slot-axis-outermost layout; without it the indirect store silently
    writes to the wrong rows (torch-spyre#3705)."""
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick

    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_slots, num_kv_heads, sticks, eps],
        stride_map=[num_kv_heads * sticks * eps, sticks * eps, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


def _maybe_compile(fn):
    """Triggers compilation when SPYRE_FORCE_COMPILE_ATTN=1.

    Used only for the online-softmax attention kernel; the reshape/cache
    kernel is compiled unconditionally instead (see _reshape_and_cache).
    """
    if _FORCE_COMPILE_ATTN:
        return torch.compile(fn, dynamic=False)
    return fn


def _reshape_and_cache_kernel(key, value, k_slots, v_slots, slot_mapping):
    k_slots.index_copy_(0, slot_mapping, key)
    v_slots.index_copy_(0, slot_mapping, value)


# ---------------------------------------------------------------------------
# Attention inside the block's graph
# ---------------------------------------------------------------------------
#
# Attention reaches the model through torch.ops.vllm.unified_attention_with_output,
# which Dynamo cannot see into, so a decoder block compiles as prologue /
# attention / epilogue, and the host-side work around the kernel — query padding,
# the permutes, the output copy — is an eager dispatch each. `Attention.forward`
# is traced Python, so putting the attention body there instead leaves one graph
# per block, with the scatter, the page loop and the softmax inside it.
#
# Two constraints make that work:
#
#   * The scatter and the page gathers must go through the *same* tensor. Given
#     the 4D pages and a 3D view of them as two graph inputs, inductor sees no
#     dependency between the store and the loads and asserts on the store's
#     stride map. Both sides use the flat slot-major view, and the mutation then
#     orders the gathers after it.
#   * Each page's slot indices must arrive as their own 1D tensor; see slot_rows.
#
# Per-step values reach the traced body through one object mutated in place.
# Dynamo lifts its tensors to graph inputs guarded on metadata and bakes its ints
# as constants, so a block recompiles when a shape or a block count changes, not
# every step. A fresh object per layer would risk an identity guard, and with it
# a recompile per layer.
_SLOT_MAPPING_ATTR = "_spyre_slot_mapping"
_K_SLOTS_ATTR = "_spyre_k_slots"
_V_SLOTS_ATTR = "_spyre_v_slots"
_FUSED_STEP_ATTR = "_spyre_fused_step"
_ATTN_KERNELS_ATTR = "_spyre_attn_kernels"

_inline_scatter_installed = False
_wired_layers: dict[str, Attention] = {}
_fusable_layers: dict[str, Attention] = {}


class FusedAttentionStep:
    """This step's attention inputs, mutated in place and shared by every layer."""

    __slots__ = (
        "num_seqs",
        "num_actual_tokens",
        "aligned_max_query_len",
        "q_starts",
        "q_ends",
        "slot_row_tables",
        "mask_tiles",
        "slot_mapping",
        "block_size",
        "num_fresh",
    )

    def __init__(self) -> None:
        self.num_seqs = 0
        self.num_actual_tokens = 0
        self.aligned_max_query_len = 0
        self.q_starts: tuple[int, ...] = ()
        self.q_ends: tuple[int, ...] = ()
        self.slot_row_tables: list[list[torch.Tensor]] = []
        self.mask_tiles: list[list[torch.Tensor]] = []
        self.slot_mapping: torch.Tensor | None = None
        self.block_size = 0
        self.num_fresh: tuple[int, ...] = ()


_fused_step = FusedAttentionStep()


def _attn_out_barrier_impl(x: torch.Tensor) -> torch.Tensor:
    # Folding the head axis into the hidden axis is a relayout under Spyre's stick
    # layouts, and in a graph it needs a flat index split by two Mods, which the
    # coordinate mapper rejects ("variable d2 ... appears in multiple Mod
    # expressions"). Eagerly it is just a retile, which is how the untraced path
    # has always done it.
    #
    # clone, because a custom op must return a fresh tensor: reshape hands back a
    # view, and Inductor assumes the result is its own buffer.
    return x.reshape(x.shape[0], -1).clone()


def _attn_out_barrier_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        x.shape[0], x.shape[1] * x.shape[2], dtype=x.dtype, device=x.device
    )


@lru_cache(maxsize=1)
def register_attn_out_barrier() -> None:
    direct_register_custom_op(
        op_name="spyre_attn_out",
        op_func=_attn_out_barrier_impl,
        fake_impl=_attn_out_barrier_fake,
        dispatch_key="CompositeExplicitAutograd",
    )


_fused_mask_cache: dict[tuple, torch.Tensor] = {}


def _fused_mask_tile(
    valid_cols, padded_query_len, query_len, block_size, causal_from, dtype, device
):
    """One additive mask tile for the traced path, cached by its content.

    ``valid_cols`` columns of the tile hold real KV. ``causal_from`` is the absolute
    position of column 0 within this step's own tokens, or None for a context page,
    which needs no causal constraint — every context position precedes every query.
    Rows past ``query_len`` are query padding and are masked out entirely.
    """
    key = (valid_cols, padded_query_len, query_len, block_size, causal_from, str(device))
    tile = _fused_mask_cache.get(key)
    if tile is None:
        cols = torch.arange(block_size)
        rows = torch.arange(padded_query_len)
        attend = (cols < valid_cols).unsqueeze(0) & (rows < query_len).unsqueeze(1)
        if causal_from is not None:
            attend = attend & ((causal_from + cols).unsqueeze(0) <= rows.unsqueeze(1))
        tile = convert(
            torch.where(
                attend,
                torch.tensor(0.0, dtype=dtype),
                torch.tensor(torch.finfo(dtype).min, dtype=dtype),
            ),
            device=device,
        )
        _fused_mask_cache[key] = tile
    return tile


def _fused_masks_for_seq(context_len, query_len, padded_query_len, block_size, dtype, device):
    """Mask tiles for one sequence: context pages first, then its own K/V."""
    tiles = []
    num_prefix = (context_len + block_size - 1) // block_size
    for b in range(num_prefix):
        valid = min(max(context_len - b * block_size, 0), block_size)
        tiles.append(
            _fused_mask_tile(valid, padded_query_len, query_len, block_size, None, dtype, device)
        )
    num_fresh = (padded_query_len + block_size - 1) // block_size
    for t in range(num_fresh):
        tiles.append(
            _fused_mask_tile(
                query_len - t * block_size,
                padded_query_len,
                query_len,
                block_size,
                t * block_size,
                dtype,
                device,
            )
        )
    return num_prefix, num_fresh, tiles


def _attn_out_barrier(x: torch.Tensor) -> torch.Tensor:
    """Keep attention's result out of o_proj's matmul.

    Handed the value, Inductor fuses attention's epilogue — the softmax divide and
    the head-to-token relayout — into o_proj's matmul, and the backend scheduler
    rejects the resulting batchmatmul (`out_reuse_dim.size() == 1` in
    L3DlOpsScheduler). Every in-graph way of forcing a materialization is undone
    by functionalization, which rewrites a store-then-read of the same buffer back
    into the value, so the boundary has to be an op Inductor cannot see through.

    It costs one dispatch per layer. Everything before it — norm, QKV, RoPE, the KV
    scatter, the page loop, the softmax — still compiles as one program, where the
    eager path spends a launch per step on the query padding and the output copy.
    """
    return torch.ops.vllm.spyre_attn_out(x)


def _traced_attention(impl, step, kernels, query, key, value, k_slots, v_slots):
    """Scatter this step's K/V and attend over the cache, in the caller's graph.

    query/key/value are [num_tokens, heads, head_size] views of this layer's
    projection output; the result is [num_tokens, num_heads * head_size].

    Pages cover the context only. This step's own K/V is attended to from the
    tensors that hold it, so nothing here reads a slot this graph is writing —
    Inductor defers a graph input's mutation, so a gather of those slots can run
    before the scatter lands.
    """
    num_tokens = step.num_actual_tokens
    k_slots.index_copy_(0, step.slot_mapping, key[:num_tokens])
    v_slots.index_copy_(0, step.slot_mapping, value[:num_tokens])

    block_size = step.block_size
    num_kv_heads = impl.num_kv_heads
    num_queries_per_kv = impl.num_queries_per_kv
    head_size = impl.head_size
    padded_query_len = step.aligned_max_query_len

    out = torch.empty(
        query.shape[0], impl.num_heads, head_size, dtype=query.dtype, device=query.device
    )
    for seq_idx in range(step.num_seqs):
        q_start = step.q_starts[seq_idx]
        q_end = step.q_ends[seq_idx]
        query_len = q_end - q_start

        if query_len == 1:
            q_dev = query.unbind(dim=0)[q_start].reshape(
                num_kv_heads, num_queries_per_kv, 1, head_size
            )
            if padded_query_len > 1:
                q_dev = torch.nn.functional.pad(q_dev, (0, 0, 0, padded_query_len - 1))
        else:
            q_seq = query[q_start:q_end]
            if padded_query_len > query_len:
                q_seq = torch.nn.functional.pad(
                    q_seq, (0, 0, 0, 0, 0, padded_query_len - query_len)
                )
            q_dev = (
                q_seq.unsqueeze(0)
                .transpose(1, 2)
                .contiguous()
                .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
            )

        # This step's K/V for this sequence, laid out like a gathered page and
        # padded to whole tiles so every tile the kernel sees is block_size wide.
        kv_rows = step.num_fresh[seq_idx] * block_size
        fresh_k = []
        fresh_v = []
        for src in (key, value):
            tile = src[q_start:q_end].permute(1, 0, 2).unsqueeze(1)
            if kv_rows > query_len:
                tile = torch.nn.functional.pad(tile, (0, 0, 0, kv_rows - query_len))
            dest = fresh_k if src is key else fresh_v
            for t in range(step.num_fresh[seq_idx]):
                dest.append(tile[:, :, t * block_size : (t + 1) * block_size, :])

        result = kernels[seq_idx](
            q_dev,
            k_slots,
            v_slots,
            step.slot_row_tables[seq_idx],
            step.mask_tiles[seq_idx],
            impl.scale,
            fresh_k=fresh_k,
            fresh_v=fresh_v,
        )
        if query_len < padded_query_len:
            # A prefix view copies its whole extent and overruns the destination
            # (torch-spyre#3826), so copy it first.
            out[q_start:q_end] = result[:query_len].clone()
        else:
            out[q_start:q_end] = result

    return _attn_out_barrier(out)


def _layer_is_fusable(layer) -> bool:
    """Whether the traced path can serve this layer, on properties fixed at build."""
    impl = layer.impl
    return (
        isinstance(impl, SpyreAttentionImpl)
        and impl.alibi_slopes is None
        and impl.attn_type == AttentionType.DECODER
        and getattr(layer, "kv_sharing_target_layer_name", None) is None
        and not getattr(layer, "calculate_kv_scales", False)
        and getattr(layer, "query_quant", None) is None
        and getattr(layer, "head_size_v", impl.head_size) == impl.head_size
    )


def fused_step_for(attn_metadata, device) -> FusedAttentionStep | None:
    """Fill the shared step context, or None if the traced path can't serve this
    step. Mirrors to device what the traced body reads, since it cannot itself."""
    md = attn_metadata
    if md.page_indices is None:
        return None
    if md.active_block_indices is not None:
        # Sliding window: pages are a filtered subset, so "the first N pages hold
        # the context" no longer holds. The eager path handles those.
        return None

    starts = md.query_start_loc.tolist()
    if len(starts) != md.num_seqs + 1:
        return None

    if md.slot_mapping_device is None:
        md.slot_mapping_device = convert(md.slot_mapping[: md.num_actual_tokens], device=device)

    register_attn_out_barrier()

    padded_query_len = md.aligned_max_query_len
    block_size = md.block_size
    dtype = md.attention_mask_tiles[0][0].dtype
    seq_lens = md.seq_lens.tolist()

    rows_per_seq = []
    masks_per_seq = []
    num_fresh_per_seq = []
    for s in range(md.num_seqs):
        query_len = starts[s + 1] - starts[s]
        context_len = seq_lens[s] - query_len
        if _FUSE_FRESH_TILES:
            num_prefix, num_fresh, tiles = _fused_masks_for_seq(
                context_len, query_len, padded_query_len, block_size, dtype, device
            )
            if num_prefix > len(md.page_indices[s]):
                return None
            pages = md.page_indices[s][:num_prefix]
        else:
            num_fresh = 0
            pages = md.page_indices[s]
            if md.attention_mask_tiles_device is None:
                tiles_cpu = md.attention_mask_tiles
                assert tiles_cpu is not None, "attention_mask_tiles must be precomputed"
                md.attention_mask_tiles_device = [
                    [convert(t, device=device) for t in seq] for seq in tiles_cpu
                ]
            tiles = md.attention_mask_tiles_device[s]
        rows_per_seq.append([slot_rows(p, block_size, device) for p in pages])
        masks_per_seq.append(tiles)
        num_fresh_per_seq.append(num_fresh)

    step = _fused_step
    step.num_seqs = md.num_seqs
    step.num_actual_tokens = md.num_actual_tokens
    step.aligned_max_query_len = padded_query_len
    step.block_size = block_size
    step.q_starts = tuple(starts[:-1])
    step.q_ends = tuple(starts[1:])
    step.slot_row_tables = rows_per_seq
    step.mask_tiles = masks_per_seq
    step.num_fresh = tuple(num_fresh_per_seq)
    step.slot_mapping = md.slot_mapping_device
    return step


def _patched_attention_forward(orig_forward):
    def forward(self, query, key, value, *args, **kwargs):
        step = getattr(self, _FUSED_STEP_ATTR, None)
        if step is not None and not args and not kwargs:
            num_heads = self.num_heads
            head_size = self.head_size
            # Already [num_tokens, num_heads * head_size]: the barrier folds the
            # head axis in, since the graph cannot express that relayout.
            return _traced_attention(
                self.impl,
                step,
                getattr(self, _ATTN_KERNELS_ATTR),
                query.view(-1, num_heads, head_size),
                key.view(-1, self.num_kv_heads, head_size),
                value.view(-1, self.num_kv_heads, head_size),
                getattr(self, _K_SLOTS_ATTR),
                getattr(self, _V_SLOTS_ATTR),
            )

        slot_mapping = getattr(self, _SLOT_MAPPING_ATTR, None)
        if slot_mapping is not None and key is not None and value is not None:
            num_tokens = slot_mapping.shape[0]
            getattr(self, _K_SLOTS_ATTR).index_copy_(
                0,
                slot_mapping,
                key.view(-1, self.num_kv_heads, self.head_size)[:num_tokens],
            )
            getattr(self, _V_SLOTS_ATTR).index_copy_(
                0,
                slot_mapping,
                value.view(-1, self.num_kv_heads, self.head_size)[:num_tokens],
            )
        return orig_forward(self, query, key, value, *args, **kwargs)

    return forward


def install_inline_kv_scatter(static_forward_context, kv_caches) -> int:
    """Move the KV scatter into the graph that produces K/V, for layers we can.

    Attaches the flattened slot-major views of each layer's pages, then wraps
    ``Attention.forward`` so it scatters before delegating. Layers left without
    the views fall back to ``SpyreAttentionImpl._reshape_and_cache``.

    Returns the number of layers wired up.
    """
    global _inline_scatter_installed

    _wired_layers.clear()
    _fusable_layers.clear()
    for layer_name, cache in kv_caches.items():
        layer = static_forward_context.get(layer_name)
        if layer is None:
            continue
        # A view keeps the slot-outermost device layout; taken here rather than in
        # the traced graph, where an in-graph view of the 4D pages makes inductor
        # assert on the store's stride map.
        slots = (-1, cache.k_pages.shape[2], cache.k_pages.shape[3])
        setattr(layer, _K_SLOTS_ATTR, cache.k_pages.view(slots))
        setattr(layer, _V_SLOTS_ATTR, cache.v_pages.view(slots))
        _wired_layers[layer_name] = layer
        if _layer_is_fusable(layer):
            _fusable_layers[layer_name] = layer

    if _wired_layers and not _inline_scatter_installed:
        Attention.forward = _patched_attention_forward(  # ty: ignore[invalid-assignment]
            Attention.forward
        )
        _inline_scatter_installed = True
    return len(_wired_layers)


def build_slot_row_tables(attn_metadata, device) -> list[list[torch.Tensor]]:
    """Slot-index tensors for every active block of every sequence."""
    page_indices = attn_metadata.page_indices
    assert page_indices is not None, "page_indices must be set by the metadata builder"
    block_size = attn_metadata.block_size
    return [[slot_rows(p, block_size, device) for p in seq] for seq in page_indices]


def wired_kv_scatter_layers() -> dict[str, Attention]:
    """Layers whose KV write the traced scatter can take over."""
    return _wired_layers


def fusable_attention_layers() -> dict[str, Attention]:
    """Layers the traced attention path can serve."""
    return _fusable_layers


def step_kernels(impl, step) -> tuple:
    """The per-sequence attention loops this step needs, built outside the graph.

    Built here rather than in the traced body: creating the closure while Dynamo
    is tracing would put a function definition in the middle of the graph.
    """
    return tuple(
        impl.traced_attn_fn(
            len(step.slot_row_tables[s]), step.aligned_max_query_len, step.num_fresh[s]
        )
        for s in range(step.num_seqs)
    )


def publish_fused_attention(step_by_layer) -> None:
    """Arm the traced path for this step, and disarm every layer it cannot serve.

    Values are (step, kernels) pairs. A context left from an earlier step would be
    applied to this step's K/V with the wrong token count and slot mapping.
    """
    for layer_name, layer in _wired_layers.items():
        armed = step_by_layer.get(layer_name)
        if armed is None:
            if hasattr(layer, _FUSED_STEP_ATTR):
                delattr(layer, _FUSED_STEP_ATTR)
            continue
        step, kernels = armed
        setattr(layer, _FUSED_STEP_ATTR, step)
        setattr(layer, _ATTN_KERNELS_ATTR, kernels)


def publish_slot_mapping(slot_mapping_by_layer) -> None:
    """Hand this step's device slot mappings to the traced scatter.

    Every wired layer is visited, every step: a mapping left over from another
    step would be applied to this step's K/V, and it is the wrong length as soon
    as the token count changes — an out-of-bounds device write, not an error. Any
    layer absent from ``slot_mapping_by_layer`` falls back to the impl's scatter.
    """
    for layer_name, layer in _wired_layers.items():
        slot_mapping = slot_mapping_by_layer.get(layer_name)
        if slot_mapping is None:
            if hasattr(layer, _SLOT_MAPPING_ATTR):
                delattr(layer, _SLOT_MAPPING_ATTR)
        else:
            setattr(layer, _SLOT_MAPPING_ATTR, slot_mapping)


# ---------------------------------------------------------------------------
# Compilable factory functions
# ---------------------------------------------------------------------------


def _create_compilable_page_attn(
    num_blocks: int,
    padded_query_len: int,
    num_heads: int,
    head_size: int,
    has_alibi: bool = False,
    logits_soft_cap: float = 0.0,
    fused_store: bool = False,
    num_fresh: int = 0,
):
    """Create online softmax attention over a fixed number of pages for torch.compile.

    Dynamo unrolls the loop because num_blocks, padded_query_len, has_alibi,
    logits_soft_cap, fused_store and num_fresh are closure constants.

    ``num_fresh`` appends tiles the caller passes in directly rather than gathering
    them from the cache — this step's own K/V, still in registers. The traced path
    uses it so attention never reads the slots the same graph is writing: Inductor
    defers a graph input's mutation, so a gather of those slots can run before the
    scatter lands. Pages then only have to cover the context, and the mask tiles
    for them stop at its end.

    With ``fused_store``, the kernel scatters its result into the caller's output
    buffer instead of returning it, so the store lands in the same jobplan rather
    than costing an extra eager dispatch and launch per layer.
    """

    def specialized_paged_attn_kernel(
        q,
        k_slots,
        v_slots,
        slot_rows_per_block,
        mask_tiles,
        scale,
        alibi_bias_tiles=None,
        out=None,
        fresh_k=None,
        fresh_v=None,
    ):
        """
        This kernels specializes for num_blocks and padded_query_len.

        Expected shapes:
            q: [num_kv_heads, num_queries_per_kv, padded_query_len, head_size]
            k_slots: [num_blocks_total * block_size, num_kv_heads, head_size]
            v_slots: [num_blocks_total * block_size, num_kv_heads, head_size]
            slot_rows_per_block: one [block_size] int32 device tensor per active
                block, holding that page's slot indices.
            mask_tiles: [num_blocks + num_fresh]
            fresh_k, fresh_v: only with num_fresh — one
                [num_kv_heads, 1, width, head_size] tile each, this step's own K/V
            alibi_bias_tiles: list of [num_kv_heads, num_queries_per_kv, 1, block_size]
                (only when has_alibi=True; None otherwise). The query-axis dim
                is 1 because softmax absorbs per-query-row constants — see
                the derivation at the bias-tile construction site in
                _online_softmax_attention.
            out: only with fused_store — the caller's output buffer, which this
                sequence owns in full ([padded_query_len, num_heads, head_size]).

        Returns [padded_query_len, num_heads, head_size], or ``out`` when
        fused_store scattered the result in place.
        """
        tile_max = None
        tile_sum = None
        tile_output = None

        for i in range(num_blocks + num_fresh):
            if i < num_blocks:
                # index_select, not `k_slots[rows]`: subscripting lowers to
                # aten.index, which upcasts the int32 index to int64 and fails eager.
                rows = slot_rows_per_block[i]
                # Token-major page to head-major for the matmuls; permutes on device.
                k_page_4d = k_slots.index_select(0, rows).permute(1, 0, 2).unsqueeze(1)
                v_page_4d = v_slots.index_select(0, rows).permute(1, 0, 2).unsqueeze(1)
            else:
                assert fresh_k is not None and fresh_v is not None
                k_page_4d = fresh_k[i - num_blocks]
                v_page_4d = fresh_v[i - num_blocks]

            mask_tile = mask_tiles[i]

            scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
            if logits_soft_cap > 0.0:
                # Pull logits into (-cap, +cap) before the mask add so masked
                # positions still map cleanly to -inf. Applied before the ALiBi
                # bias so the positional term is not squashed by the tanh.
                scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
            if has_alibi:
                # ALiBi bias slope[h] * (kv_pos - context_len). The additive
                # mask_tile below uses finfo.min for masked positions, so this
                # bias cannot un-mask them.
                assert alibi_bias_tiles is not None
                scores = scores + alibi_bias_tiles[i]
            scores = scores + mask_tile
            scores_max = torch.amax(scores, dim=-1, keepdim=True)

            if i == 0:
                tile_max = scores_max
                tile_probs = torch.exp(scores - tile_max)
                tile_output = torch.matmul(tile_probs, v_page_4d)
                tile_sum = tile_probs.sum(dim=-1, keepdim=True)
            else:
                # i > 0 only reachable after the i == 0 branch initialized these.
                assert tile_max is not None
                assert tile_sum is not None
                assert tile_output is not None
                new_max = torch.maximum(tile_max, scores_max)
                rescale = torch.exp(tile_max - new_max)
                tile_output = tile_output * rescale
                tile_sum = tile_sum * rescale
                tile_probs = torch.exp(scores - new_max)
                tile_output += torch.matmul(tile_probs, v_page_4d)
                tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
                tile_max = new_max

        assert tile_output is not None and tile_sum is not None
        attn = tile_output / tile_sum
        attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
        attn = attn.reshape(padded_query_len, num_heads, head_size)
        if fused_store:
            # A full copy, not an indirect store: a compiled index_copy_ writes
            # nothing at all when the destination has a single row, which is every
            # batch-1 decode. The caller only enables fused_store when this
            # sequence owns every row of `out`, so the shapes match exactly.
            assert out is not None
            out.copy_(attn)
            return out
        return attn

    return specialized_paged_attn_kernel


@dataclass
class SpyreAttentionMetadata(AttentionMetadata):
    """Metadata for paged online-softmax attention on Spyre."""

    # Total real (non-padding) tokens across all sequences. Used to slice
    # q/k/v to actual tokens before processing (input may have padding).
    num_actual_tokens: int

    # Number of sequences in this batch.
    num_seqs: int

    # Maximum query length among all sequences (raw, unaligned).
    max_query_len: int

    # Maximum KV sequence length among all sequences (raw, unaligned).
    max_seq_len: int

    # Per-sequence KV lengths. [num_seqs]
    seq_lens: torch.Tensor

    # Cumulative query lengths for varlen layout. query_start_loc[i]
    # is the start offset of sequence i in the flat q/k/v buffer.
    # [num_seqs + 1], last entry = total tokens.
    query_start_loc: torch.Tensor

    # Block table mapping logical blocks to physical pages.
    # [num_seqs, max_num_blocks_per_seq]
    block_table: torch.Tensor

    # Number of KV tokens per physical page.
    block_size: int

    # Flat mapping from token index to its position in the KV cache
    # (physical_block_index * block_size + block_offset). [num_actual_tokens]
    slot_mapping: torch.Tensor

    # True when causal masking is needed (prefill/mixed, i.e. max_query_len > 1).
    # Decode steps (max_query_len=1) don't need explicit causal masking because
    # the online softmax over KV pages naturally only attends to past tokens.
    apply_causal_mask: bool = False

    # Number of KV heads (for GQA).
    num_kv_heads: int = 0

    # Number of query heads.
    num_heads: int = 0

    # Pre-tiled additive attention mask. attention_mask_tiles[seq_idx][i]
    # gives the mask tile for the i-th ACTIVE block of one sequence (indexed
    # by position within active_block_indices[seq_idx], not by absolute block
    # index). Each tile: [aligned_max_query_len, block_size] on CPU. When
    # sliding_window is None, active == all blocks and the layout is
    # equivalent to indexing by absolute block index.
    attention_mask_tiles: list[list[torch.Tensor]] | None = None

    # For each sequence: absolute block indices whose mask is not fully
    # `-inf` (blocks that contribute to at least one query's attention).
    # None means all blocks are active (sliding_window is None, or the
    # window covers the whole sequence). When set, len(active_block_indices[s])
    # matches len(attention_mask_tiles[s]).
    active_block_indices: list[list[int]] | None = None

    # Global aligned query length for stable kernel compilation.
    # max_query_len rounded up to QUERY_CHUNK_SIZE (32). All queries are
    # padded to this length so the compiled attention kernel receives
    # consistent tensor shapes across steps and sequences.
    aligned_max_query_len: int = 0

    # Global aligned KV sequence length for stable kernel compilation.
    # max_seq_len rounded up to KV_LENGTH_ALIGNMENT (256). The KV mask
    # dimension is padded to this length so recompilation only happens
    # per 256-token tier, not per distinct sequence length.
    aligned_max_seq_len: int = 0

    # Physical page index of each active block, per sequence.
    page_indices: list[list[int]] | None = None

    # Device slot indices for those pages, one [block_size] int32 tensor per
    # active block. Taken from the process-wide cache on first use.
    slot_row_tables: list[list[torch.Tensor]] | None = None

    # Device mirror of slot_mapping, which vLLM hands us on the host.
    slot_mapping_device: torch.Tensor | None = None

    # True once the traced scatter in Attention.forward has taken over the KV write
    # for this step, so forward() must not do it again. See install_inline_kv_scatter.
    inline_kv_scatter: bool = False

    # Device mirror of attention_mask_tiles, filled once per step by forward().
    attention_mask_tiles_device: list[list[torch.Tensor]] | None = None

    @property
    def query_lens(self) -> torch.Tensor:
        """Per-sequence query lengths, derived from query_start_loc. [num_seqs]"""
        return self.query_start_loc[1:] - self.query_start_loc[:-1]


class SpyreAttentionMetadataBuilder(AttentionMetadataBuilder[SpyreAttentionMetadata]):
    """Builds attention metadata — only the attention mask is precomputed."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self.head_size = kv_cache_spec.head_size
        self.sliding_window = getattr(kv_cache_spec, "sliding_window", None)
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {self.sliding_window}")

        # Validate block_size alignment: Spyre stick size is 128 bytes (64 fp16 elements).
        # block_size must be a multiple of 64 to avoid restickification errors during
        # torch.compile.
        if self.block_size % 64 != 0:
            raise ValueError(
                f"block_size must be a multiple of 64 for the Spyre paged attention "
                f"backend. Got block_size={self.block_size}, head_size={self.head_size}. "
            )

        model_config = vllm_config.model_config
        self.num_heads = model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        # `model_config.dtype` is typed `ModelDType | torch.dtype`, but
        # `TorchSpyrePlatform.check_and_update_config` rejects anything but
        # `torch.float16` upstream so it's always a real torch.dtype here.
        assert isinstance(model_config.dtype, torch.dtype)
        self.model_dtype: torch.dtype = model_config.dtype

        # Shared zero tile reused for interior active blocks (fully inside the
        # window, so their mask is all-zeros). Allocated lazily on first use
        # and resized if aligned_max_query_len or block_size changes across
        # calls.
        self._zero_tile: torch.Tensor | None = None
        self._zero_tile_shape: tuple[int, int] = (0, 0)

    def _get_zero_tile(self, aligned_max_query_len: int) -> torch.Tensor:
        """Return (or create) the shared all-zero mask tile for interior blocks.

        The returned tensor is reused by reference across all interior blocks
        and sequences in a batch. Callers must treat it as read-only: any
        in-place mutation would corrupt every interior tile simultaneously.
        This is safe today because attention kernels only read mask tiles.
        """
        shape = (aligned_max_query_len, self.block_size)
        if self._zero_tile is None or self._zero_tile_shape != shape:
            self._zero_tile = torch.zeros(shape, dtype=self.model_dtype)
            self._zero_tile_shape = shape
        return self._zero_tile

    def _build_attention_mask(
        self,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        apply_causal_mask: bool,
        max_query_len: int,
        aligned_max_query_len: int,
        aligned_max_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build additive attention mask on Spyre for the non-sliding-window path.

        All sequences share the same aligned_max_query_len so every mask tile
        has a uniform query dimension — this avoids per-sequence kernel
        specializations.

        Sliding-window sequences take a different path: see
        _build_active_tiles_with_skip.

        Returns:
            - mask: [num_seqs, aligned_max_query_len, aligned_max_seq_len] additive mask
        """
        assert self.sliding_window is None
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        num_seqs = len(seq_lens)

        q_pos = torch.arange(max_query_len, device=device)
        kv_pos = torch.arange(aligned_max_seq_len, device=device)

        # Padding mask: valid positions are within actual sequence/query lengths
        q_valid = q_pos.unsqueeze(0) < query_lens.unsqueeze(1)
        kv_valid = kv_pos.unsqueeze(0) < seq_lens.unsqueeze(1)
        attend = q_valid.unsqueeze(2) & kv_valid.unsqueeze(1)

        # Causal mask: prevent attending to future tokens during generation
        if apply_causal_mask:
            context_lens = seq_lens - query_lens
            causal_limit = (context_lens.unsqueeze(1) + q_pos.unsqueeze(0)).unsqueeze(2)
            kv_pos_exp = kv_pos.unsqueeze(0).unsqueeze(0)
            causal_ok = kv_pos_exp <= causal_limit
            attend = attend & causal_ok

        # Convert to additive mask: finfo.min for masked positions, 0 for valid
        mask_bool = ~attend  # [num_seqs, max_query_len, aligned_max_seq_len]

        if aligned_max_query_len > max_query_len:
            padding = torch.ones(
                num_seqs,
                aligned_max_query_len - max_query_len,
                aligned_max_seq_len,
                dtype=torch.bool,
                device=device,
            )
            mask_bool = torch.cat([mask_bool, padding], dim=1)

        mask_additive = torch.where(
            mask_bool,
            torch.tensor(torch.finfo(self.model_dtype).min, dtype=self.model_dtype, device=device),
            torch.tensor(0.0, dtype=self.model_dtype, device=device),
        )

        return mask_additive

    def _build_single_tile(
        self,
        block_idx: int,
        kv_len: int,
        query_len: int,
        context_len: int,
        aligned_max_query_len: int,
        apply_causal_mask: bool,
    ) -> torch.Tensor:
        """Build the additive mask tile for one (sequence, block) pair.

        Returns a [aligned_max_query_len, block_size] CPU tensor.

        Only called for boundary blocks that require real mask content:
          - lower-boundary blocks (window-start cutoff falls inside them for
            at least one query), and
          - the upper-boundary block (last block: KV padding, plus causal
            during prefill).
        Interior blocks reuse the shared zero tile instead.
        """
        block_size = self.block_size
        mask_min = torch.finfo(self.model_dtype).min

        # KV positions covered by this block. May extend past kv_len (handled
        # by the kv_valid mask below).
        kv_start = block_idx * block_size
        kv_end = kv_start + block_size

        q_pos = torch.arange(aligned_max_query_len)  # [aligned_max_query_len]
        kv_pos = torch.arange(kv_start, kv_end)  # [block_size]

        # Padding mask: query rows beyond query_len are fully masked;
        # KV columns beyond kv_len are fully masked.
        q_valid = q_pos < query_len  # [aligned_max_query_len]
        kv_valid = kv_pos < kv_len  # [block_size]
        attend = q_valid.unsqueeze(1) & kv_valid.unsqueeze(0)  # [Q, B]

        # Causal mask (prefill only): query at absolute position
        # context_len + q_pos can only attend to KV positions <= that value.
        if apply_causal_mask:
            causal_limit = context_len + q_pos  # [aligned_max_query_len]
            attend = attend & (kv_pos.unsqueeze(0) <= causal_limit.unsqueeze(1))

        # Sliding window: per-query window_start.
        assert self.sliding_window is not None
        abs_q_pos = context_len + q_pos  # [aligned_max_query_len]
        window_start = (abs_q_pos - self.sliding_window + 1).clamp(min=0)
        attend = attend & (kv_pos.unsqueeze(0) >= window_start.unsqueeze(1))

        mask_bool = ~attend
        return torch.where(
            mask_bool,
            torch.tensor(mask_min, dtype=self.model_dtype),
            torch.tensor(0.0, dtype=self.model_dtype),
        )

    def _build_active_tiles_with_skip(
        self,
        kv_len: int,
        query_len: int,
        context_len: int,
        aligned_max_query_len: int,
        apply_causal_mask: bool,
    ) -> tuple[list[int], list[torch.Tensor]]:
        """Return (active_block_indices, mask_tiles) using arithmetic block-skip.

        active_block_indices: absolute block indices whose mask contributes
        to at least one query's attention (i.e. inside the window of the
        earliest query).
        mask_tiles: one tile per active block, in the same order.

        Block classification:
          - [0, first_active):
                entirely outside every query's window; skipped.
          - [first_active, last_lower_boundary]:
                lower-boundary blocks — the window cutoff falls inside them
                for at least one query. Real tile with per-query-row cutoffs.
                In decode (query_len == 1) this collapses to a single block.
          - (last_lower_boundary, last_causal_interior]:
                interior blocks — fully inside every query's window AND fully
                below the earliest query's causal limit. Mask is all-zero.
          - (last_causal_interior, last_block):
                causal-boundary blocks — inside every window, but early
                queries have causal cutoffs falling inside them (prefill
                only). Real tile.
          - last_block:
                upper-boundary block — always has KV padding (and causal
                cutoffs during prefill). Real tile.

        When any of the boundary ranges overlap (short kv_len, single-block
        sequence, etc.) real tiles are built for the union — never zero tiles.
        """
        assert self.sliding_window is not None
        block_size = self.block_size
        num_blocks = (kv_len + block_size - 1) // block_size

        # Earliest query (q_pos=0) has window
        # [max(0, context_len - W + 1), context_len].
        # Latest query (q_pos=query_len-1) has window
        # [max(0, kv_len - W), kv_len - 1].
        # A block is fully outside every query's window when its highest KV
        # position is below the earliest query's window start.
        # NOTE: using the EARLIEST query's window (not the latest, kv_len - W)
        # is required for prefill correctness. In a prefill batch with
        # query_len > 1, early queries have earlier windows and their
        # in-window blocks would otherwise be incorrectly dropped. For decode
        # (query_len == 1) both formulas coincide.
        earliest_window_start = max(0, context_len - self.sliding_window + 1)
        latest_window_start = max(0, kv_len - self.sliding_window)

        first_active = earliest_window_start // block_size
        # Every block from first_active up to the block containing the
        # latest window start can have a per-query cutoff falling inside it.
        last_lower_boundary = latest_window_start // block_size
        # A block is fully below the earliest query's causal limit
        # (abs_pos = context_len) iff (b + 1) * block_size - 1 <= context_len.
        # For decode (no causal mask) all blocks satisfy this trivially.
        if apply_causal_mask:
            last_causal_interior = (context_len + 1) // block_size - 1
        else:
            last_causal_interior = num_blocks - 1
        last_block = num_blocks - 1

        active_bs = list(range(first_active, num_blocks))
        if not active_bs:
            return [], []

        zero_tile = self._get_zero_tile(aligned_max_query_len)
        tiles: list[torch.Tensor] = []

        for b in active_bs:
            is_lower_boundary = b <= last_lower_boundary
            is_upper_boundary = (b == last_block) and not is_lower_boundary
            is_causal_boundary = apply_causal_mask and b > last_causal_interior and b != last_block
            if is_lower_boundary or is_upper_boundary or is_causal_boundary:
                tiles.append(
                    self._build_single_tile(
                        b,
                        kv_len,
                        query_len,
                        context_len,
                        aligned_max_query_len,
                        apply_causal_mask,
                    )
                )
            else:
                # Interior block: entirely within every query's window,
                # entirely filled with valid KV tokens, and (for prefill)
                # entirely below the earliest query's causal limit.
                # Mask is all-zero.
                tiles.append(zero_tile)

        return active_bs, tiles

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SpyreAttentionMetadata:
        """Build attention metadata from common metadata."""

        seq_lens = common_attn_metadata.seq_lens
        query_start_loc = common_attn_metadata.query_start_loc
        max_seq_len = common_attn_metadata.max_seq_len
        max_query_len = common_attn_metadata.max_query_len
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        causal = common_attn_metadata.causal
        if isinstance(causal, torch.Tensor):
            causal = bool(causal.item())
        # Batch-level flag: True iff the batch contains at least one prefill
        # sequence (max_query_len > 1). For decode sequences (query_len == 1)
        # in a mixed batch, the causal constraint is subsumed by the KV
        # validity mask (the single query at position context_len can only
        # attend to KV positions [0, kv_len) = [0, context_len]), so applying
        # the causal mask to them is a correct no-op.
        apply_causal_mask = causal and max_query_len > 1

        # A decode-only batch needs no padding at all: every query_len is 1.
        if max_query_len == 1:
            aligned_max_query_len = 1
        else:
            aligned_max_query_len = (
                (max_query_len + QUERY_CHUNK_SIZE - 1) // QUERY_CHUNK_SIZE * QUERY_CHUNK_SIZE
            )
        aligned_max_seq_len = (
            (max_seq_len + KV_LENGTH_ALIGNMENT - 1) // KV_LENGTH_ALIGNMENT * KV_LENGTH_ALIGNMENT
        )

        num_seqs = common_attn_metadata.num_reqs
        block_size = self.block_size
        attention_mask_tiles: list[list[torch.Tensor]] = []
        active_block_indices: list[list[int]] | None = None

        if self.sliding_window is None:
            # No sliding window: build the full additive mask and split it into
            # per-block tiles (one tile per absolute block index).
            mask_cpu = self._build_attention_mask(
                seq_lens,
                query_start_loc,
                apply_causal_mask,
                max_query_len,
                aligned_max_query_len,
                aligned_max_seq_len,
                torch.device("cpu"),
            )
            # Pre-tile the mask: split into per-block tiles.
            # Query dimension is uniform (aligned_max_query_len) for all sequences,
            # so tiling only follows the KV dimension.
            for s in range(num_seqs):
                seq_tiles: list[torch.Tensor] = []
                kv_len_s = int(seq_lens[s].item())
                num_blocks_s = (kv_len_s + block_size - 1) // block_size
                for b in range(num_blocks_s):
                    col_start = b * block_size
                    col_end = col_start + block_size
                    tile = mask_cpu[s, :aligned_max_query_len, col_start:col_end]
                    seq_tiles.append(tile.contiguous())
                attention_mask_tiles.append(seq_tiles)
            # active_block_indices stays None, so forward iterates all blocks.
        else:
            # Sliding window: arithmetic block-skip. Blocks entirely outside
            # every query's window are dropped; interior blocks share a
            # zero mask tile; only boundary blocks get real per-query cutoffs.
            active_block_indices = []
            query_lens_list = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
            seq_lens_list = seq_lens.tolist()

            for s in range(num_seqs):
                kv_len_s = int(seq_lens_list[s])
                query_len_s = int(query_lens_list[s])
                context_len_s = kv_len_s - query_len_s

                active_bs, tiles = self._build_active_tiles_with_skip(
                    kv_len_s,
                    query_len_s,
                    context_len_s,
                    aligned_max_query_len,
                    apply_causal_mask,
                )
                active_block_indices.append(active_bs)
                attention_mask_tiles.append(tiles)

        # Physical page of each active block, per sequence.
        page_indices = []
        for s, tiles in enumerate(attention_mask_tiles):
            n = len(tiles)
            blocks_s = slice(n) if active_block_indices is None else active_block_indices[s]
            page_indices.append(block_table[s, blocks_s].tolist())

        return SpyreAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_seqs=common_attn_metadata.num_reqs,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_table=block_table,
            block_size=self.block_size,
            slot_mapping=slot_mapping,
            apply_causal_mask=apply_causal_mask,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            attention_mask_tiles=attention_mask_tiles,
            active_block_indices=active_block_indices,
            page_indices=page_indices,
            aligned_max_query_len=aligned_max_query_len,
            aligned_max_seq_len=aligned_max_seq_len,
        )


class SpyreAttentionBackend(AttentionBackend):
    """Paged KV-cache attention backend for Spyre."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Spyre stick size is 128 bytes; tensors are transferred as float16 (2 bytes),
        # so block_size must be a multiple of 64 (= 128 / 2) to satisfy stick alignment.
        # This matches the constraint on head_size in supports_head_size().
        return [MultipleOf(64)]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["SpyreAttentionImpl"]:
        return SpyreAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["SpyreAttentionMetadataBuilder"]:
        return SpyreAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # K and V are separate tensors in SpyrePagedKVCache, each with the same
        # shape. The base vLLM API expects a single tuple here; callers like
        # get_kv_cache_block_dim and KV-transfer code index into it directly.
        return (num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Spyre stick size is 128 bytes; tensors are transferred as float16 (2 bytes),
        # so head_size must be a multiple of 64 (= 128 / 2) to satisfy stick alignment.
        return head_size % 64 == 0

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes


class SpyreAttentionImpl(AttentionImpl[SpyreAttentionMetadata]):
    """Online-softmax paged attention iterating over KV pages.

    KV cache is a tuple (k_pages, v_pages) where each is one dense tensor of
    shape [num_blocks, block_size, num_kv_heads, head_size] on Spyre. Pages are
    read by indirect access, indexing the dense tensor with a device-resident
    page index. No gather masks.

    On Spyre, the per-page attention loop and reshape_and_cache are compiled
    via torch.compile with fixed iteration counts. A dict
    caches compiled variants per unique loop length.
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
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        # ALiBi slopes: per-head linear-bias coefficients (BLOOM/MPT style).
        # Reshape once to [num_kv_heads, num_queries_per_kv, 1, 1] so the
        # per-block bias construction in _online_softmax_attention broadcasts
        # cleanly against the score-tile shape.
        if alibi_slopes is not None:
            slopes_t = torch.tensor(alibi_slopes, dtype=torch.float16)
            if slopes_t.numel() != num_heads:
                raise ValueError(
                    f"alibi_slopes must have length num_heads={num_heads}, got {slopes_t.numel()}"
                )
            self.alibi_slopes: torch.Tensor | None = slopes_t.view(
                num_kv_heads, self.num_queries_per_kv, 1, 1
            )
        else:
            self.alibi_slopes = None

        # Normalise the API's Optional[float] into a plain float so the kernel
        # can bake it as a closure constant. logits_soft_cap == 0.0 disables
        # soft-capping (kernel takes the same path as upstream).
        self.logits_soft_cap: float = 0.0 if logits_soft_cap is None else float(logits_soft_cap)

        # Always compiled: eager index_copy_ rejects an int32 index and falls
        # back to CPU with an int64 one.
        self._reshape_fn = torch.compile(_reshape_and_cache_kernel, dynamic=False)

        # Compiled attention loops, keyed by (num_blocks, padded_query_len, fused_store)
        self._attn_fns: dict[tuple[int, int, bool], object] = {}

        # Same loops for the traced path, left uncompiled: they are traced into
        # the block's graph, and a nested torch.compile there buys nothing.
        self._traced_attn_fns: dict[tuple[int, int], object] = {}

        logger.debug_once(
            "Using SpyreAttentionBackend with a dense paged KV cache and indirect page gather"
        )

    def _get_attn_fn(self, num_blocks: int, padded_query_len: int, fused_store: bool = False):
        # self.alibi_slopes and self.logits_soft_cap are fixed per instance, so
        # has_alibi and logits_soft_cap don't need to be part of the cache key.
        key = (num_blocks, padded_query_len, fused_store)
        if key not in self._attn_fns:
            self._attn_fns[key] = _maybe_compile(
                _create_compilable_page_attn(
                    num_blocks,
                    padded_query_len,
                    self.num_heads,
                    self.head_size,
                    has_alibi=self.alibi_slopes is not None,
                    logits_soft_cap=self.logits_soft_cap,
                    fused_store=fused_store,
                )
            )
        return self._attn_fns[key]

    def traced_attn_fn(self, num_blocks: int, padded_query_len: int, num_fresh: int):
        key = (num_blocks, padded_query_len, num_fresh)
        fn = self._traced_attn_fns.get(key)
        if fn is None:
            fn = _create_compilable_page_attn(
                num_blocks,
                padded_query_len,
                self.num_heads,
                self.head_size,
                logits_soft_cap=self.logits_soft_cap,
                num_fresh=num_fresh,
            )
            self._traced_attn_fns[key] = fn
        return fn

    # `kv_cache` widens the base's `torch.Tensor` to `SpyrePagedKVCache`,
    # which `TorchSpyreModelRunner.initialize_kv_cache_tensors` allocates
    # and `bind_kv_cache` smuggles through a dict typed `dict[str, Tensor]`.
    # The matching pair of overrides preserves the runtime contract; ty
    # cannot see the co-evolution.
    @_record_function("spyre_attn::forward")
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: SpyrePagedKVCache,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output

        k_pages, v_pages = kv_cache
        _target_device = k_pages.device
        num_actual_tokens = attn_metadata.num_actual_tokens

        # Only the first layer of a step pays for the device mirror.
        if attn_metadata.slot_row_tables is None:
            attn_metadata.slot_row_tables = build_slot_row_tables(
                attn_metadata, _target_device
            )
        if attn_metadata.slot_mapping_device is None:
            attn_metadata.slot_mapping_device = convert(
                attn_metadata.slot_mapping[:num_actual_tokens], device=_target_device
            )
        if attn_metadata.attention_mask_tiles_device is None:
            tiles_cpu = attn_metadata.attention_mask_tiles
            assert tiles_cpu is not None, (
                "attention_mask_tiles must be precomputed by the metadata builder"
            )
            attn_metadata.attention_mask_tiles_device = [
                [convert(t, device=_target_device) for t in seq_tiles] for seq_tiles in tiles_cpu
            ]

        # Step 1: Reshape and cache — scatter new tokens into their slots, unless the
        # traced scatter in Attention.forward already folded it into the graph that
        # produced K/V (see install_inline_kv_scatter).
        if not attn_metadata.inline_kv_scatter:
            self._reshape_and_cache(
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                k_pages,
                v_pages,
                attn_metadata.slot_mapping_device,
            )

        # Step 2: Online softmax attention over pages (varlen). The kernel reads
        # the flat slot-major view, the same tensor the scatter writes.
        slots = (-1, k_pages.shape[2], k_pages.shape[3])
        output = self._online_softmax_attention(
            query[:num_actual_tokens],
            k_pages.view(slots),
            v_pages.view(slots),
            attn_metadata,
            output,
            _target_device,
        )

        return output

    @_record_function("spyre_attn::reshape_and_cache")
    def _reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Scatter new K/V tokens into their cache slots.

        key, value: [num_tokens, num_kv_heads, head_size] on the pages' device,
            strided last-dim views of the fused QKV output
        k_pages, v_pages: [num_blocks, block_size, num_kv_heads, head_size]
        slot_mapping: [num_tokens] on the pages' device
        """
        # A source on the wrong device falls back to CPU silently, without raising.
        assert key.device.type == k_pages.device.type, (
            f"reshape_and_cache source is on {key.device.type}, pages on {k_pages.device.type}"
        )

        # Valid because a view keeps the slot-outermost device layout.
        slots = (-1, k_pages.shape[2], k_pages.shape[3])
        self._reshape_fn(key, value, k_pages.view(slots), v_pages.view(slots), slot_mapping)

    @_record_function("spyre_attn::online_softmax")
    def _online_softmax_attention(
        self,
        query_dev: torch.Tensor,
        k_slots: torch.Tensor,
        v_slots: torch.Tensor,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,
        _target_device: torch.device,
    ) -> torch.Tensor:
        """FlashAttention-style online softmax iterating over KV pages (varlen).

        Handles multiple sequences using query_start_loc for the varlen layout.
        k_slots/v_slots are the flat [num_blocks * block_size, num_kv_heads,
        head_size] slot-major view of the cache; each iteration gathers one
        page's rows with an int32 device index, then feeds it to bmm.

        Writes results directly into the caller's output buffer in-place.

        Query is assembled on device into the padded 4D tensor
        [num_kv_heads, num_queries_per_kv, aligned_max_query_len, head_size]
        the kernel expects.

        Args:
            query_dev: Query on the target device, [num_tokens, num_heads, D].
        """
        head_size = self.head_size
        num_kv_heads = self.num_kv_heads
        num_queries_per_kv = self.num_queries_per_kv
        block_size = attn_metadata.block_size

        num_seqs = attn_metadata.num_seqs
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        mask_tiles_all = attn_metadata.attention_mask_tiles_device
        active_block_indices_all = attn_metadata.active_block_indices
        aligned_max_query_len = attn_metadata.aligned_max_query_len
        slot_row_tables = attn_metadata.slot_row_tables
        # The kernel can store into `output` itself only when this step's single
        # sequence owns every row of it, so the store is a plain copy of matching
        # shape. A compiled kernel also reads its arguments from offset 0
        # (torch-spyre#3770), and vLLM hands out a fresh buffer per layer, so the
        # buffer is re-checked every call.
        fused_store_ok = (
            num_seqs == 1
            and aligned_max_query_len == attn_metadata.max_query_len
            and output.shape[0] == aligned_max_query_len
            and output.storage_offset() == 0
            and output.is_contiguous()
        )
        assert mask_tiles_all is not None, (
            "attention_mask_tiles_device must be mirrored by forward()"
        )
        assert slot_row_tables is not None, "slot_row_tables must be filled by forward()"

        for seq_idx in range(num_seqs):
            # Most-naive implementation: no parallelization
            # over sequences or GQA optimization
            q_start = int(query_start_loc[seq_idx].item())
            q_end = int(query_start_loc[seq_idx + 1].item())
            query_len = q_end - q_start
            kv_len = int(seq_lens[seq_idx].item())

            if query_len == 1:
                # Decode: the single real token goes at row 0 of the padded
                # buffer; the trailing padded rows are masked out downstream.
                q_dev = query_dev.unbind(dim=0)[q_start].reshape(
                    num_kv_heads, num_queries_per_kv, 1, head_size
                )
                if aligned_max_query_len > 1:
                    q_dev = torch.nn.functional.pad(q_dev, (0, 0, 0, aligned_max_query_len - 1))
            else:
                q_seq = query_dev[q_start:q_end]

                # Pad query to global aligned_max_query_len (uniform for all seqs)
                if aligned_max_query_len > query_len:
                    q_seq = torch.nn.functional.pad(
                        q_seq,
                        (0, 0, 0, 0, 0, aligned_max_query_len - query_len),
                        mode="constant",
                        value=0.0,
                    )

                # Reshape: [padded_query_len, num_heads, head_size]
                #   → [num_kv_heads, num_queries_per_kv, padded_query_len, head_size]
                q = q_seq.unsqueeze(0).transpose(1, 2).contiguous()
                q_dev = q.reshape(
                    num_kv_heads, num_queries_per_kv, aligned_max_query_len, head_size
                )

            num_blocks_needed = (kv_len + block_size - 1) // block_size

            # Restrict to active (non-fully-masked) blocks when sliding window
            # is set. When active_block_indices_all is None (no sliding), all
            # blocks are active in their natural order.
            if active_block_indices_all is not None:
                active_bs = active_block_indices_all[seq_idx]
            else:
                active_bs = list(range(num_blocks_needed))

            if len(active_bs) == 0:
                # Every KV position is outside every query's window. Attention
                # over the empty set is undefined; write zeros.
                output[q_start:q_end] = 0.0
                continue

            rows_per_block = slot_row_tables[seq_idx]
            # mask_tiles_all[seq_idx] is indexed by position within active_bs.
            mask_tiles = mask_tiles_all[seq_idx][: len(active_bs)]

            # ALiBi bias tiles: slope[h] * (kv_pos - context_len), one per block.
            #
            # The full ALiBi form is slope[h] * (kv_pos - (context_len + q_rel)),
            # which varies over both query and KV positions. The (context_len + q_rel)
            # term is a per-query-row constant, and softmax is invariant under adding
            # any per-row constant to its input (numerator and denominator both pick
            # up the same exp() factor). We therefore drop it and keep only the
            # kv-dependent term — the softmax output is bit-identical to the full
            # form, and each tile stays 1D over KV (block_size floats per head)
            # instead of 2D (aligned_max_query_len * block_size).
            #
            # Matches vllm/v1/attention/ops/triton_attention_helpers.py::apply_alibi_to_score
            # (alibi_offset = seq_offset - context_len) — the production Triton path.
            #
            # Per-tile shape: [num_kv_heads, num_queries_per_kv, 1, block_size].
            alibi_bias_tiles: list[torch.Tensor] | None = None
            if self.alibi_slopes is not None:
                context_len = kv_len - query_len
                alibi_bias_tiles = []
                for b in active_bs:
                    kv_pos = torch.arange(
                        b * block_size,
                        (b + 1) * block_size,
                        dtype=torch.float16,
                    )
                    rel = (kv_pos - context_len).view(1, 1, 1, block_size)
                    bias = self.alibi_slopes * rel
                    alibi_bias_tiles.append(convert(bias, device=_target_device))

            # Run attention on target device. When the kernel can scatter straight
            # into `output`, the store joins its jobplan instead of costing a
            # separate eager dispatch and launch per layer.
            attn_fn = self._get_attn_fn(
                len(active_bs), aligned_max_query_len, fused_store=fused_store_ok
            )
            result = attn_fn(
                q_dev,
                k_slots,
                v_slots,
                rows_per_block,
                mask_tiles,
                self.scale,
                alibi_bias_tiles=alibi_bias_tiles,
                out=output if fused_store_ok else None,
            )

            assert result.dtype == output.dtype
            if fused_store_ok:
                continue
            if query_len < aligned_max_query_len:
                # Writing a prefix view copies its whole extent and overruns the
                # destination (torch-spyre#3826), so copy it first.
                output[q_start:q_end] = result[:query_len].clone()
            else:
                output[q_start:q_end] = result

        return output
