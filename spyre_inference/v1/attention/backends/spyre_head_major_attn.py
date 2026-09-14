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

"""Head-major variant of the Spyre paged attention backend, behind SPYRE_KV_CACHE_LAYOUT.

A page is stored ``[num_kv_heads, block_size, head_size]`` rather than
``[block_size, num_kv_heads, head_size]``, so a (block, kv head) pair is one contiguous
tile. The per-sequence kernel then gathers a page that is already head-major, dropping the
permute the token-major gather pays, and the batched decode kernel can fold (seqs, kv
heads) into a single bmm batch axis -- which is a different kernel rather than a different
gather, so it brings its own index tables. The cost is the KV store: a token's heads are
now block_size rows apart, so no view keeps them adjacent and the scatter unrolls over
heads.

Everything not overridden here is inherited; the base classes are layout-neutral.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from vllm.v1.attention.backend import AttentionLayer

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadata,
    SpyreAttentionMetadataBuilder,
    SpyrePagedKVCache,
    _call_kernel,
)
from spyre_inference.v1.attention.ops.batched_decode import (
    MAX_HEAD_MAJOR_DECODE_LANES,
    batched_decode_head_major_kernel,
    head_major_decode_seq_bucket,
)
from spyre_inference.v1.attention.ops.layout import head_major_kv_layout, stick_aligned_len
from spyre_inference.v1.attention.ops.page_attn import gather_page_head_major
from spyre_inference.v1.attention.ops.reshape_and_cache import (
    reshape_and_cache_head_major_kernel,
)

_batched_decode_head_major_compiled = torch.compile(batched_decode_head_major_kernel, dynamic=False)


@dataclass
class SpyreHeadMajorAttentionMetadata(SpyreAttentionMetadata):
    """Adds the head-major decode kernel's tables, in its lane order.

    lane = seq * num_kv_heads + kv head, the order the query reshape, the tile ids and the
    mask all share. The inherited chunked fields stay None.
    """

    query_row_ids_cpu: torch.Tensor | None = None  # [B_seqs] int32
    query_row_ids_dev: torch.Tensor | None = None
    tile_ids_cpu: torch.Tensor | None = None  # [B_blocks, stick-padded B_seqs * KV] int32
    tile_ids_dev: torch.Tensor | None = None
    mask_by_block_cpu: torch.Tensor | None = None  # [B_blocks, B_seqs * KV, 1, block] fp16
    mask_by_block_dev: torch.Tensor | None = None

    def mirror_decode_tables(self, device: torch.device) -> None:
        if self.query_row_ids_dev is not None:
            return
        assert self.query_row_ids_cpu is not None
        assert self.tile_ids_cpu is not None
        assert self.mask_by_block_cpu is not None
        self.query_row_ids_dev = convert(self.query_row_ids_cpu, device=device)
        self.tile_ids_dev = convert(self.tile_ids_cpu, device=device)
        self.mask_by_block_dev = convert(self.mask_by_block_cpu, device=device)


class SpyreHeadMajorAttentionMetadataBuilder(SpyreAttentionMetadataBuilder):
    _metadata_cls: ClassVar[type[SpyreAttentionMetadata]] = SpyreHeadMajorAttentionMetadata

    def _decode_seq_bucket(self, num_decode_seqs: int) -> int | None:
        # Sized by the lane rule rather than a sequence-bucket ladder, so one compiled
        # kernel serves every batch size it accepts.
        return head_major_decode_seq_bucket(num_decode_seqs, self.num_kv_heads)

    def _decode_block_extent(self, b_seqs: int, b_blocks: int) -> tuple[int, int | None]:
        # The lane axis already fills the cores, so the block axis stays one block per
        # step and needs no chunk round-up.
        return b_blocks, None

    def _decode_kernel_tables(
        self,
        block_ids: torch.Tensor,
        mask_bs_bb: torch.Tensor,
        query_row_ids: torch.Tensor,
        num_decode_seqs: int,
        blocks_per_chunk: int | None,
        block_size: int,
    ) -> dict:
        b_blocks, b_seqs = block_ids.shape
        num_kv_heads = self.num_kv_heads
        heads = torch.arange(num_kv_heads, dtype=torch.int32)
        # Rows stick-padded: a narrower inner dim emits a Mod(d0, ...) stick expression
        # the inductor rejects. Padded columns hold page 0, inert under an all--inf mask.
        tile_ids = torch.zeros(
            b_blocks, stick_aligned_len(b_seqs * num_kv_heads), dtype=torch.int32
        )
        tile_ids[:, : b_seqs * num_kv_heads] = (
            block_ids[:, :, None] * num_kv_heads + heads
        ).reshape(b_blocks, -1)
        return {
            "query_row_ids_cpu": query_row_ids,
            "tile_ids_cpu": tile_ids,
            # 4-D, not 5-D: the kernel slices dim 0 per block, and a dim-0 slice of a
            # 5-D base fails torch-spyre layout propagation.
            "mask_by_block_cpu": (
                mask_bs_bb.permute(1, 0, 2)
                .unsqueeze(2)
                .expand(b_blocks, b_seqs, num_kv_heads, block_size)
                .reshape(b_blocks, b_seqs * num_kv_heads, 1, block_size)
                .contiguous()
            ),
        }


class SpyreHeadMajorAttentionImpl(SpyreAttentionImpl):
    _gather_page: ClassVar = staticmethod(gather_page_head_major)
    _block_axis: ClassVar[int] = 2

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reshape_fn = torch.compile(reshape_and_cache_head_major_kernel, dynamic=False)
        self._decode_fn = _batched_decode_head_major_compiled
        self._kv_tiles: SpyrePagedKVCache | None = None

    def kv_slot_views(self, kv_cache: SpyrePagedKVCache) -> SpyrePagedKVCache:
        # One row per (page, kv head, token). KV heads cannot stay a trailing axis: under
        # the head-major page order a token's heads are block_size rows apart, so no view
        # keeps them adjacent. head_size is a multiple of the stick, so collapsing the
        # leading dims is a pure re-view of the same bytes.
        if self._kv_slots is None:
            k_pages, v_pages = kv_cache
            shape = (-1, k_pages.shape[3])
            self._kv_slots = SpyrePagedKVCache(k_pages.view(shape), v_pages.view(shape))
        return self._kv_slots

    def kv_tile_views(self, kv_cache: SpyrePagedKVCache) -> SpyrePagedKVCache:
        """One row per (page, kv head), which is what makes the fold a single bmm axis."""
        if self._kv_tiles is None:
            k_pages, v_pages = kv_cache
            shape = (-1, k_pages.shape[2], k_pages.shape[3])
            self._kv_tiles = SpyrePagedKVCache(k_pages.view(shape), v_pages.view(shape))
        return self._kv_tiles

    def slot_rows(
        self, slot_mapping: torch.Tensor, kv_cache: SpyrePagedKVCache, device: torch.device
    ) -> list[torch.Tensor]:
        # One tensor per head rather than a 2-D table: an index tensor reaches the
        # hardware as a tensor argument, so a row slice would have its offset dropped
        # (torch-spyre#3770).
        block_size = kv_cache[0].shape[self._block_axis]
        pages = torch.div(slot_mapping, block_size, rounding_mode="floor")
        offsets = slot_mapping - pages * block_size
        return [
            convert((pages * self.num_kv_heads + h) * block_size + offsets, device=device)
            for h in range(self.num_kv_heads)
        ]

    def do_kv_cache_update(
        self,
        layer: AttentionLayer | None,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: SpyrePagedKVCache,
        slot_mapping: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        assert key.device.type == kv_cache[0].device.type, (
            f"kv cache update source is on {key.device.type}, pages on {kv_cache[0].device.type}"
        )
        k_slots, v_slots = self.kv_slot_views(kv_cache)
        # Cloned out here, not in the kernel: K and V are strided views into the fused QKV
        # projection, and the store's per-head slices of such a view read the wrong rows.
        # clone(), not contiguous(): at one token the view already reports contiguous.
        self._reshape_fn(key.clone(), value.clone(), k_slots, v_slots, slot_mapping)
        return k_slots

    def _run_batched_decode_dispatch(
        self,
        query_dev: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        assert isinstance(attn_metadata, SpyreHeadMajorAttentionMetadata)
        b_seqs = attn_metadata.padded_num_seqs
        assert b_seqs is not None
        assert attn_metadata.query_row_ids_dev is not None
        assert attn_metadata.tile_ids_dev is not None
        assert attn_metadata.mask_by_block_dev is not None
        # head_major_decode_seq_bucket refuses above this, so reaching it would be a wrong
        # answer rather than a slow one.
        assert b_seqs * self.num_kv_heads <= MAX_HEAD_MAJOR_DECODE_LANES, (
            f"batched decode with {b_seqs * self.num_kv_heads} lanes exceeds the "
            f"{MAX_HEAD_MAJOR_DECODE_LANES} torch-spyre returns correct values for"
        )

        # Gathers (page, kv head) tiles, so this kernel reads the tile views.
        k_tiles, v_tiles = self.kv_tile_views(SpyrePagedKVCache(k_pages, v_pages))
        store_out = (
            query_dev.shape[0] >= b_seqs
            and output.shape[0] >= b_seqs
            and output.dtype == query_dev.dtype
            and output.storage_offset() == 0
            and output.is_contiguous()
        )
        result = _call_kernel(
            "batched decode attention",
            self._decode_fn,
            query_dev,
            attn_metadata.query_row_ids_dev,
            k_tiles,
            v_tiles,
            attn_metadata.tile_ids_dev,
            attn_metadata.mask_by_block_dev,
            self.scale,
            b_seqs,
            attn_metadata.padded_batch_blocks,
            self.num_kv_heads,
            self.num_queries_per_kv,
            attn_metadata.block_size,
            self.head_size,
            self.logits_soft_cap,
            output if store_out else None,
        )
        if store_out:
            return
        num_decode_seqs = attn_metadata.num_decode_seqs
        src = result.reshape(b_seqs, self.num_heads, self.head_size)[:num_decode_seqs].clone()
        output[:num_decode_seqs].copy_(src)


class SpyreHeadMajorAttentionBackend(SpyreAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[SpyreHeadMajorAttentionImpl]:
        return SpyreHeadMajorAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[SpyreHeadMajorAttentionMetadataBuilder]:
        return SpyreHeadMajorAttentionMetadataBuilder

    @classmethod
    def get_kv_cache_allocation_shape(
        cls,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, num_kv_heads, block_size, head_size)

    @classmethod
    def get_kv_cache_device_layout(
        cls,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ):
        # 4-D allocation over a device layout that keeps (page, kv head) as one extent, so
        # the per-sequence kernel gathers one contiguous page while the flatter views the
        # store and the decode kernel take stay merges -- merging device dims is the
        # direction torch-spyre lowers, splitting device dim 0 is not.
        return head_major_kv_layout(num_blocks * num_kv_heads, block_size, head_size, dtype)
