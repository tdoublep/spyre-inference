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

"""Per-sequence paged attention over the KV cache."""

import torch


def alibi_tile_shape(
    num_kv_heads: int, num_queries_per_kv: int, block_size: int
) -> tuple[int, int, int, int]:
    """Shape of one per-block ALiBi bias tile; derived at the construction site in
    ``SpyreAttentionImpl._online_softmax_attention``."""
    return (num_kv_heads, num_queries_per_kv, 1, block_size)


def gather_page_token_major(pages, page_idx):
    """One page of a token-major cache as [num_kv_heads, 1, block_size, head_size].

    index_select, not ``pages[page_idx]``: subscripting lowers to aten.index, which
    upcasts the int32 index to int64 and fails eager. The permute is what turns the
    token-major page head-major for the matmuls; it runs on device.
    """
    return pages.index_select(0, page_idx).squeeze(0).permute(1, 0, 2).unsqueeze(1)


def gather_page_head_major(pages, page_idx):
    """The same tile out of a head-major cache, whose pages already land head-major.

    One entry, hence one contiguous page: num_kv_heads scattered entries over a
    flattened cache move the same bytes at a higher cost. squeeze/unsqueeze rather than
    a permute, so this is strictly fewer ops than the token-major gather.
    """
    return pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)


def page_attn_kernel(
    query,
    query_row_index,
    k_pages,
    v_pages,
    page_index_table,
    mask_tiles,
    scale,
    num_blocks,
    padded_query_len,
    num_heads,
    num_kv_heads,
    head_size,
    logits_soft_cap=0.0,
    alibi_bias_tiles=None,
    out=None,
    *,
    gather_page=gather_page_token_major,
):
    """Online softmax attention over ``num_blocks`` KV pages.

    Under `dynamic=False` Dynamo specializes on every non-tensor argument, so the
    page loop is unrolled per variant.

    Expected shapes:
        query: [num_tokens, num_heads, head_size], the whole batch's query
        query_row_index: int32 device tensor whose first padded_query_len
            entries are this sequence's absolute query rows.
        k_pages: the KV cache in whichever form ``gather_page`` reads, so
            [num_blocks_total, block_size, num_kv_heads, head_size] token-major
            or [num_blocks_total, num_kv_heads, block_size, head_size] head-major.
        v_pages: as k_pages.
        page_index_table: [num_blocks, INT32_ELEMS_PER_STICK] int32 device
            tensor, row i holding the i-th active block's page index at
            column 0.
        mask_tiles: [num_blocks]
        alibi_bias_tiles: list of [num_kv_heads, num_queries_per_kv, 1, block_size],
            or None for no ALiBi. The query-axis dim is 1 because softmax absorbs
            per-query-row constants — see the derivation at the bias-tile
            construction site in _online_softmax_attention.
        out: buffer to store into, or None to return the result instead.
        gather_page: the backend's page gather, specialized on by Dynamo so each cache
            layout gets its own compiled variant of this kernel. Keyword-only: it sits
            among values callers pass positionally, and defaulting it keeps a
            token-major call site from having to name it.

    Returns [padded_query_len, num_heads, head_size], or ``out`` when this
    kernel stored the result itself.
    """
    num_queries_per_kv = num_heads // num_kv_heads
    # A compiled region reads a view from offset 0, ignoring storage_offset
    # (torch-spyre#3770), so the rows are gathered here rather than sliced outside.
    q_rows = query.index_select(0, query_row_index[:padded_query_len])
    q = (
        q_rows.unsqueeze(0)
        .transpose(1, 2)
        .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
    )

    tile_max = None
    tile_sum = None
    tile_output = None

    for i in range(num_blocks):
        page_idx = page_index_table[i, 0:1]
        k_page_4d = gather_page(k_pages, page_idx)
        v_page_4d = gather_page(v_pages, page_idx)

        mask_tile = mask_tiles[i]

        scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Pull logits into (-cap, +cap) before the mask add so masked
            # positions still map cleanly to -inf. Applied before the ALiBi
            # bias so the positional term is not squashed by the tanh.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        if alibi_bias_tiles is not None:
            # ALiBi bias slope[h] * (kv_pos - context_len). The additive
            # mask_tile below uses finfo.min for masked positions, so this
            # bias cannot un-mask them.
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
    if out is not None:
        # `out` and `query` are both indexed by absolute token row. Storing the
        # full padded extent keeps the sequence's real query_len out of the
        # arguments, so it is not specialized on; rows past it duplicate the
        # sequence's last row, so index_copy_'s undefined write order for
        # duplicate indices is harmless.
        out.index_copy_(0, query_row_index[:padded_query_len], attn[:padded_query_len])
        return out
    return attn
