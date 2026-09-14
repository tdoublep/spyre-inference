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

"""Paged attention passing the GQA broadcast straight to ``spyre.batched_matmul``.

Identical to ``page_attn_kernel`` except for the two matmuls. Needs
torch-spyre#4277: before it, ``lower_bmm`` only accepted equal batch dims and a
size-1 group axis would be mis-indexed.

``page_attn_kernel`` writes the same broadcast as ``torch.matmul``, which the aten
decomposition lowers to ``expand`` + ``reshape``; a reshape across a stride-0 axis
is not a view, so aten inserts a ``clone`` — a G-fold copy of every K and V page.
torch-spyre's ``bmm_unflatten_pass`` recovers an N-D matmul but hands it the
clone, and ``lower_bmm`` realizes its operands. Calling the op directly keeps the
size-1 axis intact all the way to the lowering, which (with #4277) indexes it with
a constant 0 instead of materializing it.

Unlike the row-packed arm, the query keeps its ``[KV, G, q, D]`` shape. That is a
*split* of the head axis, which is always a view, so there is no query clone at
q>1 and the mask/ALiBi tiles broadcast unchanged. This arm is therefore the one
that works for chunked prefill, not just decode.
"""

import torch


def page_attn_native_bcast_kernel(
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
):
    """Signature, semantics and return value match ``page_attn_kernel``."""
    num_queries_per_kv = num_heads // num_kv_heads
    # spyre::batched_matmul is registered for the spyre device only, so CPU (the
    # reference path and the eager tests) keeps torch.matmul. Same arithmetic;
    # only the device path is what the clone claim is about.
    bmm = (
        torch.ops.spyre.batched_matmul
        if query.device.type == "spyre"
        else torch.matmul
    )

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
        k_page = k_pages.index_select(0, page_idx)
        v_page = v_pages.index_select(0, page_idx)
        # [1, blk, KV, D] -> [KV, 1, D, blk] / [KV, 1, blk, D]. The size-1 axis is
        # the GQA group slot, broadcast against the query's G by the lowering.
        k_page_4d = k_page.squeeze(0).permute(1, 2, 0).unsqueeze(1)
        v_page_4d = v_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)

        mask_tile = mask_tiles[i]

        scores = bmm(q, k_page_4d) * scale
        if logits_soft_cap > 0.0:
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        if alibi_bias_tiles is not None:
            scores = scores + alibi_bias_tiles[i]
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = bmm(tile_probs, v_page_4d)
            tile_sum = tile_probs.sum(dim=-1, keepdim=True)
        else:
            assert tile_max is not None
            assert tile_sum is not None
            assert tile_output is not None
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_output = tile_output * rescale
            tile_sum = tile_sum * rescale
            tile_probs = torch.exp(scores - new_max)
            tile_output += bmm(tile_probs, v_page_4d)
            tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
    attn = tile_output / tile_sum
    attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
    attn = attn.reshape(padded_query_len, num_heads, head_size)
    if out is not None:
        out.index_copy_(0, query_row_index[:padded_query_len], attn[:padded_query_len])
        return out
    return attn
