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

"""Batched multi-sequence decode over a head-major KV cache.

The reduction is ``batched_decode``'s, chunk for chunk; only the page read differs.
Gathering ``(page, kv_head)`` rows out of the folded cache lands the page already
head-major and keeps the gather's split on an axis ``probs @ V`` carries.
"""

import torch


def batched_decode_head_major_kernel(
    query,
    rep_row_ids,
    k_pages,
    v_pages,
    chunk_row_ids,
    mask_by_chunk,
    scale,
    num_seqs,
    blocks_per_chunk,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
    logits_soft_cap=0.0,
    out=None,
):
    """Shapes as in ``batched_decode_kernel``, except:

    k/v_pages: [num_pages_total * num_kv_heads, block_size, head_size]. chunk_row_ids: one
    [entries * num_kv_heads, 1] int32 tensor per chunk of ``page * num_kv_heads + kv``
    rows, entry-major and kv-minor, the order ``mask_by_chunk`` is broadcast in. One
    tensor per chunk, not slices of one: a slice's storage offset is dropped
    (torch-spyre#3770).
    """
    num_heads = num_kv_heads * num_queries_per_kv
    entries = num_seqs * blocks_per_chunk
    q = query.index_select(0, rep_row_ids).reshape(
        entries, num_kv_heads, num_queries_per_kv, head_size
    )

    tile_max = None
    tile_sum = None
    tile_output = None

    for c, rows in enumerate(chunk_row_ids):
        # Subscripting, not index_select: behind a 1-D index the entry axis splits only in
        # whole 32-entry sticks. Costs the eager path, which the preconditions decline.
        k_page = k_pages[rows].reshape(entries, num_kv_heads, block_size, head_size)
        v_page = v_pages[rows].reshape(entries, num_kv_heads, block_size, head_size)
        # Builder already broadcast across KV heads; split them back out.
        mask_tile = mask_by_chunk[c].reshape(entries, num_kv_heads, 1, block_size)

        scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Before the mask add: tanh(-inf/cap)*cap is -cap, not -inf, so
            # capping after it would un-mask the padded lanes.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        scores = scores + mask_tile
        # Leading-axis split only: torch-spyre rejects merging a permuted axis pair.
        sc = scores.reshape(
            num_seqs, blocks_per_chunk, num_kv_heads, num_queries_per_kv, block_size
        )
        chunk_max = torch.amax(torch.amax(sc, dim=-1, keepdim=True), dim=1, keepdim=True)

        # The running max drives exp(), not the chunk's own: a chunk wholly past a
        # sequence's length is -inf throughout and exp(-inf - -inf) is NaN.
        if c == 0:
            new_max = chunk_max
        else:
            assert tile_max is not None
            new_max = torch.maximum(tile_max, chunk_max)
        probs = torch.exp(sc - new_max)
        # The chunk's slots share one max, so summing them needs no rescale.
        chunk_sum = torch.sum(torch.sum(probs, dim=-1, keepdim=True), dim=1, keepdim=True)
        chunk_out = torch.sum(
            torch.matmul(
                probs.reshape(entries, num_kv_heads, num_queries_per_kv, block_size),
                v_page,
            ).reshape(num_seqs, blocks_per_chunk, num_kv_heads, num_queries_per_kv, head_size),
            dim=1,
            keepdim=True,
        )

        if c == 0:
            tile_max = new_max
            tile_sum = chunk_sum
            tile_output = chunk_out
        else:
            assert tile_max is not None
            assert tile_sum is not None
            assert tile_output is not None
            rescale = torch.exp(tile_max - new_max)
            tile_output = tile_output * rescale + chunk_out
            tile_sum = tile_sum * rescale + chunk_sum
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
    attn = (tile_output / tile_sum).reshape(num_seqs, num_heads, head_size)
    if out is not None:
        # Offset 0, so torch-spyre#3770 does not apply; rows past the batch are
        # don't-care and kept finite by the builder.
        out[:num_seqs].copy_(attn)
        return out
    return attn
