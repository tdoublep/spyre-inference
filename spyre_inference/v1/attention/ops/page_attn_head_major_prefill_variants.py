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

"""Experimental prefill kernels for the head-major KV layout.

Selected by ``SPYRE_ATTN_PREFILL_VARIANT``; ``batched`` is PR 915's shipped kernel.
Each variant keeps 915's signature so the backend can dispatch between them.
"""

import torch


def _fold_query(query, query_row_index, padded_query_len, num_kv_heads, num_queries_per_kv,
                head_size):
    """Query as ``[kv_head, group * row, head_size]``.

    Heads are kv-major, so folding the group into the row axis leaves the page a single
    batch dim and the matmul a true 3-D bmm — nothing to broadcast, so nothing to clone
    (torch-spyre#4123). Row ``g * padded_query_len + t`` is group ``g``, query row ``t``.
    """
    q_rows = query.index_select(0, query_row_index[:padded_query_len])
    return (
        q_rows.unsqueeze(0)
        .transpose(1, 2)
        .reshape(num_kv_heads, num_queries_per_kv * padded_query_len, head_size)
    )


def _unfold_out(attn, padded_query_len, num_heads, head_size):
    return (
        attn.reshape(1, num_heads, padded_query_len, head_size)
        .transpose(1, 2)
        .reshape(padded_query_len, num_heads, head_size)
    )


def _store(out, attn, query_row_index, padded_query_len):
    if out is not None:
        out.index_copy_(0, query_row_index[:padded_query_len], attn[:padded_query_len])
        return out
    return attn


def _masked_scores(q, k_page, mask_tile, scale, num_kv_heads, num_queries_per_kv,
                   padded_query_len, kv_width, logits_soft_cap, prescaled, transpose_k=True):
    """QK over a folded query, with the mask repeated over the folded group axis."""
    scores = torch.matmul(q, k_page.transpose(-2, -1) if transpose_k else k_page)
    if not prescaled:
        scores = scores * scale
    if logits_soft_cap > 0.0:
        scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
    # The mask is group-independent, but a 4-D view of the folded scores has no
    # supported stick dim, so the tile is repeated over the row axis instead.
    if num_queries_per_kv > 1:
        mask_tile = torch.cat([mask_tile] * num_queries_per_kv, dim=0)
    return scores + mask_tile


def page_attn_head_major_prefill_fold_kernel(
    query, query_row_index, k_pages, v_pages, page_index_tables, mask_tiles, scale,
    num_blocks, padded_query_len, num_heads, num_kv_heads, head_size, block_size,
    logits_soft_cap=0.0, out=None, prescale=False,
):
    """915's online softmax with the query group folded into the row axis."""
    g = num_heads // num_kv_heads
    q = _fold_query(query, query_row_index, padded_query_len, num_kv_heads, g, head_size)
    if prescale:
        q = q * scale

    tile_max = None
    tile_sum = None
    tile_out = None
    for i in range(num_blocks):
        page_idx = page_index_tables[i]
        k_page = k_pages.index_select(0, page_idx).squeeze(0)
        v_page = v_pages.index_select(0, page_idx).squeeze(0)
        scores = _masked_scores(q, k_page, mask_tiles[i], scale, num_kv_heads, g,
                                padded_query_len, block_size, logits_soft_cap, prescale)
        scores_max = torch.amax(scores, dim=-1, keepdim=True)
        if i == 0:
            tile_max = scores_max
            probs = torch.exp(scores - tile_max)
            tile_out = torch.matmul(probs, v_page)
            tile_sum = probs.sum(dim=-1, keepdim=True)
        else:
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_out = tile_out * rescale
            tile_sum = tile_sum * rescale
            probs = torch.exp(scores - new_max)
            tile_out = tile_out + torch.matmul(probs, v_page)
            tile_sum = tile_sum + probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    attn = _unfold_out(tile_out / tile_sum, padded_query_len, num_heads, head_size)
    return _store(out, attn, query_row_index, padded_query_len)


def page_attn_head_major_prefill_hoist_kernel(
    query, query_row_index, k_pages, v_pages, page_index_tables, mask_tiles, scale,
    num_blocks, padded_query_len, num_heads, num_kv_heads, head_size, block_size,
    logits_soft_cap=0.0, out=None, prescale=False,
):
    """Folded, with the row max taken over every block before any exp.

    Drops the online form's per-block rescale of the accumulator, which is a read and a
    write of the whole ``[kv, group * row, head_size]`` output per block, at the cost of
    holding every block's scores live.
    """
    g = num_heads // num_kv_heads
    q = _fold_query(query, query_row_index, padded_query_len, num_kv_heads, g, head_size)
    if prescale:
        q = q * scale

    all_scores = []
    row_max = None
    for i in range(num_blocks):
        k_page = k_pages.index_select(0, page_index_tables[i]).squeeze(0)
        scores = _masked_scores(q, k_page, mask_tiles[i], scale, num_kv_heads, g,
                                padded_query_len, block_size, logits_soft_cap, prescale)
        all_scores.append(scores)
        block_max = torch.amax(scores, dim=-1, keepdim=True)
        row_max = block_max if row_max is None else torch.maximum(row_max, block_max)

    tile_out = None
    tile_sum = None
    for i in range(num_blocks):
        v_page = v_pages.index_select(0, page_index_tables[i]).squeeze(0)
        probs = torch.exp(all_scores[i] - row_max)
        pv = torch.matmul(probs, v_page)
        psum = probs.sum(dim=-1, keepdim=True)
        if i == 0:
            tile_out, tile_sum = pv, psum
        else:
            tile_out = tile_out + pv
            tile_sum = tile_sum + psum

    attn = _unfold_out(tile_out / tile_sum, padded_query_len, num_heads, head_size)
    return _store(out, attn, query_row_index, padded_query_len)


def page_attn_head_major_prefill_slab_kernel(
    query, query_row_index, k_pages, v_pages, page_index_tables, mask_tiles, scale,
    num_blocks, padded_query_len, num_heads, num_kv_heads, head_size, block_size,
    logits_soft_cap=0.0, out=None, prescale=False,
):
    """Folded, with every page concatenated into one slab and one softmax.

    Two matmuls and one softmax for the whole context: no accumulator, no rescale, no
    per-block partial writes. Pays a slab copy of K and V and holds a full
    ``[kv, group * row, context]`` score tensor.
    """
    g = num_heads // num_kv_heads
    q = _fold_query(query, query_row_index, padded_query_len, num_kv_heads, g, head_size)
    if prescale:
        q = q * scale

    kv_width = num_blocks * block_size
    k_slab = torch.cat(
        [k_pages.index_select(0, page_index_tables[i]).squeeze(0) for i in range(num_blocks)],
        dim=1,
    )
    v_slab = torch.cat(
        [v_pages.index_select(0, page_index_tables[i]).squeeze(0) for i in range(num_blocks)],
        dim=1,
    )
    mask = torch.cat([mask_tiles[i] for i in range(num_blocks)], dim=-1)

    scores = _masked_scores(q, k_slab, mask, scale, num_kv_heads, g, padded_query_len,
                            kv_width, logits_soft_cap, prescale)
    probs = torch.exp(scores - torch.amax(scores, dim=-1, keepdim=True))
    attn = torch.matmul(probs, v_slab) / probs.sum(dim=-1, keepdim=True)
    attn = _unfold_out(attn, padded_query_len, num_heads, head_size)
    return _store(out, attn, query_row_index, padded_query_len)


def page_attn_head_major_prefill_mmpair_kernel(
    query, query_row_index, k_pages, v_pages, page_index_tables, mask_tiles, scale,
    num_blocks, padded_query_len, num_heads, num_kv_heads, head_size, block_size,
    logits_soft_cap=0.0, out=None, prescale=False,
):
    """Diagnostic, numerically wrong: both matmuls per block, no softmax at all.

    Bounds how much of the kernel is the matmuls versus the softmax epilogue.
    """
    g = num_heads // num_kv_heads
    q = _fold_query(query, query_row_index, padded_query_len, num_kv_heads, g, head_size)
    acc = None
    for i in range(num_blocks):
        k_page = k_pages.index_select(0, page_index_tables[i]).squeeze(0)
        v_page = v_pages.index_select(0, page_index_tables[i]).squeeze(0)
        pv = torch.matmul(torch.matmul(q, k_page.transpose(-2, -1)), v_page)
        acc = pv if acc is None else acc + pv
    attn = _unfold_out(acc, padded_query_len, num_heads, head_size)
    return _store(out, attn, query_row_index, padded_query_len)


def page_attn_head_major_prefill_qkonly_kernel(
    query, query_row_index, k_pages, v_pages, page_index_tables, mask_tiles, scale,
    num_blocks, padded_query_len, num_heads, num_kv_heads, head_size, block_size,
    logits_soft_cap=0.0, out=None, prescale=False,
):
    """Diagnostic, numerically wrong: QK matmul and the mask add only."""
    g = num_heads // num_kv_heads
    q = _fold_query(query, query_row_index, padded_query_len, num_kv_heads, g, head_size)
    acc = None
    for i in range(num_blocks):
        k_page = k_pages.index_select(0, page_index_tables[i]).squeeze(0)
        scores = _masked_scores(q, k_page, mask_tiles[i], scale, num_kv_heads, g,
                                padded_query_len, block_size, logits_soft_cap, False)
        acc = scores if acc is None else acc + scores
    attn = _unfold_out(acc[:, :, :head_size], padded_query_len, num_heads, head_size)
    return _store(out, attn, query_row_index, padded_query_len)


def page_attn_head_major_prefill_nopv_kernel(
    query, query_row_index, k_pages, v_pages, page_index_tables, mask_tiles, scale,
    num_blocks, padded_query_len, num_heads, num_kv_heads, head_size, block_size,
    logits_soft_cap=0.0, out=None, prescale=False,
):
    """Diagnostic, numerically wrong: full online softmax with the PV matmul removed."""
    g = num_heads // num_kv_heads
    q = _fold_query(query, query_row_index, padded_query_len, num_kv_heads, g, head_size)
    tile_max = None
    tile_sum = None
    tile_out = None
    for i in range(num_blocks):
        k_page = k_pages.index_select(0, page_index_tables[i]).squeeze(0)
        scores = _masked_scores(q, k_page, mask_tiles[i], scale, num_kv_heads, g,
                                padded_query_len, block_size, logits_soft_cap, False)
        scores_max = torch.amax(scores, dim=-1, keepdim=True)
        if i == 0:
            tile_max = scores_max
            probs = torch.exp(scores - tile_max)
            tile_out = probs[:, :, :head_size]
            tile_sum = probs.sum(dim=-1, keepdim=True)
        else:
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_out = tile_out * rescale
            tile_sum = tile_sum * rescale
            probs = torch.exp(scores - new_max)
            tile_out = tile_out + probs[:, :, :head_size]
            tile_sum = tile_sum + probs.sum(dim=-1, keepdim=True)
            tile_max = new_max
    attn = _unfold_out(tile_out / tile_sum, padded_query_len, num_heads, head_size)
    return _store(out, attn, query_row_index, padded_query_len)


def page_attn_head_major_prefill_foldkv_kernel(
    query, query_row_index, k_pages, v_pages, kv_index_tables, mask_tiles, scale,
    num_blocks, padded_query_len, num_heads, num_kv_heads, head_size, block_size,
    logits_soft_cap=0.0, out=None, prescale=False,
):
    """Folded query, and the page gathered per kv head off the folded cache.

    915 gathers one page with a ``[1]`` index on the 4-D cache, whose entry axis — the
    only axis a gather can be split on — holds a single entry. This takes 893's
    ``[num_kv_heads, 1]`` index instead, so the split lands per kv head.
    """
    g = num_heads // num_kv_heads
    q = _fold_query(query, query_row_index, padded_query_len, num_kv_heads, g, head_size)
    if prescale:
        q = q * scale

    tile_max = None
    tile_sum = None
    tile_out = None
    for i in range(num_blocks):
        kv_rows = kv_index_tables[i]
        k_page = k_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)
        v_page = v_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)
        scores = torch.matmul(q, k_page.permute(0, 2, 1))
        if not prescale:
            scores = scores * scale
        if logits_soft_cap > 0.0:
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        mask_tile = mask_tiles[i]
        if g > 1:
            mask_tile = torch.cat([mask_tile] * g, dim=0)
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-1, keepdim=True)
        if i == 0:
            tile_max = scores_max
            probs = torch.exp(scores - tile_max)
            tile_out = torch.matmul(probs, v_page)
            tile_sum = probs.sum(dim=-1, keepdim=True)
        else:
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_out = tile_out * rescale
            tile_sum = tile_sum * rescale
            probs = torch.exp(scores - new_max)
            tile_out = tile_out + torch.matmul(probs, v_page)
            tile_sum = tile_sum + probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    attn = _unfold_out(tile_out / tile_sum, padded_query_len, num_heads, head_size)
    return _store(out, attn, query_row_index, padded_query_len)


def page_attn_head_major_prefill_batchedkv_kernel(
    query, query_row_index, k_pages, v_pages, kv_index_tables, mask_tiles, scale,
    num_blocks, padded_query_len, num_heads, num_kv_heads, head_size, block_size,
    logits_soft_cap=0.0, out=None, prescale=False,
):
    """915's batched GQA form, with 893's per-kv-head gather off the folded cache."""
    g = num_heads // num_kv_heads
    q_rows = query.index_select(0, query_row_index[:padded_query_len])
    q = (
        q_rows.unsqueeze(0)
        .transpose(1, 2)
        .reshape(num_kv_heads, g, padded_query_len, head_size)
    )
    if prescale:
        q = q * scale

    tile_max = None
    tile_sum = None
    tile_out = None
    for i in range(num_blocks):
        kv_rows = kv_index_tables[i]
        k_page = k_pages[kv_rows].reshape(num_kv_heads, 1, block_size, head_size)
        v_page = v_pages[kv_rows].reshape(num_kv_heads, 1, block_size, head_size)
        scores = torch.matmul(q, k_page.transpose(-2, -1))
        if not prescale:
            scores = scores * scale
        if logits_soft_cap > 0.0:
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        scores = scores + mask_tiles[i]
        scores_max = torch.amax(scores, dim=-1, keepdim=True)
        if i == 0:
            tile_max = scores_max
            probs = torch.exp(scores - tile_max)
            tile_out = torch.matmul(probs, v_page)
            tile_sum = probs.sum(dim=-1, keepdim=True)
        else:
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_out = tile_out * rescale
            tile_sum = tile_sum * rescale
            probs = torch.exp(scores - new_max)
            tile_out = tile_out + torch.matmul(probs, v_page)
            tile_sum = tile_sum + probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    attn = _unfold_out(tile_out / tile_sum, padded_query_len, num_heads, head_size)
    return _store(out, attn, query_row_index, padded_query_len)


def page_attn_head_major_prefill_batchedt_kernel(
    query, query_row_index, k_pages, v_pages, page_index_tables, mask_tiles, scale,
    num_blocks, padded_query_len, num_heads, num_kv_heads, head_size, block_size,
    logits_soft_cap=0.0, out=None, prescale=False,
):
    """915's batched GQA form with the score matrix transposed.

    Scores come out as ``[kv, group, kv_pos, query_row]``, so the softmax reduces over
    kv_pos, an outer axis, instead of over the innermost (stick) axis. The query is
    transposed once per layer and K is consumed in the layout the cache already stores.
    """
    g = num_heads // num_kv_heads
    q_rows = query.index_select(0, query_row_index[:padded_query_len])
    qt = (
        q_rows.unsqueeze(0)
        .permute(0, 2, 3, 1)
        .reshape(num_kv_heads, g, head_size, padded_query_len)
    )
    if prescale:
        qt = qt * scale

    tile_max = None
    tile_sum = None
    tile_out = None
    for i in range(num_blocks):
        page_idx = page_index_tables[i]
        k_page = k_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
        v_page = v_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
        scores = torch.matmul(k_page, qt)
        if not prescale:
            scores = scores * scale
        if logits_soft_cap > 0.0:
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        scores = scores + mask_tiles[i].transpose(0, 1)
        scores_max = torch.amax(scores, dim=-2, keepdim=True)
        vt = v_page.transpose(-2, -1)
        if i == 0:
            tile_max = scores_max
            probs = torch.exp(scores - tile_max)
            tile_out = torch.matmul(vt, probs)
            tile_sum = probs.sum(dim=-2, keepdim=True)
        else:
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_out = tile_out * rescale
            tile_sum = tile_sum * rescale
            probs = torch.exp(scores - new_max)
            tile_out = tile_out + torch.matmul(vt, probs)
            tile_sum = tile_sum + probs.sum(dim=-2, keepdim=True)
            tile_max = new_max

    attn = (tile_out / tile_sum).transpose(-2, -1)
    attn = _unfold_out(attn, padded_query_len, num_heads, head_size)
    return _store(out, attn, query_row_index, padded_query_len)


def page_attn_head_major_prefill_foldt_kernel(
    query, query_row_index, k_pages, v_pages, kv_index_tables, mask_tiles, scale,
    num_blocks, padded_query_len, num_heads, num_kv_heads, head_size, block_size,
    logits_soft_cap=0.0, out=None, prescale=False,
):
    """Transposed scores over a folded query and 893's per-kv-head page gather."""
    g = num_heads // num_kv_heads
    q_rows = query.index_select(0, query_row_index[:padded_query_len])
    qt = (
        q_rows.unsqueeze(0)
        .permute(0, 2, 3, 1)
        .reshape(num_kv_heads, g, head_size, padded_query_len)
        .permute(0, 2, 1, 3)
        .reshape(num_kv_heads, head_size, g * padded_query_len)
    )
    if prescale:
        qt = qt * scale

    tile_max = None
    tile_sum = None
    tile_out = None
    for i in range(num_blocks):
        kv_rows = kv_index_tables[i]
        k_page = k_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)
        v_page = v_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)
        scores = torch.matmul(k_page, qt)
        if not prescale:
            scores = scores * scale
        if logits_soft_cap > 0.0:
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        mask_tile = mask_tiles[i].transpose(0, 1)
        if g > 1:
            mask_tile = torch.cat([mask_tile] * g, dim=1)
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-2, keepdim=True)
        vt = v_page.permute(0, 2, 1)
        if i == 0:
            tile_max = scores_max
            probs = torch.exp(scores - tile_max)
            tile_out = torch.matmul(vt, probs)
            tile_sum = probs.sum(dim=-2, keepdim=True)
        else:
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_out = tile_out * rescale
            tile_sum = tile_sum * rescale
            probs = torch.exp(scores - new_max)
            tile_out = tile_out + torch.matmul(vt, probs)
            tile_sum = tile_sum + probs.sum(dim=-2, keepdim=True)
            tile_max = new_max

    attn = (tile_out / tile_sum).transpose(-2, -1)
    attn = _unfold_out(attn, padded_query_len, num_heads, head_size)
    return _store(out, attn, query_row_index, padded_query_len)


def page_attn_head_major_prefill_slabb_kernel(
    query, query_row_index, k_pages, v_pages, page_index_tables, mask_tiles, scale,
    num_blocks, padded_query_len, num_heads, num_kv_heads, head_size, block_size,
    logits_soft_cap=0.0, out=None, prescale=False,
):
    """915's batched GQA form over one concatenated slab of every page.

    One softmax for the whole context, so there is no accumulator: the online form's
    per-block read-modify-write of the ``[kv, group, row, head_size]`` output and its
    per-block partial write both disappear, at the cost of a full
    ``[kv, group, row, context]`` score tensor.
    """
    g = num_heads // num_kv_heads
    q_rows = query.index_select(0, query_row_index[:padded_query_len])
    q = (
        q_rows.unsqueeze(0)
        .transpose(1, 2)
        .reshape(num_kv_heads, g, padded_query_len, head_size)
    )
    if prescale:
        q = q * scale

    k_slab = torch.cat(
        [k_pages.index_select(0, page_index_tables[i]).squeeze(0) for i in range(num_blocks)],
        dim=1,
    ).unsqueeze(1)
    v_slab = torch.cat(
        [v_pages.index_select(0, page_index_tables[i]).squeeze(0) for i in range(num_blocks)],
        dim=1,
    ).unsqueeze(1)
    mask = torch.cat([mask_tiles[i] for i in range(num_blocks)], dim=-1)

    scores = torch.matmul(q, k_slab.transpose(-2, -1))
    if not prescale:
        scores = scores * scale
    if logits_soft_cap > 0.0:
        scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
    scores = scores + mask
    probs = torch.exp(scores - torch.amax(scores, dim=-1, keepdim=True))
    attn = torch.matmul(probs, v_slab) / probs.sum(dim=-1, keepdim=True)
    attn = _unfold_out(attn, padded_query_len, num_heads, head_size)
    return _store(out, attn, query_row_index, padded_query_len)


def page_attn_head_major_prefill_qknotr_kernel(
    query, query_row_index, k_pages, v_pages, page_index_tables, mask_tiles, scale,
    num_blocks, padded_query_len, num_heads, num_kv_heads, head_size, block_size,
    logits_soft_cap=0.0, out=None, prescale=False,
):
    """Diagnostic, numerically wrong: ``qkonly`` with the page fed in untransposed.

    Same gather, same matmul extents, same mask add and accumulate. The only difference
    from ``qkonly`` is that the stationary operand arrives sticked on the axis
    ``matmul`` wants (head_size innermost, as the cache stores it) rather than needing
    the last two axes swapped, so the gap between the two is the cost of that swap.
    """
    g = num_heads // num_kv_heads
    q = _fold_query(query, query_row_index, padded_query_len, num_kv_heads, g, head_size)
    acc = None
    for i in range(num_blocks):
        k_page = k_pages.index_select(0, page_index_tables[i]).squeeze(0)
        scores = _masked_scores(q, k_page.reshape(num_kv_heads, head_size, block_size),
                                mask_tiles[i], scale, num_kv_heads, g, padded_query_len,
                                block_size, logits_soft_cap, True, transpose_k=False)
        acc = scores if acc is None else acc + scores
    attn = _unfold_out(acc[:, :, :head_size], padded_query_len, num_heads, head_size)
    return _store(out, attn, query_row_index, padded_query_len)
