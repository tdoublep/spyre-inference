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

"""Head-major paged prefill that splits the work off the KV axis.

The shipped kernels walk KV one page per step and carry an online softmax across
those steps. Here the whole KV extent is gathered once and reduced in a single
ordinary softmax, so the work is split along axes that need no running-max
bookkeeping at all: the query rows, and optionally the GQA query-group axis folded
into the matmul's M.

Head-major only: a page is ``[num_kv_heads, block_size, head_size]``, so pages
concatenate straight along the token axis.
"""

import torch


def _pv(probs, v_c, gqa_merge, num_kv_heads, num_queries_per_kv, rows, head_size):
    """probs @ V, folding the query-group axis into M when asked."""
    if gqa_merge:
        merged = probs.reshape(num_kv_heads, num_queries_per_kv * rows, probs.shape[-1])
        return torch.matmul(merged, v_c).reshape(
            num_kv_heads, num_queries_per_kv, rows, head_size
        )
    return torch.matmul(probs, v_c.unsqueeze(1))


def head_major_split_kernel(
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
    q_tile: int = 0,
    gqa_merge: bool = False,
    subtract_max: bool = True,
    kv_blocks: int = 0,
):
    """One softmax over the whole KV extent, per query tile.

    ``q_tile`` query rows per step (0 = every row in one step). ``kv_blocks`` KV pages
    per step (0 = the whole extent in one step, which needs no running max at all);
    combining the two shrinks the score tile on both axes. ``gqa_merge`` folds the
    query-group axis into the matmul's M so K is not broadcast. ``subtract_max`` off is
    a diagnostic: it removes the max pass but overflows fp16 on real logits.
    """
    num_queries_per_kv = num_heads // num_kv_heads
    block_size = k_pages.shape[2]
    kv_len = num_blocks * block_size

    q_rows = query.index_select(0, query_row_index[:padded_query_len])
    q = (
        q_rows.unsqueeze(0)
        .transpose(1, 2)
        .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
    )

    # KV is gathered one chunk at a time, outer to the query loop: each chunk is read
    # once, and its token axis is a whole dimension of the gathered tensor. Slicing a
    # wider token axis instead trips the backend's batchmatmul stick constraint
    # ("Input2 stick must contain generated_sym"), which is why the extent is not
    # gathered once and sliced.
    blocks_per_chunk = kv_blocks if kv_blocks else num_blocks
    if num_blocks % blocks_per_chunk:
        blocks_per_chunk = num_blocks
    chunk_len = blocks_per_chunk * block_size
    num_chunks = num_blocks // blocks_per_chunk

    tile = q_tile if q_tile else padded_query_len
    q_starts = list(range(0, padded_query_len, tile))
    # Per query tile: running max, running denominator, unnormalised accumulator.
    run_max: list = [None] * len(q_starts)
    run_sum: list = [None] * len(q_starts)
    acc: list = [None] * len(q_starts)

    for c in range(num_chunks):
        page_ids = page_index_table[c * blocks_per_chunk : (c + 1) * blocks_per_chunk, 0:1]
        k_c = (
            k_pages[page_ids]
            .squeeze(1)
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, chunk_len, head_size)
        )
        v_c = (
            v_pages[page_ids]
            .squeeze(1)
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, chunk_len, head_size)
        )
        k_c_t = k_c.transpose(-2, -1)
        mask_c = torch.cat(mask_tiles[c * blocks_per_chunk : (c + 1) * blocks_per_chunk], dim=-1)
        alibi_c = (
            torch.cat(
                alibi_bias_tiles[c * blocks_per_chunk : (c + 1) * blocks_per_chunk], dim=-1
            )
            if alibi_bias_tiles is not None
            else None
        )

        for t, start in enumerate(q_starts):
            stop = min(start + tile, padded_query_len)
            rows = stop - start
            q_t = q[:, :, start:stop, :]

            if gqa_merge:
                # [KV, groups * rows, D] x [KV, D, chunk_len]: no broadcast over the
                # group axis, and M is groups * rows instead of rows.
                scores = torch.matmul(
                    q_t.reshape(num_kv_heads, num_queries_per_kv * rows, head_size), k_c_t
                ).reshape(num_kv_heads, num_queries_per_kv, rows, chunk_len)
            else:
                scores = torch.matmul(q_t, k_c_t.unsqueeze(1))
            scores = scores * scale

            if logits_soft_cap > 0.0:
                scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
            if alibi_c is not None:
                scores = scores + alibi_c
            scores = scores + mask_c[start:stop]

            if num_chunks == 1:
                # One chunk over the whole extent: an ordinary softmax, no running state.
                if subtract_max:
                    scores = scores - torch.amax(scores, dim=-1, keepdim=True)
                probs = torch.exp(scores)
                run_sum[t] = probs.sum(dim=-1, keepdim=True)
                acc[t] = _pv(
                    probs, v_c, gqa_merge, num_kv_heads, num_queries_per_kv, rows, head_size
                )
                continue

            chunk_max = torch.amax(scores, dim=-1, keepdim=True) if subtract_max else None
            if acc[t] is None:
                run_max[t] = chunk_max
                probs = torch.exp(scores - chunk_max) if subtract_max else torch.exp(scores)
                run_sum[t] = probs.sum(dim=-1, keepdim=True)
                acc[t] = _pv(
                    probs, v_c, gqa_merge, num_kv_heads, num_queries_per_kv, rows, head_size
                )
            else:
                if subtract_max:
                    new_max = torch.maximum(run_max[t], chunk_max)
                    rescale = torch.exp(run_max[t] - new_max)
                    acc[t] = acc[t] * rescale
                    run_sum[t] = run_sum[t] * rescale
                    run_max[t] = new_max
                    probs = torch.exp(scores - new_max)
                else:
                    probs = torch.exp(scores)
                run_sum[t] = run_sum[t] + probs.sum(dim=-1, keepdim=True)
                acc[t] = acc[t] + _pv(
                    probs, v_c, gqa_merge, num_kv_heads, num_queries_per_kv, rows, head_size
                )

    pieces = [acc[t] / run_sum[t] for t in range(len(q_starts))]
    attn = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=2)
    attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
    attn = attn.reshape(padded_query_len, num_heads, head_size)
    if out is not None:
        out.index_copy_(0, query_row_index[:padded_query_len], attn[:padded_query_len])
        return out
    return attn


def parse_split_variant(spec: str) -> dict | None:
    """``SPYRE_ATTN_PREFILL_VARIANT`` -> kwargs, or None when no split token appears.

    Tokens, '+'-separated: ``qtile<N>`` query rows per step, ``full`` for one step
    over every row, ``merge`` to fold the GQA group axis into M, ``nomax`` to drop
    the max subtraction (diagnostic).
    """
    kwargs: dict = {"q_tile": 0, "gqa_merge": False, "subtract_max": True, "kv_blocks": 0}
    seen = False
    for tok in spec.split("+"):
        tok = tok.strip()
        if tok.startswith("qtile"):
            kwargs["q_tile"] = int(tok[len("qtile") :])
            seen = True
        elif tok == "full":
            kwargs["q_tile"] = 0
            seen = True
        elif tok == "merge":
            kwargs["gqa_merge"] = True
            seen = True
        elif tok.startswith("kv"):
            kwargs["kv_blocks"] = int(tok[len("kv") :])
            seen = True
        elif tok == "nomax":
            kwargs["subtract_max"] = False
            seen = True
    return kwargs if seen else None
