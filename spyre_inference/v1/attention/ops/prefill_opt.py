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

"""Experimental prefill attention formulations, selected by SPYRE_ATTN_PREFILL_VARIANT.

Every kernel here takes ``page_attn_kernel``'s signature, so switching formulation
needs no metadata, builder or warmup change: the chunked variants slice their page
ids out of ``page_index_table`` and concatenate their mask inside the traced region.
"""

import torch


def prefill_opt_kernel(
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
    head_major: bool = False,
    blocks_per_step: int = 1,
    fold_scale: bool = False,
    skip_zero_mask: bool = False,
):
    """Online softmax attention, parameterised over the formulations under test.

    ``blocks_per_step`` KV pages per online-softmax step; must divide ``num_blocks``.
    ``fold_scale`` scales the query once instead of every step's score tile.
    ``head_major`` selects the ``[blocks, kv_heads, block_size, head_size]`` cache.
    """
    num_queries_per_kv = num_heads // num_kv_heads
    block_size = k_pages.shape[2] if head_major else k_pages.shape[1]
    # Decode has no query axis to spread a wider score tile over, and an
    # indivisible count would leave a ragged final step. Both are trace-time
    # constants under dynamic=False, so this branch costs nothing at runtime.
    if padded_query_len <= 1 or num_blocks % blocks_per_step:
        blocks_per_step = 1
    step_len = blocks_per_step * block_size

    # Blocks below the causal diagonal are visible to every query row, so their mask
    # tile is all-zero and adding it costs a full pass over the score tile for nothing.
    # _mirror_mask_tiles hands every such block the same device buffer, so the repeated
    # object identifies them here at trace time -- no metadata, and nothing is skipped
    # if the sharing is ever absent.
    zero_tile = None
    if skip_zero_mask and len(mask_tiles) > 1:
        counts: dict[int, int] = {}
        for t in mask_tiles:
            counts[id(t)] = counts.get(id(t), 0) + 1
        hottest = max(counts, key=lambda k: counts[k])
        if counts[hottest] > 1:
            zero_tile = next(t for t in mask_tiles if id(t) == hottest)

    q_rows = query.index_select(0, query_row_index[:padded_query_len])
    q = (
        q_rows.unsqueeze(0)
        .transpose(1, 2)
        .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
    )
    if fold_scale:
        # scale is a scalar and matmul is linear in q, so one pass over q replaces
        # one pass over every step's score tile.
        q = q * scale

    tile_max = None
    tile_sum = None
    tile_output = None

    for step in range(num_blocks // blocks_per_step):
        if blocks_per_step == 1:
            page_idx = page_index_table[step, 0:1]
            k_step = k_pages.index_select(0, page_idx).squeeze(0)
            v_step = v_pages.index_select(0, page_idx).squeeze(0)
            if not head_major:
                k_step = k_step.permute(1, 0, 2)
                v_step = v_step.permute(1, 0, 2)
            k_step = k_step.unsqueeze(1)
            v_step = v_step.unsqueeze(1)
            mask_step = mask_tiles[step]
            if zero_tile is not None and mask_step is zero_tile:
                mask_step = None
        else:
            # Advanced indexing on a [blocks_per_step, 1] index: merging the gathered
            # page axis into the token axis and feeding that to a matmul faults the
            # compute block (RAS 0x7b1b), while this form lowers cleanly.
            page_ids = page_index_table[step * blocks_per_step : (step + 1) * blocks_per_step, 0:1]
            k_gathered = k_pages[page_ids].squeeze(1)
            v_gathered = v_pages[page_ids].squeeze(1)
            # Head-major keeps each page's [block_size, head_size] tile contiguous, so
            # this permute moves whole tiles; token-major transposes within each page.
            perm = (1, 0, 2, 3) if head_major else (2, 0, 1, 3)
            k_step = (
                k_gathered.permute(*perm).reshape(num_kv_heads, step_len, head_size).unsqueeze(1)
            )
            v_step = (
                v_gathered.permute(*perm).reshape(num_kv_heads, step_len, head_size).unsqueeze(1)
            )
            group = mask_tiles[step * blocks_per_step : (step + 1) * blocks_per_step]
            mask_step = (
                None
                if zero_tile is not None and all(t is zero_tile for t in group)
                else torch.cat(group, dim=-1)
            )

        scores = torch.matmul(q, k_step.transpose(-2, -1))
        if not fold_scale:
            scores = scores * scale
        if logits_soft_cap > 0.0:
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        if alibi_bias_tiles is not None:
            scores = scores + (
                alibi_bias_tiles[step]
                if blocks_per_step == 1
                else torch.cat(
                    alibi_bias_tiles[step * blocks_per_step : (step + 1) * blocks_per_step],
                    dim=-1,
                )
            )
        if mask_step is not None:
            scores = scores + mask_step
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if step == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = torch.matmul(tile_probs, v_step)
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
            tile_output += torch.matmul(tile_probs, v_step)
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


def parse_variant(spec: str) -> dict:
    """``SPYRE_ATTN_PREFILL_VARIANT`` -> kwargs for prefill_opt_kernel.

    Tokens, '+'-separated: ``chunk<N>`` for N pages per step, ``fold`` to fold the
    scale into the query. ``base`` (or empty) is the unmodified per-page formulation.
    """
    kwargs: dict = {"blocks_per_step": 1, "fold_scale": False, "skip_zero_mask": False}
    for tok in spec.split("+"):
        tok = tok.strip()
        if tok in ("", "base"):
            continue
        if tok == "fold":
            kwargs["fold_scale"] = True
        elif tok == "skipmask":
            kwargs["skip_zero_mask"] = True
        elif tok.startswith("chunk"):
            kwargs["blocks_per_step"] = int(tok[len("chunk") :])
        else:
            raise ValueError(f"unknown SPYRE_ATTN_PREFILL_VARIANT token {tok!r} in {spec!r}")
    return kwargs


def experimental_attn_fn(head_major: bool, compile_attn: bool):
    """Compiled ``prefill_opt_kernel`` for the configured variant, or None when unset."""
    import functools

    from spyre_inference import envs

    spec = envs.SPYRE_ATTN_PREFILL_VARIANT
    if not spec:
        return None

    # A split token selects the work-split kernel, which is head-major only.
    from spyre_inference.v1.attention.ops.prefill_split import (
        head_major_split_kernel,
        parse_split_variant,
    )

    split = parse_split_variant(spec)
    if split is not None:
        # Head-major only. The head-major impl's __init__ runs the token-major base
        # first, so declining here leaves the shipped kernel for that pass and the
        # subclass installs the split kernel afterwards.
        if not head_major:
            return None
        fn = functools.partial(head_major_split_kernel, **split)
    else:
        fn = functools.partial(prefill_opt_kernel, head_major=head_major, **parse_variant(spec))
    return torch.compile(fn, dynamic=False) if compile_attn else fn
