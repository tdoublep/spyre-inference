"""Candidate batched decode kernels, measured against per_seq / batched / unrolled.

Companion to microbench_batched_decode.py. That harness established the problem:
one batched call is ~2.7x slower than one compiled graph holding the num_seqs
per-sequence bodies ("unrolled"), so batching as an idea is fine and the shipped
batched kernel's shapes are what cost. This file searches for a batched kernel
that beats unrolled.

Structural difference the design turns on. Both kernels present the *same* number
of matmul batch units (32 = 4 seqs x 8 kv = 8 kv x 4 query groups); batching did
not widen the batch axis, it moved the 4 query groups out of a broadcast batch
axis and into the matmul's M axis:

    per_seq  q [KV, QPKV, 1, D]  @  kT [KV, 1, D, block]   batch (8,4)  M=1
             (K broadcast across the 4 query groups)
    batched  q [S, KV, QPKV, D]  @  kT [S, KV, D, block]   batch (4,8)  M=4
             (no broadcast axis)

So the candidates below keep M=1, and none of them merges two axes across a
permute -- (S,KV) cannot be flattened after permute(0,2,1,3) without a copy,
since that needs stride[0] == size[1]*stride[1], which holds only for block==1.

Variants
  per_seq        reference: num_seqs sequential _page_attn_kernel calls
  batched        the shipped _batched_decode_kernel (1 launch, M=4)
  unrolled       the bar to beat: 1 graph, num_seqs per-sequence bodies
  batched_masklist
                 `batched` with the per-block mask passed as a Python list
                 instead of sliced in-graph out of one [nb, S*KV, 1, block]
                 buffer. Isolates the cost of the in-graph mask slice, which
                 per_seq never pays (its mask_tiles[i] is a trace-time list
                 index, so each block's mask is its own kernel argument).
  qgroup_loop    CANDIDATE 1. Batched gather (one index_select per block for all
                 num_seqs pages), batch (S,KV), M=1 by looping the 4 query groups
                 inside the block loop. The gathered page is reused across the 4
                 groups, recovering per_seq's broadcast reuse explicitly. No copy.
  gather_shared  CANDIDATE 2. One batched gather per block, then num_seqs
                 per-sequence bodies over slices of it -- per_seq's exact matmul
                 shape (broadcast included) with 4x fewer gathers and 1 launch.
  merged_sk      CANDIDATE 3. Pay the (S,KV) merge copy to get a single batch
                 axis of 32 with M=1: q [32, QPKV, 1, D] @ kT [32, 1, D, block].
                 per_seq's exact shape family, 4x wider. Measures whether the
                 copy is cheaper than the lossy division it removes.
  merged_slot_gather
                 CANDIDATE 4. merged_sk's shape without its copy, by gathering
                 at slot granularity so (S,KV) needs no merge. Backend rejects
                 the interleaved index.
  seq_on_m       CANDIDATE 5. 3-D bmm, batch=KV, all sequences' queries on M and
                 all their keys on N with a block-diagonal mask; num_seqs x the
                 MACs but maximal operand reuse. Backend compiler crashes.
  merged_cat     CANDIDATE 6. Width-1 gathers only, cat'd into the (S*KV) batch
                 axis -- unrolled's gathers with a quarter of its matmuls.
  gather_all_once
                 DIAGNOSTIC, not viable: one gather for the whole call. Scratch
                 goes as max_batch x max_model_len, i.e. un-pages the cache.
                 Kept only as an upper bound; chunk=8 already matches it.
  chunked_gather CANDIDATE 8, best batched result. Gather `chunk` blocks for all
                 sequences per index_select, then static per-block slices.
                 Scratch bounded by `chunk`, independent of max_model_len.
  batched_ktile  CANDIDATE 9. Batch the sequences but process kv_tile slots at a
                 time, holding each per-op tensor at the per-sequence size.
                 kv_tile must be a multiple of the fp16 stick (64).
  chunked_ktile  CANDIDATE 10. Both fixes at once: few wide gathers over blocks
                 and small tiled operands.

Run (needs a Spyre device; #4347 needs LAYOUT_SOLVER=greedy to be active at all,
since GreedyLayoutSolver is the only solver with supports_paired_buffers=True):
    LAYOUT_SOLVER=greedy SPYRE_LX_PLANNER_RELAYOUT=1 \
        python experimental/microbench_batched_decode_v2.py

RESULTS (tpa-spyre-dev-2, granite shapes: 4 seqs, 16 blocks x 128, 8 KV heads,
4 queries/KV, head 128; #4347 active via LAYOUT_SOLVER=greedy). Wall median, ms:

    chunked_ktile chunk=8 tile=64  2.707 - 2.716  <- BEATS THE BAR, 0.93x
    unrolled (the bar)            2.923 - 3.006   (five runs)
    per_seq                       3.067 - 3.082
    chunked_gather chunk=8        3.623           (block-major index)
    chunked_gather chunk=8        3.902           (sequence-major index)
    chunked_gather chunk=16       3.912
    chunked_gather chunk=4        4.822
    chunked_gather chunk=2        6.261
    batched_ktile tile=64         7.252           (tiling with no chunking)
    batched_masklist              8.039
    batched (shipped)             8.129
    merged_sk                     8.243
    merged_cat                    8.822
    gather_shared                 8.941
    qgroup_loop                  11.114           (also numerically WRONG, below)

What the regression is NOT. Each was varied on its own and none tracks runtime:
  * work division. qgroup_loop removed all 32 lossy divisions and got SLOWER
    (11.11). Conversely `batched` at num_seqs=1 emits the same 32 lossy
    divisions and runs 10x faster. Dead end.
  * the in-graph mask slice. batched_masklist 8.04 vs batched 8.13.
  * matmul batch dims / M / the K broadcast. gather_shared reproduces per_seq's
    exact matmul, broadcast included, and still costs 8.94.
  * gather index width per se. merged_cat uses unrolled's own 128 width-1
    gathers and still costs 8.82.
  * lower_bmm does cap batch axes at 2 (3-D@3-D, 4-D@4-D, 3-D@2-D; anything else
    raises Unsupported), so batching sequences and keeping the query-group
    broadcast are mutually exclusive in one bmm -- but that is not what costs.

What it IS. Two independent effects, found by scaling num_seqs 1/2/4:

    num_seqs      per_seq   unrolled   batched
    1               0.842      0.885     0.832   <- batched WINS at batch 1
    2               1.571      1.619     7.316
    4               3.067      2.953     8.129

  1. Gather op count. The jump is a cliff at num_seqs>=2, then nearly flat --
     matching the isolated gather curve (0.146 ms at width 1, 0.380 at width 2,
     flat to width 8; adjacent pages are no cheaper than spread, so it is index
     count and not contiguity). The cost is per gather op and roughly flat in
     width, so gathering `chunk` blocks at once takes 8.13 -> 3.90 and then
     flattens at chunk=8. Note chunk=16 buys nothing over chunk=8, so the
     unbounded-scratch "gather everything once" variant has no advantage and
     paging costs nothing.
  2. A residual that scales superlinearly in num_seqs. With gather cost
     removed, batched compute is 3.90 at S=4 against 4 x 0.832 = 3.33 expected,
     while per-sequence compute scales sublinearly (0.885 -> 2.953). This points
     at per-op working set (a batch-wide per-block tensor is 1 MB here against
     256 KB per sequence) rather than at any shape property.

THE ANSWER: fix both, and a batched kernel beats the unrolled one.

    batched (shipped)                          8.129   2.76x the bar
      + gather chunk=8, block-major index      3.623   1.23x
      + KV tiling kv_tile=64                   2.716   0.92x  <- chunked_ktile

Reproduced back-to-back in one process at 30 iterations, which is the fairest
comparison available (same process, same device state):

    unrolled        2.923
    chunked_ktile   2.707      0.926x

Neither fix alone is enough, and they are superadditive -- tiling on its own is
worth only 11% because the 32 wide gathers hide it, and it is worth a further 25%
once they are gone:

    batched                      8.129
    batched + tiling only        7.252   -11%
    batched + chunking only      3.623   -55%
    batched + both               2.716   -67%

That mutual masking is why every single-axis fix tried before this failed.
`chunked_ktile` is the design --
batch every sequence into one launch, gather `chunk` blocks per index_select so
gather ops fall from 2*num_blocks to 2*num_blocks/chunk, and keep every matmul
operand and score tile at kv_tile slots so the per-op working set does not grow
with the batch.

Both knobs are constrained:
  * kv_tile must be a multiple of the fp16 stick (64). kv_tile=32 fails with
    "no mechanism to resolve stick incompatibility", and kv_tile=128 is no
    tiling at all (3.586 ms, and the 32 lossy divisions come back), so at
    block_size 128 the value 64 is the only tiling available and is the optimum.
  * the chunk index must be BLOCK-major. Sequence-major makes the per-block
    slice's host dim 0 strided, which insert_restickify_padding rejects; going
    block-major was also worth 7% on chunked_gather (3.902 -> 3.623).
  * chunk must divide num_blocks, and scratch is num_seqs*chunk*block*KV*D per
    tensor -- bounded by chunk and the batch, never by max_model_len, so the
    cache stays paged. Note chunking is arithmetically the same as a chunk x
    larger block_size for the gather, but is the better lever because it leaves
    cache allocation granularity (and fragmentation) alone.

    The win needs chunk=8; smaller chunks do not reach the bar. At kv_tile=64:

        chunk    gathers   wall ms   beats the 2.92 bar?
          2        16       5.632    no
          4         8       3.681    no
          8         4       2.71     yes
         16         2       (see below)

    So the demonstrated advantage is specific to chunk=8, which at batch 4 costs
    16.8 MB of transient for K+V. That figure grows with the batch (~134 MB at
    batch 32), so chunk has to shrink as batch grows and the advantage shown here
    should not be assumed to carry to large batch without re-measuring -- or
    without a larger block_size doing the same job for free.

Backend limits hit while exploring (all torch-spyre, not local bugs):
  * merged_slot_gather: insert_restickify_padding rejects the interleaved slot
    index (coord 128*d0 + d3, "carries multiple free symbols").
  * seq_on_m: dxp_standalone exits 1 on the block-diagonal 3-D bmm shape.
  * batched_ktile at kv_tile=32: sub-stick score tile.
  * chunked_ktile with a sequence-major chunk: strided host dim 0.

Note the KV cache here is zero-filled, so timing is data-independent but no
variant's output means anything -- correctness lives in
check_batched_decode_candidates.py, which uses random K/V and a real mask.
Against an eager per-sequence reference (absmax 0.118) it reports:

    chunked_gather, merged_sk, merged_cat   max_abs 0.00000   bit-exact
    batched, batched_masklist               max_abs 0.00049
    qgroup_loop                             max_abs 0.08268   WRONG

qgroup_loop is numerically broken, not merely slow -- do not read its timing as a
valid data point for the query-group-loop idea. It was already the slowest
variant so nothing here rests on it, and it has not been debugged.
"""

import argparse
import logging
import statistics
import time

import torch

from spyre_inference.custom_ops.utils import convert
from spyre_inference.custom_ops.utils import register as register_convert_op
from spyre_inference.v1.attention.backends.spyre_attn import (
    INT32_ELEMS_PER_STICK,
    _batched_decode_kernel,
    _page_attn_kernel,
    _stick_aligned_len,
    slot_major_kv_layout,
)

DEV = torch.device("spyre")


def _online_init(scores, v_page):
    m = torch.amax(scores, dim=-1, keepdim=True)
    p = torch.exp(scores - m)
    return m, p.sum(dim=-1, keepdim=True), torch.matmul(p, v_page)


def _online_step(state, scores, v_page):
    tile_max, tile_sum, tile_out = state
    new_max = torch.maximum(tile_max, torch.amax(scores, dim=-1, keepdim=True))
    rescale = torch.exp(tile_max - new_max)
    p = torch.exp(scores - new_max)
    return (
        new_max,
        tile_sum * rescale + p.sum(dim=-1, keepdim=True),
        tile_out * rescale + torch.matmul(p, v_page),
    )


# ---------------------------------------------------------------------------
# Diagnostic variant
# ---------------------------------------------------------------------------


def _batched_masklist_kernel(
    query, k_pages, v_pages, block_ids, mask_list, scale,
    num_seqs, num_blocks, num_kv_heads, num_queries_per_kv, block_size, head_size,
):
    """`_batched_decode_kernel` with the mask as a trace-time list, not a slice."""
    num_heads = num_kv_heads * num_queries_per_kv
    q = query[:num_seqs].reshape(num_seqs, num_kv_heads, num_queries_per_kv, head_size)

    state = None
    for i in range(num_blocks):
        page_idx = block_ids[i, 0:num_seqs]
        k_page = k_pages.index_select(0, page_idx).permute(0, 2, 1, 3)
        v_page = v_pages.index_select(0, page_idx).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale + mask_list[i]
        state = _online_init(scores, v_page) if i == 0 else _online_step(state, scores, v_page)

    tile_max, tile_sum, tile_out = state
    return (tile_out / tile_sum).reshape(num_seqs, num_heads, head_size)


# ---------------------------------------------------------------------------
# Candidate 1: batched gather, M=1, query groups as an unrolled loop
# ---------------------------------------------------------------------------


def _qgroup_loop_kernel(
    query, k_pages, v_pages, block_ids, mask_list, scale,
    num_seqs, num_blocks, num_kv_heads, num_queries_per_kv, block_size, head_size,
):
    """Batch (S,KV) with M=1; the gathered page is shared by the QPKV groups.

    mask_list[i]: [num_seqs, num_kv_heads, 1, block_size].
    """
    num_heads = num_kv_heads * num_queries_per_kv
    q_all = query[:num_seqs].reshape(num_seqs, num_kv_heads, num_queries_per_kv, head_size)

    states = [None] * num_queries_per_kv
    for i in range(num_blocks):
        page_idx = block_ids[i, 0:num_seqs]
        # Gathered once per block and reused by every query group below, which is
        # the reuse per_seq gets for free from its broadcast K axis.
        k_page = k_pages.index_select(0, page_idx).permute(0, 2, 1, 3)
        v_page = v_pages.index_select(0, page_idx).permute(0, 2, 1, 3)
        k_t = k_page.transpose(-2, -1)
        mask_tile = mask_list[i]
        for g in range(num_queries_per_kv):
            q_g = q_all[:, :, g : g + 1, :]
            scores = torch.matmul(q_g, k_t) * scale + mask_tile
            states[g] = (
                _online_init(scores, v_page)
                if i == 0
                else _online_step(states[g], scores, v_page)
            )

    outs = [s[2] / s[1] for s in states]
    return torch.cat(outs, dim=2).reshape(num_seqs, num_heads, head_size)


# ---------------------------------------------------------------------------
# Candidate 2: one batched gather per block, per-sequence bodies over slices
# ---------------------------------------------------------------------------


def _gather_shared_kernel(
    query, k_pages, v_pages, block_ids, mask_tiles, scale,
    num_seqs, num_blocks, num_kv_heads, num_queries_per_kv, block_size, head_size,
):
    """per_seq's exact matmul shape, but the gather is done once for all seqs.

    mask_tiles[i]: [1, block_size], as _page_attn_kernel takes.
    """
    num_heads = num_kv_heads * num_queries_per_kv
    q_all = query[:num_seqs].reshape(num_seqs, num_kv_heads, num_queries_per_kv, head_size)

    states = [None] * num_seqs
    for i in range(num_blocks):
        page_idx = block_ids[i, 0:num_seqs]
        k_all = k_pages.index_select(0, page_idx)
        v_all = v_pages.index_select(0, page_idx)
        mask_tile = mask_tiles[i]
        for s in range(num_seqs):
            # [block, KV, D] -> [KV, 1, block, D]: _page_attn_kernel's k_page_4d.
            k_page = k_all[s].permute(1, 0, 2).unsqueeze(1)
            v_page = v_all[s].permute(1, 0, 2).unsqueeze(1)
            q_s = q_all[s].unsqueeze(2)  # [KV, QPKV, 1, D]
            scores = torch.matmul(q_s, k_page.transpose(-2, -1)) * scale + mask_tile
            states[s] = (
                _online_init(scores, v_page)
                if i == 0
                else _online_step(states[s], scores, v_page)
            )

    outs = [(s[2] / s[1]).reshape(1, num_heads, head_size) for s in states]
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Candidate 3: merge (S,KV) into one batch axis of 32, M=1
# ---------------------------------------------------------------------------


def _merged_sk_kernel(
    query, k_pages, v_pages, block_ids, mask_tiles, scale,
    num_seqs, num_blocks, num_kv_heads, num_queries_per_kv, block_size, head_size,
):
    """Single batch axis of S*KV with M=1 -- per_seq's shape, 4x wider.

    The (S,KV) merge cannot be a view after permute(0,2,1,3), so reshape copies
    here on purpose; the point is to price that copy against the lossy division.
    mask_tiles[i]: [1, block_size].
    """
    g = num_seqs * num_kv_heads
    num_heads = num_kv_heads * num_queries_per_kv
    q = query[:num_seqs].reshape(g, num_queries_per_kv, head_size).unsqueeze(2)

    state = None
    for i in range(num_blocks):
        page_idx = block_ids[i, 0:num_seqs]
        k_page = (
            k_pages.index_select(0, page_idx)
            .permute(0, 2, 1, 3)
            .reshape(g, block_size, head_size)
            .unsqueeze(1)
        )
        v_page = (
            v_pages.index_select(0, page_idx)
            .permute(0, 2, 1, 3)
            .reshape(g, block_size, head_size)
            .unsqueeze(1)
        )
        scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale + mask_tiles[i]
        state = _online_init(scores, v_page) if i == 0 else _online_step(state, scores, v_page)

    tile_max, tile_sum, tile_out = state
    attn = (tile_out / tile_sum).reshape(num_seqs, num_kv_heads, num_queries_per_kv, head_size)
    return attn.reshape(num_seqs, num_heads, head_size)


# ---------------------------------------------------------------------------
# Candidate 4: merged (S,KV) batch axis, but gathered straight into that layout
# ---------------------------------------------------------------------------


def _merged_slot_gather_kernel(
    query, k_pages_flat, v_pages_flat, slot_index, mask_tiles, scale,
    num_seqs, num_blocks, num_kv_heads, num_queries_per_kv, block_size, head_size,
):
    """merged_sk's shape without merged_sk's copy.

    (S,KV) cannot be merged as a view after a page gather, because S and KV are
    separated by the block axis. So gather at slot granularity instead: the
    device layout is contiguous, so [num_slots*KV, D] is a legal view of the
    cache, and row (s,kv,t) of the target sits at source row
    (page_s*block + t)*KV + kv. slot_index carries that, precomputed per block.

    k/v_pages_flat: [num_slots * KV, D] view of the cache.
    slot_index[i]:  [S*KV*block] int32, the rows for block i in target order.
    mask_tiles[i]:  [1, block_size].
    """
    g = num_seqs * num_kv_heads
    num_heads = num_kv_heads * num_queries_per_kv
    q = query[:num_seqs].reshape(g, num_queries_per_kv, head_size).unsqueeze(2)

    state = None
    for i in range(num_blocks):
        idx = slot_index[i]
        # Free reshape: the gather already lands in (S,KV)-major order.
        k_page = k_pages_flat.index_select(0, idx).reshape(g, block_size, head_size).unsqueeze(1)
        v_page = v_pages_flat.index_select(0, idx).reshape(g, block_size, head_size).unsqueeze(1)
        scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale + mask_tiles[i]
        state = _online_init(scores, v_page) if i == 0 else _online_step(state, scores, v_page)

    tile_max, tile_sum, tile_out = state
    attn = (tile_out / tile_sum).reshape(num_seqs, num_kv_heads, num_queries_per_kv, head_size)
    return attn.reshape(num_seqs, num_heads, head_size)


# ---------------------------------------------------------------------------
# Candidate 10: chunked gather AND KV tiling -- the two fixes combined
# ---------------------------------------------------------------------------


def _chunked_ktile_kernel(
    query, k_pages, v_pages, chunk_pages, mask_list, scale,
    num_seqs, num_blocks, chunk, kv_tile, num_kv_heads, num_queries_per_kv,
    block_size, head_size,
):
    """Few gathers (wide, over blocks) and small per-op tensors (tiled KV).

    The two penalties measured here are independent:
      * gather cost is per-op, so gathering `chunk` blocks at once takes the
        batched kernel from 8.13 to 3.90 ms and then flattens;
      * what is left scales superlinearly in num_seqs, which per-sequence
        compute does not, pointing at per-op working set rather than shape.
    This fixes both at once: `chunk` sets the number of gathers, `kv_tile` sets
    the size of every matmul operand and score tile.

    Scratch stays bounded by num_seqs*chunk*block_size*KV*D and never depends on
    max_model_len, so the cache stays paged.

    Shapes per tile (granite at batch 4, chunk 8, kv_tile 64):
        gather      [chunk*S] pages          -> [chunk, S, block, KV, D]
        k_ch[j]     [S, block, KV, D]           contiguous view, static index
        k_t         [S, KV, kv_tile, D]         permuted view
        q           [S, KV, QPKV, D]
        scores      [S, KV, QPKV, kv_tile]      batch (S,KV), M=QPKV, no broadcast
        out         [S, num_heads, D]
    Note this is the batch-(S,KV), M=4, no-broadcast shape that earlier looked
    like the culprit; it is the fastest one measured, so matmul shape was never
    what cost.

    Requirements: chunk divides num_blocks; kv_tile is a multiple of the fp16
    stick (64); chunk_pages is block-major.

    mask_list[i]: [num_seqs, num_kv_heads, 1, block_size]; sliced per tile.
    """
    num_heads = num_kv_heads * num_queries_per_kv
    q = query[:num_seqs].reshape(num_seqs, num_kv_heads, num_queries_per_kv, head_size)
    tiles = block_size // kv_tile

    state = None
    first = True
    for c in range(len(chunk_pages)):
        idx = chunk_pages[c]
        k_ch = k_pages.index_select(0, idx).reshape(
            chunk, num_seqs, block_size, num_kv_heads, head_size)
        v_ch = v_pages.index_select(0, idx).reshape(
            chunk, num_seqs, block_size, num_kv_heads, head_size)
        for j in range(chunk):
            i = c * chunk + j
            for tt in range(tiles):
                lo = tt * kv_tile
                # Both indices static -> views, no gather and no copy. Block-major
                # so the leading index keeps host dim 0 contiguous.
                k_t = k_ch[j, :, lo : lo + kv_tile].permute(0, 2, 1, 3)
                v_t = v_ch[j, :, lo : lo + kv_tile].permute(0, 2, 1, 3)
                scores = torch.matmul(q, k_t.transpose(-2, -1)) * scale
                scores = scores + mask_list[i][:, :, :, lo : lo + kv_tile]
                state = (_online_init(scores, v_t) if first
                         else _online_step(state, scores, v_t))
                first = False

    tile_max, tile_sum, tile_out = state
    return (tile_out / tile_sum).reshape(num_seqs, num_heads, head_size)


# ---------------------------------------------------------------------------
# Candidate 9: batch the sequences, tile the KV axis to hold the working set
# ---------------------------------------------------------------------------


def _batched_ktile_kernel(
    query, k_pages, v_pages, block_ids, mask_list, scale,
    num_seqs, num_blocks, kv_tile, num_kv_heads, num_queries_per_kv,
    block_size, head_size,
):
    """Batched over sequences, but only kv_tile slots of a block at a time.

    Every variant that batched sequences made each per-block tensor num_seqs
    times larger (256 KB -> 1 MB here) and every one of them was ~2.7x slower,
    independent of gather width, matmul shape or work division. If the binding
    constraint is the per-block working set rather than any of those, then
    splitting the block's slot axis into kv_tile-sized pieces should recover the
    fast path while keeping the sequence batching: each tensor is then
    [S, kv_tile, KV, D], which at kv_tile = block_size/num_seqs is the same size
    as the per-sequence kernel's per-block tensor.

    Costs num_blocks * (block_size/kv_tile) matmul pairs instead of num_blocks,
    and the online softmax simply runs over more, smaller tiles -- it is already
    a streaming reduction, so correctness does not care where the tile boundaries
    fall.

    mask_list[i]: [num_seqs, num_kv_heads, 1, block_size]; sliced per tile.
    """
    num_heads = num_kv_heads * num_queries_per_kv
    q = query[:num_seqs].reshape(num_seqs, num_kv_heads, num_queries_per_kv, head_size)
    tiles = block_size // kv_tile

    state = None
    for i in range(num_blocks):
        page_idx = block_ids[i, 0:num_seqs]
        k_page = k_pages.index_select(0, page_idx)
        v_page = v_pages.index_select(0, page_idx)
        for tt in range(tiles):
            lo = tt * kv_tile
            # Static slice of the slot axis -> a view.
            k_t = k_page[:, lo : lo + kv_tile].permute(0, 2, 1, 3)
            v_t = v_page[:, lo : lo + kv_tile].permute(0, 2, 1, 3)
            scores = torch.matmul(q, k_t.transpose(-2, -1)) * scale
            scores = scores + mask_list[i][:, :, :, lo : lo + kv_tile]
            state = (_online_init(scores, v_t) if (i == 0 and tt == 0)
                     else _online_step(state, scores, v_t))

    tile_max, tile_sum, tile_out = state
    return (tile_out / tile_sum).reshape(num_seqs, num_heads, head_size)


# ---------------------------------------------------------------------------
# Candidate 8: chunked gather -- widen over blocks, with bounded scratch
# ---------------------------------------------------------------------------


def _chunked_gather_kernel(
    query, k_pages, v_pages, chunk_pages, mask_tiles, scale,
    num_seqs, num_blocks, chunk, num_kv_heads, num_queries_per_kv, block_size, head_size,
):
    """Gather `chunk` blocks for every sequence at once, then slice inside it.

    The gather penalty is per-op and flat in width, so widen the gather instead
    of repeating it -- but widen over a *fixed* number of blocks, not over the
    whole sequence. Scratch is num_seqs * chunk * block_size * KV * D, set by the
    compile-time `chunk` and independent of max_model_len, so the KV cache stays
    paged and no capacity has to be reserved for a contiguous worst case.

    Gathers per call fall from 2*num_blocks to 2*ceil(num_blocks/chunk).

    chunk_pages[c]: [chunk * num_seqs] int32, BLOCK-major, i.e. entry
        j*num_seqs + s is sequence s's page for block c*chunk + j. Block-major
        rather than sequence-major so the per-block slice below is dim-0
        contiguous: slicing a [S, chunk, ...] gather at [:, j] leaves host dim 0
        strided, which insert_restickify_padding rejects ("strided input on host
        dim 0 ... is not supported"). In production this is still just a slice of
        the block table, transposed.
    mask_tiles[i]: [1, block_size].
    """
    g = num_seqs * num_kv_heads
    num_heads = num_kv_heads * num_queries_per_kv
    q = query[:num_seqs].reshape(g, num_queries_per_kv, head_size).unsqueeze(2)

    state = None
    for c in range(len(chunk_pages)):
        idx = chunk_pages[c]
        k_ch = k_pages.index_select(0, idx).reshape(
            chunk, num_seqs, block_size, num_kv_heads, head_size)
        v_ch = v_pages.index_select(0, idx).reshape(
            chunk, num_seqs, block_size, num_kv_heads, head_size)
        for j in range(chunk):
            # Static leading index -> a contiguous view, not a gather.
            k_page = (k_ch[j].permute(0, 2, 1, 3)
                      .reshape(g, block_size, head_size).unsqueeze(1))
            v_page = (v_ch[j].permute(0, 2, 1, 3)
                      .reshape(g, block_size, head_size).unsqueeze(1))
            scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale
            scores = scores + mask_tiles[c * chunk + j]
            state = (_online_init(scores, v_page) if (c == 0 and j == 0)
                     else _online_step(state, scores, v_page))

    tile_max, tile_sum, tile_out = state
    attn = (tile_out / tile_sum).reshape(num_seqs, num_kv_heads, num_queries_per_kv, head_size)
    return attn.reshape(num_seqs, num_heads, head_size)


# ---------------------------------------------------------------------------
# Diagnostic only, NOT a viable design: one gather for the whole call.
# Its scratch is num_seqs * num_blocks * block_size * KV * D, i.e. proportional
# to max_batch x max_model_len, which un-pages the cache and would force
# reserving a contiguous worst-case region. Kept as an upper bound on what
# removing the gather penalty entirely is worth.
# ---------------------------------------------------------------------------


def _gather_all_once_kernel(
    query, k_pages, v_pages, all_pages, mask_tiles, scale,
    num_seqs, num_blocks, num_kv_heads, num_queries_per_kv, block_size, head_size,
):
    """Two gathers per call instead of two per block.

    The measured gather penalty is per-op and flat in width (a width-8 select
    costs what a width-2 one does, for 4x the bytes), so the objective is to
    minimise the *number* of gathers, not their width. Gather every
    (sequence, block) page once up front, then index blocks out of the result
    with static slices, which are views rather than gathers.

    all_pages: [num_seqs * num_blocks] int32, sequence-major page ids.
    mask_tiles[i]: [1, block_size].
    """
    g = num_seqs * num_kv_heads
    num_heads = num_kv_heads * num_queries_per_kv
    q = query[:num_seqs].reshape(g, num_queries_per_kv, head_size).unsqueeze(2)

    # One gather each, then a free reshape splitting seq from block.
    k_all = k_pages.index_select(0, all_pages).reshape(
        num_seqs, num_blocks, block_size, num_kv_heads, head_size)
    v_all = v_pages.index_select(0, all_pages).reshape(
        num_seqs, num_blocks, block_size, num_kv_heads, head_size)

    state = None
    for i in range(num_blocks):
        # Static index -> a view, not a gather.
        k_page = k_all[:, i].permute(0, 2, 1, 3).reshape(g, block_size, head_size).unsqueeze(1)
        v_page = v_all[:, i].permute(0, 2, 1, 3).reshape(g, block_size, head_size).unsqueeze(1)
        scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale + mask_tiles[i]
        state = _online_init(scores, v_page) if i == 0 else _online_step(state, scores, v_page)

    tile_max, tile_sum, tile_out = state
    attn = (tile_out / tile_sum).reshape(num_seqs, num_kv_heads, num_queries_per_kv, head_size)
    return attn.reshape(num_seqs, num_heads, head_size)


# ---------------------------------------------------------------------------
# Candidate 6: per-page (width-1) gathers, concatenated into one batched matmul
# ---------------------------------------------------------------------------


def _merged_cat_kernel(
    query, k_pages, v_pages, page_index_tables, mask_tiles, scale,
    num_seqs, num_blocks, num_kv_heads, num_queries_per_kv, block_size, head_size,
):
    """The one shape that batches the compute without ever widening a gather.

    A width-1 page index_select is a contiguous block_size-slot run and is cheap;
    any wider index is a real gather and costs a flat ~0.19 ms in-graph whatever
    its width. So gather one page at a time exactly as the per-sequence kernel
    does, then cat the per-sequence pages into the (S*KV) batch axis -- cat is a
    copy, but copies measured free next to a wide gather.

    Result: unrolled's gathers (num_seqs*num_blocks width-1 selects) with a
    quarter of unrolled's matmuls, since one bmm now covers all num_seqs.

    page_index_tables[s]: [num_blocks, INT32_ELEMS_PER_STICK], page at col 0.
    mask_tiles[i]: [1, block_size].
    """
    g = num_seqs * num_kv_heads
    num_heads = num_kv_heads * num_queries_per_kv
    q = query[:num_seqs].reshape(g, num_queries_per_kv, head_size).unsqueeze(2)

    state = None
    for i in range(num_blocks):
        ks, vs = [], []
        for s in range(num_seqs):
            page_idx = page_index_tables[s][i, 0:1]
            # [1, block, KV, D] -> [KV, block, D]; cat along dim 0 then yields
            # (S,KV)-major order, which is the merge the gather cannot express.
            ks.append(k_pages.index_select(0, page_idx).squeeze(0).permute(1, 0, 2))
            vs.append(v_pages.index_select(0, page_idx).squeeze(0).permute(1, 0, 2))
        k_page = torch.cat(ks, dim=0).unsqueeze(1)
        v_page = torch.cat(vs, dim=0).unsqueeze(1)
        scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale + mask_tiles[i]
        state = _online_init(scores, v_page) if i == 0 else _online_step(state, scores, v_page)

    tile_max, tile_sum, tile_out = state
    attn = (tile_out / tile_sum).reshape(num_seqs, num_kv_heads, num_queries_per_kv, head_size)
    return attn.reshape(num_seqs, num_heads, head_size)


# ---------------------------------------------------------------------------
# Candidate 5: batch over KV only, all sequences on the M axis
# ---------------------------------------------------------------------------


def _seq_on_m_kernel(
    query_kv_major, k_pages, v_pages, block_ids, mask_diag, scale,
    num_seqs, num_blocks, num_kv_heads, num_queries_per_kv, block_size, head_size,
):
    """3-D bmm with batch=KV; every sequence's query rows share one K tile.

    Merging (S,block) IS free (they are adjacent and contiguous in the gather
    output), so this needs no copy in the block loop. The price is arithmetic:
    every sequence's queries multiply every sequence's keys and the
    block-diagonal mask throws away the (S-1)/S that do not belong, so the
    matmuls do num_seqs x the necessary MACs. Worth it only if the kernel is
    limited by how operand tiles are fetched rather than by arithmetic.

    query_kv_major: [KV, S*QPKV, D], built once per call outside the loop.
    mask_diag[i]:   [1, S*QPKV, S*block] additive, -inf off the diagonal blocks.
    """
    num_heads = num_kv_heads * num_queries_per_kv

    state = None
    for i in range(num_blocks):
        page_idx = block_ids[i, 0:num_seqs]
        # [S, block, KV, D] -> [S*block, KV, D] is a free merge, then a view
        # permute puts KV on the batch axis.
        k_h = k_pages.index_select(0, page_idx).flatten(0, 1).permute(1, 0, 2)
        v_h = v_pages.index_select(0, page_idx).flatten(0, 1).permute(1, 0, 2)
        scores = torch.matmul(query_kv_major, k_h.transpose(-2, -1)) * scale + mask_diag[i]
        state = _online_init(scores, v_h) if i == 0 else _online_step(state, scores, v_h)

    tile_max, tile_sum, tile_out = state
    # [KV, S*QPKV, D] -> [S, num_heads, D]
    attn = (tile_out / tile_sum).reshape(num_kv_heads, num_seqs, num_queries_per_kv, head_size)
    return attn.permute(1, 0, 2, 3).reshape(num_seqs, num_heads, head_size)


# ---------------------------------------------------------------------------
# Reference: unrolled per-sequence bodies (the bar to beat)
# ---------------------------------------------------------------------------


def _unrolled_decode_kernel(
    query, query_row_indices, k_pages, v_pages, page_index_tables, mask_tiles,
    scale, num_seqs, num_blocks, num_heads, num_kv_heads, head_size,
):
    outs = []
    for s in range(num_seqs):
        outs.append(
            _page_attn_kernel(
                query, query_row_indices[s], k_pages, v_pages, page_index_tables[s],
                mask_tiles, scale, num_blocks, 1, num_heads, num_kv_heads, head_size,
            )
        )
    return torch.cat(outs, dim=0)


class LossyCounter(logging.Handler):
    PATTERNS = ("lossy work-division", "RetileWarning", "re-tiling")

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.hits = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if any(p in msg for p in self.PATTERNS):
            self.hits.append(msg)

    def reset(self):
        self.hits = []


def build_inputs(a):
    register_convert_op()

    num_heads = a.num_kv_heads * a.num_queries_per_kv
    dtype = torch.float16
    num_pages = a.num_seqs * a.num_blocks + 1

    torch.manual_seed(0)
    # Allocated as initialize_kv_cache_tensors does: host zeros then .to(device,
    # device_layout=slot-major). A plain convert() leaves the default tiled layout,
    # which spreads the slot index across two device dims (torch-spyre#3705) and
    # makes the in-graph page gather slow for every variant.
    layout = slot_major_kv_layout(
        num_pages * a.block_size, a.num_kv_heads, a.head_size, dtype
    )
    k_pages = torch.zeros(
        num_pages, a.block_size, a.num_kv_heads, a.head_size, dtype=dtype
    ).to(DEV, device_layout=layout)
    v_pages = torch.zeros(
        num_pages, a.block_size, a.num_kv_heads, a.head_size, dtype=dtype
    ).to(DEV, device_layout=layout)

    # Production staging width (max_num_batched_tokens + 1), not num_seqs: every
    # variant either prefixes or gathers out of this buffer and the width shows.
    query = convert(
        torch.randn(a.staging_rows, num_heads, a.head_size).to(dtype), device=DEV
    )

    idx_len = _stick_aligned_len(1)
    row_idx = []
    for s in range(a.num_seqs):
        t = torch.zeros(idx_len, dtype=torch.int32)
        t[0] = s
        row_idx.append(convert(t, device=DEV))

    page_tables = []
    for s in range(a.num_seqs):
        t = torch.zeros(a.num_blocks, INT32_ELEMS_PER_STICK, dtype=torch.int32)
        for b in range(a.num_blocks):
            t[b, 0] = s * a.num_blocks + b
        page_tables.append(convert(t, device=DEV))

    # [1, block_size] additive tiles, as _page_attn_kernel takes.
    mask_tiles = [
        convert(torch.zeros(1, a.block_size, dtype=dtype), device=DEV)
        for _ in range(a.num_blocks)
    ]

    bid = torch.zeros(a.num_blocks, _stick_aligned_len(a.num_seqs), dtype=torch.int32)
    for b in range(a.num_blocks):
        for s in range(a.num_seqs):
            bid[b, s] = s * a.num_blocks + b
    block_ids = convert(bid, device=DEV)

    # chunked_gather: one index per chunk of `chunk` blocks, BLOCK-major so the
    # gather output reshapes to [chunk, S, block, KV, D] and the per-block slice
    # stays dim-0 contiguous (a strided host dim 0 is rejected by restickify).
    chunk = a.chunk
    chunk_pages = []
    if chunk > 0 and a.num_blocks % chunk == 0:
        for c in range(a.num_blocks // chunk):
            ids = [s * a.num_blocks + c * chunk + j
                   for j in range(chunk) for s in range(a.num_seqs)]
            chunk_pages.append(convert(torch.tensor(ids, dtype=torch.int32), device=DEV))

    # gather_all_once: every (seq, block) page id in one sequence-major index.
    all_pages = convert(
        torch.tensor(
            [s * a.num_blocks + b for s in range(a.num_seqs) for b in range(a.num_blocks)],
            dtype=torch.int32,
        ),
        device=DEV,
    )

    mask_by_block = convert(
        torch.zeros(
            a.num_blocks, a.num_seqs * a.num_kv_heads, 1, a.block_size, dtype=dtype
        ),
        device=DEV,
    )
    # Same values as mask_by_block, but one tensor per block so the kernel does a
    # trace-time list index instead of an in-graph slice.
    mask_list_4d = [
        convert(
            torch.zeros(a.num_seqs, a.num_kv_heads, 1, a.block_size, dtype=dtype),
            device=DEV,
        )
        for _ in range(a.num_blocks)
    ]

    # --- merged_slot_gather: flat cache view + per-block slot index -----------
    nkv, hs, bs, ns = a.num_kv_heads, a.head_size, a.block_size, a.num_seqs
    k_flat = k_pages.view(num_pages * bs * nkv, hs)
    v_flat = v_pages.view(num_pages * bs * nkv, hs)
    # base[kv, t] = t*KV + kv, the within-page part of the source row.
    base = (torch.arange(bs, dtype=torch.int32).view(1, bs) * nkv
            + torch.arange(nkv, dtype=torch.int32).view(nkv, 1))
    slot_index = []
    for b in range(a.num_blocks):
        pages = torch.tensor([s * a.num_blocks + b for s in range(ns)], dtype=torch.int32)
        idx = pages.view(ns, 1, 1) * (bs * nkv) + base.view(1, nkv, bs)
        slot_index.append(convert(idx.reshape(-1).contiguous(), device=DEV))

    # --- seq_on_m: KV-major query and a block-diagonal mask ------------------
    qpkv = a.num_queries_per_kv
    q_km = (
        q_host_kv_major := torch.zeros(nkv, ns * qpkv, hs, dtype=dtype)
    )
    # Filled from the same rows the other variants read, so all variants attend
    # to identical data: query[s, kv*qpkv + g, :] -> q_km[kv, s*qpkv + g, :].
    q_cpu = query.cpu()
    for s in range(ns):
        for kv in range(nkv):
            for gg in range(qpkv):
                q_km[kv, s * qpkv + gg] = q_cpu[s, kv * qpkv + gg]
    query_kv_major = convert(q_km, device=DEV)
    neg = torch.finfo(dtype).min
    diag = torch.full((1, ns * qpkv, ns * bs), neg, dtype=dtype)
    for s in range(ns):
        diag[0, s * qpkv : (s + 1) * qpkv, s * bs : (s + 1) * bs] = 0.0
    mask_diag = [convert(diag.clone(), device=DEV) for _ in range(a.num_blocks)]

    return dict(
        query=query, k_pages=k_pages, v_pages=v_pages, row_idx=row_idx,
        page_tables=page_tables, mask_tiles=mask_tiles, block_ids=block_ids,
        mask_by_block=mask_by_block, mask_list_4d=mask_list_4d, num_heads=num_heads,
        k_flat=k_flat, v_flat=v_flat, slot_index=slot_index,
        query_kv_major=query_kv_major, mask_diag=mask_diag, all_pages=all_pages,
        chunk_pages=chunk_pages,
    )


def make_callables(a, t):
    scale = a.head_size**-0.5
    nh, nkv, qpkv, bs, hs = t["num_heads"], a.num_kv_heads, a.num_queries_per_kv, a.block_size, a.head_size
    ns, nb = a.num_seqs, a.num_blocks

    per_seq_c = torch.compile(_page_attn_kernel, dynamic=False)
    batched_c = torch.compile(_batched_decode_kernel, dynamic=False)
    unrolled_c = torch.compile(_unrolled_decode_kernel, dynamic=False)
    masklist_c = torch.compile(_batched_masklist_kernel, dynamic=False)
    qgroup_c = torch.compile(_qgroup_loop_kernel, dynamic=False)
    gshared_c = torch.compile(_gather_shared_kernel, dynamic=False)
    merged_c = torch.compile(_merged_sk_kernel, dynamic=False)
    merged_cat_c = torch.compile(_merged_cat_kernel, dynamic=False)
    slotgather_c = torch.compile(_merged_slot_gather_kernel, dynamic=False)
    seqonm_c = torch.compile(_seq_on_m_kernel, dynamic=False)
    gatherall_c = torch.compile(_gather_all_once_kernel, dynamic=False)
    chunked_c = torch.compile(_chunked_gather_kernel, dynamic=False)
    ktile_c = torch.compile(_batched_ktile_kernel, dynamic=False)
    chunked_ktile_c = torch.compile(_chunked_ktile_kernel, dynamic=False)

    def per_seq():
        outs = []
        for s in range(ns):
            outs.append(per_seq_c(
                t["query"], t["row_idx"][s], t["k_pages"], t["v_pages"],
                t["page_tables"][s], t["mask_tiles"], scale, nb, 1, nh, nkv, hs,
            ))
        return outs[-1]

    def batched():
        return batched_c(
            t["query"], None, t["k_pages"], t["v_pages"], t["block_ids"],
            t["mask_by_block"], scale, ns, nb, nkv, qpkv, bs, hs,
        )

    def unrolled():
        return unrolled_c(
            t["query"], t["row_idx"], t["k_pages"], t["v_pages"], t["page_tables"],
            t["mask_tiles"], scale, ns, nb, nh, nkv, hs,
        )

    def batched_masklist():
        return masklist_c(
            t["query"], t["k_pages"], t["v_pages"], t["block_ids"],
            t["mask_list_4d"], scale, ns, nb, nkv, qpkv, bs, hs,
        )

    def qgroup_loop():
        return qgroup_c(
            t["query"], t["k_pages"], t["v_pages"], t["block_ids"],
            t["mask_list_4d"], scale, ns, nb, nkv, qpkv, bs, hs,
        )

    def gather_shared():
        return gshared_c(
            t["query"], t["k_pages"], t["v_pages"], t["block_ids"],
            t["mask_tiles"], scale, ns, nb, nkv, qpkv, bs, hs,
        )

    def merged_sk():
        return merged_c(
            t["query"], t["k_pages"], t["v_pages"], t["block_ids"],
            t["mask_tiles"], scale, ns, nb, nkv, qpkv, bs, hs,
        )

    def merged_cat():
        return merged_cat_c(
            t["query"], t["k_pages"], t["v_pages"], t["page_tables"],
            t["mask_tiles"], scale, ns, nb, nkv, qpkv, bs, hs,
        )

    def merged_slot_gather():
        return slotgather_c(
            t["query"], t["k_flat"], t["v_flat"], t["slot_index"],
            t["mask_tiles"], scale, ns, nb, nkv, qpkv, bs, hs,
        )

    def chunked_gather():
        return chunked_c(
            t["query"], t["k_pages"], t["v_pages"], t["chunk_pages"],
            t["mask_tiles"], scale, ns, nb, a.chunk, nkv, qpkv, bs, hs,
        )

    def chunked_ktile():
        return chunked_ktile_c(
            t["query"], t["k_pages"], t["v_pages"], t["chunk_pages"],
            t["mask_list_4d"], scale, ns, nb, a.chunk, a.kv_tile, nkv, qpkv, bs, hs,
        )

    def batched_ktile():
        return ktile_c(
            t["query"], t["k_pages"], t["v_pages"], t["block_ids"],
            t["mask_list_4d"], scale, ns, nb, a.kv_tile, nkv, qpkv, bs, hs,
        )

    def gather_all_once():
        return gatherall_c(
            t["query"], t["k_pages"], t["v_pages"], t["all_pages"],
            t["mask_tiles"], scale, ns, nb, nkv, qpkv, bs, hs,
        )

    def seq_on_m():
        return seqonm_c(
            t["query_kv_major"], t["k_pages"], t["v_pages"], t["block_ids"],
            t["mask_diag"], scale, ns, nb, nkv, qpkv, bs, hs,
        )

    return {
        "per_seq": per_seq, "batched": batched, "unrolled": unrolled,
        "batched_masklist": batched_masklist, "qgroup_loop": qgroup_loop,
        "gather_shared": gather_shared, "merged_sk": merged_sk,
        "merged_cat": merged_cat, "merged_slot_gather": merged_slot_gather,
        "seq_on_m": seq_on_m, "gather_all_once": gather_all_once,
        "chunked_gather": chunked_gather, "batched_ktile": batched_ktile,
        "chunked_ktile": chunked_ktile,
    }


def wall_ms(fn, iters):
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.spyre.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-seqs", type=int, default=4)
    p.add_argument("--num-blocks", type=int, default=16)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--num-queries-per-kv", type=int, default=4)
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--staging-rows", type=int, default=513)
    p.add_argument("--chunk", type=int, default=4,
                   help="chunked_gather: blocks gathered per index_select. Scratch is "
                        "num_seqs*chunk*block_size*KV*D, so this bounds the extra "
                        "footprint independently of max_model_len")
    p.add_argument("--kv-tile", type=int, default=64,
                   help="batched_ktile: slots of a block processed per matmul. Must be "
                        "a multiple of the fp16 stick (64) -- a sub-stick tile makes the "
                        "score tile's last dim stick-incompatible and fails to compile")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--variants",
                   default="per_seq,batched,unrolled,batched_masklist,"
                           "qgroup_loop,gather_shared,merged_sk")
    p.add_argument("--check", action="store_true",
                   help="compare each variant's output against per_seq")
    a = p.parse_args()

    kv_mb = a.num_blocks * a.block_size * a.num_kv_heads * a.head_size * 2 * 2 / 1e6
    print("shapes: num_seqs=%d num_blocks=%d block_size=%d kv_heads=%d "
          "queries_per_kv=%d head_size=%d"
          % (a.num_seqs, a.num_blocks, a.block_size, a.num_kv_heads,
             a.num_queries_per_kv, a.head_size))
    print("KV per layer: %.2f MB/seq, %.2f MB for %d seqs\n"
          % (kv_mb, kv_mb * a.num_seqs, a.num_seqs))

    counter = LossyCounter()
    logging.getLogger().addHandler(counter)
    logging.getLogger().setLevel(logging.DEBUG)

    t = build_inputs(a)
    fns = make_callables(a, t)

    rows = []
    for name in [v.strip() for v in a.variants.split(",") if v.strip()]:
        fn = fns[name]
        counter.reset()
        t0 = time.perf_counter()
        try:
            for _ in range(a.warmup):
                fn()
            torch.spyre.synchronize()
        except Exception as exc:
            print("  FAILED %-17s %s: %s" % (name, type(exc).__name__, str(exc)[:400]))
            continue
        compile_s = time.perf_counter() - t0
        lossy = len(counter.hits)
        w = wall_ms(fn, a.iters)
        rows.append((name, statistics.median(w), min(w), lossy, compile_s))
        print("  ran %-17s wall med %8.3f ms  lossy %3d  warmup %6.1f s"
              % (name, statistics.median(w), lossy, compile_s), flush=True)

    print("\n%-17s %11s %10s %7s %9s"
          % ("variant", "wall med ms", "wall min", "lossy", "warmup s"))
    for name, wmed, wmin, lossy, cs in rows:
        print("%-17s %11.3f %10.3f %7d %9.1f" % (name, wmed, wmin, lossy, cs))

    bar = next((r for r in rows if r[0] == "unrolled"), None)
    if bar:
        print("\nrelative to unrolled (wall median; <1.00 beats the bar):")
        for name, wmed, *_ in rows:
            print("  %-17s %6.2fx   -> %8.1f ms/step across %d layers"
                  % (name, wmed / bar[1], wmed * a.layers, a.layers))


if __name__ == "__main__":
    main()
