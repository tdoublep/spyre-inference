"""Microbenchmark: per-sequence vs batched vs unrolled decode attention.

Isolates why SPYRE_BATCHED_DECODE=1 loses to the per-sequence path even though
it cuts kernel launches 4:1. Drives the real kernels from spyre_attn, so it
tracks the production code rather than a paraphrase.

Variants
  per_seq   num_seqs sequential calls to _page_attn_kernel (what production does
            with SPYRE_BATCHED_DECODE=0)
  batched   one call to _batched_decode_kernel (SPYRE_BATCHED_DECODE=1)
  unrolled  one compiled graph containing the num_seqs per-sequence bodies:
            a single launch that keeps the per-sequence shapes

Defaults reproduce granite-3.3-8b decode at batch 4 / 2048 KV / block 128.

torch-spyre must be built with USE_SPYRE_PROFILER=1 (this repo pins "0"), or the
`dev ms/call` column is empty: device timings come from the Kineto
AIUActivityProfiler that build variable compiles in. See MISSION.md.

Run it through the wrapper, which sets the environment the numbers below were
taken with:

    bash experimental/run_microbench_batched_decode.sh

Knobs are forwarded, e.g.:

    bash experimental/run_microbench_batched_decode.sh --num-blocks 8 --block-size 256
    bash experimental/run_microbench_batched_decode.sh --variants per_seq,batched
    bash experimental/run_microbench_batched_decode.sh --show-warnings

STATUS
------
tpa-spyre-dev-2, granite shapes, LAYOUT_SOLVER=greedy, 2026-09-10. num_seqs=4,
block 128, 2048 KV:

    variant   dev ms/call  kernels  GB/s  lossy  correctness
    per_seq         2.374        4  14.1      0  ok
    batched         7.647        1   4.4     32  ok
    unrolled        2.267        2  14.8      0  ok
    kvm4d           7.105        1   4.7     32  ok
    flatbc          1.344        1  25.0      0  ok
    flat3d             --       --    --     --  Incompatible host_size and dim_order

flatbc is a single dispatch that beats the per-sequence bar by 1.77x here. It is
not a drop-in: see "What governs flatbc" below.

Fidelity rules (unchanged, and still load-bearing): k/v_pages allocated as
initialize_kv_cache_tensors does (host zeros then .to(device,
device_layout=slot_major_kv_layout(...))), and `query` at the production staging
width of max_num_batched_tokens + 1 = 513 rows. Break either and per_seq
collapses to 1.7 GB/s and the ranking inverts. The kvm/flat variants allocate
their own pages in a different page order, which is the point of them, through
the same host-zeros-then-device_layout path.

Every variant is checked against an fp32 CPU reference on randn pages
(--no-check to skip). The timed pages are all zeros, which makes every kernel
return zeros: without the check a wrong kernel ranks first, and that is exactly
how flatbc's >32-lane bug showed up.

What the batched regression actually is
--------------------------------------
    num_seqs   per_seq   batched   batched GB/s
           1     0.593     0.450           18.7
           2     1.185     6.885            2.4
           4     2.374     7.647            4.4
           8     4.766     9.378            7.3

per_seq is linear through the origin at ~0.59 ms/seq. batched is ~6.6 ms fixed +
~0.33 ms/seq: its marginal cost per sequence is *better* than per_seq's, so
batching does share work as intended. What kills it is a penalty that switches
on at num_seqs >= 2 and is then nearly independent of batch size (~412 us per
block-loop iteration). Today's batched kernel would not overtake per_seq until
batch ~25.

The penalty is the bmm form: a 4-D bmm whose two batch axes both vary.

  * batched at num_seqs=1 has batch axes (1, KV), one effective axis, and is the
    fastest variant at that shape (0.450 vs 0.593).
  * kvm4d keeps batch axes (num_seqs, KV) over KV-major pages with the permute
    gone entirely, and is still 7.105 -- the permute is worth 7%, not 3x.
  * per_seq has batch axes (KV, queries_per_kv) but K/V is size 1 on the second,
    so lower_bmm reads it through a stride-0 broadcast: the stationary operand
    effectively has one batch axis.
  * flatbc merges (num_seqs, KV) into one flat batch axis and restores per_seq's
    M=1 broadcast form. 5.7x faster than batched.

That matches the KG3 weight-stationary dataflow both attention matmuls lower to
(see the knowledgebase page on attention on Spyre): the K/V tile is held
stationary in the 64KB per-core XRF while Q streams through it, so one flat lane
per core keeps each core's stationary tile distinct and reads KV once.

What governs flatbc's speed: lanes per bmm group
-----------------------------------------------
flatbc's parallelism is lanes = num_seqs * num_kv_heads against the 32 cores.
Hold lanes at 32 and move the split between sequences and KV heads, and flatbc
does not care while per_seq does:

    num_seqs  kv_heads  lanes   per_seq   flatbc
           1         8      8     0.593    1.213
           2         8     16     1.185    2.552
           2        16     32     4.214    1.377
           4         8     32     2.371    1.342
           8         4     32     3.257    1.344
           8         8     64     4.766    5.743

At every lanes=32 point flatbc is ~1.35 ms at 25.0 GB/s, so speed is a function
of lanes per bmm group and not of batch size; below 32 it under-fills the cores
and above 32 it falls off. Nothing forces one group, though: flatchunk cuts the
lane axis into fixed groups of 32 inside the same dispatch, which restores the
rate at any batch and removes num_kv_heads and num_seqs from the question of
whether the kernel is usable at all:

    num_seqs=8, kv_heads=8, lanes=64      dev ms/call  kernels   GB/s
    per_seq                                     4.753        8   14.1
    batched                                     9.316        1    7.2
    flatbc  (one group of 64)                   5.743        1   11.7
    flatchunk (two groups of 32)                2.651        2   25.3

So the merged flat batch axis is worth 1.79x over per_seq at batch 8 once the
groups are sized to the cores, and 25.3 GB/s is the same rate the lanes=32
single-group case gets.

Correctness limit: total lanes > 32, and it blocks all of this
-------------------------------------------------------------
Above 32 lanes the flat variants return the WRONG ANSWER -- rel l2 ~0.71-0.80
against the CPU reference where a correct variant sits at 0.004. Two runs
separate lanes from the folded cache's row count (num_pages * num_kv_heads,
the gather's index range), which is the other thing that grows here:

    num_seqs  kv_heads  num_blocks  lanes  cache rows  correct?
           4         8          32     32        1032  ok, rel l2 0.0046
           8         8           8     64         520  WRONG, rel l2 0.7140

520 and 1032 rows each appear on both sides, so the index range is not it: the
trigger is lanes = num_seqs * num_kv_heads exceeding the 32 cores. Every
lanes <= 32 point measured is correct (at 136, 264, 516, 520, 528 and 1032 rows)
and every lanes = 64 point is wrong (at 520, 1032 and 1040 rows), with the
failures agreeing to four figures in both time and error across different
(num_seqs, kv_heads).

Chunking the bmm does NOT avoid it. flatchunk runs groups of exactly 32 lanes --
the width that is correct on its own -- and is still wrong whenever the total is
64, so what matters is the lane extent present in the graph rather than the width
of any one matmul. That is why flatchunk buys the rate but not usability, and it
is the single thing blocking the flat form. It wants a standalone repro filed
against torch-spyre, and note it is a numerical failure, distinct from the
gather-per-core-view saturation PR #783 documents in
scripts/probes/repro_gather_view_width.py.

Relation to PR #783
-------------------
PR #783 (SPYRE_LX_KV_LAYOUT) already folds the cache on (page, kv_head), which
is the same page order the flat variants here use, so these numbers are evidence
for that layout rather than a separate proposal. Two things it reports line up:
its note that the batched query form "makes inductor clone the page out to an
axis it does not have" matches flat3d failing to compile where only the M=1
broadcast form reaches the merged axis, and its SPYRE_ATTN_MAX_CORES matches the
under-fill measured below 32 lanes. It needs torch-spyre #4153; the runs here
are on the #4347 tip, which descends from #4153, which is the likely reason the
flat form reaches 25 GB/s at the full 32 cores without capping them.

Leads closed, negatively
------------------------
  * LAYOUT_SOLVER (lead 1). cpsat does not help batched (7.761 vs 7.647) and
    makes per_seq 1.9x worse (4.465 vs 2.374). greedy was the right default.
  * Work-division fallbacks. Not causal. Cleanest disproof is
    --num-queries-per-kv 1, where per_seq emits 32 fallbacks and batched emits
    0 and per_seq is 3.6x faster (1.851 vs 6.710). The warning also fires for
    any contraction-axis split by construction: the "output" pass in
    finalize_work_division_for_scheduler labels a reduction symbol absent while
    splits_by_index_coeff still transports it through the reduction dict.
  * The permute (lead 4). kvm4d prices it at 7%.
  * Block-loop trip count (lead 5). --num-blocks 8 --block-size 256 makes
    everything slower (per_seq 4.210, flatbc 4.139, both ~8 GB/s) and erases
    flatbc's margin; block 128 is the better operating point. At block 256 a
    [256, 128] fp16 K tile is exactly the 64KB XRF capacity.
  * Folding (num_seqs, KV) into one bmm batch axis by reshaping the gathered
    pages still does not compile on token-major pages, and flat3d shows the
    merged 3-D form (M=queries_per_kv) does not compile on KV-major pages
    either: RuntimeError: Incompatible host_size and dim_order. Only the M=1
    broadcast form (flatbc) reaches the merged axis.

Cost of the KV-major page order flatbc needs
--------------------------------------------
flatbc needs [num_pages, KV, block_size, head_size] pages so the gather
addresses num_seqs*KV independent [block_size, head_size] tiles. --store prices
the write side, since a token's KV heads stop being contiguous and one
index_copy_ of KV*head_size per token becomes KV copies of head_size:

    KV cache store          token-major   KV-major   ratio
    decode width, 4 tokens      0.0041      0.0046    1.13x
    prefill width, 512 tokens   0.0507      0.0707    1.40x

That is +0.0005 ms per layer at decode against flatbc's -1.040 ms, so the store
is not what decides this. The read side is what costs:

    variant         dev ms/call   GB/s   page order   bmm form
    per_seq               2.375   14.1   token-major  M=1, K/V stride-0 broadcast
    per_seq_kvm           3.089   10.9   KV-major     M=1, K/V stride-0 broadcast
    per_seq_kvm3d         3.786    8.9   KV-major     3-D, M=queries_per_kv

Reshaping the per-sequence kernel to the form KV-major pages ought to suit -- a
plain 3-D bmm with the queries as M, no permute and no broadcast -- makes it
worse, not better, so per_seq's M=1 broadcast shape is genuinely its best form
and the 1.30x is the price of the page order rather than a fixable mismatch.
The page order charges every path that still gathers one page per sequence,
prefill included, so it has to be decided globally rather than per kernel.

Next
----
  * Reduce the lanes > 32 wrong answer to a standalone repro for torch-spyre.
    Nothing here can land until it is fixed: granite's kv_heads=8 puts the limit
    at num_seqs = 4, and chunking the bmm does not work around it.
  * Once it is fixed, chunk the lane axis rather than sizing lanes to the cores:
    flatchunk already shows the rate holds at batch 8 (2.651 ms, 25.3 GB/s
    against per_seq's 4.753) and collapses to flatbc at one group.
  * Decide the page order globally, not per kernel: the flat form wants it,
    _page_attn_kernel is 1.30x worse under it, and PR #783 already carries it
    behind a flag.
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
    _reshape_and_cache_kernel,
    _page_attn_kernel,
    _stick_aligned_len,
    slot_major_kv_layout,
)

DEV = torch.device("spyre")


def _unrolled_decode_kernel(
    query,
    query_row_indices,
    k_pages,
    v_pages,
    page_index_tables,
    mask_tiles,
    scale,
    num_seqs,
    num_blocks,
    num_heads,
    num_kv_heads,
    head_size,
):
    """One graph, num_seqs per-sequence bodies. Static ints so Dynamo unrolls."""
    outs = []
    for s in range(num_seqs):
        outs.append(
            _page_attn_kernel(
                query,
                query_row_indices[s],
                k_pages,
                v_pages,
                page_index_tables[s],
                mask_tiles,
                scale,
                num_blocks,
                1,
                num_heads,
                num_kv_heads,
                head_size,
            )
        )
    return torch.cat(outs, dim=0)


def _kvmajor_decode_kernel(
    query,
    k_pages,
    v_pages,
    block_ids,
    mask_by_block,
    scale,
    num_seqs,
    num_blocks,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
    merge_batch_axes,
):
    """Batched decode over KV-major pages: [num_pages, KV, block_size, head_size].

    Same math and same one-dispatch shape as _batched_decode_kernel; the page
    axis order is the only change. With KV inside the page the gather already
    lands head-major, so the permute(0, 2, 1, 3) disappears, and (num_seqs, KV)
    become adjacent and merge into a single bmm batch axis with a legal view --
    the merge that fails on the token-major cache ("Incompatible host_size and
    dim_order"). merge_batch_axes picks 3-D (merged) or 4-D (two batch axes) so
    the permute and the merge can be priced separately.
    """
    S, KV, QPK, D = num_seqs, num_kv_heads, num_queries_per_kv, head_size
    q = query[:S].reshape(S, KV, QPK, D)
    if merge_batch_axes:
        q = q.reshape(S * KV, QPK, D)

    tile_max = None
    tile_sum = None
    tile_output = None

    for i in range(num_blocks):
        page_idx = block_ids[i, 0:S]
        k_page = k_pages.index_select(0, page_idx)
        v_page = v_pages.index_select(0, page_idx)
        if merge_batch_axes:
            k_page = k_page.reshape(S * KV, block_size, D)
            v_page = v_page.reshape(S * KV, block_size, D)
            mask_tile = mask_by_block[i].reshape(S * KV, 1, block_size)
        else:
            mask_tile = mask_by_block[i].reshape(S, KV, 1, block_size)

        scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = torch.matmul(tile_probs, v_page)
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
            tile_output += torch.matmul(tile_probs, v_page)
            tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
    return (tile_output / tile_sum).reshape(S, KV * QPK, D)


def _flat_decode_kernel(
    query,
    k_pages,
    v_pages,
    flat_ids,
    mask_by_block,
    scale,
    num_seqs,
    num_blocks,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
    form,
    out=None,
):
    """Batched decode with (num_seqs, KV) pre-merged into one bmm batch axis.

    The cache is [num_pages * KV, block_size, head_size]: one contiguous
    [block_size, head_size] tile per (page, kv head), which is the layout other
    paged backends already use. That buys the whole point of this variant --
    the gather indexes num_seqs*KV independent tiles directly, so there is no
    permute and no axis merge, and the bmm batch axis is a single flat extent
    the planner can split one tile per core. _batched_decode_kernel instead
    hands it two batch axes it does not split on num_seqs, which is what makes
    that kernel re-stream KV per sequence.

    flat_ids: [num_blocks, stick-padded num_seqs*KV] int32, row i column
    s*KV + h holding block i's tile index for sequence s, kv head h.
    form is "3d" (M=num_queries_per_kv) or "bcast" (M=1, K/V broadcast along
    the query axis, which is byte for byte the shape per_seq feeds lower_bmm).
    """
    S, KV, QPK, D = num_seqs, num_kv_heads, num_queries_per_kv, head_size
    lanes = S * KV
    q = query[:S].reshape(S, KV, QPK, D).reshape(lanes, QPK, D)
    if form == "bcast":
        q = q.unsqueeze(2)

    tile_max = None
    tile_sum = None
    tile_output = None

    for i in range(num_blocks):
        idx = flat_ids[i, 0:lanes]
        k_tile = k_pages.index_select(0, idx)
        v_tile = v_pages.index_select(0, idx)
        if form == "bcast":
            k_tile = k_tile.unsqueeze(1)
            v_tile = v_tile.unsqueeze(1)
            mask_tile = mask_by_block[i].reshape(lanes, 1, 1, block_size)
        else:
            mask_tile = mask_by_block[i].reshape(lanes, 1, block_size)

        scores = torch.matmul(q, k_tile.transpose(-2, -1)) * scale
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = torch.matmul(tile_probs, v_tile)
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
            tile_output += torch.matmul(tile_probs, v_tile)
            tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
    attn = (tile_output / tile_sum).reshape(S, KV * QPK, D)
    if out is not None:
        # Matches _run_batched_decode_dispatch: a contiguous prefix copy into
        # vLLM's own output buffer at offset 0, not the 513-row staging buffer.
        out[:num_seqs].copy_(attn)
        return out
    return attn


def _per_seq_kvm_kernel(
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
    out=None,
):
    """_page_attn_kernel over KV-major pages, to price the layout change itself.

    flatbc needs [num_pages, KV, block_size, head_size] pages, so the question
    for a migration is whether the path that works today gets worse under that
    order. Identical to _page_attn_kernel except the gather already lands
    head-major, so squeeze/unsqueeze replaces squeeze/permute/unsqueeze. The
    per-sequence page index table is unchanged -- the page axis is still dim 0.
    """
    num_queries_per_kv = num_heads // num_kv_heads
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
        k_page_4d = k_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
        v_page_4d = v_pages.index_select(0, page_idx).squeeze(0).unsqueeze(1)
        mask_tile = mask_tiles[i]

        scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = torch.matmul(tile_probs, v_page_4d)
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
            tile_output += torch.matmul(tile_probs, v_page_4d)
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


def _per_seq_fold3d_kernel(
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
    out=None,
):
    """The per-sequence kernel exactly as PR 855 ships it, over FLAT folded pages.

    The variant the microbench was missing. per_seq_kvm keeps the page axis at dim 0
    of a 4-D [pages, KV, block, D] cache and gathers ONE entry per block; PR 855's
    initialize_kv_cache_tensors materialises the cache flattened to
    [pages * KV, block, D], so the kernel gathers num_kv_heads separate rows per
    block. Same bytes, same device layout, num_kv_heads times the index entries --
    and under the 32-entry stick rule neither 1 nor num_kv_heads entries is
    core-splittable, so nothing compensates.

    page_index_table row i holds block i's `page * num_kv_heads + h` in columns
    0..num_kv_heads-1.
    """
    num_queries_per_kv = num_heads // num_kv_heads
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
        page_idx = page_index_table[i, 0:num_kv_heads]
        k_page_4d = k_pages.index_select(0, page_idx).unsqueeze(1)
        v_page_4d = v_pages.index_select(0, page_idx).unsqueeze(1)
        mask_tile = mask_tiles[i]

        scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * scale
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = torch.matmul(tile_probs, v_page_4d)
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
            tile_output += torch.matmul(tile_probs, v_page_4d)
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


def _per_seq_kvm3d_kernel(
    query,
    query_row_index,
    k_pages,
    v_pages,
    page_index_table,
    mask_tiles,
    scale,
    num_blocks,
    num_heads,
    num_kv_heads,
    head_size,
    out=None,
):
    """Per-sequence decode over KV-major pages using a 3-D bmm, M=queries_per_kv.

    per_seq's M=1 plus stride-0 broadcast on K exists because token-major pages
    force the KV axis inside the page: reaching a contiguous [block, head_size]
    tile needs the permute, so the query axis has to carry the broadcast. Under
    KV-major the gather already yields [KV, block_size, head_size], so the
    natural form is a plain 3-D bmm with the queries as M and no broadcast --
    the shape flat3d wanted but could not have, since that one needed an axis
    merge and this one does not. Decode only (padded_query_len == 1).
    """
    num_queries_per_kv = num_heads // num_kv_heads
    q_rows = query.index_select(0, query_row_index[:1])
    q = q_rows.reshape(num_kv_heads, num_queries_per_kv, head_size)

    tile_max = None
    tile_sum = None
    tile_output = None

    for i in range(num_blocks):
        page_idx = page_index_table[i, 0:1]
        k_tile = k_pages.index_select(0, page_idx).squeeze(0)
        v_tile = v_pages.index_select(0, page_idx).squeeze(0)
        mask_tile = mask_tiles[i]

        scores = torch.matmul(q, k_tile.transpose(-2, -1)) * scale
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = torch.matmul(tile_probs, v_tile)
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
            tile_output += torch.matmul(tile_probs, v_tile)
            tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
    attn = (tile_output / tile_sum).reshape(1, num_heads, head_size)
    if out is not None:
        out.index_copy_(0, query_row_index[:1], attn)
        return out
    return attn


def _flat_chunked_decode_kernel(
    query,
    k_pages,
    v_pages,
    flat_ids_chunks,
    mask_chunks,
    scale,
    num_seqs,
    num_blocks,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
    lanes_per_chunk,
):
    """flatbc with the lane axis cut into fixed groups of lanes_per_chunk.

    flatbc puts all num_seqs*KV lanes in one bmm batch axis, which ties its lane
    count to the model (num_kv_heads) and the batch, and above 32 lanes the
    backend miscomputes. Nothing requires one group: this keeps the single
    dispatch and runs ceil(lanes / lanes_per_chunk) groups of exactly the shape
    that works, so num_seqs and num_kv_heads stop deciding whether the kernel is
    usable. Per-chunk page tables and masks are passed as lists, the way the
    per-sequence path already passes mask tiles, so no slice of a shared tensor
    lands at a non-stick-aligned offset.
    """
    KV, QPK, D = num_kv_heads, num_queries_per_kv, head_size
    L = lanes_per_chunk
    num_chunks = (num_seqs * KV) // L
    q_all = query[:num_seqs].reshape(num_chunks, L * QPK, D)

    chunk_outs = []
    for c in range(num_chunks):
        q = q_all[c].reshape(L, QPK, 1, D)
        ids_c = flat_ids_chunks[c]
        mask_c = mask_chunks[c]

        tile_max = None
        tile_sum = None
        tile_output = None

        for i in range(num_blocks):
            idx = ids_c[i, 0:L]
            k_tile = k_pages.index_select(0, idx).unsqueeze(1)
            v_tile = v_pages.index_select(0, idx).unsqueeze(1)
            mask_tile = mask_c[i].reshape(L, 1, 1, block_size)

            scores = torch.matmul(q, k_tile.transpose(-2, -1)) * scale
            scores = scores + mask_tile
            scores_max = torch.amax(scores, dim=-1, keepdim=True)

            if i == 0:
                tile_max = scores_max
                tile_probs = torch.exp(scores - tile_max)
                tile_output = torch.matmul(tile_probs, v_tile)
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
                tile_output += torch.matmul(tile_probs, v_tile)
                tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
                tile_max = new_max

        assert tile_output is not None and tile_sum is not None
        chunk_outs.append((tile_output / tile_sum).reshape(L * QPK, D))

    return torch.cat(chunk_outs, dim=0).reshape(num_seqs, KV * QPK, D)


class LossyCounter(logging.Handler):
    """Collects torch-spyre work-division records emitted while compiling.

    `hits` are the lossy-transport fallbacks. `splits` are the planner's chosen
    per-op work divisions, which is what actually decides how many cores run an
    op and along which axes -- and therefore whether each core reloads the same
    KV tile. Both only see an actual compile: a warm inductor cache reports
    nothing.
    """

    PATTERNS = ("lossy work-division", "RetileWarning", "re-tiling")

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.hits = []
        self.splits = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if any(p in msg for p in self.PATTERNS):
            self.hits.append(msg)
        if "work_division" in msg and "cores=" in msg:
            self.splits.append(msg)

    def reset(self):
        self.hits = []
        self.splits = []


def summarize_splits(msgs):
    """Group work-division records by (cores, iteration space, chosen splits)."""
    import collections
    import re

    def grab(pat, m):
        hit = re.search(pat, m)
        return hit.group(1) if hit else "?"

    counts = collections.Counter()
    for m in msgs:
        counts[(
            grab(r"cores=(\d+)", m),
            grab(r"iteration_space=(\{[^}]*\})", m),
            grab(r"min_splits=(\{[^}]*\})", m),
        )] += 1
    return counts


def build_inputs(a):
    """Device tensors matching each kernel's documented shape contract."""
    # convert() routes through torch.ops.vllm.spyre_convert, which only exists
    # once this runs; the runner normally does it during platform setup.
    register_convert_op()

    num_heads = a.num_kv_heads * a.num_queries_per_kv
    dtype = torch.float16
    # One distinct page per (seq, block) so no sequence shares KV with another.
    num_pages = a.num_seqs * a.num_blocks + 1

    torch.manual_seed(0)
    # Allocated exactly as TorchSpyreModelRunner.initialize_kv_cache_tensors does:
    # host zeros then .to(device, device_layout=slot-major). The default tiled
    # layout spreads the slot index across two device dims (torch-spyre#3705),
    # which changes what the in-graph page gather costs -- so a plain convert()
    # here does NOT measure the same thing production does.
    layout = slot_major_kv_layout(
        num_pages * a.block_size, a.num_kv_heads, a.head_size, dtype
    )
    page_shape = (num_pages, a.block_size, a.num_kv_heads, a.head_size)
    k_pages = torch.zeros(page_shape, dtype=dtype).to(DEV, device_layout=layout)
    v_pages = torch.zeros(page_shape, dtype=dtype).to(DEV, device_layout=layout)

    # KV-major pages for the kvmajor variants: the same outermost-major device
    # layout, with KV moved inside the page. Only the page axis order differs.
    layout_kvm = slot_major_kv_layout(
        num_pages * a.num_kv_heads, a.block_size, a.head_size, dtype
    )
    kvm_shape = (num_pages, a.num_kv_heads, a.block_size, a.head_size)
    k_pages_kvm = torch.zeros(kvm_shape, dtype=dtype).to(DEV, device_layout=layout_kvm)
    v_pages_kvm = torch.zeros(kvm_shape, dtype=dtype).to(DEV, device_layout=layout_kvm)

    # Same bytes as the kvm pages, addressed as one tile per (page, kv head).
    flat_shape = (num_pages * a.num_kv_heads, a.block_size, a.head_size)
    k_pages_flat = torch.zeros(flat_shape, dtype=dtype).to(DEV, device_layout=layout_kvm)
    v_pages_flat = torch.zeros(flat_shape, dtype=dtype).to(DEV, device_layout=layout_kvm)

    # flat_ids[i, s * KV + h] = tile index of block i, sequence s, kv head h.
    lanes = a.num_seqs * a.num_kv_heads
    fid = torch.zeros(a.num_blocks, _stick_aligned_len(lanes), dtype=torch.int32)
    for b in range(a.num_blocks):
        for sq in range(a.num_seqs):
            for h in range(a.num_kv_heads):
                fid[b, sq * a.num_kv_heads + h] = (sq * a.num_blocks + b) * a.num_kv_heads + h
    flat_ids = convert(fid, device=DEV)

    # Per-chunk page tables and masks for the chunked variant: each chunk owns a
    # stick-wide table starting at column 0.
    lanes_per_chunk = min(INT32_ELEMS_PER_STICK, lanes)
    num_chunks = max(lanes // lanes_per_chunk, 1)
    flat_ids_chunks = [
        convert(
            fid[:, c * lanes_per_chunk : (c + 1) * lanes_per_chunk].contiguous(), device=DEV
        )
        for c in range(num_chunks)
    ]
    mask_chunks = [
        convert(
            torch.zeros(a.num_blocks, lanes_per_chunk, 1, a.block_size, dtype=dtype),
            device=DEV,
        )
        for _ in range(num_chunks)
    ]

    # Decode: one query row per sequence in rows 0..num_seqs-1, but the buffer is
    # the production staging buffer width (max_num_batched_tokens + 1), not
    # num_seqs: the batched kernel takes a query[:num_seqs] prefix of it and the
    # per-sequence kernel gathers out of it, and both cost differently against a
    # 513-row source than against a snug one.
    q_host = torch.randn(a.staging_rows, num_heads, a.head_size).to(dtype)
    query = convert(q_host, device=DEV)
    out_buf = convert(
        torch.zeros(a.staging_rows, num_heads, a.head_size).to(dtype), device=DEV
    )

    # --- per-sequence / unrolled inputs -------------------------------------
    # query_row_index: stick-aligned int32, first padded_query_len entries are
    # this sequence's absolute query rows (padded_query_len == 1 for decode).
    q_len = a.query_len
    idx_len = _stick_aligned_len(q_len)
    row_idx = []
    for s in range(a.num_seqs):
        t = torch.zeros(idx_len, dtype=torch.int32)
        # Absolute query rows, contiguous per sequence, exactly as
        # _build_query_row_tables lays them out from query_start_loc.
        t[:q_len] = torch.arange(s * q_len, (s + 1) * q_len, dtype=torch.int32)
        row_idx.append(convert(t, device=DEV))

    # page_index_table: [num_blocks, INT32_ELEMS_PER_STICK], page index at col 0.
    page_tables = []
    for s in range(a.num_seqs):
        t = torch.zeros(a.num_blocks, INT32_ELEMS_PER_STICK, dtype=torch.int32)
        for b in range(a.num_blocks):
            t[b, 0] = s * a.num_blocks + b
        page_tables.append(convert(t, device=DEV))

    # Folded per-sequence page tables: row b holds block b's
    # `page * num_kv_heads + h` in columns 0..num_kv_heads-1, exactly as PR 855's
    # metadata builder fills them for _page_attn_kernel over the flat cache.
    heads_ar = torch.arange(a.num_kv_heads, dtype=torch.int32)
    page_tables_fold = []
    for s in range(a.num_seqs):
        tf = torch.zeros(a.num_blocks, INT32_ELEMS_PER_STICK, dtype=torch.int32)
        for b in range(a.num_blocks):
            tf[b, 0 : a.num_kv_heads] = (s * a.num_blocks + b) * a.num_kv_heads + heads_ar
        page_tables_fold.append(convert(tf, device=DEV))

    # One allocation serving both kernels: the flat batched kernel wants
    # [rows, block, D], the per-sequence kernel wants the 4-D form. Merging
    # (page, kv head) is the direction known to lower (kv_slot_views does the
    # same); splitting device dim 0 is not, so allocate 4-D and merge down.
    # Built outside any graph -- Inductor cannot lower through a view of a
    # Spyre-layout tensor created inside one.
    try:
        k_kvm_flatview = k_pages_kvm.view(flat_shape)
        v_kvm_flatview = v_pages_kvm.view(flat_shape)
    except Exception as exc:  # noqa: BLE001 - reported as an unavailable variant
        print(f"note: 4-D -> flat view unavailable ({exc}); flat_from4dview disabled")
        k_kvm_flatview = v_kvm_flatview = None

    # mask_tiles: [padded_query_len, block_size] additive, zeros = nothing masked.
    mask_tiles = [
        convert(torch.zeros(q_len, a.block_size, dtype=dtype), device=DEV)
        for _ in range(a.num_blocks)
    ]

    # --- batched inputs ------------------------------------------------------
    # block_ids: [num_blocks, stick-padded num_seqs], row i = block i's page per seq.
    bid = torch.zeros(a.num_blocks, _stick_aligned_len(a.num_seqs), dtype=torch.int32)
    for b in range(a.num_blocks):
        for s in range(a.num_seqs):
            bid[b, s] = s * a.num_blocks + b
    block_ids = convert(bid, device=DEV)

    # mask_by_block: [num_blocks, num_seqs * num_kv_heads, 1, block_size],
    # pre-broadcast across KV heads by the production builder.
    mask_by_block = convert(
        torch.zeros(
            a.num_blocks, a.num_seqs * a.num_kv_heads, 1, a.block_size, dtype=dtype
        ),
        device=DEV,
    )

    return dict(
        query=query,
        q_host=q_host,
        out_buf=out_buf,
        k_pages=k_pages,
        v_pages=v_pages,
        k_pages_kvm=k_pages_kvm,
        v_pages_kvm=v_pages_kvm,
        k_pages_flat=k_pages_flat,
        v_pages_flat=v_pages_flat,
        flat_ids=flat_ids,
        flat_ids_chunks=flat_ids_chunks,
        mask_chunks=mask_chunks,
        lanes_per_chunk=lanes_per_chunk,
        flat_shape=flat_shape,
        layout=layout,
        layout_kvm=layout_kvm,
        page_shape=page_shape,
        num_pages=num_pages,
        row_idx=row_idx,
        page_tables=page_tables,
        page_tables_fold=page_tables_fold,
        k_kvm_flatview=k_kvm_flatview,
        v_kvm_flatview=v_kvm_flatview,
        mask_tiles=mask_tiles,
        block_ids=block_ids,
        mask_by_block=mask_by_block,
        num_heads=num_heads,
    )


def cpu_reference(a, q_host, k_host, v_host, num_heads):
    """fp32 CPU attention over each sequence's num_blocks pages.

    Head h reads KV head h // num_queries_per_kv, which is how every variant
    splits num_heads into (num_kv_heads, num_queries_per_kv).
    """
    scale = a.head_size**-0.5
    out = torch.zeros(a.num_seqs, num_heads, a.head_size, dtype=torch.float32)
    for s in range(a.num_seqs):
        pages = [s * a.num_blocks + b for b in range(a.num_blocks)]
        k = torch.cat([k_host[pg] for pg in pages], 0).float()
        v = torch.cat([v_host[pg] for pg in pages], 0).float()
        for h in range(num_heads):
            kv = h // a.num_queries_per_kv
            probs = torch.softmax((k[:, kv] @ q_host[s, h].float()) * scale, 0)
            out[s, h] = probs @ v[:, kv]
    return out


def run_check(a, t, fns, names, num_heads):
    """Compare each variant against the CPU reference on non-zero KV.

    The timed caches are all zeros, which makes every variant return zeros --
    that hides a wrong kernel completely. This re-runs the same compiled
    callables (same shapes, so no recompile) against randn pages allocated
    through the same layouts, then restores the timing tensors.
    """
    dtype = torch.float16
    torch.manual_seed(1)
    k_host = torch.randn(t["page_shape"]).to(dtype)
    v_host = torch.randn(t["page_shape"]).to(dtype)
    keys = ("k_pages", "v_pages", "k_pages_kvm", "v_pages_kvm",
            "k_pages_flat", "v_pages_flat", "k_kvm_flatview", "v_kvm_flatview")
    saved = {k: t[k] for k in keys}
    t["k_pages"] = k_host.to(DEV, device_layout=t["layout"])
    t["v_pages"] = v_host.to(DEV, device_layout=t["layout"])
    t["k_pages_kvm"] = (
        k_host.permute(0, 2, 1, 3).contiguous().to(DEV, device_layout=t["layout_kvm"])
    )
    t["v_pages_kvm"] = (
        v_host.permute(0, 2, 1, 3).contiguous().to(DEV, device_layout=t["layout_kvm"])
    )
    t["k_pages_flat"] = (
        k_host.permute(0, 2, 1, 3).contiguous().reshape(t["flat_shape"])
        .to(DEV, device_layout=t["layout_kvm"])
    )
    t["v_pages_flat"] = (
        v_host.permute(0, 2, 1, 3).contiguous().reshape(t["flat_shape"])
        .to(DEV, device_layout=t["layout_kvm"])
    )

    # Rebuilt onto the randn allocations: a view captured in build_inputs still
    # points at the all-zero timing cache, which would return zeros and score a
    # rel l2 of exactly 1.0 -- a harness artefact, not a wrong kernel.
    if t["k_kvm_flatview"] is not None:
        t["k_kvm_flatview"] = t["k_pages_kvm"].view(t["flat_shape"])
        t["v_kvm_flatview"] = t["v_pages_kvm"].view(t["flat_shape"])

    ref = cpu_reference(a, t["q_host"], k_host, v_host, num_heads)
    print("\ncorrectness vs fp32 CPU reference (randn KV, production layouts):")
    print("%-9s %12s %12s %8s" % ("variant", "max abs err", "rel l2", "verdict"))
    ok = True
    for name in names:
        val = fns[name]()
        torch.spyre.synchronize()
        if isinstance(val, list):
            got = torch.cat([o.to("cpu").float() for o in val], 0)
        else:
            got = val.to("cpu").float()
        if got.shape[0] == a.staging_rows:
            # Storing variants return the staging buffer; row s is sequence s
            # because row_idx[s][0] == s.
            got = got[: a.num_seqs]
        got = got.reshape(a.num_seqs, num_heads, a.head_size)
        aerr = (got - ref).abs().max().item()
        rel = ((got - ref).norm() / ref.norm()).item()
        good = rel < 2e-2
        ok = ok and good
        print("%-9s %12.5f %12.5f %8s"
              % (name, aerr, rel, "ok" if good else "MISMATCH"))
    t.update(saved)
    return ok


def _store_kvm_kernel(key2d, value2d, k_flat, v_flat, flat_slots):
    """KV-major equivalent of _reshape_and_cache_kernel.

    Under [num_pages, KV, block_size, head_size] a token's KV heads are no
    longer contiguous, so one index_copy_ of KV*head_size per token becomes KV
    copies of head_size. Same bytes, KV times as many indices.
    """
    k_flat.index_copy_(0, flat_slots, key2d)
    v_flat.index_copy_(0, flat_slots, value2d)
    return k_flat


def _flat_2d_layout(rows, head_size, dtype):
    """[rows, head_size] with the row axis outermost, matching slot_major_kv_layout."""
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick

    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[rows, sticks, eps],
        stride_map=[sticks * eps, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


def run_store(a, t, num_tokens, label):
    """Price the KV cache store under both page orders.

    flatbc needs KV-major pages, so the read-side win is only real if the store
    it forces does not give the win back. Both paths move the same bytes.
    """
    dtype = torch.float16
    KV, D, BS = a.num_kv_heads, a.head_size, a.block_size
    num_pages, num_slots = t["num_pages"], t["num_pages"] * BS
    torch.manual_seed(2)

    key_h = torch.randn(num_tokens, KV, D).to(dtype)
    val_h = torch.randn(num_tokens, KV, D).to(dtype)
    # Slots must be distinct: index_copy_ leaves the write order undefined for
    # duplicate indices, so a colliding fixture makes the two page orders
    # disagree for reasons that have nothing to do with the layout. Walking
    # pages first gives one token per page at decode width, as decode does.
    slots_h = torch.tensor(
        [(i % num_pages) * BS + (i // num_pages) for i in range(num_tokens)],
        dtype=torch.int64,
    )
    assert len(set(slots_h.tolist())) == num_tokens, "store fixture has duplicate slots"
    # (page * KV + h) * block_size + offset
    flat_h = torch.empty(num_tokens * KV, dtype=torch.int64)
    for i in range(num_tokens):
        pg, off = int(slots_h[i]) // BS, int(slots_h[i]) % BS
        for h in range(KV):
            flat_h[i * KV + h] = (pg * KV + h) * BS + off

    layout_slots = slot_major_kv_layout(num_slots, KV, D, dtype)
    layout_flat = _flat_2d_layout(num_pages * KV * BS, D, dtype)

    k_slots = torch.zeros(num_slots, KV, D, dtype=dtype).to(DEV, device_layout=layout_slots)
    v_slots = torch.zeros(num_slots, KV, D, dtype=dtype).to(DEV, device_layout=layout_slots)
    k_flat = torch.zeros(num_pages * KV * BS, D, dtype=dtype).to(DEV, device_layout=layout_flat)
    v_flat = torch.zeros(num_pages * KV * BS, D, dtype=dtype).to(DEV, device_layout=layout_flat)

    key = convert(key_h, device=DEV)
    val = convert(val_h, device=DEV)
    key2d = convert(key_h.reshape(num_tokens * KV, D), device=DEV)
    val2d = convert(val_h.reshape(num_tokens * KV, D), device=DEV)
    slots = convert(slots_h, device=DEV)
    flat_slots = convert(flat_h, device=DEV)

    tok_c = torch.compile(_reshape_and_cache_kernel, dynamic=False)
    kvm_c = torch.compile(_store_kvm_kernel, dynamic=False)
    fns = {
        "store_tok": lambda: tok_c(key, val, k_slots, v_slots, slots),
        "store_kvm": lambda: kvm_c(key2d, val2d, k_flat, v_flat, flat_slots),
    }

    mb = num_tokens * KV * D * 2 * 2 / 1e6
    print("\nKV cache store, %s (%d tokens, %.3f MB written):" % (label, num_tokens, mb))
    print("%-10s %12s %8s %12s" % ("variant", "dev ms/call", "kernels", "wall med ms"))
    rows = []
    for name, fn in fns.items():
        try:
            for _ in range(a.warmup):
                fn()
            torch.spyre.synchronize()
            dev_ms, nkern = device_kernel_ms(fn, a.iters)
            w = wall_ms(fn, a.iters)
            rows.append((name, dev_ms))
            print("%-10s %12.4f %8d %12.4f" % (name, dev_ms, nkern, statistics.median(w)))
        except Exception as exc:
            lines = [ln for ln in str(exc).strip().splitlines() if ln.strip()]
            reason = next((ln for ln in lines if "TORCHDYNAMO_VERBOSE" not in ln), "")
            print("%-10s did not compile: %s" % (name, reason[:200]))
    if len(rows) == 2:
        print("  store_kvm / store_tok = %.2fx" % (rows[1][1] / rows[0][1]))

    # Both page orders must end up holding the same logical cache.
    got_tok = k_slots.to("cpu").reshape(num_pages, BS, KV, D)
    got_kvm = k_flat.to("cpu").reshape(num_pages, KV, BS, D).permute(0, 2, 1, 3)
    same = torch.equal(got_tok, got_kvm.contiguous())
    print("  stores agree: %s" % ("yes" if same else "NO -- results not comparable"))


def make_callables(a, t):
    """One zero-arg callable per variant, each compiled with dynamic=False."""
    scale = a.head_size**-0.5
    nh, nkv, hs = t["num_heads"], a.num_kv_heads, a.head_size

    def _out_buf():
        """A private staging buffer per storing variant.

        Production always hands _page_attn_kernel the staging buffer, so the
        in-kernel index_copy_ output scatter is part of every call it makes. The
        harness used to omit `out`, which compiled a different graph and is why the
        per_seq bar came out ~23% under production. One buffer per variant, so a
        variant that writes nothing cannot inherit another's rows and pass --check.
        """
        return convert(
            torch.zeros(a.staging_rows, nh, hs, dtype=torch.float16, device="cpu"), device=DEV
        )

    def _batched_out():
        """vLLM's per-layer output buffer: num_seqs rows, offset 0."""
        return convert(
            torch.zeros(a.num_seqs, nh, hs, dtype=torch.float16, device="cpu"), device=DEV
        )

    per_seq_c = torch.compile(_page_attn_kernel, dynamic=False)
    batched_c = torch.compile(_batched_decode_kernel, dynamic=False)
    unrolled_c = torch.compile(_unrolled_decode_kernel, dynamic=False)

    per_seq_out = _out_buf()

    def per_seq():
        for s in range(a.num_seqs):
            per_seq_c(
                t["query"], t["row_idx"][s], t["k_pages"], t["v_pages"],
                t["page_tables"][s], t["mask_tiles"], scale,
                a.num_blocks, a.query_len, nh, nkv, hs, 0.0, None, per_seq_out,
            )
        return per_seq_out

    def batched():
        return batched_c(
            t["query"], None, t["k_pages"], t["v_pages"], t["block_ids"],
            t["mask_by_block"], scale, a.num_seqs, a.num_blocks, nkv,
            a.num_queries_per_kv, a.block_size, hs,
        )

    def unrolled():
        return unrolled_c(
            t["query"], t["row_idx"], t["k_pages"], t["v_pages"], t["page_tables"],
            t["mask_tiles"], scale, a.num_seqs, a.num_blocks, nh, nkv, hs,
        )

    kvm_c = torch.compile(_kvmajor_decode_kernel, dynamic=False)

    def _kvm(merge):
        def run():
            return kvm_c(
                t["query"], t["k_pages_kvm"], t["v_pages_kvm"], t["block_ids"],
                t["mask_by_block"], scale, a.num_seqs, a.num_blocks, nkv,
                a.num_queries_per_kv, a.block_size, hs, merge,
            )
        return run

    per_seq_kvm_c = torch.compile(_per_seq_kvm_kernel, dynamic=False)

    per_seq_kvm_out = _out_buf()

    def per_seq_kvm():
        for sq in range(a.num_seqs):
            per_seq_kvm_c(
                t["query"], t["row_idx"][sq], t["k_pages_kvm"], t["v_pages_kvm"],
                t["page_tables"][sq], t["mask_tiles"], scale,
                a.num_blocks, a.query_len, nh, nkv, hs, per_seq_kvm_out,
            )
        return per_seq_kvm_out

    per_seq_fold3d_c = torch.compile(_per_seq_fold3d_kernel, dynamic=False)
    per_seq_fold3d_out = _out_buf()

    def per_seq_fold3d():
        for sq in range(a.num_seqs):
            per_seq_fold3d_c(
                t["query"], t["row_idx"][sq], t["k_pages_flat"], t["v_pages_flat"],
                t["page_tables_fold"][sq], t["mask_tiles"], scale,
                a.num_blocks, a.query_len, nh, nkv, hs, per_seq_fold3d_out,
            )
        return per_seq_fold3d_out

    per_seq_kvm3d_c = torch.compile(_per_seq_kvm3d_kernel, dynamic=False)

    def _rows_buf(rows):
        return convert(
            torch.zeros(rows, nh, hs, dtype=torch.float16, device="cpu"), device=DEV
        )

    def _per_seq_out(rows, kvm):
        buf = _rows_buf(rows)
        fn = per_seq_kvm_c if kvm else per_seq_c
        kpg, vpg = ("k_pages_kvm", "v_pages_kvm") if kvm else ("k_pages", "v_pages")

        def run():
            for s in range(a.num_seqs):
                args = (
                    t["query"], t["row_idx"][s], t[kpg], t[vpg],
                    t["page_tables"][s], t["mask_tiles"], scale,
                    a.num_blocks, a.query_len, nh, nkv, hs,
                )
                if kvm:
                    fn(*args, buf)
                else:
                    fn(*args, 0.0, None, buf)
            return buf

        return run

    out_sweep = {}
    for _rows in (a.num_seqs, 33, 129, a.staging_rows):
        out_sweep[f"per_seq_out{_rows}"] = _per_seq_out(_rows, False)
        out_sweep[f"per_seq_kvm_out{_rows}"] = _per_seq_out(_rows, True)

    per_seq_narrow_out = _batched_out()

    def per_seq_narrowout():
        for s in range(a.num_seqs):
            per_seq_c(
                t["query"], t["row_idx"][s], t["k_pages"], t["v_pages"],
                t["page_tables"][s], t["mask_tiles"], scale,
                a.num_blocks, a.query_len, nh, nkv, hs, 0.0, None, per_seq_narrow_out,
            )
        return per_seq_narrow_out

    per_seq_kvm_narrow_out = _batched_out()

    def per_seq_kvm_narrowout():
        for sq in range(a.num_seqs):
            per_seq_kvm_c(
                t["query"], t["row_idx"][sq], t["k_pages_kvm"], t["v_pages_kvm"],
                t["page_tables"][sq], t["mask_tiles"], scale,
                a.num_blocks, a.query_len, nh, nkv, hs, per_seq_kvm_narrow_out,
            )
        return per_seq_kvm_narrow_out

    per_seq_kvm3d_out = _out_buf()

    def per_seq_kvm3d():
        for sq in range(a.num_seqs):
            per_seq_kvm3d_c(
                t["query"], t["row_idx"][sq], t["k_pages_kvm"], t["v_pages_kvm"],
                t["page_tables"][sq], t["mask_tiles"], scale,
                a.num_blocks, nh, nkv, hs, per_seq_kvm3d_out,
            )
        return per_seq_kvm3d_out

    flat_c = torch.compile(_flat_decode_kernel, dynamic=False)
    flatchunk_c = torch.compile(_flat_chunked_decode_kernel, dynamic=False)

    def _flat(form):
        out_buf = _batched_out()

        def run():
            return flat_c(
                t["query"], t["k_pages_flat"], t["v_pages_flat"], t["flat_ids"],
                t["mask_by_block"], scale, a.num_seqs, a.num_blocks, nkv,
                a.num_queries_per_kv, a.block_size, hs, form, out_buf,
            )
        return run

    flat_view_out = _batched_out()

    def flat_from4dview():
        """flatbc over a merged view of the 4-D allocation, not its own cache.

        Proves whether ONE allocation can serve both kernels: the per-sequence
        kernel reading it 4-D with a 1-entry gather, the batched kernel reading
        the same bytes as [rows, block, D].
        """
        assert t["k_kvm_flatview"] is not None, "4-D -> flat view did not lower"
        return flat_c(
            t["query"], t["k_kvm_flatview"], t["v_kvm_flatview"], t["flat_ids"],
            t["mask_by_block"], scale, a.num_seqs, a.num_blocks, nkv,
            a.num_queries_per_kv, a.block_size, hs, "bcast", flat_view_out,
        )

    def flatchunk():
        return flatchunk_c(
            t["query"], t["k_pages_flat"], t["v_pages_flat"], t["flat_ids_chunks"],
            t["mask_chunks"], scale, a.num_seqs, a.num_blocks, nkv,
            a.num_queries_per_kv, a.block_size, hs, t["lanes_per_chunk"],
        )

    return {
        "per_seq": per_seq,
        "batched": batched,
        "unrolled": unrolled,
        "kvm4d": _kvm(False),
        "kvm3d": _kvm(True),
        "per_seq_kvm": per_seq_kvm,
        "per_seq_kvm3d": per_seq_kvm3d,
        "per_seq_fold3d": per_seq_fold3d,
        "per_seq_narrowout": per_seq_narrowout,
        "per_seq_kvm_narrowout": per_seq_kvm_narrowout,
        **out_sweep,
        "flat_from4dview": flat_from4dview,
        "flat3d": _flat("3d"),
        "flatbc": _flat("bcast"),
        "flatchunk": flatchunk,
    }


def device_kernel_ms(fn, iters):
    """Sum of device kernel durations per call, via the torch profiler."""
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "PrivateUse1"):
        acts.append(ProfilerActivity.PrivateUse1)
    with profile(activities=acts, record_shapes=False) as prof:
        for _ in range(iters):
            fn()
        torch.spyre.synchronize()
    total, n = 0.0, 0
    for e in prof.events():
        # Device kernels only; CPU ops carry the same names in some builds.
        if getattr(e, "device_type", None) is not None and "spyre_kernel" in str(e.key):
            total += e.self_device_time_total or 0.0
            n += 1
    return total / 1000.0 / iters, n // max(iters, 1)


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
    p.add_argument("--num-blocks", type=int, default=16, help="2048 KV / block 128")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--num-queries-per-kv", type=int, default=4, help="32 heads / 8 kv")
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=40, help="only scales the report")
    p.add_argument("--query-len", type=int, default=1,
                   help="padded_query_len: 1 is decode, >1 prices the prefill shape "
                        "the per-sequence kernel also runs (batched variants are "
                        "decode-only and are skipped)")
    p.add_argument("--staging-rows", type=int, default=513,
                   help="production is max_num_batched_tokens + 1 (512 + 1)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--variants", default="per_seq,batched,unrolled")
    p.add_argument("--show-warnings", action="store_true")
    p.add_argument("--store", action="store_true",
                   help="also price the KV cache store under both page orders")
    p.add_argument("--store-prefill-tokens", type=int, default=512)
    p.add_argument("--show-splits", action="store_true",
                   help="print the planner's chosen work division per variant")
    p.add_argument("--no-check", action="store_true",
                   help="skip the correctness comparison against the CPU reference")
    a = p.parse_args()

    kv_mb = a.num_blocks * a.block_size * a.num_kv_heads * a.head_size * 2 * 2 / 1e6
    print("shapes: num_seqs=%d num_blocks=%d block_size=%d kv_heads=%d "
          "queries_per_kv=%d head_size=%d" %
          (a.num_seqs, a.num_blocks, a.block_size, a.num_kv_heads,
           a.num_queries_per_kv, a.head_size))
    print("KV per layer: %.2f MB/seq, %.2f MB for %d seqs\n"
          % (kv_mb, kv_mb * a.num_seqs, a.num_seqs))

    counter = LossyCounter()
    logging.getLogger().addHandler(counter)
    logging.getLogger().setLevel(logging.DEBUG)
    # torch-spyre's own loggers sit under "spyre" and carry their own level, so
    # a DEBUG root alone never sees the work-division records.
    logging.getLogger("spyre").setLevel(logging.DEBUG)

    t = build_inputs(a)
    fns = make_callables(a, t)

    rows = []
    failed = []
    for name in [v.strip() for v in a.variants.split(",") if v.strip()]:
        fn = fns[name]
        counter.reset()
        t0 = time.perf_counter()
        # A variant that will not compile must not cost the run its other
        # results: several plausible reformulations hit backend limits, and
        # those are findings worth keeping next to the numbers.
        try:
            for _ in range(a.warmup):
                fn()
            torch.spyre.synchronize()
            compile_s = time.perf_counter() - t0
            lossy = len(counter.hits)
            if a.show_warnings:
                for h in counter.hits[:40]:
                    print("   [%s] %s" % (name, h[:150]))
            if a.show_splits:
                for (cores, itsp, spl), n in summarize_splits(counter.splits).most_common(8):
                    print("   [%s] x%-3d cores=%-3s splits=%s it_space=%s"
                          % (name, n, cores, spl, itsp[:110]))

            dev_ms, nkern = device_kernel_ms(fn, a.iters)
            w = wall_ms(fn, a.iters)
            rows.append(
                (name, dev_ms, nkern, statistics.median(w), min(w), lossy, compile_s)
            )
        except Exception as exc:
            # First informative line, not the last: inductor appends a
            # "Set TORCHDYNAMO_VERBOSE=1" hint that hides the actual reason.
            lines = [ln for ln in str(exc).strip().splitlines() if ln.strip()]
            reason = next((ln for ln in lines if "TORCHDYNAMO_VERBOSE" not in ln), "")
            failed.append((name, (reason or type(exc).__name__)[:220]))
            print("   [%s] FAILED: %s" % (name, failed[-1][1]))

    print("%-9s %11s %8s %11s %10s %7s %9s"
          % ("variant", "dev ms/call", "kernels", "wall med ms", "wall min", "lossy", "warmup s"))
    for name, dev, nk, wmed, wmin, lossy, cs in rows:
        print("%-9s %11.3f %8d %11.3f %10.3f %7d %9.1f"
              % (name, dev, nk, wmed, wmin, lossy, cs))

    for name, msg in failed:
        print("%-9s %s" % (name, "did not compile: " + msg))

    base = next((r for r in rows if r[0] == "per_seq"), None)
    if base:
        print("\nrelative to per_seq (device time):")
        for name, dev, *_ in rows:
            print("  %-9s %6.2fx   -> %8.1f ms/step across %d layers"
                  % (name, dev / base[1], dev * a.layers, a.layers))
        print("\neffective KV bandwidth:")
        for name, dev, *_ in rows:
            print("  %-9s %6.1f GB/s" % (name, kv_mb * a.num_seqs / 1e3 / (dev / 1e3)))

    if not a.no_check:
        names = [r[0] for r in rows]
        if not run_check(a, t, fns, names, t["num_heads"]):
            print("\nWARNING: a variant disagrees with the reference; its timing is moot")

    if a.store:
        run_store(a, t, a.num_seqs, "decode width")
        run_store(a, t, a.store_prefill_tokens, "prefill width")

    print("\nreference (granite-3.3-8b, batch 4, from the profiled runs):")
    print("  per_seq : 763.4 us x 4 seqs = 3.05 ms/layer -> 122.1 ms/step, 11.0 GB/s")
    print("  batched : 7581.1 us x 1     = 7.58 ms/layer -> 303.2 ms/step,  4.4 GB/s")


if __name__ == "__main__":
    main()
