"""Microbenchmark: per-sequence vs batched decode attention.

Isolates why SPYRE_BATCHED_DECODE=1 loses to the per-sequence path even though
it cuts kernel launches 4:1. Drives the real kernels from spyre_attn, so it
tracks the production code rather than a paraphrase.

Variants
  per_seq   num_seqs sequential calls to _page_attn_kernel (what production does
            with SPYRE_BATCHED_DECODE=0)
  batched   one call to _batched_decode_kernel (SPYRE_BATCHED_DECODE=1)
  unrolled  one compiled graph containing the num_seqs per-sequence bodies:
            a single launch that keeps the per-sequence shapes
  pretrans  batched over a pre-transposed cache, so no permute is needed
  blockpar  batched, gathering num_seqs*nc pages per step through a 2-D index
  pretrans_bp  pretrans + blockpar together
  headmaj / headmaj_bp  the same two over a [pages, KV, block, D] cache, the one
            layout K and V can share and the one with a working store
  per_seq_hm  the per-sequence kernel over that cache, to price what the shared
            layout costs prefill and the decode fallback

Defaults reproduce granite-3.3-8b decode at batch 4 / 2048 KV / block 128.

Run it through the wrapper, which sets the environment the numbers below were
taken with:

    bash experimental/run_microbench_batched_decode.sh

Knobs are forwarded, e.g.:

    bash experimental/run_microbench_batched_decode.sh --num-blocks 8 --block-size 256
    bash experimental/run_microbench_batched_decode.sh --variants per_seq,batched
    bash experimental/run_microbench_batched_decode.sh --show-warnings
    bash experimental/run_microbench_batched_decode.sh --check

`dev ms/call` is zero unless torch-spyre was built with USE_SPYRE_PROFILER=1;
pyproject.toml forces it to 0, so a stock install measures wall clock only.

STATUS
------
tpa-spyre-dev-2, granite shapes, LAYOUT_SOLVER=greedy, torch-spyre 94b7969 built
USE_SPYRE_PROFILER=1, nc=8. Cold inductor cache, so the lossy column is real:

    variant      dev ms/call  kernels  wall med ms  lossy  warmup s
    per_seq            2.351        4        3.137      0      35.0
    batched            7.722        1        7.995     32      40.4
    pretrans           6.131        1        6.478     32      37.5
    blockpar           1.953        1        2.207      4      12.4
    pretrans_bp        1.525        1        1.731      4      11.1
    headmaj            7.253        1        7.460      0      28.1
    headmaj_bp         1.645        1        1.892      0      10.5

The baseline reproduces the recorded regression (2.343 / 7.693) to within 0.5%,
on the committed torch-spyre pin rather than the fork the original numbers used.

Decode attention is not the only per-step cost the cache layout moves, so the
comparison that decides anything is attention + store (`--store`, verified to
place exactly, not merely to run):

    path          attention   store   total   vs bar
    per_seq (bar)     2.351   0.004   2.355   1.00x
    batched           7.722   0.004   7.726   3.28x worse
    blockpar          1.953   0.004   1.957   1.20x faster
    headmaj_bp        1.645   0.020   1.665   1.41x faster
    pretrans_bp       1.525       -       -   no store compiles

**headmaj_bp is the best deliverable kernel: 1.41x faster than per_seq including
the store, one kernel, one cache layout shared by K and V, zero work-division
fallbacks.** Its store is PR #783's `_folded_reshape_and_cache_kernel` pattern --
one index_copy_ per KV head against a flat view -- which fuses into a single
kernel and costs 0.016 ms/layer more than production's, 5% of the 0.308 ms/layer
that headmaj_bp saves over blockpar.

**blockpar is the safe default: 1.20x with no cache change at all**, production
cache byte-for-byte, so neither the store nor the per-sequence kernel is exposed.
Pick headmaj_bp for the extra 1.18x only alongside the prefill and fallback costs
measured below.

`pretrans_bp` has the fastest attention and is still not deliverable: putting K's
token axis last is exactly what makes its store fail to lower ("Scatter: no
mechanism to resolve stick incompatibility"), so the cache cannot be written.

What the shared layout costs the per-sequence kernel
  The cache is shared, so headmaj also changes what per_seq reads -- and per_seq
  is what prefill, mixed batches and every decode fallback run. `per_seq_hm` is a
  copy of the production kernel over a headmaj cache (bit-identical output,
  deviation 0.00e+00) and it is SLOWER, despite strictly fewer ops: the 3-axis
  permute collapses to one transpose(0, 1), and it still loses.

    shape                          per_seq  per_seq_hm  delta
    decode  (q_len=1,   4 seqs)      2.375       3.075  1.29x slower
    prefill (q_len=512, 1 seq)       8.016       8.549  1.07x slower

  So the layout is not free, and "fewer ops" was the wrong prediction. It is
  still net-positive, because the two costs have different frequencies:

    decode win,   headmaj_bp vs per_seq   -0.690 ms/layer   every output token
    prefill cost, per_seq_hm  vs per_seq  +0.533 ms/layer   once per request

  One prefill step is repaid by the second decode step, so headmaj wins for any
  realistic generation length. The decode-fallback case is the one to watch: a
  batch that misses the batched path pays 1.29x on a shared layout with no
  compensating win, so headmaj should not ship without the batched path covering
  the buckets that actually occur.

  The block-parallel trick does not transfer to prefill, and does not need to.
  lower_bmm's 4-D form has two batch axes; prefill already spends both on (KV, G)
  with M=q_len, so an nc axis needs a third or an axis merge the backend rejects
  (the same limit that kills nc=16 below). Decode needs the gather for
  parallelism precisely because its M is 1; prefill gets it from the query axis.
  Reasoning from the documented constraints, not measured here.

Two things must match production or the whole comparison is worthless -- with
either wrong, per_seq collapsed to 1.7 GB/s and the ranking INVERTED (batched
looked 1.7x faster than per_seq):

  1. k/v_pages must be allocated the way initialize_kv_cache_tensors does:
     host zeros then .to(device, device_layout=slot_major_kv_layout(...)).
     A plain convert() leaves the default tiled layout, which spreads the slot
     index across two device dims (torch-spyre#3705) and makes the in-graph
     page gather slow for every variant.
  2. `query` must be the production staging width (max_num_batched_tokens + 1
     = 513 rows), not num_seqs. The batched kernel takes a query[:num_seqs]
     prefix of it and the per-sequence kernel gathers out of it.

What causes the 3.3x regression
  The gather's entry width and the sequential trip count, not the work division
  and not the layout solver.

  * `blockpar` changes only how KV is fetched: num_seqs*nc pages per step through
    a 2-D index instead of num_seqs through a 1-D one, which also cuts the block
    loop from num_blocks steps to num_blocks/nc. Worth 3.95x on its own
    (7.722 -> 1.953). A gather can only be core-split on its index-entry axis,
    and for a 1-D int32 index that axis is counted in whole 32-entry sticks, so
    the batched kernel's 4-entry gather has zero splittable units and lands on
    one core. The 2-D index removes the /32.
  * The nc sweep confirms that is the mechanism rather than a coincidence:

        nc   entries  chunks   blockpar dev ms
         4        16       4   3.330   (worse than per_seq)
         8        32       2   1.953
        16        64       1   fails: Incompatible host_size and dim_order

    16 entries is below the core count and loses; 32 matches it and wins. nc=16
    would make the block loop fully parallel -- the online softmax is
    associative, so nothing forces it to be sequential -- but that shape hits the
    same backend axis-merge limit already documented under Constraints.
  * The permute is secondary: 1.26x alone (batched -> pretrans), 1.28x on top of
    blockpar (1.953 -> 1.525).

The 32 work-division fallbacks are priced, and they are not the cause
  Nothing had measured them; these variants do. `pretrans` emits the identical 32
  and runs 1.26x faster than `batched`; `blockpar` still emits 4 and is the
  fastest of the unchanged-cache variants; `headmaj` emits 0 and is barely faster
  than `batched` (7.253 vs 7.722). The count is just 2 per bmm with M > 1
  (16 blocks x 2 matmuls = 32; 2 chunks x 2 matmuls = 4), which is why per_seq
  scores 0 -- its M is 1. It tracks the shape, not the cost. Stop chasing it.

The layout solver is not the lever: cpsat leaves `batched` unchanged (7.673 vs
7.656) and makes `per_seq` 1.9x worse (4.440 vs 2.342), so the pathology is
solver-independent and greedy is the right setting at these shapes.

What the store costs, by layout (`--store`)
  `placed` compares the written rows against the source, so a store that lands
  the right bytes in the wrong rows is caught rather than timed:

    layout             dev ms  kernels  placed
    tokmaj              0.004        1  ok       production today
    headmaj_perhead     0.020        1  ok       PR #783's per-KV-head index_copy_
    headmaj                 -        -  -        index_put_: dxp scheduler failure
    pretrans                -        -  -        index_put_: stick incompatibility

  Two things this settles. The per-head form fuses into ONE kernel, so the
  headmaj layout is storable for +0.016 ms/layer -- do not reject it as "KV times
  the work". And the earlier index_put_ failures were a formulation gap for
  headmaj but are structural for pretrans, whose token axis is last.

  The layout choice is still gated on prefill, which shares the cache and runs
  the per-sequence kernel at query_len > 1; this harness is decode-only, so that
  is untested here. blockpar is the option with no exposure either way.

`--check` compares every variant against a CPU fp32 reference over random pages.
The timing path deliberately keeps the all-zero pages the recorded numbers used,
which cannot distinguish a correct kernel from a wrong one, so a new variant is
only a result once it passes --check as well. All variants pass: relative L2 vs
fp32 is 3.7e-03 to 4.1e-03 and each lands within 7.32e-04 of per_seq, one fp16
tick at this magnitude. The blockpar variants are marginally the most accurate,
having fewer sequential rescale steps.
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


def pretrans_kv_layout(num_pages, num_kv_heads, rows, cols, dtype):
    """Page-outermost layout for a pre-transposed cache page [KV, rows, cols].

    Same discipline as slot_major_kv_layout: the gathered (page) axis has to sit
    at device position 0 or torch-spyre restickifies the whole cache per gather.
    The default layout for this shape puts it at position 3.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick

    eps = get_elem_in_stick(dtype)
    sticks = (cols + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_pages, num_kv_heads, rows, sticks, eps],
        stride_map=[num_kv_heads * rows * cols, rows * cols, cols, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


def headmaj_kv_layout(num_pages, num_kv_heads, block_size, head_size, dtype):
    """Page-outermost layout for a [pages, KV, block_size, head_size] cache.

    The KV-head axis moves out from between block_size and head_size, which is
    what the decode matmuls want, but the token axis stays ahead of head_size --
    so one decode token is still num_kv_heads contiguous runs of head_size, not
    an element-strided scatter the way [pages, KV, head_size, block_size] is.
    K and V can share this layout; K only needs transpose(-2, -1), which matmul
    absorbs.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick

    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_pages, num_kv_heads, block_size, sticks, eps],
        stride_map=[
            num_kv_heads * block_size * head_size,
            block_size * head_size,
            head_size,
            eps,
            1,
        ],
        device_dtype=get_device_dtype(dtype),
    )


def _store_tokmaj(key, value, k_slots, v_slots, slot_mapping):
    """Production store: contiguous rows into the token-major cache."""
    k_slots.index_copy_(0, slot_mapping, key)
    v_slots.index_copy_(0, slot_mapping, value)


def _store_headmaj(key, value, k_pages, v_pages, page_idx, offset_idx):
    """Store into [pages, KV, block_size, head_size]: KV runs of head_size."""
    k_pages[page_idx, :, offset_idx, :] = key
    v_pages[page_idx, :, offset_idx, :] = value


def _store_headmaj_perhead(key, value, k_flat, v_flat, per_head_rows, num_kv_heads):
    """PR #783's folded store, against a flat [P*KV*block, head_size] view.

    One index_copy_ per KV head, the head sliced in-graph: every form that instead
    flattens the source leaves a work-division dim out of the tensor's scales and
    fails to lower. The per-head ops fuse into one kernel, so this costs no extra
    launches than the two-op slot-major store.
    """
    for h in range(num_kv_heads):
        rows = per_head_rows[h]
        k_flat.index_copy_(0, rows, key.select(1, h))
        v_flat.index_copy_(0, rows, value.select(1, h))


def _store_pretrans(key, value, k_pages, v_pages, page_idx, offset_idx):
    """Store into K [pages, KV, head_size, block_size] / V [pages, KV, block, D].

    K's token axis is last, so this writes KV*head_size elements block_size apart.
    """
    k_pages[page_idx, :, :, offset_idx] = key
    v_pages[page_idx, :, offset_idx, :] = value


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


def _per_seq_headmaj_kernel(
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
):
    """_page_attn_kernel against a headmaj cache, to price the layout for prefill.

    A line-for-line copy of the production kernel with only the two page lines
    changed: a [pages, KV, block, D] page needs one transpose(0, 1) where the
    token-major one needs squeeze + 3-axis permute + unsqueeze. Everything else,
    including the online softmax, is identical, so the pair isolates the layout.
    Kept as a copy rather than a flag on the production kernel because the mission
    is not to touch spyre_inference until a win is shown; it can drift, so re-read
    _page_attn_kernel before trusting a stale number.
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
        # [1, KV, block, D] -> [KV, 1, block, D]: no 3-axis permute to undo.
        k_page_4d = k_pages.index_select(0, page_idx).transpose(0, 1)
        v_page_4d = v_pages.index_select(0, page_idx).transpose(0, 1)

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
    return attn.reshape(padded_query_len, num_heads, head_size)


def _pretrans_decode_kernel(
    query,
    k_pages_t,
    v_pages_t,
    block_ids,
    mask_by_block,
    scale,
    num_seqs,
    num_blocks,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
    k_needs_transpose=False,
):
    """_batched_decode_kernel with the cache already in the matmul's frame.

    Identical to it op for op except that the two .permute(0, 2, 1, 3) calls are
    gone, which is the only difference and so makes the pair a measurement.

    k_needs_transpose False ("pretrans"): K [pages, KV, head_size, block_size],
    nothing left to do. True ("headmaj"): K [pages, KV, block_size, head_size],
    so K keeps transpose(-2, -1), which matmul absorbs. V is
    [pages, KV, block_size, head_size] either way.
    """
    num_heads = num_kv_heads * num_queries_per_kv
    q = query[:num_seqs].reshape(num_seqs, num_kv_heads, num_queries_per_kv, head_size)

    tile_max = None
    tile_sum = None
    tile_output = None

    for i in range(num_blocks):
        page_idx = block_ids[i, 0:num_seqs]
        k_t = k_pages_t.index_select(0, page_idx)
        if k_needs_transpose:
            k_t = k_t.transpose(-2, -1)
        v_t = v_pages_t.index_select(0, page_idx)
        mask_tile = mask_by_block[i].reshape(num_seqs, num_kv_heads, 1, block_size)

        scores = torch.matmul(q, k_t) * scale
        scores = scores + mask_tile
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = torch.matmul(tile_probs, v_t)
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
            tile_output += torch.matmul(tile_probs, v_t)
            tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
    return (tile_output / tile_sum).reshape(num_seqs, num_heads, head_size)


def _blockpar_decode_kernel(
    query,
    rep_row_idx,
    k_pages,
    v_pages,
    chunk_page_idx,
    chunk_masks,
    scale,
    num_seqs,
    num_chunks,
    nc,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
    cache_form="tokmaj",
):
    """Batched decode gathering num_seqs*nc pages per step instead of num_seqs.

    A gather can only be core-split on its index-entry dim, and for a 1-D int32
    index that dim is measured in whole 32-entry sticks, so the batched kernel's
    num_seqs-entry gather has zero splittable units. Two changes here: the entry
    count rises to num_seqs*nc, and the index is 2-D ([E, 1]) so the entry var is
    no longer the trailing device coord and splits without the /32.

    The nc blocks inside a chunk are independent partial softmaxes merged by one
    reduction over the chunk axis; only num_blocks/nc steps stay sequential.
    """
    num_heads = num_kv_heads * num_queries_per_kv
    entries = num_seqs * nc
    # rep_row_idx repeats each sequence's query row nc times, so the replication
    # rides the gather already needed to reach the staging buffer.
    q = query.index_select(0, rep_row_idx).reshape(
        entries, num_kv_heads, num_queries_per_kv, head_size
    )

    tile_max = None
    tile_sum = None
    tile_output = None

    for c in range(num_chunks):
        idx2 = chunk_page_idx[c]
        k_g = k_pages[idx2].squeeze(1)
        v_g = v_pages[idx2].squeeze(1)
        if cache_form == "pretrans":
            k_t, v_t = k_g, v_g
        elif cache_form == "headmaj":
            k_t, v_t = k_g.transpose(-2, -1), v_g
        else:
            k_t = k_g.permute(0, 2, 1, 3).transpose(-2, -1)
            v_t = v_g.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k_t) * scale
        scores = scores + chunk_masks[c]
        # Split the entry axis back into (seq, chunk slot) so one sequence's nc
        # partial softmaxes can be merged. Leading-axis views only: merging a
        # permuted axis pair is what torch-spyre rejects.
        sc = scores.reshape(num_seqs, nc, num_kv_heads, num_queries_per_kv, block_size)
        chunk_max = torch.amax(torch.amax(sc, dim=-1, keepdim=True), dim=1, keepdim=True)
        probs = torch.exp(sc - chunk_max)
        chunk_sum = torch.sum(
            torch.sum(probs, dim=-1, keepdim=True), dim=1, keepdim=True
        ).squeeze(1)
        out_e = torch.matmul(
            probs.reshape(entries, num_kv_heads, num_queries_per_kv, block_size), v_t
        )
        chunk_out = torch.sum(
            out_e.reshape(num_seqs, nc, num_kv_heads, num_queries_per_kv, head_size),
            dim=1,
        )
        chunk_max = chunk_max.squeeze(1)

        if c == 0:
            tile_max = chunk_max
            tile_sum = chunk_sum
            tile_output = chunk_out
        else:
            assert tile_max is not None
            assert tile_sum is not None
            assert tile_output is not None
            new_max = torch.maximum(tile_max, chunk_max)
            rescale_old = torch.exp(tile_max - new_max)
            rescale_new = torch.exp(chunk_max - new_max)
            tile_output = tile_output * rescale_old + chunk_out * rescale_new
            tile_sum = tile_sum * rescale_old + chunk_sum * rescale_new
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
    return (tile_output / tile_sum).reshape(num_seqs, num_heads, head_size)


class LossyCounter(logging.Handler):
    """Counts torch-spyre work-division fallbacks emitted while compiling."""

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


def build_inputs(a, random_kv=False):
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
    shape = (num_pages, a.block_size, a.num_kv_heads, a.head_size)
    if random_kv:
        k_host = torch.randn(*shape).to(dtype)
        v_host = torch.randn(*shape).to(dtype)
    else:
        k_host = torch.zeros(*shape, dtype=dtype)
        v_host = torch.zeros(*shape, dtype=dtype)
    k_pages = k_host.to(DEV, device_layout=layout)
    v_pages = v_host.to(DEV, device_layout=layout)

    # Same contents stored in the frame the matmuls want: K as [D, block] and V
    # as [block, D] per (page, kv head). Only the pretrans variants read these.
    k_pages_t = k_host.permute(0, 2, 3, 1).contiguous().to(
        DEV,
        device_layout=pretrans_kv_layout(
            num_pages, a.num_kv_heads, a.head_size, a.block_size, dtype
        ),
    )
    v_pages_t = v_host.permute(0, 2, 1, 3).contiguous().to(
        DEV,
        device_layout=pretrans_kv_layout(
            num_pages, a.num_kv_heads, a.block_size, a.head_size, dtype
        ),
    )

    # Only the KV-head axis moves out; the token axis stays ahead of head_size, so
    # the store stays contiguous per KV head. K and V share this layout.
    hm_layout = headmaj_kv_layout(
        num_pages, a.num_kv_heads, a.block_size, a.head_size, dtype
    )
    kv_hm_host = k_host.permute(0, 2, 1, 3).contiguous()
    k_pages_hm = kv_hm_host.to(DEV, device_layout=hm_layout)
    v_pages_hm = v_host.permute(0, 2, 1, 3).contiguous().to(DEV, device_layout=hm_layout)

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
    # query_len > 1 is the prefill shape: each sequence owns query_len contiguous
    # staging rows, which is what makes per_seq the kernel prefill runs.
    q_len = a.query_len
    assert a.num_seqs * q_len <= a.staging_rows, (
        f"num_seqs*query_len ({a.num_seqs * q_len}) exceeds the staging buffer "
        f"({a.staging_rows}); use --num-seqs 1 for a prefill-length run"
    )
    idx_len = _stick_aligned_len(q_len)
    row_idx = []
    for s in range(a.num_seqs):
        t = torch.zeros(idx_len, dtype=torch.int32)
        for j in range(q_len):
            t[j] = s * q_len + j
        row_idx.append(convert(t, device=DEV))

    def page_of(s, b):
        return s * a.num_blocks + b

    # page_index_table: [num_blocks, INT32_ELEMS_PER_STICK], page index at col 0.
    page_tables = []
    for s in range(a.num_seqs):
        t = torch.zeros(a.num_blocks, INT32_ELEMS_PER_STICK, dtype=torch.int32)
        for b in range(a.num_blocks):
            t[b, 0] = page_of(s, b)
        page_tables.append(convert(t, device=DEV))

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
            bid[b, s] = page_of(s, b)
    block_ids = convert(bid, device=DEV)

    # mask_by_block: [num_blocks, num_seqs * num_kv_heads, 1, block_size],
    # pre-broadcast across KV heads by the production builder.
    mask_by_block = convert(
        torch.zeros(
            a.num_blocks, a.num_seqs * a.num_kv_heads, 1, a.block_size, dtype=dtype
        ),
        device=DEV,
    )

    # --- blockpar inputs -----------------------------------------------------
    # nc blocks per sequence per step, so entries = num_seqs*nc and only
    # num_blocks/nc steps stay sequential. nc must divide num_blocks.
    nc = min(a.nc, a.num_blocks)
    while a.num_blocks % nc:
        nc -= 1
    num_chunks = a.num_blocks // nc
    entries = a.num_seqs * nc
    rep = torch.zeros(entries, dtype=torch.int32)
    for s in range(a.num_seqs):
        for j in range(nc):
            rep[s * nc + j] = s
    rep_row_idx = convert(rep, device=DEV)

    chunk_page_idx = []
    for c in range(num_chunks):
        t = torch.zeros(entries, 1, dtype=torch.int32)
        for s in range(a.num_seqs):
            for j in range(nc):
                t[s * nc + j, 0] = page_of(s, c * nc + j)
        chunk_page_idx.append(convert(t, device=DEV))
    chunk_masks = [
        convert(
            torch.zeros(entries, a.num_kv_heads, 1, a.block_size, dtype=dtype),
            device=DEV,
        )
        for _ in range(num_chunks)
    ]

    # --- store inputs --------------------------------------------------------
    # One new token per sequence per step, landing in the block after the context.
    new_k = convert(torch.randn(a.num_seqs, a.num_kv_heads, a.head_size).to(dtype), DEV)
    new_v = convert(torch.randn(a.num_seqs, a.num_kv_heads, a.head_size).to(dtype), DEV)
    store_page = torch.zeros(a.num_seqs, dtype=torch.int32)
    store_off = torch.zeros(a.num_seqs, dtype=torch.int32)
    slot_map = torch.zeros(a.num_seqs, dtype=torch.int32)
    for s in range(a.num_seqs):
        pg, off = page_of(s, a.num_blocks - 1), a.block_size - 1
        store_page[s], store_off[s] = pg, off
        slot_map[s] = pg * a.block_size + off
    # Row of (page, kv_head, offset) in the flattened headmaj cache, per KV head.
    per_head_rows = []
    for h in range(a.num_kv_heads):
        r = torch.zeros(a.num_seqs, dtype=torch.int32)
        for s in range(a.num_seqs):
            r[s] = (int(store_page[s]) * a.num_kv_heads + h) * a.block_size + int(store_off[s])
        per_head_rows.append(convert(r, device=DEV))

    return dict(
        query=query,
        out_buf=out_buf,
        k_pages=k_pages,
        v_pages=v_pages,
        k_pages_t=k_pages_t,
        v_pages_t=v_pages_t,
        k_pages_hm=k_pages_hm,
        v_pages_hm=v_pages_hm,
        k_slots=k_pages.view(num_pages * a.block_size, a.num_kv_heads, a.head_size),
        v_slots=v_pages.view(num_pages * a.block_size, a.num_kv_heads, a.head_size),
        new_k=new_k,
        new_v=new_v,
        per_head_rows=per_head_rows,
        # Flattened over (page, kv_head, offset): all three are outermost and
        # adjacent in the headmaj device layout, so the store's indexed axis lands
        # at device dim 0, which is what the destination check requires.
        k_flat_hm=k_pages_hm.view(-1, a.head_size),
        v_flat_hm=v_pages_hm.view(-1, a.head_size),
        store_page=convert(store_page, device=DEV),
        store_off=convert(store_off, device=DEV),
        slot_map=convert(slot_map, device=DEV),
        row_idx=row_idx,
        page_tables=page_tables,
        mask_tiles=mask_tiles,
        block_ids=block_ids,
        mask_by_block=mask_by_block,
        rep_row_idx=rep_row_idx,
        chunk_page_idx=chunk_page_idx,
        chunk_masks=chunk_masks,
        nc=nc,
        num_chunks=num_chunks,
        num_heads=num_heads,
        q_host=q_host,
        k_host=k_host,
        v_host=v_host,
        page_of=page_of,
    )


def reference_attention(a, t):
    """CPU fp32 decode attention over the same pages. Masks are all zeros."""
    num_heads = t["num_heads"]
    q = t["q_host"][: a.num_seqs].float()
    k_host, v_host, page_of = t["k_host"].float(), t["v_host"].float(), t["page_of"]
    out = torch.zeros(a.num_seqs, num_heads, a.head_size)
    for s in range(a.num_seqs):
        pages = [page_of(s, b) for b in range(a.num_blocks)]
        k = torch.cat([k_host[p] for p in pages], dim=0)  # [ctx, KV, D]
        v = torch.cat([v_host[p] for p in pages], dim=0)
        for h in range(num_heads):
            kv = h // a.num_queries_per_kv
            scores = (k[:, kv, :] @ q[s, h, :]) * (a.head_size**-0.5)
            out[s, h, :] = torch.softmax(scores, dim=0) @ v[:, kv, :]
    return out


def make_callables(a, t):
    """One zero-arg callable per variant, each compiled with dynamic=False."""
    scale = a.head_size**-0.5
    nh, nkv, hs = t["num_heads"], a.num_kv_heads, a.head_size

    per_seq_c = torch.compile(_page_attn_kernel, dynamic=False)
    batched_c = torch.compile(_batched_decode_kernel, dynamic=False)
    unrolled_c = torch.compile(_unrolled_decode_kernel, dynamic=False)
    pretrans_c = torch.compile(_pretrans_decode_kernel, dynamic=False)
    blockpar_c = torch.compile(_blockpar_decode_kernel, dynamic=False)
    per_seq_hm_c = torch.compile(_per_seq_headmaj_kernel, dynamic=False)

    def per_seq():
        outs = []
        for s in range(a.num_seqs):
            outs.append(
                per_seq_c(
                    t["query"], t["row_idx"][s], t["k_pages"], t["v_pages"],
                    t["page_tables"][s], t["mask_tiles"], scale,
                    a.num_blocks, a.query_len, nh, nkv, hs,
                )
            )
        # Returned unconcatenated: the recorded baseline timed these 4 calls and
        # nothing else, and a cat would add a launch to the bar. --check joins them.
        return outs

    def per_seq_hm():
        outs = []
        for s in range(a.num_seqs):
            outs.append(
                per_seq_hm_c(
                    t["query"], t["row_idx"][s], t["k_pages_hm"], t["v_pages_hm"],
                    t["page_tables"][s], t["mask_tiles"], scale,
                    a.num_blocks, a.query_len, nh, nkv, hs,
                )
            )
        return outs

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

    def pretrans():
        return pretrans_c(
            t["query"], t["k_pages_t"], t["v_pages_t"], t["block_ids"],
            t["mask_by_block"], scale, a.num_seqs, a.num_blocks, nkv,
            a.num_queries_per_kv, a.block_size, hs,
        )

    def blockpar():
        return blockpar_c(
            t["query"], t["rep_row_idx"], t["k_pages"], t["v_pages"],
            t["chunk_page_idx"], t["chunk_masks"], scale, a.num_seqs,
            t["num_chunks"], t["nc"], nkv, a.num_queries_per_kv, a.block_size, hs,
        )

    def pretrans_bp():
        return blockpar_c(
            t["query"], t["rep_row_idx"], t["k_pages_t"], t["v_pages_t"],
            t["chunk_page_idx"], t["chunk_masks"], scale, a.num_seqs,
            t["num_chunks"], t["nc"], nkv, a.num_queries_per_kv, a.block_size, hs,
            "pretrans",
        )

    def headmaj():
        return pretrans_c(
            t["query"], t["k_pages_hm"], t["v_pages_hm"], t["block_ids"],
            t["mask_by_block"], scale, a.num_seqs, a.num_blocks, nkv,
            a.num_queries_per_kv, a.block_size, hs, True,
        )

    def headmaj_bp():
        return blockpar_c(
            t["query"], t["rep_row_idx"], t["k_pages_hm"], t["v_pages_hm"],
            t["chunk_page_idx"], t["chunk_masks"], scale, a.num_seqs,
            t["num_chunks"], t["nc"], nkv, a.num_queries_per_kv, a.block_size, hs,
            "headmaj",
        )

    return {
        "per_seq": per_seq,
        "per_seq_hm": per_seq_hm,
        "batched": batched,
        "unrolled": unrolled,
        "pretrans": pretrans,
        "blockpar": blockpar,
        "pretrans_bp": pretrans_bp,
        "headmaj": headmaj,
        "headmaj_bp": headmaj_bp,
    }


def make_store_callables(a, t):
    """One zero-arg callable per cache layout's decode store."""
    tokmaj_c = torch.compile(_store_tokmaj, dynamic=False)
    headmaj_c = torch.compile(_store_headmaj, dynamic=False)
    perhead_c = torch.compile(_store_headmaj_perhead, dynamic=False)
    pretrans_c = torch.compile(_store_pretrans, dynamic=False)

    return {
        "tokmaj": lambda: tokmaj_c(
            t["new_k"], t["new_v"], t["k_slots"], t["v_slots"], t["slot_map"]
        ),
        "headmaj_perhead": lambda: perhead_c(
            t["new_k"], t["new_v"], t["k_flat_hm"], t["v_flat_hm"],
            t["per_head_rows"], a.num_kv_heads,
        ),
        "headmaj": lambda: headmaj_c(
            t["new_k"], t["new_v"], t["k_pages_hm"], t["v_pages_hm"],
            t["store_page"], t["store_off"],
        ),
        "pretrans": lambda: pretrans_c(
            t["new_k"], t["new_v"], t["k_pages_t"], t["v_pages_t"],
            t["store_page"], t["store_off"],
        ),
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
    p.add_argument("--staging-rows", type=int, default=513,
                   help="production is max_num_batched_tokens + 1 (512 + 1)")
    p.add_argument("--query-len", type=int, default=1,
                   help="padded_query_len for the per_seq variants; >1 is the "
                        "prefill shape (needs --num-seqs 1 at 512)")
    p.add_argument("--nc", type=int, default=8,
                   help="blocks gathered per step by the blockpar variants")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--variants",
                   default="per_seq,batched,pretrans,blockpar,pretrans_bp,"
                           "headmaj,headmaj_bp")
    p.add_argument("--show-warnings", action="store_true")
    p.add_argument("--store", action="store_true",
                   help="time the decode KV store into each cache layout instead "
                        "of the attention kernels: a pre-transposed cache is only "
                        "a win if it does not give the saving back here")
    p.add_argument("--check", action="store_true",
                   help="compare each variant against a CPU fp32 reference over "
                        "random pages instead of timing it")
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

    t = build_inputs(a, random_kv=a.check)
    if a.store:
        print("decode KV store, %d tokens per step (K and V together)\n" % a.num_seqs)
        store_fns = make_store_callables(a, t)
        # A store that writes the right bytes to the wrong rows would time as well
        # as a correct one, so check placement before believing any of these.
        want_k = t["new_k"].to("cpu").float()
        want_v = t["new_v"].to("cpu").float()
        pages, offs = t["store_page"].to("cpu"), t["store_off"].to("cpu")

        def placed(name):
            if name == "tokmaj":
                kc, vc = t["k_pages"].to("cpu").float(), t["v_pages"].to("cpu").float()
                got_k = torch.stack([kc[pages[s], offs[s]] for s in range(a.num_seqs)])
                got_v = torch.stack([vc[pages[s], offs[s]] for s in range(a.num_seqs)])
            else:  # headmaj: [pages, KV, block, head_size]
                kc = t["k_pages_hm"].to("cpu").float()
                vc = t["v_pages_hm"].to("cpu").float()
                got_k = torch.stack([kc[pages[s], :, offs[s]] for s in range(a.num_seqs)])
                got_v = torch.stack([vc[pages[s], :, offs[s]] for s in range(a.num_seqs)])
            return max((got_k - want_k).abs().max().item(),
                       (got_v - want_v).abs().max().item())

        print("%-16s %11s %8s %11s %7s %9s"
              % ("layout", "dev ms/call", "kernels", "wall med ms", "lossy", "placed"))
        for name, fn in store_fns.items():
            counter.reset()
            try:
                for _ in range(a.warmup):
                    fn()
                torch.spyre.synchronize()
            except Exception as e:
                print("%-16s FAILED %s: %s" % (name, type(e).__name__, str(e)[:110]))
                continue
            try:
                err = placed(name)
                verdict = "ok" if err == 0.0 else "WRONG %.1e" % err
            except Exception as e:
                verdict = "uncheckable %s" % type(e).__name__
            dev_ms, nkern = device_kernel_ms(fn, a.iters)
            w = wall_ms(fn, a.iters)
            print("%-16s %11.3f %8d %11.3f %7d %9s"
                  % (name, dev_ms, nkern, statistics.median(w), len(counter.hits),
                     verdict))
        return

    fns = make_callables(a, t)
    names = [v.strip() for v in a.variants.split(",") if v.strip()]

    if a.check:
        print("nc=%d num_chunks=%d\n" % (t["nc"], t["num_chunks"]))
        ref = reference_attention(a, t)
        # Averaging 2048 random V rows leaves |out| ~ 0.02, which is only ~20 fp16
        # ticks, so per-element relative error is meaningless here (the shipped
        # per_seq kernel scores 7e-2 on it). Judge on relative L2 against fp32 and
        # on agreement with per_seq, which is the kernel production actually runs.
        print("%-12s %10s %10s %10s  %s"
              % ("variant", "max abs", "relL2 cpu", "max d(ps)", "verdict"))
        base_l2 = None
        for name in names:
            try:
                got = fns[name]()
                if isinstance(got, list):
                    got = torch.cat(got, dim=0)
                got = got.to("cpu").float()[: a.num_seqs]
            except Exception as e:
                print("%-12s %10s %10s %10s  FAILED %s: %s"
                      % (name, "-", "-", "-", type(e).__name__, str(e)[:80]))
                continue
            rel_l2 = ((got - ref).norm() / ref.norm()).item()
            if name == "per_seq":
                base_l2, per_seq_out = rel_l2, got
            d_ps = (got - per_seq_out).abs().max().item() if base_l2 is not None else float("nan")
            ok = rel_l2 <= 3.0 * base_l2 if base_l2 else rel_l2 < 0.05
            print("%-12s %10.2e %10.2e %10.2e  %s"
                  % (name, (got - ref).abs().max().item(), rel_l2, d_ps,
                     "ok" if ok else "MISMATCH"))
        return

    print("nc=%d num_chunks=%d\n" % (t["nc"], t["num_chunks"]))
    rows = []
    for name in names:
        fn = fns[name]
        counter.reset()
        t0 = time.perf_counter()
        try:
            for _ in range(a.warmup):
                fn()
            torch.spyre.synchronize()
        except Exception as e:
            print("%s: FAILED to compile/run -- %s: %s"
                  % (name, type(e).__name__, str(e)[:300]))
            continue
        compile_s = time.perf_counter() - t0
        lossy = len(counter.hits)
        if a.show_warnings:
            for h in counter.hits[:40]:
                print("   [%s] %s" % (name, h[:150]))

        dev_ms, nkern = device_kernel_ms(fn, a.iters)
        w = wall_ms(fn, a.iters)
        rows.append((name, dev_ms, nkern, statistics.median(w), min(w), lossy, compile_s))

    print("%-12s %11s %8s %11s %10s %7s %9s"
          % ("variant", "dev ms/call", "kernels", "wall med ms", "wall min", "lossy",
             "warmup s"))
    for name, dev, nk, wmed, wmin, lossy, cs in rows:
        print("%-12s %11.3f %8d %11.3f %10.3f %7d %9.1f"
              % (name, dev, nk, wmed, wmin, lossy, cs))

    base = next((r for r in rows if r[0] == "per_seq"), None)
    if base and base[1] > 0:
        print("\nrelative to per_seq (device time):")
        for name, dev, *_ in rows:
            print("  %-12s %6.2fx   -> %8.1f ms/step across %d layers"
                  % (name, dev / base[1], dev * a.layers, a.layers))
        print("\neffective KV bandwidth:")
        for name, dev, *_ in rows:
            print("  %-12s %6.1f GB/s" % (name, kv_mb * a.num_seqs / 1e3 / (dev / 1e3)))
    elif base:
        print("\nno device time recorded: torch-spyre must be built with "
              "USE_SPYRE_PROFILER=1 (pyproject.toml forces it to 0)")

    print("\nreference (granite-3.3-8b, batch 4, from the profiled runs):")
    print("  per_seq : 763.4 us x 4 seqs = 3.05 ms/layer -> 122.1 ms/step, 11.0 GB/s")
    print("  batched : 7581.1 us x 1     = 7.58 ms/layer -> 303.2 ms/step,  4.4 GB/s")


if __name__ == "__main__":
    main()
