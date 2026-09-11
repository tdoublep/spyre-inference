# A ~20% decode-attention win on main, independent of any layout change

The per-sequence attention kernel writes its result with
`out.index_copy_(0, query_row_index[:padded_query_len], attn)` into `out_staging`,
which is `max_num_batched_tokens + 1` = **513 rows**. Narrowing that destination to
a decode-sized buffer is worth **21% of every per-sequence call**, changing nothing
else.

Measured on `experimental/microbench_batched_decode.py` (granite-3.3-8b shapes:
4 seqs, 16 blocks x 128, KV=8, G=4, D=128), torch-spyre `AdnanHoque` @ `927c3b83`
built `USE_SPYRE_PROFILER=1`, `LAYOUT_SOLVER=greedy`, `SPYRE_NUM_CPUS=8`:

| out rows | dev ms/layer/step | ms/step (40 layers) |
|---|---|---|
| 513 (today) | 3.066 | 122.6 |
| 129 | 2.569 | 102.8 |
| **33** | **2.423** | **96.9** |
| 4 (`num_seqs`) | 2.361 | 94.4 |

    bash experimental/run_microbench_batched_decode.sh \
        --variants per_seq_out513,per_seq_out33,per_seq_out4

A 33-row buffer keeps essentially all of the win, so this does not need to be sized
per batch.

## Why it is safe

`out_staging` is 513 rows because a **mixed** batch puts sequence `s`'s rows at
`query_start_loc[s]`, an arbitrary offset, and a compiled region reads its arguments
from storage offset 0 (torch-spyre#3770) — so the destination must be allocated
whole and indexed by absolute row.

A **pure decode** batch has `Q=1`, so `query_start_loc[:num_seqs] == range(num_seqs)`
and every row index is `< num_seqs`. That is the same property
`_run_batched_decode_dispatch` already relies on to write vLLM's `output` directly
("result scatter is a single contiguous copy_ at offset 0, valid because Q=1 forces
`query_row_ids_cpu[:num_seqs] == range(num_seqs)`").

## The change

In `SpyreAttentionImpl` (`v1/attention/backends/spyre_attn.py`):

1. Add a second staging destination of `max_num_seqs` rows, allocated once beside
   the existing pair in `_staging_buffers`. Keep it a **run constant** — sizing it to
   the step's token count would make Dynamo guard on the model graph's token bucket,
   which is the whole reason `staging_rows` exists.
2. In `_online_softmax_attention`, where it currently does
   `dest = out_staging if store_out else output`, select the narrow buffer when the
   batch is decode-only (`max(aligned_query_lens, default=1) == 1`), and copy back
   `dest[:num_query_rows]` instead of `out_staging[:batch_rows]`.
3. Leave `q_staging` at 513 rows. The win measured here comes from the output
   destination alone; the query gather source is untouched, so torch-spyre#4033's
   strict-subset rule still holds.
4. Add the narrow-destination shape to `_record_one` so warmup records it — one extra
   variant, not one per bucket. Without it the first decode step compiles mid-request.

## Scope

Decode-only batches. Mixed and prefill batches keep the 513-row path unchanged, so
the fallback behaviour is exactly today's.
