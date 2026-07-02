# Progress

## Round 1 — Orchestrator
- **reasoning**: Single-prompt decode has `max_query_len=1` but the code pads to `QUERY_CHUNK_SIZE=32`, so every one of the hundreds of decode steps runs the attention kernel with 32× more query rows than necessary. The code itself flags this at `spyre_attn.py:51-53` as the obvious next win. This is a small, self-contained change (one file, ~2 conditional edits), targets the biggest single waste on the decode hot path, is guarded by the pytest correctness suite, and preserves the prefill code path untouched. The `_get_attn_fn` cache is already keyed by `padded_query_len`, so the new fast-path kernel coexists with existing prefill compilations without any cache-management work. Deferring mask-tile caching (M2) and CPU-staging removal (M3) to later rounds keeps this round narrow enough to verify.

### Task
Add a decode-only fast path to `spyre_inference/v1/attention/backends/spyre_attn.py` that avoids padding query length up to `QUERY_CHUNK_SIZE=32` when every sequence in the batch has `query_len == 1` (i.e. steady-state decode).

Concrete plan:

1. In `SpyreAttentionMetadataBuilder.build`, when `max_query_len == 1`, set `aligned_max_query_len = 1` (bypassing the round-up to 32). Keep the `>1` path unchanged (prefill/mixed still pad to `QUERY_CHUNK_SIZE`). This automatically shrinks the pre-tiled `attention_mask_tiles` query dimension to 1 as well.

2. In `SpyreAttentionImpl._online_softmax_attention`, for the branch `aligned_max_query_len == 1`, skip the `torch.nn.functional.pad(...)` call entirely (nothing to pad — `query_len == 1 == aligned_max_query_len`). The rest of the reshape (`unsqueeze(0).transpose(1,2).contiguous().reshape(num_kv_heads, num_queries_per_kv, 1, head_size)`) still works with `aligned_max_query_len=1` and is what gets compiled — this is exactly the win: the compiled attn kernel receives a query tensor with 1 row instead of 32, so the score/softmax/output matmuls are 32× smaller along the query dimension.

3. The compiled kernel key is already `(num_blocks, padded_query_len)`, so the decode-fast-path variant will naturally live at `padded_query_len=1` and coexist with the prefill variants — no cache-key changes needed.

4. Do NOT change QUERY_CHUNK_SIZE for the prefill path. Do NOT touch KV_LENGTH_ALIGNMENT. Do NOT touch the CPU staging buffer or mask-tile transfer this round (those are M2/M3). Keep everything else in `_online_softmax_attention` identical.

5. Static-check post-edit that:
   - `QUERY_CHUNK_SIZE = 32` constant is still present (prefill uses it).
   - No new `torch.compile` calls in `spyre_inference/platform.py`.
   - No new imports besides what's needed.

6. Run the correctness gate: `uv run --no-sync pytest -m "not upstream" -q` — must pass with no new failures.

7. Run the offline benchmark once: `uv run --no-sync python examples/offline_inference/torch_spyre_inference.py --num-prompts 1` and record the `Time elapsed for <N> generated tokens is <T> sec` line. Do NOT use `--compare-with-cpu`.

Watch out for: `FallbackWarning` — the smaller query dim (1 rather than 32) shouldn't trigger any new fallback (matmul with a 1-row query is a supported shape family), but grep the run output. If it does emit a FallbackWarning on the hot path, park M1 and open a note; do not merge with a hot-path fallback.

Files touched (should be exactly one):
- `spyre_inference/v1/attention/backends/spyre_attn.py`

Non-goals: do not restructure attention. Do not fuse RMSNorm. Do not touch custom_ops/. Do not adjust MAX_MODEL_LEN_CAP or MAX_NUM_SEQS_CAP.

### Pass criteria
1. `uv run --no-sync pytest -m "not upstream" -q` passes with at least the same pass count as the pre-change baseline; no new failures introduced.
2. Only `spyre_inference/v1/attention/backends/spyre_attn.py` is modified. No other tracked files change.
3. `QUERY_CHUNK_SIZE = 32` constant remains in `spyre_inference/v1/attention/backends/spyre_attn.py` (prefill path still uses it).
4. `KV_LENGTH_ALIGNMENT = 256` remains unchanged in `spyre_inference/v1/attention/backends/spyre_attn.py`.
5. No new `torch.compile(...)` calls in `spyre_inference/platform.py`.
6. When `max_query_len == 1`, `aligned_max_query_len` is set to 1 (not rounded up to 32). Verify by static inspection of `SpyreAttentionMetadataBuilder.build` in `spyre_inference/v1/attention/backends/spyre_attn.py`.
7. Offline benchmark `uv run --no-sync python examples/offline_inference/torch_spyre_inference.py --num-prompts 1` runs to completion and prints a `Time elapsed for <N> generated tokens is <T> sec` line. No `FallbackWarning` from `torch_spyre.ops.fallbacks` appears in the run's stderr for the decode-path attention kernel.
8. Median tok/s from the benchmark (N/T) is ≥ 1.10× the baseline tok/s reported at setup — i.e. at least a 10% speedup, reflecting the ~32× smaller decode query dimension propagating into observable end-to-end gain (some gain is diluted by non-attention work).

## Round 1 — Implementer (attempt 1)
- **expected_behavior**: Correctness gate `uv run --no-sync pytest -m "not upstream" -q` passes: 265 passed, 24 skipped, 4 xfailed (same as baseline). Offline benchmark `uv run --no-sync python examples/offline_inference/torch_spyre_inference.py --num-prompts 1` completes and prints `Time elapsed for 20 generated tokens is 51.36 sec` (≈0.389 tok/s). No new FallbackWarning appears on the hot path — only pre-existing fallbacks (aten.embedding.default in vLLM vocab_parallel_embedding, aten.argmax.default at spyre_attn.py:252 from `scores.max(dim=-1)` which existed before this change). The decode-path compiled attention kernel now receives a query tensor with 1 row instead of 32, coexisting with prefill variants in the `(num_blocks, padded_query_len)` cache.

### Summary
Added decode-only fast path in `SpyreAttentionMetadataBuilder.build`: when `max_query_len == 1`, set `aligned_max_query_len = 1` instead of rounding up to `QUERY_CHUNK_SIZE=32`. The prefill/mixed path (`max_query_len > 1`) still rounds to 32. No other changes: `QUERY_CHUNK_SIZE=32` and `KV_LENGTH_ALIGNMENT=256` constants remain, no new imports, no `torch.compile` calls added to platform.py. The existing `if aligned_max_query_len > query_len:` gate in `_online_softmax_attention` naturally skips the pad call when both equal 1. Only one file modified: `spyre_inference/v1/attention/backends/spyre_attn.py`.

