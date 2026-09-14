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

"""KV store into the head-major paged cache. See ``SpyreHeadMajorAttentionImpl``."""


def reshape_and_cache_head_major_kernel(key, value, k_rows, v_rows, row_index):
    """Scatter one row per (token, kv_head) into the flat row view of the cache.

    A token's destinations are no longer one contiguous run: head-major puts the same
    token's heads block_size rows apart, so this stores one head at a time. Flattening
    (T, KV) into a single index instead does not compile at all: UnalignedStickSplit on
    the merged row axis.

    The `* 1.0` is the price of the layout, and it is load-bearing. `key` arrives as a
    strided view of the fused QKV projection; slicing a head out of that view stores
    wrong values on device (a whole strided view is a fine source — the token-major
    kernel passes one — it is the per-head slice of one that is not), so the source has
    to be materialized first. It cannot be `contiguous()` or `clone()`: at one token the
    view is *flagged* contiguous while still sitting at a nonzero storage offset, so
    both are elided and the slice reads the wrong data. An elementwise pass lowers to a
    real device op, which nothing elides. If a future Inductor folds it away, the
    single-token fused-QKV cases in tests/attention/test_spyre_head_major_attn.py fail.

    k/v_rows: [num_blocks * num_kv_heads * block_size, head_size] views.
    row_index: one [T] int64 tensor per KV head, each its own allocation — see
    ``SpyreHeadMajorAttentionImpl.kv_write_index`` for why rows of one [KV, T]
    tensor are not usable here.
    """
    key = key * 1.0
    value = value * 1.0
    for h, idx in enumerate(row_index):
        k_rows.index_copy_(0, idx, key[:, h])
        v_rows.index_copy_(0, idx, value[:, h])
