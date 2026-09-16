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
    """Store one head at a time: head-major puts a token's heads block_size rows apart.

    k/v_rows are [num_blocks * num_kv_heads * block_size, head_size] views; row_index is
    one [T] int64 tensor per KV head. A single index over a flattened (T, KV) source does
    not compile (UnalignedStickSplit on the merged row axis).
    """
    # `key` is a strided view of the fused QKV projection, and the per-head slice of one
    # stores wrong values on device, so the source is materialized here. Not contiguous():
    # at one token the view already reports contiguous. clone() does not fix it either.
    key = key * 1.0
    value = value * 1.0
    for h, idx in enumerate(row_index):
        k_rows.index_copy_(0, idx, key[:, h])
        v_rows.index_copy_(0, idx, value[:, h])


def reshape_and_cache_head_major_kt_kernel(
    key, value, k_pages, v_rows, block_ids, v_row_index, num_blocks, block_size
):
    """Store into a K cache whose pages are ``[head_size, block_size]``.

    K's token axis is innermost there, so a token's key is a column rather than a row and
    the per-token row copy V uses cannot express it. This writes whole pages instead,
    which is exact when the step's tokens are a whole number of block-aligned pages.

    The materializing pass runs *after* the permute on purpose: before it, the scatter is
    left reconciling two stick axes and fails to lower ("no mechanism to resolve stick
    incompatibility").
    """
    kv, head_size = key.shape[1], key.shape[2]
    pages = (
        key.reshape(num_blocks, block_size, kv, head_size).permute(0, 2, 3, 1) * 1.0
    ).reshape(num_blocks, kv, head_size, block_size)
    k_pages.index_copy_(0, block_ids, pages)
    value = value * 1.0
    for h, idx in enumerate(v_row_index):
        v_rows.index_copy_(0, idx, value[:, h])
    return k_pages


def reshape_and_cache_head_major_kt_v_only_kernel(key, value, v_rows, v_row_index):
    """V store alone, for a step whose tokens are not a whole number of pages.

    Only reachable when nothing reads the K it skips: at ``--output-len 1`` the first
    token comes out of the last prefill chunk and no decode step runs.
    """
    value = value * 1.0
    for h, idx in enumerate(v_row_index):
        v_rows.index_copy_(0, idx, value[:, h])
    return v_rows
