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

"""The fast path's grid layout is a contract between two places in the runner:
``_preprocess`` scatters the packed tokens into ``[B*L]``, and ``_unpad_encoder_hidden``
gathers them back for the pooler. They share one row-index table, so a round trip is
the thing worth testing.
"""

import pytest
import torch

from spyre_inference.v1.worker.spyre_shape_bucketer import (
    encoder_dense_row_indices,
    expand_packed_to_encoder_grid,
)


@pytest.mark.parametrize(
    "query_lens",
    [
        [64, 64],
        [1, 64, 32],
        [10],
        [64] * 4,
        [7, 1, 63, 64],
    ],
)
def test_expand_then_gather_round_trips(query_lens):
    extent, width = 64, len(query_lens) + 2  # spare batch-pad lanes
    total = sum(query_lens)
    ids = torch.arange(1, total + 1, dtype=torch.int64)
    positions = torch.cat([torch.arange(n, dtype=torch.int64) for n in query_lens])

    grid_ids, grid_pos = expand_packed_to_encoder_grid(
        ids, positions, query_lens, width, extent, pad_token_id=0
    )
    assert grid_ids.shape[0] == width * extent

    rows = encoder_dense_row_indices(query_lens, extent)
    assert torch.equal(grid_ids[rows], ids)
    assert torch.equal(grid_pos[rows], positions)


def test_real_pad_continues_positions_and_batch_pad_restarts():
    """Real pad rows continue the sequence's own positions so a position embedding
    stays in range; batch-pad lanes are filler whose output is never read."""
    extent, width = 64, 3
    query_lens = [4, 2]
    ids = torch.arange(1, 7, dtype=torch.int64)
    positions = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.int64)

    grid_ids, grid_pos = expand_packed_to_encoder_grid(
        ids, positions, query_lens, width, extent, pad_token_id=9
    )

    # Sequence 0: 4 real tokens, then its pad rows carry positions 4..63.
    assert grid_ids[:4].tolist() == [1, 2, 3, 4]
    assert grid_pos[:8].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert grid_ids[4:extent].unique().tolist() == [9]
    # The third lane is pure batch pad: filler ids, positions 0..L-1.
    base = 2 * extent
    assert grid_ids[base : base + extent].unique().tolist() == [9]
    assert grid_pos[base : base + 5].tolist() == [0, 1, 2, 3, 4]
    # Every position stays inside the declared length, which is what keeps an
    # offset position table in bounds.
    assert int(grid_pos.max()) == extent - 1


def test_positions_never_exceed_the_declared_length():
    extent, width = 128, 4
    query_lens = [100, 5, 128]
    ids = torch.ones(sum(query_lens), dtype=torch.int64)
    positions = torch.cat([torch.arange(n, dtype=torch.int64) for n in query_lens])
    _, grid_pos = expand_packed_to_encoder_grid(ids, positions, query_lens, width, extent)
    assert int(grid_pos.max()) == extent - 1


def test_more_sequences_than_the_rectangle_is_rejected():
    with pytest.raises(ValueError, match="exceeds batch_bucket"):
        expand_packed_to_encoder_grid(
            torch.zeros(3, dtype=torch.int64),
            torch.zeros(3, dtype=torch.int64),
            [1, 1, 1],
            2,
            64,
        )


def test_a_length_past_the_extent_is_rejected():
    with pytest.raises(ValueError, match="exceeds len_bucket"):
        encoder_dense_row_indices([65], 64)


def test_empty_batch_yields_no_rows():
    assert encoder_dense_row_indices([], 64).numel() == 0
