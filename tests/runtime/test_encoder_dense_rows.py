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

"""Every shape the pooling path hands downstream must be content-independent.

A throughput regression violated this: the boundary gather re-compacted the dense grid to
``[sum(query_lens), hidden]``, whose first dimension tracked the real token count and so
recompiled the pooler on nearly every step once lengths varied. Uniform-length tests
cannot see that, hence the ragged cases here.
"""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

HIDDEN = 8


def _unpad(query_lens: list[int], len_bucket: int, batch_bucket: int) -> torch.Tensor:
    """Drive ``_unpad_encoder_hidden`` against a stub runner.

    Unbound call with a stub self: the method's whole surface is
    ``_encoder_grid``, so this stays host-only instead of building a runner.
    """
    rows = batch_bucket * len_bucket
    hidden = torch.arange(rows * HIDDEN, dtype=torch.float16).reshape(rows, HIDDEN)
    runner = SimpleNamespace(_encoder_grid=(len_bucket, batch_bucket, query_lens))
    return TorchSpyreModelRunner._unpad_encoder_hidden(
        cast(TorchSpyreModelRunner, runner), hidden, sum(query_lens)
    )


# Ragged, and with a different token sum in every case: a shape that tracked the
# sum would produce five different first dimensions here.
_RAGGED = [
    [64, 64, 64, 64],
    [37, 64, 12, 5],
    [1, 1, 1, 1],
    [64, 1, 33, 20],
    [50, 50, 50, 64],
]


@pytest.mark.parametrize("query_lens", _RAGGED)
def test_row_count_is_constant_whatever_the_lengths(query_lens):
    out = _unpad(query_lens, len_bucket=64, batch_bucket=4)
    assert out.shape == (4 * 64, HIDDEN), (
        "gather width tracks content -- this recompiles the pooler per step"
    )


def test_every_ragged_case_agrees_on_one_shape():
    """Stated as a set so the point is the *invariant*, not five constants."""
    shapes = {tuple(_unpad(lens, 64, 4).shape) for lens in _RAGGED}
    assert len(shapes) == 1, f"pooling saw {len(shapes)} distinct shapes: {shapes}"


def test_real_rows_are_compacted_to_the_front():
    """The cursor addresses rows by cumsum of real lengths, so order matters."""
    query_lens = [3, 2]
    out = _unpad(query_lens, len_bucket=64, batch_bucket=4)
    # Sequence 0 is grid rows 0..2, sequence 1 is grid rows 64..65.
    expected_first = [0, 1, 2, 64, 65]
    for compacted, grid_row in enumerate(expected_first):
        assert out[compacted, 0].item() == pytest.approx(grid_row * HIDDEN), (
            f"compacted row {compacted} is not grid row {grid_row}"
        )


def test_a_full_grid_is_returned_untouched():
    """Nothing to compact when every sequence fills its bucket: skip the gather."""
    rows = 2 * 64
    hidden = torch.zeros(rows, HIDDEN, dtype=torch.float16)
    runner = SimpleNamespace(_encoder_grid=(64, 2, [64, 64]))
    out = TorchSpyreModelRunner._unpad_encoder_hidden(
        cast(TorchSpyreModelRunner, runner), hidden, rows
    )
    assert out is hidden


def test_filler_rows_are_in_range():
    """Out-of-range filler indices would fault rather than merely go unread."""
    out = _unpad([1, 1, 1, 1], len_bucket=64, batch_bucket=4)
    assert torch.isfinite(out).all()
