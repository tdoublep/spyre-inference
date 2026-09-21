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

"""``PoolingSpyreScheduler`` admits only batches a declared ``(L, B)`` shape covers.

That guarantee is what lets the attention impl drop its runtime fallback: if a
batch reaching the runner always fits a compiled shape, ``pick_encoder_shape``
cannot miss and nothing compiles mid-request.

Host-only. The base ``Scheduler`` is expensive to construct, so these drive the
override directly with a stub self and a recording base ``schedule``.
"""

from collections import deque
from types import SimpleNamespace
from typing import cast

import pytest

from spyre_inference.v1.core.scheduler import PoolingSpyreScheduler, TorchSpyreScheduler

# (prompt_length, batch_size), sorted by (B, L) as encoder_warmup_shapes returns.
SHAPES = [(512, 2), (256, 8), (64, 32)]


def _req(num_prompt_tokens: int, name: str = ""):
    return SimpleNamespace(num_prompt_tokens=num_prompt_tokens, name=name)


def _run(shapes, waiting, running=(), monkeypatch=None):
    """Return (admitted, left_waiting) for one ``schedule()`` call."""
    sched = PoolingSpyreScheduler.__new__(PoolingSpyreScheduler)
    sched.spyre_warmup_shapes = list(shapes)
    sched.waiting = deque(waiting)
    sched.running = list(running)

    admitted: list = []

    def _record(self, *args, **kwargs):
        admitted.extend(self.waiting)
        self.waiting.clear()
        return "scheduler-output"

    monkeypatch.setattr(TorchSpyreScheduler, "schedule", _record)
    out = PoolingSpyreScheduler.schedule(cast(PoolingSpyreScheduler, sched))
    assert out == "scheduler-output"
    return admitted, list(sched.waiting)


class TestFittingShapes:
    def test_needs_the_length_to_fit_and_the_batch_to_have_room(self):
        sched = PoolingSpyreScheduler.__new__(PoolingSpyreScheduler)
        # A 300-token request only fits L=512, whose width is 2.
        assert sched._fitting_shapes(_req(300), SHAPES, 0) == [(512, 2)]
        # With two already admitted, that shape has no room left.
        assert sched._fitting_shapes(_req(300), SHAPES, 2) == []

    def test_a_short_request_fits_every_shape(self):
        sched = PoolingSpyreScheduler.__new__(PoolingSpyreScheduler)
        assert sched._fitting_shapes(_req(10), SHAPES, 0) == SHAPES


class TestAdmission:
    def test_fills_up_to_the_widest_shape_that_still_fits(self, monkeypatch):
        """Ten short requests: (64, 32) holds them all."""
        admitted, left = _run(SHAPES, [_req(10) for _ in range(10)], monkeypatch=monkeypatch)
        assert len(admitted) == 10
        assert left == []

    def test_stops_at_the_width_of_the_surviving_shape(self, monkeypatch):
        """300-token requests only fit L=512, so at most 2 per batch."""
        reqs = [_req(300, f"r{i}") for i in range(5)]
        admitted, left = _run(SHAPES, reqs, monkeypatch=monkeypatch)
        assert len(admitted) == 2, "a third would not fit any declared shape"
        assert len(left) == 3

    def test_every_admitted_batch_is_covered_by_one_shape(self, monkeypatch):
        """The invariant the attention impl relies on."""
        reqs = [_req(300), _req(10), _req(10), _req(500)]
        admitted, _left = _run(SHAPES, reqs, monkeypatch=monkeypatch)
        assert admitted, "nothing admitted -- the assertion below would be vacuous"
        covering = [
            (length, batch)
            for length, batch in SHAPES
            if batch >= len(admitted) and length >= max(r.num_prompt_tokens for r in admitted)
        ]
        assert covering, f"batch of {len(admitted)} fits no declared shape"

    def test_skips_an_incompatible_request_and_keeps_filling(self, monkeypatch):
        """A long request that blocks the batch is retried later, not dropped.

        One 500-token request forces L=512 (width 2). The short ones that follow
        still fit, so the batch is topped up rather than abandoned.
        """
        reqs = [_req(500, "long"), _req(10, "a"), _req(10, "b")]
        admitted, left = _run(SHAPES, reqs, monkeypatch=monkeypatch)
        assert len(admitted) == 2
        assert len(admitted) + len(left) == 3, "a request was lost"

    def test_holds_everything_back_while_a_batch_is_in_flight(self, monkeypatch):
        """Pooling has no decode step, so a running batch owns the device."""
        reqs = [_req(10) for _ in range(4)]
        admitted, left = _run(SHAPES, reqs, running=[_req(10)], monkeypatch=monkeypatch)
        assert admitted == []
        assert len(left) == 4, "requests must be returned to waiting, not dropped"

    def test_no_request_is_ever_lost(self, monkeypatch):
        """Whatever the mix, admitted + still-waiting accounts for everything."""
        reqs = [_req(n) for n in (500, 10, 300, 64, 10, 512, 20)]
        admitted, left = _run(SHAPES, reqs, monkeypatch=monkeypatch)
        assert len(admitted) + len(left) == len(reqs)

    def test_a_single_declared_shape_still_batches(self, monkeypatch):
        """The default config is one shape; it must not degrade to one per step."""
        admitted, left = _run(
            [(512, 8)], [_req(400) for _ in range(8)], monkeypatch=monkeypatch
        )
        assert len(admitted) == 8
        assert left == []


def test_pooling_scheduler_keeps_the_partial_prefill_base(monkeypatch):
    """It subclasses TorchSpyreScheduler rather than replacing it.

    An earlier revision of this branch dropped TorchSpyreScheduler when adding the
    gate, silently removing the decoder's partial-prefill cap.
    """
    assert issubclass(PoolingSpyreScheduler, TorchSpyreScheduler)
    assert hasattr(TorchSpyreScheduler, "schedule")


@pytest.mark.parametrize("num_reqs", [1, 2, 3, 8, 32])
def test_admitted_count_never_exceeds_the_widest_declared_batch(num_reqs, monkeypatch):
    admitted, _left = _run(SHAPES, [_req(10) for _ in range(num_reqs)], monkeypatch=monkeypatch)
    assert len(admitted) <= max(batch for _, batch in SHAPES)
