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

That guarantee is what lets the attention impl drop its runtime fallback. The base
``Scheduler`` is expensive to construct, so these drive the override with a stub self.
"""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.request_queue import FCFSRequestQueue

from spyre_inference.v1.core.scheduler import PoolingSpyreScheduler, TorchSpyreScheduler

# (prompt_length, batch_size), sorted by (B, L) as encoder_warmup_shapes returns.
SHAPES = [(512, 2), (256, 8), (64, 32)]


def _req(num_prompt_tokens: int, name: str = ""):
    return SimpleNamespace(num_prompt_tokens=num_prompt_tokens, name=name)


def _run(shapes, waiting, running=(), monkeypatch=None, max_num_running_reqs=None):
    """Return (admitted, left_waiting) for one ``schedule()`` call.

    The stub base admits up to ``max_num_running_reqs``, as upstream's waiting loop does.
    """
    sched = PoolingSpyreScheduler.__new__(PoolingSpyreScheduler)
    sched.spyre_warmup_shapes = list(shapes)
    # The real queue type, so the RequestQueue API is exercised rather than deque ops.
    sched.waiting = FCFSRequestQueue(waiting)
    sched.running = list(running)
    sched.max_num_running_reqs = (
        max(batch for _, batch in shapes) if max_num_running_reqs is None else max_num_running_reqs
    )

    admitted: list = []

    def _record(self, *args, **kwargs):
        while self.waiting and len(self.running) + len(admitted) < self.max_num_running_reqs:
            admitted.append(self.waiting.pop_request())
        return "scheduler-output"

    monkeypatch.setattr(TorchSpyreScheduler, "schedule", _record)
    out = PoolingSpyreScheduler.schedule(sched)
    assert out == "scheduler-output"
    assert sched.max_num_running_reqs == (
        max(batch for _, batch in shapes) if max_num_running_reqs is None else max_num_running_reqs
    ), "the clamp must be restored"
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
        admitted, left = _run([(512, 8)], [_req(400) for _ in range(8)], monkeypatch=monkeypatch)
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


class TestUpstreamAdmissionIsClampedToTheGate:
    """The gate's approved set is pinned into ``max_num_running_reqs`` for the base call.

    Upstream admits on ``max_num_seqs``, the widest declared shape rather than the widest
    covering *this* batch's length, so anything past the gate reaches the runner with no
    shape covering it.
    """

    SHAPES = [(512, 2), (256, 8), (64, 32)]

    def _observed_cap(self, waiting, running, monkeypatch, max_num_running_reqs=32):
        sched = PoolingSpyreScheduler.__new__(PoolingSpyreScheduler)
        sched.spyre_warmup_shapes = list(self.SHAPES)
        sched.waiting = FCFSRequestQueue(waiting)
        sched.running = list(running)
        sched.max_num_running_reqs = max_num_running_reqs
        seen: list[int] = []

        def _peek(self, *args, **kwargs):
            seen.append(self.max_num_running_reqs)
            return "scheduler-output"

        monkeypatch.setattr(TorchSpyreScheduler, "schedule", _peek)
        PoolingSpyreScheduler.schedule(sched)
        assert sched.max_num_running_reqs == max_num_running_reqs, "cap not restored"
        return seen[0]

    def test_the_cap_matches_the_approved_batch_not_max_num_seqs(self, monkeypatch):
        # Five 512-token requests: only (512, 2) fits, so the gate approves 2. Left at
        # 32, upstream would admit five and the runner would have no shape for them.
        assert self._observed_cap([_req(512) for _ in range(5)], [], monkeypatch) == 2

    def test_a_wide_short_batch_keeps_its_full_width(self, monkeypatch):
        assert self._observed_cap([_req(10) for _ in range(10)], [], monkeypatch) == 10

    def test_nothing_is_admitted_while_a_batch_is_in_flight(self, monkeypatch):
        # The gate is skipped and waiting is held back, so the cap is just the running set.
        running = [_req(10), _req(10)]
        assert self._observed_cap([_req(10) for _ in range(4)], running, monkeypatch) == 2

    def test_an_already_lower_cap_is_not_raised(self, monkeypatch):
        cap = self._observed_cap(
            [_req(10) for _ in range(10)], [], monkeypatch, max_num_running_reqs=3
        )
        assert cap == 3

    def test_the_cap_is_restored_even_when_the_base_raises(self, monkeypatch):
        sched = PoolingSpyreScheduler.__new__(PoolingSpyreScheduler)
        sched.spyre_warmup_shapes = list(self.SHAPES)
        sched.waiting = FCFSRequestQueue([_req(10)])
        sched.running = []
        sched.max_num_running_reqs = 32

        def _boom(self, *args, **kwargs):
            raise RuntimeError("upstream blew up")

        monkeypatch.setattr(TorchSpyreScheduler, "schedule", _boom)
        with pytest.raises(RuntimeError, match="upstream blew up"):
            PoolingSpyreScheduler.schedule(sched)
        assert sched.max_num_running_reqs == 32


class TestHeldBackRequestsSurviveAFailure:
    """A raise in the base scheduler must not drop the requests the gate held back.

    The gate drains ``self.waiting`` into local deques before delegating, so anything it
    does not re-add is lost to the engine entirely rather than retried next step.
    """

    SHAPES = [(512, 2), (256, 8), (64, 32)]

    def _stub(self, reqs):
        sched = PoolingSpyreScheduler.__new__(PoolingSpyreScheduler)
        sched.spyre_warmup_shapes = list(self.SHAPES)
        sched.waiting = FCFSRequestQueue(list(reqs))
        sched.running = []
        sched.max_num_running_reqs = 32
        return sched

    def _raise_from_base(self, sched, monkeypatch):
        def _boom(self, *args, **kwargs):
            raise RuntimeError("upstream blew up")

        monkeypatch.setattr(TorchSpyreScheduler, "schedule", _boom)
        with pytest.raises(RuntimeError, match="upstream blew up"):
            PoolingSpyreScheduler.schedule(sched)

    def test_holdback_is_restored_when_the_base_raises(self, monkeypatch):
        # Only (512, 2) fits a 512-token request, so 2 are approved and 3 held back.
        reqs = [_req(512, f"r{i}") for i in range(5)]
        sched = self._stub(reqs)
        self._raise_from_base(sched, monkeypatch)
        assert {r.name for r in sched.waiting} == {r.name for r in reqs}

    def test_skipped_requests_are_restored_too(self, monkeypatch):
        # r0 takes the only (512, 2) shape; r1/r3 are short and skipped past it; r2 is
        # another 512 that cannot join.
        reqs = [_req(512, "r0"), _req(10, "r1"), _req(512, "r2"), _req(10, "r3")]
        sched = self._stub(reqs)
        self._raise_from_base(sched, monkeypatch)
        assert {r.name for r in sched.waiting} == {"r0", "r1", "r2", "r3"}
