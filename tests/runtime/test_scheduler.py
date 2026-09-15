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

"""Unit tests for TorchSpyreScheduler's partial-prefill cap."""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from spyre_inference.v1.core.scheduler import TorchSpyreScheduler


def _request(num_computed_tokens: int, num_prompt_tokens: int = 1984) -> SimpleNamespace:
    return SimpleNamespace(
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=num_prompt_tokens,
    )


def _scheduler(running, max_num_partial_prefills=1, max_num_seqs=4):
    scheduler = TorchSpyreScheduler.__new__(TorchSpyreScheduler)
    scheduler.running = running
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_num_running_reqs = max_num_seqs
    scheduler.max_num_partial_prefills = max_num_partial_prefills
    return scheduler


@pytest.fixture()
def observed_cap(monkeypatch):
    """Record the `max_num_running_reqs` the upstream scheduler loop would see."""
    seen = []
    monkeypatch.setattr(
        Scheduler, "schedule", lambda self, *a, **kw: seen.append(self.max_num_running_reqs)
    )
    return seen


@pytest.mark.parametrize(
    ("running", "expected"),
    [
        # Nothing running: admit exactly one prefill.
        ([], 1),
        # One sequence mid-prefill: no slot left, so no further admission.
        ([_request(512)], 1),
        # Its final chunk landed, so it decodes from now on: admit the next prefill.
        ([_request(1984)], 2),
        # Decodes never block an admission, but they do occupy running slots.
        ([_request(1984), _request(1984), _request(512)], 3),
    ],
)
def test_cap_leaves_one_prefill_slot(running, expected, observed_cap):
    _scheduler(running).schedule()
    assert observed_cap == [expected]


def test_cap_never_exceeds_max_num_seqs(observed_cap):
    _scheduler([_request(1984)] * 4, max_num_seqs=4).schedule()
    assert observed_cap == [4]


def test_higher_cap_allows_more_concurrent_prefills(observed_cap):
    _scheduler([_request(512)], max_num_partial_prefills=3).schedule()
    assert observed_cap == [3]


def test_zero_disables_the_cap(observed_cap):
    _scheduler([_request(512)], max_num_partial_prefills=0).schedule()
    assert observed_cap == [4]


def test_streaming_slots_counted_as_occupancy(observed_cap):
    scheduler = _scheduler([_request(1984)], max_num_seqs=8)
    scheduler.num_waiting_for_streaming_input = 2
    scheduler.schedule()
    assert observed_cap == [4]
