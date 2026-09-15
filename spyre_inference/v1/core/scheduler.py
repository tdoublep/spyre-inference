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

# SPDX-License-Identifier: Apache-2.0

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler

from spyre_inference import envs


class TorchSpyreScheduler(Scheduler):
    """V1 scheduler that caps how many sequences may prefill in one batch.

    ``max_num_batched_tokens`` is capped at the largest compiled body bucket, so a
    long prompt is chunked. Whenever the last chunk leaves budget unspent, upstream
    tops the batch up with the next waiting request, giving a batch of two ragged
    prefills. Attention runs one kernel per sequence and pads each to its own query
    bucket, so that leftover chunk costs a full-width prefill however few tokens it
    carries. Serialising prefills spends the same budget in one kernel instead.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_num_partial_prefills = envs.SPYRE_MAX_NUM_PARTIAL_PREFILLS

    def schedule(self, *args, **kwargs) -> SchedulerOutput:
        if self.max_num_partial_prefills <= 0:
            return super().schedule(*args, **kwargs)

        prefilling = sum(
            request.num_computed_tokens < request.num_prompt_tokens for request in self.running
        )
        free_slots = max(self.max_num_partial_prefills - prefilling, 0)
        # The waiting loop re-reads `max_num_running_reqs` on every iteration and
        # appends exactly one request to `self.running` per admission, so lowering
        # the cap to the current occupancy plus the free prefill slots stops it
        # after that many admissions while still letting it skip over candidates it
        # cannot schedule. A step that preempts is not a concern: upstream skips the
        # waiting loop entirely on those.
        max_num_running_reqs = self.max_num_running_reqs
        self.max_num_running_reqs = min(
            max_num_running_reqs,
            len(self.running) + self.num_waiting_for_streaming_input + free_slots,
        )
        try:
            return super().schedule(*args, **kwargs)
        finally:
            self.max_num_running_reqs = max_num_running_reqs
