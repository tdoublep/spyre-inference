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

from collections import deque

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from spyre_inference import envs
from spyre_inference.v1.worker.spyre_shape_bucketer import encoder_warmup_shapes

logger = init_logger(__name__)


class TorchSpyreScheduler(Scheduler):
    """V1 scheduler that caps how many sequences may prefill in one batch.

    Attention runs one kernel per sequence, padding each to its own query bucket, so
    the short leftover chunk upstream uses to top up a batch costs a full-width
    prefill however few tokens it carries.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Pooling runners never decode, so there is no leftover chunk to avoid and
        # serialising their prefills only gives up batching.
        self.max_num_partial_prefills = (
            0
            if self.vllm_config.model_config.runner_type == "pooling"
            else envs.SPYRE_MAX_NUM_PARTIAL_PREFILLS
        )

    def schedule(self, *args, **kwargs) -> SchedulerOutput:
        if self.max_num_partial_prefills <= 0:
            return super().schedule(*args, **kwargs)

        prefilling = sum(
            request.num_computed_tokens < request.num_prompt_tokens for request in self.running
        )
        free_slots = max(self.max_num_partial_prefills - prefilling, 0)
        # The waiting loop re-reads this every iteration and appends one request per
        # admission, so a lowered cap stops it after `free_slots` of them while still
        # letting it skip candidates. Steps that preempt skip the loop entirely.
        max_num_running_reqs = self.max_num_running_reqs
        self.max_num_running_reqs = min(
            max_num_running_reqs,
            len(self.running) + self.num_waiting_for_streaming_input + free_slots,
        )
        try:
            return super().schedule(*args, **kwargs)
        finally:
            self.max_num_running_reqs = max_num_running_reqs


class PoolingSpyreScheduler(TorchSpyreScheduler):
    """Only admit pooling batches a declared compile shape covers.

    Constraining the batch is what removes the runtime fallback: ``pick_encoder_shape``
    cannot miss, so nothing compiles mid-request. A request longer than every declared
    length is rejected earlier by vLLM's ``max_model_len`` check.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spyre_warmup_shapes: list[tuple[int, int]] = encoder_warmup_shapes()

    def _fitting_shapes(
        self,
        request: Request,
        shapes: list[tuple[int, int]],
        current_batch_size: int,
    ) -> list[tuple[int, int]]:
        """Shapes that hold this request *and* leave room for it in the batch."""
        return [
            (length, batch)
            for length, batch in shapes
            if request.num_prompt_tokens <= length and current_batch_size < batch
        ]

    def schedule(self, *args, **kwargs) -> SchedulerOutput:
        """Admit a shape-compatible batch, then delegate to the base scheduler.

        The waiting queue is drained into a holdback deque so the base scheduler only sees
        requests sharing a shape; the rest are returned with priority order intact.
        """
        # `pop_request` / `add_request`, not deque ops: `self.waiting` is a
        # RequestQueue, and PriorityRequestQueue is not a deque.
        holdback_queue: deque[Request] = deque()
        while self.waiting:
            holdback_queue.append(self.waiting.pop_request())

        # Requests that fit no surviving shape but might fit a later batch.
        skip_queue: deque[Request] = deque()

        # Only start a batch when nothing is in flight: pooling has no decode step,
        # so a running batch owns the device until it finishes.
        if len(self.running) == 0:
            available = list(self.spyre_warmup_shapes)
            last_available = available

            while holdback_queue:
                request = holdback_queue[0]
                available = self._fitting_shapes(request, available, len(self.waiting))

                if available:
                    self.waiting.add_request(holdback_queue.popleft())
                    last_available = available
                    continue

                # No shape holds this request alongside the ones already admitted.
                # If the widest surviving shape still has room, skip past it and try
                # the next request; otherwise the batch is full.
                max_batch = max(batch for _, batch in last_available)
                if len(self.waiting) < max_batch:
                    available = last_available
                    skip_queue.append(holdback_queue.popleft())
                else:
                    break

            logger.debug(
                "Scheduling a new batch of %d requests, holding back %d",
                len(self.waiting),
                len(holdback_queue) + len(skip_queue),
            )

        outputs = super().schedule(*args, **kwargs)

        # Skipped first, then never-considered: preserves the original priority.
        while skip_queue:
            self.waiting.add_request(skip_queue.popleft())
        while holdback_queue:
            self.waiting.add_request(holdback_queue.popleft())

        return outputs
