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

"""Scheduler that only admits pooling batches a declared compile shape covers.

Encoder attention runs on a dense ``[B, L]`` grid whose ``(L, B)`` pairs are
declared up front (``SPYRE_WARMUP_PROMPT_LENS`` / ``SPYRE_WARMUP_BATCH_SIZES``).
Rather than pick a shape after the fact and compile whatever the scheduler
happened to batch, this constrains the batch so a declared shape always fits.
That is what removes the runtime fallback: ``pick_encoder_shape`` cannot miss, so
nothing compiles mid-request.

A request longer than every declared length never reaches here --
``max_model_len`` is derived from the longest shape, so vLLM's own length check
rejects it.
"""

from __future__ import annotations

from collections import deque

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from spyre_inference.v1.worker.spyre_shape_bucketer import encoder_warmup_shapes

logger = init_logger(__name__)


class PoolingSpyreScheduler(Scheduler):
    """Gate admission so every batch matches one declared ``(L, B)`` shape."""

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

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """Admit a shape-compatible batch, then delegate to the base scheduler.

        The whole waiting queue is drained into a holdback deque first so the base
        scheduler only ever sees requests that share a shape. Anything held back is
        returned afterwards with its priority order intact.
        """
        holdback_queue: deque[Request] = deque()
        while self.waiting:
            holdback_queue.append(self.waiting.popleft())

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
                    self.waiting.append(holdback_queue.popleft())
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

        outputs = super().schedule(throttle_prefills=throttle_prefills)

        # Skipped first, then never-considered: preserves the original priority.
        while skip_queue:
            self.waiting.append(skip_queue.popleft())
        while holdback_queue:
            self.waiting.append(holdback_queue.popleft())

        return outputs
