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

"""This step's batched-decode inputs, in the form the block graph reads them.

The batched decode kernel is the one attention shape with no per-sequence Python
loop, so it can be traced *into* the block graph rather than dispatched through the
opaque ``unified_attention_with_output``. That removes one device-program boundary
per block per step, along with the query/output staging copies the opaque call needs
to keep its own shapes constant.

The cost is that the kernel's shape parameters become guards on the block graph, so
one armed plan shape is one more block-graph variant. ``dummy_arm`` exists so warmup
can trace each reachable shape instead of leaving a 40-block Inductor compile in the
serving path.
"""

from __future__ import annotations

import torch

from spyre_inference.custom_ops.utils import convert

# Entries target the core count: fewer under-fills them, more than one stick's
# worth hits a backend axis-merge limit.
_SPYRE_CORE_COUNT = 32


def chunk_geometry(num_seqs: int, num_blocks: int) -> tuple[int, int]:
    """``(blocks_per_chunk, num_chunks)`` for a padded ``(num_seqs, num_blocks)``.

    Shared by the metadata builder and warmup so a recorded variant is always one
    ``build()`` can produce. ``blocks_per_chunk`` need not divide ``num_blocks``, so
    the block axis pads up to a whole chunk; padding columns gather page 0 under an
    all-``-inf`` mask and contribute zero.
    """
    blocks_per_chunk = max(1, min(_SPYRE_CORE_COUNT // num_seqs, num_blocks))
    num_chunks = (num_blocks + blocks_per_chunk - 1) // blocks_per_chunk
    return blocks_per_chunk, num_chunks


class BatchedDecodePlan:
    """Device-side batched-decode inputs, published per step and read while tracing.

    Mirrors ``SlotMapping``: the metadata builder publishes, the patched
    ``Attention.forward`` reads. Every field must exist before the first trace, since
    Dynamo lifts the tensors as graph inputs and guards the ints.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.armed = False
        self.num_seqs = 0
        self.blocks_per_chunk = 0
        self.block_size = 0
        self.rep_row_ids: torch.Tensor | None = None
        self.chunk_page_ids: list[torch.Tensor] = []
        self.chunk_slot_ids: list[torch.Tensor] = []
        self.mask_by_chunk: torch.Tensor | None = None

    def arm(
        self,
        num_seqs: int,
        blocks_per_chunk: int,
        block_size: int,
        rep_row_ids_cpu: torch.Tensor,
        chunk_page_ids_cpu: list[torch.Tensor],
        mask_by_chunk_cpu: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Mirror one step's inputs to device. Left disarmed when not enabled.

        The per-chunk page ids are expanded to slot ids here: the fused kernel gathers
        the cache slot-major, so that it reads the very tensor the scatter wrote.
        """
        if not self.enabled:
            return
        self.num_seqs = num_seqs
        self.blocks_per_chunk = blocks_per_chunk
        self.block_size = block_size
        self.rep_row_ids = convert(rep_row_ids_cpu, device=device)
        offsets = torch.arange(block_size, dtype=torch.int32)
        self.chunk_slot_ids = [
            convert(page_ids * block_size + offsets, device=device)
            for page_ids in chunk_page_ids_cpu
        ]
        self.chunk_page_ids = [convert(t, device=device) for t in chunk_page_ids_cpu]
        self.mask_by_chunk = convert(mask_by_chunk_cpu, device=device)
        self.armed = True

    def disarm(self) -> None:
        """Send the next trace down the opaque path. Tensors are left in place."""
        self.armed = False


def dummy_arm(
    plan: BatchedDecodePlan,
    num_seqs: int,
    num_blocks: int,
    num_kv_heads: int,
    block_size: int,
    device: torch.device,
) -> None:
    """Arm ``plan`` on zero-filled inputs of one reachable shape, for warmup tracing.

    Zero is the one mask value that cannot leave a row fully masked, which would make
    the softmax denominator zero and the traced result NaN.
    """
    blocks_per_chunk, num_chunks = chunk_geometry(num_seqs, num_blocks)
    entries = num_seqs * blocks_per_chunk
    plan.arm(
        num_seqs=num_seqs,
        blocks_per_chunk=blocks_per_chunk,
        block_size=block_size,
        rep_row_ids_cpu=torch.zeros(entries, dtype=torch.int32),
        chunk_page_ids_cpu=[torch.zeros(entries, 1, dtype=torch.int32) for _ in range(num_chunks)],
        mask_by_chunk_cpu=torch.zeros(
            num_chunks, entries * num_kv_heads, 1, block_size, dtype=torch.float16
        ),
        device=device,
    )
