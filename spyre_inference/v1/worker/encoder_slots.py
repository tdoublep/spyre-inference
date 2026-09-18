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

"""Token-level (slot) padding for encoder-only steps.

vLLM hands the body a var-len *packed* token list, so encoder attention has to
lift each request out of it (`index_select`), attend, and put it back
(`index_copy_`). Those two moves are 60% of the attention kernel's device time
and run at 16-33 GB/s -- a plain copy of the same bytes runs at 116 GB/s.

Giving every request its own fixed ``extent``-row slot instead makes both moves
free ``.view()``s. The padding is not new work: the body bucket already rounds
the step up to a power of two, and ``num_slots * extent`` is that same bucket.

The reorder is applied to ``input_ids`` and ``positions`` while they are still
host int tensors, so the embedding emits slot-aligned hidden states at zero
device cost -- cheaper than permuting the embeddings afterwards.

The layout is published here rather than threaded through the model because the
two readers (the attention impl and the pooler) sit either side of a compiled
region that must not take it as an argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EncoderSlotLayout:
    """Where each request's rows live in the slot-padded body buffer."""

    extent: int
    num_slots: int
    num_seqs: int
    query_lens: tuple[int, ...]

    @property
    def total_rows(self) -> int:
        return self.num_slots * self.extent

    @property
    def row_starts(self) -> tuple[int, ...]:
        return tuple(i * self.extent for i in range(self.num_seqs))

    def kv_lens(self) -> list[int]:
        """Per-slot real KV length; unused slots attend over themselves.

        A fully masked row would make softmax produce NaN, so an unused slot is
        given a full-width mask. Its output is garbage no reader touches.
        """
        return [*self.query_lens, *([self.extent] * (self.num_slots - self.num_seqs))]


_current: EncoderSlotLayout | None = None


def set_current(layout: EncoderSlotLayout | None) -> None:
    global _current
    _current = layout


def current() -> EncoderSlotLayout | None:
    return _current


def repack(vec: torch.Tensor, starts: list[int], lens: list[int], layout: EncoderSlotLayout, pad):
    """Scatter a packed host vector into its slots. Loops over <= max_num_seqs."""
    out = vec.new_full((layout.total_rows,), pad)
    for i, (start, length) in enumerate(zip(starts, lens)):
        out[i * layout.extent : i * layout.extent + length] = vec[start : start + length]
    return out


def plan(query_lens: list[int], find_bucket, max_extent: int, max_slots: int):
    """Pick the slot layout for a step, or ``None`` to keep the packed layout.

    ``extent`` follows the *longest* request in the batch, not ``max_model_len``,
    so a batch of short requests still lands on a small body bucket. The total is
    rounded to a body bucket because that is the only width the model is compiled
    for -- which is also why this costs no extra GEMM work: the packed layout was
    being rounded to the same bucket anyway.
    """
    from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
        ENCODER_LEN_ALIGNMENT,
        _alignment_units_for,
    )

    if not query_lens:
        return None
    extent = _alignment_units_for(max(query_lens)) * ENCODER_LEN_ALIGNMENT
    if extent > max_extent:
        return None
    total = find_bucket(len(query_lens) * extent)
    if total is None or total % extent:
        return None
    num_slots = total // extent
    if num_slots < len(query_lens) or num_slots > max_slots:
        return None
    return EncoderSlotLayout(
        extent=extent,
        num_slots=num_slots,
        num_seqs=len(query_lens),
        query_lens=tuple(query_lens),
    )
