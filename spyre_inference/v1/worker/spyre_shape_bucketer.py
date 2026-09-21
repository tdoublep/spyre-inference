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

"""Shape bucketing for compilation warmup and runtime dispatch.

Decoder: sorted ``compile_sizes`` token counts; pad the packed batch to the nearest
bucket.

Pooling: ``(prompt_length, batch_size)`` pairs declared through
``SPYRE_WARMUP_PROMPT_LENS`` / ``SPYRE_WARMUP_BATCH_SIZES``, zipped pairwise.
``max_model_len``, ``max_num_seqs`` and the token budget are derived from them, and every
sequence is padded to ``L`` before the model runs, so the body sees exactly ``B * L`` rows.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from vllm.config import VllmConfig
from vllm.logger import init_logger

from spyre_inference import envs

logger = init_logger(__name__)

# Spyre stick, in fp16 elements. Declared prompt lengths must be a multiple of it.
ENCODER_SEQ_ALIGNMENT = 64


def _align_up(n: int, align: int = ENCODER_SEQ_ALIGNMENT) -> int:
    return max(align, (n + align - 1) // align * align)


def next_bucket(n: int, buckets: list[int]) -> int:
    """Smallest bucket ``>= n``. If ``n`` exceeds every bucket, stick-align ``n``."""
    if n < 1:
        n = 1
    ordered = sorted({b for b in buckets if b > 0})
    for bucket in ordered:
        if bucket >= n:
            return bucket
    return _align_up(n)


def encoder_warmup_shapes(
    prompt_lens: Sequence[int] | None = None,
    batch_sizes: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    """Declared ``(prompt_length, batch_size)`` pairs, sorted by ``(B, L)``.

    The two env lists are zipped, not crossed: index ``i`` is one compiled graph. Sorting
    by width makes ``pick_encoder_shape`` prefer the cheapest covering shape. Args are for
    tests; production reads the env.
    """
    lens = list(envs.SPYRE_WARMUP_PROMPT_LENS if prompt_lens is None else prompt_lens)
    batches = list(envs.SPYRE_WARMUP_BATCH_SIZES if batch_sizes is None else batch_sizes)

    if len(lens) != len(batches):
        raise ValueError(
            "SPYRE_WARMUP_PROMPT_LENS and SPYRE_WARMUP_BATCH_SIZES must have equal "
            f"length; got {len(lens)} lengths {lens} and {len(batches)} batch sizes "
            f"{batches}. They are zipped pairwise, not crossed."
        )
    if not lens:
        raise ValueError("SPYRE_WARMUP_PROMPT_LENS is empty; at least one shape is required")
    bad = [length for length in lens if length % ENCODER_SEQ_ALIGNMENT or length <= 0]
    if bad:
        raise ValueError(
            f"All SPYRE_WARMUP_PROMPT_LENS must be positive multiples of "
            f"{ENCODER_SEQ_ALIGNMENT} (the Spyre stick); got {bad}"
        )
    bad_batches = [batch for batch in batches if batch <= 0]
    if bad_batches:
        raise ValueError(f"All SPYRE_WARMUP_BATCH_SIZES must be positive; got {bad_batches}")

    return sorted(
        {(int(length), int(batch)) for length, batch in zip(lens, batches)},
        key=lambda pair: (pair[1], pair[0]),
    )


def pick_encoder_shape(
    num_seqs: int,
    max_len: int,
    shapes: Sequence[tuple[int, int]],
) -> tuple[int, int] | None:
    """First declared ``(L, B)`` covering the batch, or ``None``.

    ``None`` needs no fallback: the scheduler gate and vLLM's ``max_model_len`` check
    both reject such a batch before it reaches the runner.
    """
    if num_seqs < 1 or max_len < 1:
        return None
    for length, batch in shapes:
        if batch >= num_seqs and length >= max_len:
            return length, batch
    return None


def encoder_body_sizes(shapes: Sequence[tuple[int, int]]) -> list[int]:
    """Distinct ``B * L`` counts: the body is flat, so equal-area shapes share a graph."""
    return sorted({length * batch for length, batch in shapes})


def logits_row_buckets(bucket_sizes: Sequence[int], max_num_reqs: int) -> list[int]:
    """Row widths the lm_head can see: each body bucket clipped to ``max_num_reqs``."""
    cap = max(1, max_num_reqs)
    return sorted({min(size, cap) for size in bucket_sizes if size > 0})


class EncoderBucketPad(NamedTuple):
    """Runtime pad of a pooling batch onto a declared ``(B, L)`` shape."""

    batch_bucket: int
    len_bucket: int
    orig_query_lens: list[int]
    orig_num_tokens: int
    orig_num_reqs: int

    @property
    def num_tokens(self) -> int:
        return self.batch_bucket * self.len_bucket


def expand_packed_to_encoder_bucket(
    input_ids: list[int],
    positions: list[int],
    query_lens: list[int],
    batch_bucket: int,
    len_bucket: int,
    pad_token_id: int = 0,
) -> tuple[list[int], list[int]]:
    """Pad each sequence to ``L`` and the batch to ``B``; return ``[B*L]`` lists.

    Real pad tokens continue positions from the true length. Dummy sequences
    (batch pad) are ``pad_token_id`` with positions ``0 .. L-1``.
    """
    if len(query_lens) > batch_bucket:
        raise ValueError(f"num_seqs={len(query_lens)} exceeds batch_bucket={batch_bucket}")
    if any(length > len_bucket for length in query_lens):
        raise ValueError(f"a query length exceeds len_bucket={len_bucket}: {query_lens}")

    total = batch_bucket * len_bucket
    padded_ids = [int(pad_token_id)] * total
    padded_pos = [0] * total
    src = 0
    for seq_idx, length in enumerate(query_lens):
        dst = seq_idx * len_bucket
        padded_ids[dst : dst + length] = list(input_ids[src : src + length])
        padded_pos[dst : dst + length] = list(positions[src : src + length])
        for offset in range(length, len_bucket):
            padded_pos[dst + offset] = offset
        src += length
    for seq_idx in range(len(query_lens), batch_bucket):
        dst = seq_idx * len_bucket
        for offset in range(len_bucket):
            padded_pos[dst + offset] = offset
    return padded_ids, padded_pos


def encoder_bucket_valid_row_indices(
    orig_query_lens: list[int],
    len_bucket: int,
) -> list[int]:
    """Row indices of real tokens inside a ``B×L`` packed hidden state."""
    indices: list[int] = []
    for seq_idx, length in enumerate(orig_query_lens):
        start = seq_idx * len_bucket
        indices.extend(range(start, start + length))
    return indices


@dataclass(frozen=True)
class SpyreBucketDescriptor:
    """Descriptor for a 1D (decoder) compilation bucket."""

    actual_num_tokens: int
    padded_num_tokens: int


@dataclass(frozen=True)
class EncoderBucketDescriptor:
    """Descriptor for a 2D encoder ``(B, L)`` compilation bucket."""

    batch_bucket: int
    len_bucket: int
    actual_num_seqs: int
    actual_max_len: int

    @property
    def padded_num_tokens(self) -> int:
        return self.batch_bucket * self.len_bucket


class SpyreShapeBucketer:
    """Dispatches runtime batches to pre-compiled 1D bucket sizes.

    Pooling shares this: with every sequence padded to ``L`` before the model, the
    body's token count is exactly ``B * L``, which is one of the declared shapes'
    products, so the same 1D lookup serves both runners.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        compilation_config = vllm_config.compilation_config
        sizes: list[int] = [int(s) for s in (compilation_config.compile_sizes or [])]
        self._bucket_sizes = sorted(set(sizes))
        self._max_bucket_size = self._bucket_sizes[-1] if self._bucket_sizes else 0
        self._is_warmed_up = False

        logger.info(
            "SpyreShapeBucketer initialized with %d bucket sizes: min=%d, max=%d",
            len(self._bucket_sizes),
            self._bucket_sizes[0] if self._bucket_sizes else 0,
            self._max_bucket_size,
        )

    @property
    def bucket_sizes(self) -> list[int]:
        return self._bucket_sizes

    @property
    def max_bucket_size(self) -> int:
        return self._max_bucket_size

    @property
    def is_warmed_up(self) -> bool:
        return self._is_warmed_up

    def mark_warmed_up(self) -> None:
        self._is_warmed_up = True

    def find_bucket(self, num_tokens: int) -> int | None:
        """Find the smallest 1D bucket size >= num_tokens.

        Returns None if num_tokens exceeds the largest compiled bucket.
        The caller (execute_model) handles the None case by running the
        forward pass without bucket padding, which may trigger Dynamo
        recompilation for the unseen shape.
        """
        idx = bisect.bisect_left(self._bucket_sizes, num_tokens)
        if idx < len(self._bucket_sizes):
            return self._bucket_sizes[idx]
        return None

    def dispatch(self, num_tokens: int) -> SpyreBucketDescriptor | None:
        """Compute padded batch descriptor for the given token count.

        Returns None if no suitable bucket exists.
        """
        padded = self.find_bucket(num_tokens)
        if padded is None:
            return None
        return SpyreBucketDescriptor(
            actual_num_tokens=num_tokens,
            padded_num_tokens=padded,
        )
