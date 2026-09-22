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

Pooling: ``(prompt_length, batch_size)`` shapes, the cross product of
``SPYRE_ATTN_QUERY_BUCKETS`` and ``SPYRE_ATTN_NUM_SEQS_BUCKETS``, both defaulting to one
entry. Every sequence is padded to ``L`` before the model, so the body sees ``B * L`` rows.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

from spyre_inference import envs

logger = init_logger(__name__)

# Spyre stick, in fp16 elements. Declared prompt lengths are rounded up to it.
ENCODER_SEQ_ALIGNMENT = 64


def _align_up(n: int, align: int = ENCODER_SEQ_ALIGNMENT) -> int:
    return max(align, (n + align - 1) // align * align)


def powers_of_two_up_to(n: int, start: int = 1) -> tuple[int, ...]:
    """Powers of 2 in [start, n] (start rounded up to a power of 2), plus n itself.

    The geometric ladder both bucketers default to.
    """
    if n < 1:
        return ()
    v = 1
    while v < start:
        v *= 2
    result = []
    while v < n:
        result.append(v)
        v *= 2
    result.append(n)
    return tuple(result)


def next_bucket(n: int, buckets: list[int]) -> int:
    """Smallest bucket ``>= n``. If ``n`` exceeds every bucket, stick-align ``n``."""
    if n < 1:
        n = 1
    ordered = sorted({b for b in buckets if b > 0})
    for bucket in ordered:
        if bucket >= n:
            return bucket
    return _align_up(n)


def _resolve_encoder_buckets(
    override: Sequence[int] | None,
    env_value: str | None,
    env_name: str,
    limit: int,
    align: bool,
    default: Sequence[int],
    append_limit: bool,
) -> list[int]:
    """One encoder ladder: the override, else the env list, else ``default``.

    ``append_limit`` is for the length axis: ``max_model_len`` is an input, so a shape has
    to cover it, and it is exempt from ``align`` because it is not ours to raise. Widths
    need no such entry -- ``max_num_seqs`` is written back *from* the ladder.
    """
    if override is not None:
        raw = [int(v) for v in override]
    elif env_value:
        raw = [int(v) for v in env_value.split(",") if v.strip()]
    else:
        raw = list(default)
    values = {_align_up(v) if align else v for v in raw if v > 0}
    kept = sorted(v for v in values if v <= limit)
    dropped = sorted(v for v in values if v > limit)
    if dropped:
        logger.warning(
            "%s entries %s exceed %d and are unreachable for a pooling batch; dropping.",
            env_name,
            dropped,
            limit,
        )
    if append_limit and limit not in kept:
        kept.append(limit)
    return kept or [limit]


# Memo for `encoder_warmup_shapes`. The platform hook, the scheduler, the runner and
# every attention layer all re-derive the same list from the same config, so a
# per-layer model paid for it once per layer at startup. Keyed on the inputs, env
# values included, so nothing goes stale.
_ENCODER_SHAPES_MEMO: dict[tuple, list[tuple[int, int]]] = {}


def encoder_warmup_shapes(
    vllm_config: VllmConfig,
    *,
    length_buckets: Sequence[int] | None = None,
    num_seqs_buckets: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    """Declared ``(prompt_length, batch_size)`` shapes, cheapest covering shape first.

    Both ladders default to powers of two: lengths from the stick up to
    ``max_model_len``, widths from 1 up to ``max_num_seqs``. A shape puts ``B * L`` dense
    rows through the body, so ``max_num_batched_tokens`` filters the pairs rather than
    capping a ladder — a short length can carry a wider batch than a long one.

    Idempotent once the caller writes the widest width back to ``max_num_seqs``
    (``TorchSpyrePlatform._apply_pooling_shape_defaults``).
    """
    memo_key = (
        int(vllm_config.model_config.max_model_len),
        int(vllm_config.scheduler_config.max_num_batched_tokens),
        int(vllm_config.scheduler_config.max_num_seqs),
        envs.SPYRE_ATTN_QUERY_BUCKETS,
        envs.SPYRE_ATTN_NUM_SEQS_BUCKETS,
        None if length_buckets is None else tuple(length_buckets),
        None if num_seqs_buckets is None else tuple(num_seqs_buckets),
    )
    memoized = _ENCODER_SHAPES_MEMO.get(memo_key)
    if memoized is not None:
        # A copy: callers own the list they get back.
        return list(memoized)

    max_model_len = int(vllm_config.model_config.max_model_len)
    # Encoder prefill cannot be chunked, so a budget under max_model_len head-of-line
    # blocks the scheduler forever.
    budget = max(int(vllm_config.scheduler_config.max_num_batched_tokens), max_model_len)

    lengths = _resolve_encoder_buckets(
        length_buckets,
        envs.SPYRE_ATTN_QUERY_BUCKETS,
        "SPYRE_ATTN_QUERY_BUCKETS",
        limit=max_model_len,
        align=True,
        default=powers_of_two_up_to(max_model_len, start=ENCODER_SEQ_ALIGNMENT),
        append_limit=True,
    )
    max_num_seqs = max(1, int(vllm_config.scheduler_config.max_num_seqs))
    width_limit = max_num_seqs
    widths = _resolve_encoder_buckets(
        num_seqs_buckets,
        envs.SPYRE_ATTN_NUM_SEQS_BUCKETS,
        "SPYRE_ATTN_NUM_SEQS_BUCKETS",
        limit=width_limit,
        align=False,
        default=powers_of_two_up_to(width_limit),
        append_limit=False,
    )
    # `(max_model_len, 1)` always survives, because the budget is floored at
    # max_model_len, so the set is never empty.
    shapes = sorted(
        {(length, batch) for length in lengths for batch in widths if length * batch <= budget},
        key=lambda pair: (pair[1], pair[0]),
    )
    _ENCODER_SHAPES_MEMO[memo_key] = shapes
    return list(shapes)


def encoder_shape_covers(shape: tuple[int, int], num_seqs: int, max_len: int) -> bool:
    """Whether a declared ``(L, B)`` holds ``num_seqs`` sequences of up to ``max_len``.

    The one place this rule lives. `PoolingSpyreScheduler` admits on it and
    `pick_encoder_shape` dispatches on it, so the gate cannot approve a batch the
    runner then rejects.
    """
    length, batch = shape
    return batch >= num_seqs and length >= max_len


def pick_encoder_shape(
    num_seqs: int,
    max_len: int,
    shapes: Sequence[tuple[int, int]],
) -> tuple[int, int] | None:
    """First declared ``(L, B)`` covering the batch, or ``None``.

    ``None`` needs no fallback: the scheduler gate and vLLM's ``max_model_len`` check both
    reject such a batch before it reaches the runner.
    """
    if num_seqs < 1 or max_len < 1:
        return None
    for shape in shapes:
        if encoder_shape_covers(shape, num_seqs, max_len):
            return shape
    return None


def logits_row_buckets(bucket_sizes: Sequence[int], max_num_reqs: int) -> list[int]:
    """Row widths the lm_head can see: each body bucket clipped to ``max_num_reqs``."""
    cap = max(1, max_num_reqs)
    return sorted({min(size, cap) for size in bucket_sizes if size > 0})


def encoder_dense_row_indices(query_lens: Sequence[int], len_bucket: int) -> torch.Tensor:
    """Dense row of every packed token: ``seq_idx * L + offset within the sequence``.

    Both directions of the grid need it — the scatter that pads the batch out to
    ``B * L`` and the gather that compacts the hidden states back — so it is built once
    per step with tensor ops rather than a Python loop per row.
    """
    lens = torch.as_tensor(list(query_lens), dtype=torch.int64)
    if lens.numel() == 0:
        return torch.empty(0, dtype=torch.int64)
    if int(lens.max()) > len_bucket:
        raise ValueError(f"a query length exceeds len_bucket={len_bucket}: {list(query_lens)}")
    # Each sequence's rows shift by a constant, so one `repeat_interleave` of the
    # per-sequence shift beats gathering a start per token.
    packed_starts = torch.cumsum(lens, 0) - lens
    shifts = torch.arange(lens.numel(), dtype=torch.int64) * len_bucket - packed_starts
    packed = torch.arange(int(lens.sum()), dtype=torch.int64)
    return packed + torch.repeat_interleave(shifts, lens)


def expand_packed_to_encoder_grid(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    query_lens: Sequence[int],
    batch_bucket: int,
    len_bucket: int,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad each sequence to ``L`` and the batch to ``B``; return two ``[B*L]`` tensors.

    Real pad tokens continue positions from the true length. Dummy sequences (batch pad)
    are ``pad_token_id`` with positions ``0 .. L-1``. Both fall out of seeding the
    position grid with a tiled ``arange`` and scattering the real rows over it.
    """
    if len(query_lens) > batch_bucket:
        raise ValueError(f"num_seqs={len(query_lens)} exceeds batch_bucket={batch_bucket}")

    rows = encoder_dense_row_indices(query_lens, len_bucket)
    total = batch_bucket * len_bucket
    ids = torch.full((total,), int(pad_token_id), dtype=input_ids.dtype)
    ids[rows] = input_ids
    positions_grid = torch.arange(len_bucket, dtype=positions.dtype).repeat(batch_bucket)
    positions_grid[rows] = positions
    return ids, positions_grid


@dataclass(frozen=True)
class SpyreBucketDescriptor:
    """Descriptor for a 1D (decoder) compilation bucket."""

    actual_num_tokens: int
    padded_num_tokens: int


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

        # Pooling dispatches on the declared (L, B) shapes, not this 1-D ladder; the
        # instance exists there only to carry `is_warmed_up`.
        if vllm_config.model_config.runner_type == "pooling":
            logger.debug("SpyreShapeBucketer constructed for pooling; the 1-D ladder is unused.")
        else:
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
