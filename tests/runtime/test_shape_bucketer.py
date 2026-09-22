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

"""Unit tests for SpyreShapeBucketer."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from vllm.config import VllmConfig

from spyre_inference.v1.worker.spyre_shape_bucketer import (
    SpyreShapeBucketer,
    encoder_bucket_valid_row_indices,
    encoder_warmup_shapes,
    expand_packed_to_encoder_bucket,
    logits_row_buckets,
    next_bucket,
    pick_encoder_shape,
)


@pytest.fixture()
def mock_vllm_config():
    """Create a minimal VllmConfig mock with compile_sizes."""
    config = MagicMock()
    config.compilation_config.compile_sizes = [1, 2, 4, 8, 16]
    return config


@pytest.fixture()
def bucketer(mock_vllm_config):
    return SpyreShapeBucketer(mock_vllm_config)


class TestFindBucket:
    def test_exact_match(self, bucketer):
        assert bucketer.find_bucket(8) == 8

    def test_rounds_up_to_next_bucket(self, bucketer):
        assert bucketer.find_bucket(3) == 4
        assert bucketer.find_bucket(5) == 8
        assert bucketer.find_bucket(9) == 16

    def test_smallest_token_count(self, bucketer):
        assert bucketer.find_bucket(1) == 1

    def test_exceeds_max_returns_none(self, bucketer):
        assert bucketer.find_bucket(17) is None
        assert bucketer.find_bucket(100) is None

    def test_zero_tokens(self, bucketer):
        assert bucketer.find_bucket(0) == 1


class TestDispatch:
    def test_returns_descriptor_with_padding(self, bucketer):
        desc = bucketer.dispatch(5)
        assert desc is not None
        assert desc.actual_num_tokens == 5
        assert desc.padded_num_tokens == 8

    def test_exact_match_no_padding(self, bucketer):
        desc = bucketer.dispatch(4)
        assert desc is not None
        assert desc.actual_num_tokens == 4
        assert desc.padded_num_tokens == 4

    def test_exceeds_max_returns_none(self, bucketer):
        assert bucketer.dispatch(20) is None

    def test_descriptor_is_frozen(self, bucketer):
        desc = bucketer.dispatch(3)
        with pytest.raises(FrozenInstanceError):
            desc.actual_num_tokens = 10


class TestBucketerState:
    def test_initial_state_not_warmed_up(self, bucketer):
        assert not bucketer.is_warmed_up

    def test_mark_warmed_up(self, bucketer):
        bucketer.mark_warmed_up()
        assert bucketer.is_warmed_up

    def test_bucket_sizes_sorted(self, bucketer):
        assert bucketer.bucket_sizes == [1, 2, 4, 8, 16]

    def test_max_bucket_size(self, bucketer):
        assert bucketer.max_bucket_size == 16


class TestEdgeCases:
    def test_empty_compile_sizes(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = []
        b = SpyreShapeBucketer(config)
        assert b.bucket_sizes == []
        assert b.max_bucket_size == 0
        assert b.find_bucket(1) is None
        assert b.dispatch(1) is None

    def test_single_bucket(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = [8]
        b = SpyreShapeBucketer(config)
        assert b.find_bucket(1) == 8
        assert b.find_bucket(8) == 8
        assert b.find_bucket(9) is None

    def test_unsorted_input_gets_sorted(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = [16, 2, 8, 1, 4]
        b = SpyreShapeBucketer(config)
        assert b.bucket_sizes == [1, 2, 4, 8, 16]


def _pooling_config(max_model_len=512, max_num_seqs=8, max_num_batched_tokens=4096):
    return cast(
        VllmConfig,
        SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=max_model_len),
            scheduler_config=SimpleNamespace(
                max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens
            ),
        ),
    )


class TestEncoderWarmupShapes:
    """``(L, B)`` shapes from ``max_model_len`` / ``max_num_seqs`` and the two ladders."""

    def test_default_is_both_ladders_crossed(self):
        """Powers of two on each axis: lengths from the stick, widths from 1."""
        shapes = encoder_warmup_shapes(_pooling_config(512, 8, 4096))
        assert sorted({length for length, _ in shapes}) == [64, 128, 256, 512]
        assert sorted({batch for _, batch in shapes}) == [1, 2, 4, 8]
        # Nothing is dropped here: the widest pair is exactly the budget.
        assert len(shapes) == 16

    def test_the_token_budget_filters_pairs_rather_than_capping_the_width(self):
        # 2048 tokens hold four 512-token sequences but thirty-two 64-token ones, so the
        # width available depends on the length instead of being capped globally.
        shapes = encoder_warmup_shapes(_pooling_config(512, 256, 2048))
        assert all(length * batch <= 2048 for length, batch in shapes)
        assert max(batch for length, batch in shapes if length == 512) == 4
        assert max(batch for length, batch in shapes if length == 64) == 32

    def test_a_budget_below_max_model_len_still_leaves_one_sequence(self):
        # Encoder prefill cannot be chunked, so the budget is floored at max_model_len.
        shapes = encoder_warmup_shapes(_pooling_config(8192, 32, 2048))
        assert (8192, 1) in shapes
        assert all(length * batch <= 8192 for length, batch in shapes)

    def test_the_widest_batch_can_exceed_the_width_at_the_longest_length(self):
        """Why `PoolingSpyreScheduler` has to clamp `max_num_running_reqs`.

        `max_num_seqs` is written back from the widest shape overall, which a long
        request cannot reach, so upstream would otherwise admit a batch no shape covers.
        """
        shapes = encoder_warmup_shapes(_pooling_config(512, 32, 2048))
        assert max(batch for _, batch in shapes) == 32
        assert max(batch for length, batch in shapes if length == 512) == 4

    def test_the_ladders_are_crossed_and_sorted_by_batch_then_length(self):
        # Sorted by (B, L) so pick_encoder_shape's first match is the cheapest graph.
        shapes = encoder_warmup_shapes(
            _pooling_config(512, 8, 4096), length_buckets=[64, 256], num_seqs_buckets=[1, 4]
        )
        # The length ladder gains max_model_len (512); the width ladder is taken as given,
        # so 3 lengths x 2 widths.
        assert shapes == [
            (64, 1),
            (256, 1),
            (512, 1),
            (64, 4),
            (256, 4),
            (512, 4),
        ]

    def test_only_the_length_ladder_gains_its_limit(self):
        """``max_model_len`` is an input a shape must cover; ``max_num_seqs`` is not.

        ``max_num_seqs`` is written back *from* the widest declared width, so a width
        ladder that stops short simply lowers it -- nothing is left uncovered.
        """
        shapes = encoder_warmup_shapes(
            _pooling_config(512, 8, 4096), length_buckets=[64], num_seqs_buckets=[2]
        )
        assert shapes == [(64, 2), (512, 2)]
        assert max(batch for _, batch in shapes) == 2

    def test_lengths_round_up_to_the_stick(self):
        shapes = encoder_warmup_shapes(_pooling_config(512, 2, 1024), length_buckets=[100])
        assert sorted({length for length, _ in shapes}) == [128, 512]

    def test_unreachable_entries_are_dropped(self):
        # 9999 exceeds max_model_len and 99 exceeds max_num_seqs, so neither is
        # schedulable; max_model_len is still appended to the length ladder.
        shapes = encoder_warmup_shapes(
            _pooling_config(512, 4, 2048), length_buckets=[64, 9999], num_seqs_buckets=[2, 99]
        )
        assert shapes == [(64, 2), (512, 2)]

    def test_non_positive_entries_are_ignored(self):
        assert encoder_warmup_shapes(
            _pooling_config(512, 8, 4096), length_buckets=[0, -64], num_seqs_buckets=[0]
        ) == [(512, 8)]

    def test_is_idempotent_under_the_platform_write_back(self):
        """The runner, scheduler and attention impl all re-derive post-startup."""
        for args in [(512, 256, 2048), (512, 8, 8192), (8192, 32, 2048), (384, 6, 2304)]:
            config = _pooling_config(*args)
            first = encoder_warmup_shapes(config)
            # The only write-backs: max_num_seqs down to the widest shape, and the
            # budget up to max_model_len when it was below it.
            config.scheduler_config.max_num_seqs = max(b for _, b in first)
            config.scheduler_config.max_num_batched_tokens = max(
                config.scheduler_config.max_num_batched_tokens,
                config.model_config.max_model_len,
            )
            assert encoder_warmup_shapes(config) == first, args

    def test_the_memo_hands_out_a_fresh_list(self):
        """Four call sites re-derive the same shapes; none may mutate another's copy."""
        config = _pooling_config(512, 8, 4096)
        first = encoder_warmup_shapes(config)
        expected = list(first)
        first.append((1, 1))
        assert encoder_warmup_shapes(config) == expected

    def test_the_memo_follows_the_environment(self, monkeypatch):
        """Same config, different ladder: the memo key carries the env values."""
        from spyre_inference import envs

        config = _pooling_config(256, 2, 512)
        assert sorted({length for length, _ in encoder_warmup_shapes(config)}) == [64, 128, 256]
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "128,256")
        envs.clear_env_cache()
        try:
            assert sorted({length for length, _ in encoder_warmup_shapes(config)}) == [128, 256]
        finally:
            envs.clear_env_cache()

    def test_reads_the_environment(self, monkeypatch):
        from spyre_inference import envs

        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "128,256")
        monkeypatch.setenv("SPYRE_ATTN_NUM_SEQS_BUCKETS", "2")
        envs.clear_env_cache()
        try:
            assert encoder_warmup_shapes(_pooling_config(256, 2, 512)) == [(128, 2), (256, 2)]
        finally:
            envs.clear_env_cache()


class TestPickEncoderShape:
    SHAPES = [(512, 2), (256, 8), (64, 32)]

    def test_first_match_wins_so_narrow_shapes_are_preferred(self):
        # One short request could use any of the three; the 2-wide one is cheapest.
        assert pick_encoder_shape(1, 10, self.SHAPES) == (512, 2)

    def test_needs_both_axes_to_fit(self):
        # Eight sequences rule out (512, 2); 100 tokens rule out (64, 32).
        assert pick_encoder_shape(8, 100, self.SHAPES) == (256, 8)

    def test_wide_short_batch_takes_the_wide_shape(self):
        assert pick_encoder_shape(32, 60, self.SHAPES) == (64, 32)

    def test_none_when_no_shape_is_wide_enough(self):
        # The scheduler gate is what keeps this out of the serving path.
        assert pick_encoder_shape(20, 300, self.SHAPES) is None

    def test_none_when_the_request_is_longer_than_every_shape(self):
        # vLLM's own max_model_len check rejects these first.
        assert pick_encoder_shape(1, 600, self.SHAPES) is None

    def test_none_on_a_degenerate_batch(self):
        assert pick_encoder_shape(0, 10, self.SHAPES) is None
        assert pick_encoder_shape(1, 0, self.SHAPES) is None


class TestDenseExpansion:
    def test_pads_each_sequence_to_l_and_the_batch_to_b(self):
        ids, pos = expand_packed_to_encoder_bucket(
            input_ids=[1, 2, 3, 7, 8],
            positions=[0, 1, 2, 0, 1],
            query_lens=[3, 2],
            batch_bucket=3,
            len_bucket=4,
            pad_token_id=0,
        )
        assert len(ids) == len(pos) == 12
        # Sequence 0 occupies rows 0..3, sequence 1 rows 4..7, dummy rows 8..11.
        assert ids[0:4] == [1, 2, 3, 0]
        assert ids[4:8] == [7, 8, 0, 0]
        assert ids[8:12] == [0, 0, 0, 0]

    def test_positions_continue_through_real_pad(self):
        _ids, pos = expand_packed_to_encoder_bucket(
            input_ids=[1, 2], positions=[0, 1], query_lens=[2], batch_bucket=1, len_bucket=4
        )
        assert pos == [0, 1, 2, 3]

    def test_dummy_sequences_get_their_own_position_range(self):
        _ids, pos = expand_packed_to_encoder_bucket(
            input_ids=[1], positions=[0], query_lens=[1], batch_bucket=2, len_bucket=3
        )
        assert pos[3:6] == [0, 1, 2]

    def test_rejects_a_batch_wider_than_the_bucket(self):
        with pytest.raises(ValueError, match="exceeds batch_bucket"):
            expand_packed_to_encoder_bucket([1, 2], [0, 0], [1, 1], 1, 4)

    def test_rejects_a_sequence_longer_than_the_bucket(self):
        with pytest.raises(ValueError, match="exceeds len_bucket"):
            expand_packed_to_encoder_bucket([1, 2, 3], [0, 1, 2], [3], 1, 2)

    def test_valid_row_indices_invert_the_expansion(self):
        """The gather that re-compacts the grid before pooling."""
        assert encoder_bucket_valid_row_indices([3, 2], 4) == [0, 1, 2, 4, 5]

    def test_valid_row_indices_skip_a_zero_length_sequence(self):
        assert encoder_bucket_valid_row_indices([2, 0, 1], 4) == [0, 1, 8]


class TestNextBucket:
    def test_returns_the_smallest_bucket_at_or_above(self):
        assert next_bucket(100, [64, 128, 512]) == 128

    def test_stick_aligns_when_past_every_bucket(self):
        assert next_bucket(600, [64, 128, 512]) == 640

    def test_floors_at_one(self):
        assert next_bucket(0, [64]) == 64


class TestLogitsRowBuckets:
    def test_clips_prefill_bucket_to_max_num_reqs(self):
        # The 512-token prefill bucket samples at most max_num_seqs rows.
        assert logits_row_buckets([1, 2, 4, 8, 512], max_num_reqs=8) == [1, 2, 4, 8]

    def test_keeps_a_non_power_of_two_max(self):
        assert logits_row_buckets([1, 2, 4, 6, 512], max_num_reqs=6) == [1, 2, 4, 6]

    def test_ignores_non_positive_sizes(self):
        assert logits_row_buckets([0, -1, 4], max_num_reqs=8) == [4]
