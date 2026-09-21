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
from unittest.mock import MagicMock

import pytest

from spyre_inference.v1.worker.spyre_shape_bucketer import (
    EncoderBucketDescriptor,
    SpyreShapeBucketer,
    encoder_body_sizes,
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


class TestEncoderWarmupShapes:
    """Declared ``(L, B)`` pairs, zipped pairwise rather than crossed."""

    def test_zips_pairwise_and_sorts_by_batch_then_length(self):
        # Sorted by (B, L) so pick_encoder_shape's first match is the cheapest graph.
        assert encoder_warmup_shapes([64, 256, 512], [32, 8, 2]) == [
            (512, 2),
            (256, 8),
            (64, 32),
        ]

    def test_is_not_a_cross_product(self):
        # Three lengths and three batches give three shapes, not nine.
        assert len(encoder_warmup_shapes([64, 128, 512], [1, 2, 4])) == 3

    def test_deduplicates_repeated_pairs(self):
        assert encoder_warmup_shapes([64, 64], [2, 2]) == [(64, 2)]

    def test_rejects_unequal_list_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            encoder_warmup_shapes([64, 128], [2])

    def test_rejects_a_length_off_the_stick(self):
        # 100 is not a multiple of 64, so attention's L would need padding the
        # dense expansion does not do.
        with pytest.raises(ValueError, match="multiples of 64"):
            encoder_warmup_shapes([100], [2])

    def test_rejects_non_positive_values(self):
        with pytest.raises(ValueError, match="multiples of 64"):
            encoder_warmup_shapes([0], [2])
        with pytest.raises(ValueError, match="must be positive"):
            encoder_warmup_shapes([64], [0])

    def test_rejects_an_empty_list(self):
        with pytest.raises(ValueError, match="at least one shape"):
            encoder_warmup_shapes([], [])

    def test_reads_the_environment_by_default(self, monkeypatch):
        from spyre_inference import envs

        monkeypatch.setenv("SPYRE_WARMUP_PROMPT_LENS", "128,256")
        monkeypatch.setenv("SPYRE_WARMUP_BATCH_SIZES", "4,2")
        envs.clear_env_cache()
        try:
            assert encoder_warmup_shapes() == [(256, 2), (128, 4)]
        finally:
            envs.clear_env_cache()

    def test_default_is_a_single_bucket(self, monkeypatch):
        """Matches sendnn-inference's default: one (512, 8) graph."""
        from spyre_inference import envs

        monkeypatch.delenv("SPYRE_WARMUP_PROMPT_LENS", raising=False)
        monkeypatch.delenv("SPYRE_WARMUP_BATCH_SIZES", raising=False)
        envs.clear_env_cache()
        try:
            assert encoder_warmup_shapes() == [(512, 8)]
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


class TestEncoderBodySizes:
    def test_distinct_products_only(self):
        """The body is flat ``[B*L, hidden]``, so equal-area shapes share a graph."""
        assert encoder_body_sizes([(512, 2), (256, 4), (64, 32)]) == [1024, 2048]

    def test_single_shape(self):
        assert encoder_body_sizes([(512, 8)]) == [4096]


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


class TestEncoderBucketDescriptor:
    def test_padded_tokens_is_the_product(self):
        desc = EncoderBucketDescriptor(
            batch_bucket=4, len_bucket=128, actual_num_seqs=3, actual_max_len=100
        )
        assert desc.padded_num_tokens == 512

    def test_is_frozen(self):
        desc = EncoderBucketDescriptor(
            batch_bucket=1, len_bucket=64, actual_num_seqs=1, actual_max_len=1
        )
        with pytest.raises(FrozenInstanceError):
            desc.batch_bucket = 2


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
