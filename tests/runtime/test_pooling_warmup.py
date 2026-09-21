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

"""``_warmup_pooling_bucket_shapes`` traces exactly one dummy per declared ``(L, B)``.

Driven with a stub self; the method's whole surface is four attributes.
"""

from types import SimpleNamespace
from typing import cast

import pytest

from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

SHAPES = [(64, 2), (512, 2), (64, 8), (512, 8)]


def _warmup(shapes, has_decoder_attn=False, max_num_seqs=8):
    calls: list[tuple[int, bool]] = []

    def dummy_run(num_tokens, **kwargs):
        calls.append((num_tokens, bool(kwargs.get("force_attention"))))
        return object(), object()

    runner = SimpleNamespace(
        _encoder_shapes=list(shapes),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        _spyre_kv_caches=({"layers.0.self_attn": object()} if has_decoder_attn else {}),
        _dummy_run=dummy_run,
        _dummy_pooler_run=lambda hidden: None,
    )
    TorchSpyreModelRunner._warmup_pooling_bucket_shapes(cast(TorchSpyreModelRunner, runner))
    return calls, runner


def test_one_dummy_per_shape_at_its_full_area():
    calls, _runner = _warmup(SHAPES)
    assert calls == [(128, True), (1024, True), (512, True), (4096, True)]


def test_max_num_seqs_is_pinned_to_each_shape_then_restored():
    """Upstream's dummy builds ``max_num_seqs`` requests, so it must be the shape's width."""
    widths: list[int] = []

    def dummy_run(num_tokens, **kwargs):
        widths.append(runner.scheduler_config.max_num_seqs)
        return object(), object()

    runner = SimpleNamespace(
        _encoder_shapes=list(SHAPES),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        _spyre_kv_caches={},
        _dummy_run=dummy_run,
        _dummy_pooler_run=lambda hidden: None,
    )
    TorchSpyreModelRunner._warmup_pooling_bucket_shapes(cast(TorchSpyreModelRunner, runner))
    assert widths == [2, 2, 8, 8]
    assert runner.scheduler_config.max_num_seqs == 8


def test_max_num_seqs_is_restored_even_when_a_dummy_raises():
    def dummy_run(num_tokens, **kwargs):
        raise RuntimeError("compile blew up")

    runner = SimpleNamespace(
        _encoder_shapes=[(64, 2)],
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        _spyre_kv_caches={},
        _dummy_run=dummy_run,
        _dummy_pooler_run=lambda hidden: None,
    )
    with pytest.raises(RuntimeError, match="compile blew up"):
        TorchSpyreModelRunner._warmup_pooling_bucket_shapes(cast(TorchSpyreModelRunner, runner))
    assert runner.scheduler_config.max_num_seqs == 8


class TestDecoderTypeTextTower:
    """A pooling model with a real KV cache (e.g. CLIP's text tower).

    Upstream's ``_dummy_run`` broadcasts one aggregate ``seq_lens`` across a uniform
    multi-request batch, overestimating ``num_blocks`` for a paged KV layer, so those
    variants go to ``_record_attention_graphs`` instead.
    """

    def test_multi_request_shapes_skip_force_attention(self):
        calls, _runner = _warmup(SHAPES, has_decoder_attn=True)
        assert calls == [(128, False), (1024, False), (512, False), (4096, False)]

    def test_a_single_request_shape_is_unaffected(self):
        calls, _runner = _warmup([(512, 1), (512, 4)], has_decoder_attn=True)
        assert calls == [(512, True), (2048, False)]

    def test_encoder_only_pooling_always_forces_attention(self):
        calls, _runner = _warmup([(512, 1), (512, 4)], has_decoder_attn=False)
        assert all(force for _tokens, force in calls)
