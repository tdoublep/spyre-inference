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

"""Tests for the HuggingFace Transformers backend (model_impl='transformers').

TODO: Delete this file once https://github.com/torch-spyre/spyre-inference/issues/324
is resolved and re-enable the upstream tests in upstream_tests.yaml.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.uses_subprocess
@pytest.mark.parametrize(
    "model",
    [
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
)
def test_transformers_generate(model: str) -> None:
    """Verify model_impl='transformers' loads and generates non-empty output."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        dtype="float16",
        enforce_eager=False,
        max_model_len=128,
        max_num_seqs=2,
        model_impl="transformers",
    )
    model_config = llm.llm_engine.model_config
    assert model_config.using_transformers_backend()

    sp = SamplingParams(max_tokens=8, temperature=0.0)
    outputs = llm.generate(["Hello, world!"], sp)
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0


@pytest.fixture()
def spyre_device():
    from spyre_testing_plugin.pytest_plugin import spyre_available

    if not spyre_available():
        pytest.skip("Spyre device not available")
    return torch.device("spyre")


def _hf_rope(head_dim: int, n_heads: int = 4):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    cfg = LlamaConfig(
        hidden_size=head_dim * n_heads,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads // 2,
        head_dim=head_dim,
        max_position_embeddings=512,
    )
    return LlamaRotaryEmbedding(config=cfg)


def _spyre_rope_parts(hf_rope, head_dim: int):
    """Build the patched rope pair the way ``_patch_rope`` does."""
    from spyre_inference.hf_adapters import (
        _STICK,
        _make_spyre_apply_rotary,
        _qk_expand_matrix,
        _SpyreRotaryEmbedding,
    )

    stick_aligned = ((head_dim + 2 * _STICK - 1) // (2 * _STICK)) * (2 * _STICK)
    padded = stick_aligned if stick_aligned > head_dim else None
    qk_expand = _qk_expand_matrix(head_dim, padded) if padded is not None else None
    return (
        _SpyreRotaryEmbedding(
            hf_rope.inv_freq,
            getattr(hf_rope, "attention_scaling", 1.0),
            padded_head_dim=padded,
        ),
        _make_spyre_apply_rotary(qk_expand),
        padded,
    )


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_rope_matches_hf_reference_cpu(head_dim: int) -> None:
    """CPU-only: the 2x2 rotation reproduces HF's own ``apply_rotary_pos_emb``.

    ``head_dim=64`` (inner dim 32) exercises the expand/contract path; 128 and 256 are
    already stick-aligned and rotate in place.
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    torch.manual_seed(11)
    n_heads, n_kv, seq, bsz = 4, 2, 7, 2
    hf_rope = _hf_rope(head_dim, n_heads)
    query = torch.randn(bsz, n_heads, seq, head_dim)
    key = torch.randn(bsz, n_kv, seq, head_dim)
    positions = torch.arange(seq).unsqueeze(0).expand(bsz, seq)

    cos, sin = hf_rope(query, positions)
    expected_q, expected_k = apply_rotary_pos_emb(query, key, cos, sin)

    spyre_rope, apply_spyre, _ = _spyre_rope_parts(hf_rope, head_dim)
    rot, second = spyre_rope(query.half(), positions)
    assert second is None, "must return (rotation, None) to fit HF's (cos, sin) contract"
    actual_q, actual_k = apply_spyre(query.half(), key.half(), rot)

    assert actual_q.shape == query.shape and actual_k.shape == key.shape
    torch.testing.assert_close(actual_q.float(), expected_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_k.float(), expected_k, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_padded_rotation_lowers_on_spyre(spyre_device, mode: str) -> None:
    """The stick-aligned rotation (inner dim 64) lowers on Spyre."""
    from spyre_inference.hf_adapters import _rotate

    torch.manual_seed(11)
    query = torch.randn(1, 4, 8, 128, dtype=torch.float16).to(spyre_device)
    rot = torch.randn(1, 8, 2, 2, 64, dtype=torch.float16).to(spyre_device)

    fn = torch.compile(_rotate, dynamic=False) if mode == "compile" else _rotate
    assert tuple(fn(query, rot).to("cpu").shape) == (1, 4, 8, 128)


@pytest.mark.parametrize("mode", ["eager", "compile"])
@pytest.mark.xfail(
    strict=True,
    reason="Spyre cannot lower the 2x2 rotation at a sub-stick inner dim: the "
    "[.., 2, 32] pairing view yields stick expression '32*d3 + d4' where the backend "
    "requires Mod(var, 64). This is why _patch_rope expands Q/K to a stick-aligned "
    "head_dim. When this XPASSes, drop _qk_expand_matrix and the expand/contract "
    "matmuls from _make_spyre_apply_rotary.",
)
def test_unpadded_rotation_rejected_on_spyre(spyre_device, mode: str) -> None:
    """Probe: rotating an unpadded head_dim=64 (inner dim 32) on Spyre."""
    from spyre_inference.hf_adapters import _rotate

    torch.manual_seed(11)
    query = torch.randn(1, 4, 8, 64, dtype=torch.float16).to(spyre_device)
    rot = torch.randn(1, 8, 2, 2, 32, dtype=torch.float16).to(spyre_device)

    fn = torch.compile(_rotate, dynamic=False) if mode == "compile" else _rotate
    assert tuple(fn(query, rot).to("cpu").shape) == (1, 4, 8, 64)
