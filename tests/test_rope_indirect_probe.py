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

"""Probe: does RoPE actually need an opaque op?

PR #479 keeps an opaque ``spyre_rope_gather`` on the claim that handing the
device-resident rotation cache to a compiled graph as a tensor input makes the
compiled kernel segfault in libsenlib. This probe puts the gather *and* the 2x2
rotation in one ``fullgraph=True`` compiled region with the real device cache as
a graph input, and checks the result against vLLM's ``forward_native``.
"""

import pytest
import torch

from spyre_testing_plugin.pytest_plugin import spyre_available

_EXPAND_MATRIX_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "head_size=64 gives inner=32 != padded=64, so _rotate_neox_2x2 takes the "
        "_get_expand_matrix branch. That helper is called *inside* the traced "
        "region and materializes its constant on the host, so the graph carries a "
        "CPU ComputedBuffer and fails to lower. Two independent fixes: pad head_dim "
        "to 128 (spyre_inference PR #551, which makes inner == padded so the branch "
        "is never taken), or hoist the expand matrix onto the device at module init. "
        "Stick-aligned head sizes (128, 256) take the pure-view branch and pass."
    ),
)

HEAD_SIZES = [
    pytest.param(64, marks=_EXPAND_MATRIX_XFAIL),
    128,
    256,
]


@pytest.fixture()
def spyre_device():
    if not spyre_available():
        pytest.skip("Spyre device not available")
    return torch.device("spyre")


@pytest.mark.rotary
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("num_tokens", [1, 32])
def test_rope_gather_and_rotate_in_fullgraph(
    default_vllm_config, spyre_device, head_size, num_tokens
):
    """The whole RoPE (indirect gather + 2x2 rotation) inside one compiled
    fullgraph, cache as a graph input, no opaque op."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    from spyre_inference.custom_ops.rotary_embedding import _rotate_neox_2x2

    torch.manual_seed(11)
    max_position, num_heads = 2048, 4
    rope = get_rope(head_size, max_position, is_neox_style=True, dtype=torch.float16)

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.int32)
    query = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)

    # The device-resident cache, exactly as the production module builds it.
    cache_dev = rope._get_device_rotation_cache(spyre_device)

    def rope_fwd(cache, pos, q, k):
        rot = cache.index_select(0, pos)
        return _rotate_neox_2x2(q, rot, head_size), _rotate_neox_2x2(k, rot, head_size)

    compiled = torch.compile(rope_fwd, dynamic=False, fullgraph=True)
    got_q, got_k = compiled(
        cache_dev,
        positions.to(spyre_device),
        query.to(spyre_device),
        key.to(spyre_device),
    )
    assert got_q.device.type == "spyre", "rotation left the device"

    exp_q, exp_k = RotaryEmbedding.forward_native(
        rope, positions.long(), query.clone(), key.clone()
    )
    torch.testing.assert_close(got_q.cpu().float(), exp_q.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(got_k.cpu().float(), exp_k.float(), atol=2e-2, rtol=2e-2)
