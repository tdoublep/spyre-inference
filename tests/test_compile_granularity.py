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

"""Per-block compile granularity: block discovery and artifact reuse. No device needed."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.models.utils import PPMissingLayer

from spyre_inference.v1.worker.spyre_model_runner import _repeated_block_lists


def _fake_attention() -> Attention:
    """Skips ``Attention.__init__``, which needs a full model config."""
    attn = Attention.__new__(Attention)
    nn.Module.__init__(attn)
    return attn


class _Block(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.attn = _fake_attention()
        self.o_proj = nn.Linear(hidden, hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, 2 * hidden)
        self.down = nn.Linear(2 * hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layernorm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        x = x + self.o_proj(torch.softmax(q * k, dim=-1) * v)
        h = self.post_attention_layernorm(x)
        return x + self.down(torch.relu(self.up(h)))


class _Backbone(nn.Module):
    def __init__(self, hidden: int, num_layers: int, num_missing: int = 0):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, hidden)
        self.layers = nn.ModuleList(
            [_Block(hidden) for _ in range(num_layers)]
            + [PPMissingLayer() for _ in range(num_missing)]
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            x = layer(x)
        return self.norm(x)


class _Model(nn.Module):
    def __init__(self, hidden: int = 32, num_layers: int = 4, num_missing: int = 0):
        super().__init__()
        self.model = _Backbone(hidden, num_layers, num_missing)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)


def test_finds_the_transformer_block_list() -> None:
    model = _Model(num_layers=4)
    found = _repeated_block_lists(model)
    assert len(found) == 1
    assert found[0] is model.model.layers


def test_ignores_module_lists_without_attention() -> None:
    class NoAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    assert _repeated_block_lists(NoAttention()) == []


def test_pp_missing_layers_do_not_break_discovery() -> None:
    model = _Model(num_layers=2, num_missing=2)
    assert _repeated_block_lists(model) == [model.model.layers]


def test_compile_blocks_wraps_every_block_in_place() -> None:
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    model = _Model(num_layers=4)
    originals = list(model.model.layers)

    runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
    runner.model = model
    assert runner._compile_blocks() == 4

    for i, original in enumerate(originals):
        wrapped = model.model.layers[i]
        assert isinstance(wrapped, torch._dynamo.eval_frame.OptimizedModule)
        assert wrapped._orig_mod is original


def test_pp_missing_layers_are_not_compiled() -> None:
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    model = _Model(num_layers=2, num_missing=2)
    runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
    runner.model = model

    assert runner._compile_blocks() == 2
    assert all(isinstance(layer, PPMissingLayer) for layer in model.model.layers[2:])


def test_identical_blocks_share_one_compiled_artifact() -> None:
    """Dynamo may trace several graphs, but only the first reaches the backend."""
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters
    from torch._inductor.utils import fresh_cache

    def compile_counts(num_layers: int) -> tuple[int, int]:
        dynamo.reset()
        counters.clear()
        with fresh_cache():
            model = _Model(hidden=32, num_layers=num_layers)
            for i, layer in enumerate(model.model.layers):
                model.model.layers[i] = cast(
                    nn.Module,
                    torch.compile(layer, backend="inductor", fullgraph=True, dynamic=False),
                )
            with torch.inference_mode():
                model(torch.zeros(2, 4, dtype=torch.long))
            return (
                counters["stats"]["unique_graphs"],
                counters["inductor"]["fxgraph_cache_miss"],
            )

    shallow_graphs, shallow_backend = compile_counts(2)
    deep_graphs, deep_backend = compile_counts(8)

    assert shallow_backend == deep_backend == 1
    assert shallow_graphs == deep_graphs
