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

"""Per-block compile granularity: block discovery, in-place wrapping, artifact reuse.

Only the artifact-reuse test traces, and tracing resolves the accelerator stream,
which opens the single contested Spyre card -- hence its ``compile`` marker. The
rest is device-free and stays in the smoke job.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from vllm.config import CompilationMode
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.models.utils import PPMissingLayer

from spyre_inference.v1.worker.spyre_model_runner import (
    TorchSpyreModelRunner,
    _head_and_tail_modules,
    _repeated_block_lists,
)


@pytest.fixture
def isolated_dynamo_state():
    """Bound this test's Dynamo cache churn; save and restore the global counters."""
    import copy

    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters

    saved = copy.deepcopy(counters)
    dynamo.reset()
    try:
        yield
    finally:
        dynamo.reset()
        counters.clear()
        counters.update(saved)


def _fake_attention() -> Attention:
    """Skips ``Attention.__init__``, which needs a full model config."""
    attn = Attention.__new__(Attention)
    nn.Module.__init__(attn)
    return attn


class _Block(nn.Module):
    """Threads a residual like a real decoder layer: ``residual is None`` on layer 0
    is a trace-time branch, so layer 0 compiles to its own graph."""

    def __init__(self, hidden: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.attn = _fake_attention()
        self.o_proj = nn.Linear(hidden, hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, 2 * hidden)
        self.down = nn.Linear(2 * hidden, hidden)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = x
            h = self.input_layernorm(x)
        else:
            residual = x + residual
            h = self.input_layernorm(residual)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        x = self.o_proj(torch.softmax(q * k, dim=-1) * v)
        h = self.post_attention_layernorm(x)
        return self.down(torch.relu(self.up(h))), residual


class _MambaBlock(nn.Module):
    """Attention-free layer, as in a hybrid Mamba+attention stack."""

    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = x
        else:
            residual = x + residual
        return self.proj(self.norm(residual)), residual


class _Backbone(nn.Module):
    def __init__(self, hidden: int, num_layers: int, num_missing: int = 0, hybrid: bool = False):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, hidden)
        blocks: list[nn.Module] = []
        for i in range(num_layers):
            blocks.append(_MambaBlock(hidden) if hybrid and i % 2 else _Block(hidden))
        self.layers = nn.ModuleList(blocks + [PPMissingLayer() for _ in range(num_missing)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            x, residual = layer(x, residual)
        return self.norm(x + residual)


class _Model(nn.Module):
    def __init__(
        self, hidden: int = 32, num_layers: int = 4, num_missing: int = 0, hybrid: bool = False
    ):
        super().__init__()
        self.model = _Backbone(hidden, num_layers, num_missing, hybrid)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)


def _runner(model: nn.Module, enforce_eager: bool = False) -> TorchSpyreModelRunner:
    """A runner with only the attributes _compile_for_spyre reads."""
    runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
    runner.model = model
    runner.compilation_config = types.SimpleNamespace(mode=CompilationMode.STOCK_TORCH_COMPILE)
    runner.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(enforce_eager=enforce_eager)
    )
    return runner


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


def test_ignores_a_module_list_of_bare_attention_layers() -> None:
    """Zamba2's shared ``dpa_list`` holds bare Attention layers, not blocks."""

    class SharedAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.dpa_list = nn.ModuleList([_fake_attention() for _ in range(2)])

    assert _repeated_block_lists(SharedAttention()) == []


def test_finds_heterogeneous_hybrid_stacks() -> None:
    """Hybrid Mamba+attention stacks (Granite 4.0, Jamba, Nemotron-H) mix classes."""
    model = _Model(num_layers=4, hybrid=True)
    assert len({type(b) for b in model.model.layers}) == 2
    assert _repeated_block_lists(model) == [model.model.layers]


def test_compile_blocks_wraps_every_block_in_place() -> None:
    model = _Model(num_layers=4)
    originals = list(model.model.layers)

    assert _runner(model)._compile_blocks() == 4

    for i, original in enumerate(originals):
        assert model.model.layers[i] is original
        assert isinstance(original, _Block)
        assert original._compiled_call_impl is not None


def test_compile_blocks_preserves_parameter_names() -> None:
    """An ``_orig_mod.`` segment in a parameter path breaks weight save/reload."""
    model = _Model(num_layers=4)
    before = [name for name, _ in model.named_parameters()]

    _runner(model)._compile_blocks()

    after = [name for name, _ in model.named_parameters()]
    assert after == before
    assert not any("_orig_mod" in name for name in after)


def test_pp_missing_layers_are_not_compiled() -> None:
    model = _Model(num_layers=2, num_missing=2)
    assert _runner(model)._compile_blocks() == 2
    assert all(isinstance(layer, PPMissingLayer) for layer in model.model.layers[2:])


def test_rejects_an_unknown_granularity_even_when_eager(monkeypatch) -> None:
    """Validation runs before the eager short-circuit, so typos are never silent."""
    monkeypatch.setenv("SPYRE_COMPILE_GRANULARITY", "blocks")
    with pytest.raises(ValueError, match="SPYRE_COMPILE_GRANULARITY"):
        _runner(_Model(num_layers=2), enforce_eager=True)._compile_for_spyre()


def test_empty_granularity_falls_back_to_block(monkeypatch) -> None:
    """`export SPYRE_COMPILE_GRANULARITY=$UNSET` must mean unset, not invalid."""
    monkeypatch.setenv("SPYRE_COMPILE_GRANULARITY", "")
    model = _Model(num_layers=2)
    _runner(model)._compile_for_spyre()
    assert all(block._compiled_call_impl is not None for block in model.model.layers)


def test_model_granularity_compiles_the_whole_model(monkeypatch) -> None:
    monkeypatch.setenv("SPYRE_COMPILE_GRANULARITY", "model")
    compiled: list[nn.Module] = []
    monkeypatch.setattr(torch, "compile", lambda m, **kw: compiled.append(m) or m)

    model = _Model(num_layers=2)
    runner = _runner(model)
    runner._compile_for_spyre()

    assert compiled == [model]
    assert all(block._compiled_call_impl is None for block in model.model.layers)
    assert model.model.embed_tokens._compiled_call_impl is None
    assert model.model.norm._compiled_call_impl is None


def test_falls_back_to_whole_model_when_no_blocks_are_found(monkeypatch) -> None:
    """The path every MLA and vision-tower model takes."""
    compiled: list[nn.Module] = []
    monkeypatch.setattr(torch, "compile", lambda m, **kw: compiled.append(m) or m)

    class NoBlocks(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

    model = NoBlocks()
    _runner(model)._compile_for_spyre()
    assert compiled == [model]


def test_eager_compiles_nothing(monkeypatch) -> None:
    compiled: list[nn.Module] = []
    monkeypatch.setattr(torch, "compile", lambda m, **kw: compiled.append(m) or m)

    model = _Model(num_layers=2)
    _runner(model, enforce_eager=True)._compile_for_spyre()

    assert compiled == []
    assert all(block._compiled_call_impl is None for block in model.model.layers)
    assert model.model.embed_tokens._compiled_call_impl is None
    assert model.model.norm._compiled_call_impl is None


def test_finds_the_head_and_tail_beside_the_block_list() -> None:
    model = _Model(num_layers=4)
    assert _head_and_tail_modules(model, model.model.layers) == [
        model.model.embed_tokens,
        model.model.norm,
    ]


def test_finds_a_head_nested_above_the_block_list() -> None:
    """BERT keeps the list under ``encoder`` and the embeddings on the grandparent."""

    class Embeddings(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.word_embeddings = nn.Embedding(16, hidden)
            self.LayerNorm = nn.LayerNorm(hidden)

    class Encoder(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.layer = nn.ModuleList([_Block(hidden) for _ in range(2)])

    class Bert(nn.Module):
        def __init__(self, hidden: int = 32):
            super().__init__()
            self.embeddings = Embeddings(hidden)
            self.encoder = Encoder(hidden)

    model = Bert()
    assert _repeated_block_lists(model) == [model.encoder.layer]
    # The whole embeddings module is the head; BERT has no post-block norm.
    assert _head_and_tail_modules(model, model.encoder.layer) == [model.embeddings]


def test_the_lm_head_is_not_mistaken_for_the_input_embedding() -> None:
    """``ParallelLMHead`` subclasses ``VocabParallelEmbedding``, so a type check alone
    would wrap the logits projection, which never runs inside the model forward."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    lm_head = ParallelLMHead.__new__(ParallelLMHead)
    nn.Module.__init__(lm_head)

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = lm_head
            self.layers = nn.ModuleList([_Block(32) for _ in range(2)])

    model = Tied()
    assert _head_and_tail_modules(model, model.layers) == []


def test_a_tail_norm_is_found_under_any_of_its_zoo_names() -> None:
    for attr in ("norm", "final_layernorm", "final_layer_norm", "ln_f", "norm_f", "final_norm"):
        model = nn.Module()
        model.layers = nn.ModuleList([_Block(32) for _ in range(2)])
        setattr(model, attr, nn.LayerNorm(32))
        assert _head_and_tail_modules(model, model.layers) == [getattr(model, attr)], attr


def test_compile_head_and_tail_wraps_in_place() -> None:
    model = _Model(num_layers=2)
    embed, norm = model.model.embed_tokens, model.model.norm

    assert _runner(model)._compile_head_and_tail() == 2

    assert model.model.embed_tokens is embed
    assert model.model.norm is norm
    assert embed._compiled_call_impl is not None
    assert norm._compiled_call_impl is not None


def test_compile_head_and_tail_preserves_parameter_names() -> None:
    """An ``_orig_mod.`` segment breaks weight save/reload, and the embedding weight is
    the one a tied lm-head shares."""
    model = _Model(num_layers=2)
    before = [name for name, _ in model.named_parameters()]

    _runner(model)._compile_head_and_tail()

    assert [name for name, _ in model.named_parameters()] == before


def test_block_granularity_compiles_the_head_and_tail_too() -> None:
    model = _Model(num_layers=2)
    _runner(model)._compile_for_spyre()

    assert all(block._compiled_call_impl is not None for block in model.model.layers)
    assert model.model.embed_tokens._compiled_call_impl is not None
    assert model.model.norm._compiled_call_impl is not None


def test_head_and_tail_are_left_eager_under_model_granularity(monkeypatch) -> None:
    monkeypatch.setenv("SPYRE_COMPILE_GRANULARITY", "model")
    monkeypatch.setattr(torch, "compile", lambda m, **kw: m)

    model = _Model(num_layers=2)
    _runner(model)._compile_for_spyre()

    assert model.model.embed_tokens._compiled_call_impl is None
    assert model.model.norm._compiled_call_impl is None


@pytest.mark.compile
def test_identical_blocks_share_compiled_artifacts_regardless_of_depth(
    isolated_dynamo_state,
) -> None:
    """Backend compile count must not grow with layer count -- and is 2, not 1,
    because layer 0 specializes on ``residual is None``."""
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters
    from torch._inductor.utils import fresh_cache

    def compile_counts(num_layers: int) -> tuple[int, int]:
        dynamo.reset()
        counters.clear()
        with fresh_cache():
            model = _Model(hidden=32, num_layers=num_layers)
            _runner(model)._compile_blocks()
            with torch.inference_mode():
                model(torch.zeros(2, 4, dtype=torch.long))
            return (
                counters["stats"]["unique_graphs"],
                counters["inductor"]["fxgraph_cache_miss"],
            )

    shallow_graphs, shallow_backend = compile_counts(2)
    deep_graphs, deep_backend = compile_counts(8)

    assert deep_backend == shallow_backend
    assert deep_graphs == shallow_graphs
    assert deep_backend < 8
