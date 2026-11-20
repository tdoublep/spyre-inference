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

"""Unfuses BertSelfAttention's QKV: the fused output goes to HBM and Q, K, V each re-read it all."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.models.bert import BertAttention, BertSelfAttention

from spyre_inference import envs
from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod, spyre_linear_t
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    _encoder_rect_kernel,
    encoder_grid,
    encoder_inline_active,
)

logger = init_logger(__name__)

_NAMES = ("q", "k", "v")


def _param(t: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(t.contiguous(), requires_grad=False)


def _split_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    q, k, v = (
        spyre_linear_t(hidden_states, getattr(self, f"{n}_weight_t"), getattr(self, f"{n}_bias"))
        for n in _NAMES
    )
    return self.attn(q, k, v)


def split_bert_qkv(model: nn.Module) -> None:
    """Runs on CPU, before the device move."""
    count = 0
    for module in model.modules():
        if not isinstance(module, BertSelfAttention):
            continue
        proj = module.qkv_proj
        if type(proj.quant_method) is not SpyreUnquantizedLinearMethod:
            continue
        if proj.weight.device.type != "cpu":
            continue
        sizes = [module.q_size, module.kv_size, module.kv_size]
        weight = cast(torch.Tensor, proj.weight).data  # stored transposed: [in, q + 2kv]
        weights = torch.split(weight, sizes, dim=1)
        biases = torch.split(proj.bias.data, sizes) if proj.bias is not None else (None,) * 3
        for n, w, b in zip(_NAMES, weights, biases):
            module.register_parameter(f"{n}_weight_t", _param(w))
            module.register_parameter(f"{n}_bias", None if b is None else _param(b))
        del module.qkv_proj
        module.forward = _split_forward.__get__(module)  # ty: ignore[invalid-assignment]
        count += 1
    if count:
        logger.info("Split the fused QKV projection of %d BertSelfAttention layers.", count)
    groups = envs.SPYRE_ENCODER_HEAD_GROUPS
    if count and groups > 1:
        _group_heads(model, groups)


def _grouped_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    sa, so = self.self, self.output
    groups = sa.spyre_head_groups
    qkv = [
        [
            spyre_linear_t(
                hidden_states, getattr(sa, f"{n}_weight_t_{g}"), getattr(sa, f"{n}_bias_{g}")
            )
            for n in _NAMES
        ]
        for g in range(groups)
    ]
    mask = encoder_grid().mask
    if mask is None or not encoder_inline_active():
        q, k, v = (torch.cat([qkv[g][i] for g in range(groups)], dim=-1) for i in range(3))
        return so(sa.attn(q, k, v), hidden_states)
    heads, kv_heads, d = sa.num_heads // groups, sa.num_kv_heads // groups, sa.head_dim
    out = so.dense.bias
    for g, (q, k, v) in enumerate(qkv):
        a = _encoder_rect_kernel(
            q.view(-1, heads, d),
            k.view(-1, kv_heads, d),
            v.view(-1, kv_heads, d),
            mask,
            sa.attn.impl.scale,
            mask.shape[0],
            mask.shape[-1],
            heads,
            kv_heads,
            d,
        )
        out = torch.matmul(a.reshape(-1, heads * d), getattr(so, f"wo_t_{g}")) + out
    return so.LayerNorm(out + hidden_states)


def _group_heads(model: nn.Module, groups: int) -> None:
    """Each group's attention fits LX untiled, so Q, K and V stay out of HBM."""
    if get_tensor_model_parallel_world_size() > 1:
        logger.warning("SPYRE_ENCODER_HEAD_GROUPS ignored at TP>1 (out-proj all-reduce).")
        return
    count = 0
    for module in model.modules():
        if not isinstance(module, BertAttention):
            continue
        sa, so = cast(BertSelfAttention, module.self), module.output
        if not hasattr(sa, "q_weight_t") or sa.num_kv_heads % groups:
            continue
        if type(so.dense.quant_method) is not SpyreUnquantizedLinearMethod:
            continue
        for n in _NAMES:
            w, b = getattr(sa, f"{n}_weight_t"), getattr(sa, f"{n}_bias")
            width = w.shape[1] // groups
            for g in range(groups):
                cols = slice(g * width, (g + 1) * width)
                sa.register_parameter(f"{n}_weight_t_{g}", _param(w.data[:, cols]))
                sa.register_parameter(f"{n}_bias_{g}", None if b is None else _param(b.data[cols]))
            delattr(sa, f"{n}_weight_t")
            delattr(sa, f"{n}_bias")
        wo = cast(torch.Tensor, so.dense.weight).data  # stored transposed: [q_size, hidden]
        rows = wo.shape[0] // groups
        for g in range(groups):
            so.register_parameter(f"wo_t_{g}", _param(wo[g * rows : (g + 1) * rows]))
        sa.spyre_head_groups = groups
        module.forward = _grouped_forward.__get__(module)  # ty: ignore[invalid-assignment]
        count += 1
    if count:
        logger.info(
            "Split the attention of %d BertAttention layers into %d head groups.", count, groups
        )
