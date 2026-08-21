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

"""Spyre RoPE patch for vLLM's Transformers modeling backend.

The Transformers backend keeps an HF model's structure and swaps in vLLM
implementations for the layers it has replacements for — linears, convs, GLU MLPs,
fused QKV, RMSNorm, input embeddings, LM head — plus vLLM's ``Attention``. Layers it
has no replacement for keep running HF's code: RoPE, ``nn.LayerNorm``, and non-gated
MLP activations. RoPE is the only one of those the Spyre compiler cannot lower, so it
is the only one patched here.

HF applies the rotation via ``rotate_half``: a last-dim slice at ``head_dim/2`` plus
a ``cat``, whose sub-stick stride the compiler rejects (and which silently falls back
to CPU, changing the fp16 accumulation order).

``HfAdaptersForCausalLM`` subclasses ``TransformersForCausalLM`` and patches both HF
RoPE call sites — the ``rotary_emb`` that produces ``(cos, sin)`` and the
module-level ``apply_rotary_pos_emb`` that consumes it — to an equivalent 2x2
rotation-matrix formulation that needs no slicing. Where ``head_dim/2`` is not
stick-aligned, Q/K are expanded into a stick-aligned dimension for the rotation and
contracted back afterwards, leaving attention and the KV cache at the native
``head_dim`` (the platform's ``_maybe_pad_head_dim`` no-ops on this backend).

Everything else comes from upstream — model creation, weight loading, attention
routing, KV cache, scheduling, forward execution — and the Spyre OOT layers
(SpyreRMSNorm, SpyreParallelLMHead, ...) apply automatically because the backend
instantiates the vLLM classes they are registered against.

Activated when ``model_impl="transformers"`` on the Spyre platform via
``register_hf_adapters()``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from vllm.logger import init_logger
from vllm.model_executor.models.transformers import TransformersForCausalLM

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Spyre stick size at fp16: 128 bytes / 2 bytes per element. The 2x2 rotation's
# inner dim is head_dim/2, so a stick-aligned head_dim is a multiple of 2*_STICK.
_STICK = 64


def _text_backbone(model: nn.Module) -> nn.Module:
    """The transformer backbone holding ``rotary_emb``.

    Causal-LM wrappers nest it at ``.model``; multimodal wrappers nest the text
    decoder one level deeper at ``.model.language_model``.
    """
    inner = getattr(model, "model", model)
    return getattr(inner, "language_model", inner)


def _rotate(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` [B, H, L, D] with rotation matrices ``rot`` [B, L, 2, 2, D/2].

    Pairs dim ``j`` with ``j + D/2`` through a broadcast multiply-reduce rather than
    HF's ``rotate_half`` slice-and-concat, so nothing is sliced on the sticked dim.
    """
    b, h, length, d = x.shape
    pairs = x.transpose(1, 2).reshape(b, length, h, 2, d // 2)
    out = rot[:, :, None].mul(pairs.unsqueeze(-3)).sum(4, keepdim=True).flatten(3)
    return out.transpose(1, 2)


class _SpyreRotaryEmbedding(nn.Module):
    """Drop-in for HF's ``RotaryEmbedding`` that emits 2x2 rotation matrices.

    Builds a ``[max_pos, 2, 2, D/2]`` cache of ``[[cos, -sin], [sin, cos]]`` on the
    host from the HF rope's ``inv_freq``, and indexes it by ``position_ids``.
    ``forward`` returns ``(rotation_matrices, None)`` to satisfy HF's ``(cos, sin)``
    contract: the tuple is threaded unchanged through the decoder layer, and the
    patched ``apply_rotary_pos_emb`` reads the first element and ignores the second.

    When ``padded_head_dim`` is set, the rotation is widened with identity entries so
    that rotating the stick-aligned Q/K passes the padded dims through unchanged.
    """

    def __init__(
        self,
        inv_freq: torch.Tensor,
        scaling: float = 1.0,
        padded_head_dim: int | None = None,
    ):
        super().__init__()
        # Plain attributes, not buffers: the cache has no Spyre kernel and must stay
        # on the host across ``model.to(device)``.
        self._inv_freq = inv_freq.detach().to(device="cpu", dtype=torch.float32)
        self._scaling = scaling
        self._padded_half = padded_head_dim // 2 if padded_head_dim is not None else None
        self._cache: torch.Tensor | None = None
        self._cached_len = 0

    def _apply(self, fn, recurse=True):
        return self

    def _extend_cache(self, length: int) -> None:
        if length <= self._cached_len:
            return
        target = max(length, self._cached_len * 2, 2048)
        freqs = torch.outer(torch.arange(target, dtype=torch.float32), self._inv_freq)
        half = freqs.shape[-1]
        cos, sin = torch.cos(freqs) * self._scaling, torch.sin(freqs) * self._scaling
        rot = torch.stack([cos, -sin, sin, cos], dim=1).view(target, 2, 2, half)

        if self._padded_half is not None and self._padded_half > half:
            ident = torch.zeros(target, 2, 2, self._padded_half - half)
            ident[:, 0, 0, :] = 1.0
            ident[:, 1, 1, :] = 1.0
            rot = torch.cat([rot, ident], dim=-1)

        # The platform enforces float16, so the cache dtype is fixed.
        self._cache = rot.contiguous().to(torch.float16)
        self._cached_len = target

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        positions = position_ids.to("cpu")
        self._extend_cache(int(positions.max().item()) + 1)
        assert self._cache is not None  # _extend_cache always populates it
        return self._cache[positions].to(x.device), None


def _qk_expand_matrix(orig_hd: int, padded_hd: int) -> torch.Tensor:
    """Interleaved expand matrix for Q/K (RoPE-compatible half-split)."""
    half, phalf = orig_hd // 2, padded_hd // 2
    m = torch.zeros(orig_hd, padded_hd)
    m[:half, :phalf] = torch.eye(half, phalf)
    m[half:, phalf:] = torch.eye(half, phalf)
    return m


def _make_spyre_apply_rotary(qk_expand=None):
    """Replace apply_rotary_pos_emb with matmul-based RoPE.

    When *qk_expand* is provided (head_dim/2 is not stick-aligned), Q/K are
    temporarily padded into the stick-aligned dimension for the rotation,
    then contracted back to the original size.
    """
    qk_contract = qk_expand.t().contiguous() if qk_expand is not None else None
    _cached: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def wrapper(q, k, cos, sin=None, *args, **kwargs):
        if qk_expand is not None:
            assert qk_contract is not None  # set together with qk_expand above
            dev = q.device
            if dev not in _cached:
                _cached[dev] = (
                    qk_expand.to(device=dev, dtype=q.dtype),
                    qk_contract.to(device=dev, dtype=q.dtype),
                )
            exp, con = _cached[dev]
            q = torch.matmul(q, exp)
            k = torch.matmul(k, exp)

        q, k = _rotate(q, cos), _rotate(k, cos)

        if qk_expand is not None:
            q = torch.matmul(q, con)
            k = torch.matmul(k, con)

        return q, k

    wrapper._spyre_patched = True
    return wrapper


class HfAdaptersForCausalLM(TransformersForCausalLM):
    """TransformersForCausalLM wrapper to use HF adapters."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self._fix_generic_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        logger.debug("HfAdaptersForCausalLM ready: %s", type(self.model).__name__)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights and patch rope."""
        result = super().load_weights(weights)
        self._patch_rope()
        return result

    @staticmethod
    def _fix_generic_config(vllm_config: VllmConfig) -> None:
        """Re-resolve generic PretrainedConfig produced by vLLM's
        config parser for some models where both config.json and params.json exists
        and force HF-format weight loading."""
        hf_config = vllm_config.model_config.hf_config
        if type(hf_config) is not PretrainedConfig:
            return

        model_id = vllm_config.model_config.hf_config_path or vllm_config.model_config.model
        try:
            resolved = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=vllm_config.model_config.trust_remote_code,
                revision=vllm_config.model_config.revision,
            )
        except Exception:
            logger.warning("AutoConfig re-resolve failed for %s", model_id, exc_info=True)
            return

        skip = {"model_type", "_name_or_path", "transformers_version", "auto_map", "architectures"}
        for key, val in hf_config.to_dict().items():
            if key not in skip and val is not None:
                setattr(resolved, key, val)

        vllm_config.model_config.hf_config = resolved
        vllm_config.model_config.hf_text_config = resolved.get_text_config()
        if vllm_config.load_config.load_format in ("auto", "mistral"):
            vllm_config.load_config.load_format = "hf"
        logger.debug(
            "Re-resolved config: %s (model_type=%s), load_format=hf",
            type(resolved).__name__,
            resolved.model_type,
        )

    # TODO: Add support for models with fused QKV / gate_up projections
    # (e.g. Phi-3) by splitting them into separate modules with TP-aware
    # weight redistribution and partial-rotary dimension permutation.

    def _patch_rope(self):
        """Point both HF RoPE call sites at the 2x2 rotation.

        A ``head_dim/2`` that is not a multiple of ``_STICK`` also gets an
        expand/contract matrix pair, so Q/K are widened into a stick-aligned
        dimension for the rotation and contracted back; attention and the KV cache
        keep the native head_dim.
        """
        cfg = self.model.config
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

        stick_aligned = ((head_dim + 2 * _STICK - 1) // (2 * _STICK)) * (2 * _STICK)
        padded_head_dim = stick_aligned if stick_aligned > head_dim else None
        qk_exp = (
            _qk_expand_matrix(head_dim, padded_head_dim) if padded_head_dim is not None else None
        )

        backbone = _text_backbone(self.model)
        hf_rope = backbone.rotary_emb
        # One shared instance: the rotation cache is position-indexed, so every layer
        # can read the same one.
        spyre_rope = _SpyreRotaryEmbedding(
            hf_rope.inv_freq,  # ty: ignore[invalid-argument-type]
            getattr(hf_rope, "attention_scaling", 1.0),
            padded_head_dim=padded_head_dim,
        )
        backbone.rotary_emb = spyre_rope

        patched_mods: set[int] = set()
        for name, module in self.model.named_modules():
            if isinstance(module, _SpyreRotaryEmbedding):
                continue

            cls_name = module.__class__.__name__

            # Architectures that hold a RoPE module per decoder layer rather than one
            # on the backbone.
            if cls_name.endswith("RotaryEmbedding"):
                pname, _, attr = name.rpartition(".")
                parent = self.model.get_submodule(pname) if pname else self.model
                setattr(parent, attr, spyre_rope)
                continue

            if "Attention" not in cls_name:
                continue

            if not hasattr(module, "rotary_emb"):
                module.rotary_emb = spyre_rope

            # apply_rotary_pos_emb is a module-level function, so patch it once per HF
            # modeling module rather than once per attention instance.
            mod = sys.modules.get(type(module).__module__)
            if mod is None or id(mod) in patched_mods:
                continue
            orig = getattr(mod, "apply_rotary_pos_emb", None)
            if orig is None or getattr(orig, "_spyre_patched", False):
                continue
            mod.apply_rotary_pos_emb = _make_spyre_apply_rotary(qk_exp)
            patched_mods.add(id(mod))


# vLLM's Transformers backend test checks ModelConfig.using_transformers_backend()
# compares _ModelInfo.architecture (set to model_cls.__name__) against "TransformersForCausalLM".
# Without this, the subclass name "HfAdaptersForCausalLM" causes that check to return False.
HfAdaptersForCausalLM.__name__ = "TransformersForCausalLM"
