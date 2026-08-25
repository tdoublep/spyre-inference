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

"""Spyre OOT replacement for ParallelLMHead.

Spyre Device Constraints:
    - Tensor Parallelism: TP>=1 supported with vocabulary sharding (each rank
      computes logits for its vocab partition)
    - Quantization: Fp8Config supported (resolves to UnquantizedEmbeddingMethod).
      Other quantization methods raise NotImplementedError.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from .linear import SpyreTransposedWeightMethod


logger = init_logger(__name__)


class SpyreUnquantizedLMHeadMethod(SpyreTransposedWeightMethod, UnquantizedEmbeddingMethod):
    """LM-head projection via the shared transposed-weight fast path."""

    WEIGHT_T_ATTR = "padded_weight_t"
    ROW_ALIGN = 64 * 32


@ParallelLMHead.register_oot(name="ParallelLMHead")
class SpyreParallelLMHead(ParallelLMHead):
    """Out-of-tree (OOT) ParallelLMHead implementation for IBM's Spyre device.

    The projection lives in `SpyreUnquantizedLMHeadMethod.apply`, reached via
    `LogitsProcessor._apply_head` -> `lm_head.quant_method.apply`. The base
    `ParallelLMHead.forward` raises and is unused.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Only UnquantizedEmbeddingMethod supported. Fp8Config resolves to it;
        # other quantization methods are rejected.
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                f"SpyreParallelLMHead does not support {type(self.quant_method).__name__}."
            )

        logger.debug("Building SpyreParallelLMHead with TP size %d ", self.tp_size)

        # Set the custom quantization method to route through spyre
        self.quant_method = SpyreUnquantizedLMHeadMethod()


def install_tied_lm_head_projection(model: torch.nn.Module) -> None:
    """Route a tied lm_head through the padded transposed weight.

    `tie_word_embeddings` sets `lm_head = embed_tokens`, leaving the head a
    VocabParallelEmbedding that never reaches SpyreParallelLMHead: logits go through
    `F.linear(x, weight)`, which relayouts the `[vocab, hidden]` table with vocab
    innermost on every call. Work division may only split a stick dim by a divisor of
    its stick count, so a 151936 vocab (2374 = 2 * 1187 sticks) is stuck at 2 cores
    and exceeds the 256 MB per-core span once hidden reaches 2048.

    `weight` is left as-is: it is still `embed_tokens.weight`, and still needs the
    row-gathered layout for the gather.
    """
    for module in model.modules():
        head = getattr(module, "lm_head", None)
        # An untied head is a ParallelLMHead, which already resolves to the OOT class.
        if not isinstance(head, VocabParallelEmbedding) or isinstance(head, ParallelLMHead):
            continue
        if isinstance(head.quant_method, SpyreUnquantizedLMHeadMethod):
            continue
        head.quant_method = SpyreUnquantizedLMHeadMethod()
        head.quant_method.process_weights_after_loading(head)
        vocab, hidden = head.weight.shape
        logger.info(
            "Tied lm_head (vocab=%d, hidden=%d): projecting through padded_weight_t.",
            vocab,
            hidden,
        )
