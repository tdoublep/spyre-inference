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

"""Spyre OOT replacement for VocabParallelEmbedding."""

from functools import lru_cache

from typing import cast

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)
from vllm.utils.torch_utils import direct_register_custom_op

from torch_spyre._C import get_elem_in_stick

from .utils import convert

logger = init_logger(__name__)


def row_gather_layout(num_rows: int, row_width: int, dtype: torch.dtype):
    """Row-axis-outermost layout, so a row gather reads only the rows it wants."""
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    eps = get_elem_in_stick(dtype)
    sticks = row_width // eps
    return SpyreTensorLayout(
        device_size=[num_rows, sticks, eps],
        stride_map=[sticks * eps, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


@VocabParallelEmbedding.register_oot(name="VocabParallelEmbedding")
class SpyreVocabParallelEmbedding(VocabParallelEmbedding):
    """Out-of-tree (OOT) VocabParallelEmbedding implementation for IBM's Spyre device."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                f"SpyreVocabParallelEmbedding does not support quantized "
                f"embeddings (got {type(self.quant_method).__name__})."
            )

    def _apply(self, fn, recurse=True):
        # The vocab table is only ever gathered from, so once it lands on device give it
        # a layout whose gather reads the wanted rows instead of the whole table. Spyre
        # requires the indexed dim outermost; the default layout puts it inwards.
        cpu_weight = cast(torch.Tensor, self.weight).data
        super()._apply(fn, recurse)
        moved = cast(torch.Tensor, self.weight).data
        if cpu_weight.device.type != "cpu" or moved.device.type != "spyre":
            return self
        num_rows, row_width = cpu_weight.shape
        if row_width % get_elem_in_stick(moved.dtype):
            # A partial trailing stick needs padding this layout cannot express.
            return self
        self.weight.data = cpu_weight.to(moved.dtype).to(  # ty: ignore[no-matching-overload]
            moved.device,
            device_layout=row_gather_layout(num_rows, row_width, moved.dtype),
        )
        return self

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            # The per-rank mask still runs on CPU: upstream get_masked_input_and_mask
            # does `input_ >= start` under torch.compile, which Spyre's inductor backend
            # rejects for int64 constants (see test_int64_compiled_compare_against_python_int).
            # The embedding gather itself runs on-device below.
            masked_input, keep = torch.ops.vllm.spyre_vocab_mask(
                convert(input_, device="cpu"),
                self.shard_indices.org_vocab_start_index,  # ty: ignore[invalid-argument-type]
                self.shard_indices.org_vocab_end_index,  # ty: ignore[invalid-argument-type]
                self.shard_indices.num_org_vocab_padding,  # ty: ignore[invalid-argument-type]
                self.shard_indices.added_vocab_start_index,  # ty: ignore[invalid-argument-type]
                self.shard_indices.added_vocab_end_index,  # ty: ignore[invalid-argument-type]
                self.weight.data.dtype,  # ty: ignore[invalid-argument-type]
            )
            masked_input = convert(masked_input, device=input_.device)
            keep = convert(keep, device=input_.device)
        else:
            masked_input = input_
            keep = None

        output = self.quant_method.embedding(self, masked_input.long())

        if keep is not None:
            output = output * keep
            output = tensor_model_parallel_all_reduce(output)
        return output


def _vocab_mask_op_func(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = input_.device
    masked_input, input_mask = get_masked_input_and_mask(
        input_,
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
    )
    keep = (~input_mask).to(dtype=dtype).unsqueeze(-1)
    return masked_input.to(device), keep.to(device)


def _vocab_mask_op_fake(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_input = torch.empty(input_.shape, dtype=input_.dtype, device=input_.device)
    keep = torch.empty((*input_.shape, 1), dtype=dtype, device=input_.device)
    return masked_input, keep


@lru_cache(maxsize=1)
def register():
    """Register the spyre_vocab_mask custom op with vLLM."""
    direct_register_custom_op(
        op_name="spyre_vocab_mask",
        op_func=_vocab_mask_op_func,
        fake_impl=_vocab_mask_op_fake,
        mutates_args=[],
        dispatch_key="CPU",
    )
    logger.debug_once("Registered custom op: spyre_vocab_mask")
