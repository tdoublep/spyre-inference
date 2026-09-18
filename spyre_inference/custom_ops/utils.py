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

"""Utility functions for Spyre custom operations.

This module provides helper functions for preparing tensors and data structures
for execution on IBM's Spyre device, primarily handling device transfer and
dtype conversion.
"""

from functools import lru_cache

import torch
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _convert_op_func(
    tensor: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Opaque-op body: device/dtype conversion with the Spyre dtype detour.

    Hidden behind `torch.ops.vllm.spyre_convert` so the device transfers and
    the spyre `torch.Tensor.to` monkey-patch are not traced into outer
    torch.compile graphs (no DeviceCopy nodes leak into the Inductor IR).
    """
    target_device = device if device is not None else tensor.device
    target_dtype = dtype if dtype is not None else tensor.dtype

    if tensor.device.type == target_device.type and tensor.dtype == target_dtype:
        # Inlined attention traces _build_plans, so this op is replayed from the
        # compiled graph on operands that are already converted. The assert is a
        # waste-detector, not a correctness requirement, so it steps aside there.
        from spyre_inference import envs

        if envs.SPYRE_ATTN_INLINE:
            # `clone`, not `tensor`: `_convert_op_fake` returns a fresh `torch.empty`,
            # so the schema promises a non-aliasing output. Handing back an input
            # makes functionalization and the memory planner both believe they own
            # the same buffer, which frees it twice.
            return tensor.clone()
        raise RuntimeError(
            f"Trying to convert a tensor to the same device ({tensor.device.type}) "
            + f"and same dtype ({tensor.dtype}), should never happen!"
        )

    # Spyre requires CPU for dtype changes
    if tensor.device.type == "spyre" and tensor.dtype != target_dtype:
        tensor = tensor.to(device="cpu")

    if tensor.dtype != target_dtype:
        tensor = tensor.to(dtype=target_dtype)

    if tensor.device.type != target_device.type:
        tensor = tensor.to(device=target_device)

    return tensor


def _convert_op_fake(
    tensor: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_device = device if device is not None else tensor.device
    target_dtype = dtype if dtype is not None else tensor.dtype
    return torch.empty(tensor.shape, dtype=target_dtype, device=target_device)


def convert(tensor, device=None, dtype=None):
    """Convert tensor device and/or dtype. No-op when both are None.

    Routes through the opaque custom op `torch.ops.vllm.spyre_convert` so the
    transfer is invisible to torch.compile / Dynamo. None tensors are
    short-circuited at the Python boundary because `infer_schema` does not
    accept Optional[Tensor] returns.

    Args:
        tensor: Input tensor, or None (passed through as None).
        device: Target device as `str` or `torch.device` (None = keep current).
        dtype: Target dtype (None = keep current).

    Returns:
        Converted tensor, or None if input is None.
    """
    if tensor is None:
        return None
    if isinstance(device, str):
        device = torch.device(device)
    # Short-circuit a true no-op at the call site so Inductor never emits a
    # same-device/dtype spyre_convert FallbackKernel into the graph.
    target_device = device if device is not None else tensor.device
    target_dtype = dtype if dtype is not None else tensor.dtype
    if tensor.device.type == target_device.type and tensor.dtype == target_dtype:
        return tensor
    return torch.ops.vllm.spyre_convert(
        tensor,
        device,  # ty: ignore[invalid-argument-type]
        dtype,  # ty: ignore[invalid-argument-type]
    )


def _offset_positions_op_func(
    position_ids: torch.Tensor, offset: int, device: torch.device
) -> torch.Tensor:
    """Opaque-op body: ``position_ids + offset`` via the host, back as int64.

    Stock torch-spyre cannot schedule an SDSC int32 add, and an int64 add
    CPU-falls-back through ``spyre::to_dtype_cpu``, which is registered only for
    PrivateUse1 -- so tracing the add emits a fallback the dispatcher then
    refuses on CPU. Keeping the whole detour behind an opaque op is what lets a
    whole-model graph (``SPYRE_COMPILE_GRANULARITY=model``) compile and run.
    """
    pos = position_ids
    if pos.device.type != "cpu":
        pos = pos.to(device="cpu")
    pos = pos + int(offset)
    return pos.to(device=device, dtype=torch.int64)


def _offset_positions_op_fake(
    position_ids: torch.Tensor, offset: int, device: torch.device
) -> torch.Tensor:
    return torch.empty(position_ids.shape, dtype=torch.int64, device=device)


def offset_positions(position_ids: torch.Tensor, offset: int, device: torch.device):
    """``position_ids + offset`` as int64 on ``device``."""
    return torch.ops.vllm.spyre_offset_positions(
        position_ids,
        offset,
        device,  # ty: ignore[invalid-argument-type]
    )


@lru_cache(maxsize=1)
def register():
    """Register the spyre_convert custom op with vLLM."""
    # CompositeExplicitAutograd so the op dispatches regardless of input device
    # (convert is called with both CPU and Spyre input tensors).
    direct_register_custom_op(
        op_name="spyre_convert",
        op_func=_convert_op_func,
        fake_impl=_convert_op_fake,
        dispatch_key="CompositeExplicitAutograd",
    )
    logger.debug_once("Registered custom op: spyre_convert")
    direct_register_custom_op(
        op_name="spyre_offset_positions",
        op_func=_offset_positions_op_func,
        fake_impl=_offset_positions_op_fake,
        dispatch_key="CompositeExplicitAutograd",
    )
    logger.debug_once("Registered custom op: spyre_offset_positions")


def place_row_gathered(src: torch.Tensor, fn, name: str) -> torch.Tensor:
    """Move a 2D gather source to device with its rows outermost."""
    # TODO(tdoublep): can this be moved upstream?
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick

    # fn cannot carry a device_layout, so probe it on one row to learn the destination.
    probe = fn(src[:1])
    if probe.device.type != "spyre":
        return fn(src)

    num_rows, row_width = src.shape
    # Spyre needs a gather's indexed dim at device position 0, and a row must fill
    # whole sticks: 64 elements for fp16 (128-byte stick / 2 bytes per element).
    elems_per_stick = get_elem_in_stick(probe.dtype)
    if row_width % elems_per_stick:
        logger.warning_once(
            "%s: row width %d is not a multiple of %d, keeping the default layout (slower gather).",
            name,
            row_width,
            elems_per_stick,
        )
        return fn(src)

    return src.to(  # ty: ignore[no-matching-overload]
        probe.device,
        dtype=probe.dtype,
        device_layout=SpyreTensorLayout(
            device_size=[num_rows, row_width // elems_per_stick, elems_per_stick],
            stride_map=[row_width, elems_per_stick, 1],
            device_dtype=get_device_dtype(probe.dtype),
        ),
    )
