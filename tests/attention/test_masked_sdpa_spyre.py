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

"""Compiled SDPA on Spyre honours an additive key-pad ``attn_mask``.

The encoder path previously hand-rolled a two-graph softmax on the belief that the mask
was dropped, and nothing tested it. These cases pin it at the text-encoder shapes with a
live pad (``real_len < L``).

SDPA is compile-only on Spyre, so the device side is compiled and the reference eager CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.utils.torch_utils import set_random_seed

from spyre_inference.custom_ops.utils import convert

pytestmark = [pytest.mark.attention, pytest.mark.encoder_attention]


@pytest.fixture(autouse=True)
def _isolate_dynamo():
    """One compiled artifact per case.

    Every case compiles the same ``_sdpa`` code object at a different shape, and
    with ``dynamic=False`` the second entry trips "Guard failed on the same frame
    it was created". Production never sees this: each shape is warmed once.

    The default device is cleared too. ``set_default_device`` pushes a
    torch-function mode, so a sibling test file that leaves one behind makes a
    guard created under one mode stack fail its check under another
    (``___check_torch_function_mode_stack``). Normalising here keeps this file
    order-independent.
    """
    torch.set_default_device(None)
    torch._dynamo.reset()
    original = torch._dynamo.config.accumulated_recompile_limit
    torch._dynamo.config.accumulated_recompile_limit = 1024
    yield
    torch._dynamo.config.accumulated_recompile_limit = original
    torch._dynamo.reset()
    torch.set_default_device(None)


def _sdpa(q, k, v, mask, scale):
    """``[B, H, L, D]`` in and out. Mask is additive ``[B, 1, 1, L]``."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def _key_pad_mask(kv_lens: list[int], aligned_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Additive ``[B, 1, 1, L]``: 0 on real keys, ``finfo.min / 2`` on pad.

    Half of ``finfo.min`` rather than the full value (and rather than ``-inf``):
    the torch-spyre decomposition does ``amax`` then ``exp(scores - max)``, so a
    fully masked query row yields ``-inf - -inf`` = NaN. Its own passing tests are
    named ``..._with_finite_mask`` for this reason.
    """
    cpu = torch.device("cpu")
    neg = torch.finfo(dtype).min / 2
    kv = torch.tensor(kv_lens, dtype=torch.int32, device=cpu).unsqueeze(1)
    pos = torch.arange(aligned_len, dtype=torch.int32, device=cpu).unsqueeze(0)
    zero = torch.zeros((), dtype=dtype, device=cpu)
    fill = torch.tensor(neg, dtype=dtype, device=cpu)
    row = torch.where(pos < kv, zero, fill)
    return row.view(len(kv_lens), 1, 1, aligned_len).contiguous()


def _row_cosine(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Per-row cosine over the last dim, both flattened to ``[rows, D]`` fp32."""
    a = actual.reshape(-1, actual.shape[-1]).float()
    e = expected.reshape(-1, expected.shape[-1]).float()
    return F.cosine_similarity(a, e, dim=-1)


# (num_heads, head_size, aligned_len, real_lens) -- head_size 64 is BGE/e5,
# 32 is MiniLM (padded to the stick by the caller in production).
_CASES = [
    # Every sequence exactly fills the grid: the no-live-pad baseline.
    pytest.param(12, 64, 128, [128, 128, 128, 128], id="full-B4"),
    # Live pad, the case the encoder docstrings claim collapses.
    pytest.param(12, 64, 128, [100, 128, 37, 64], id="livepad-B4"),
    # One sequence plus batch pad: dummy rows get kv_len=1, never 0.
    pytest.param(12, 64, 128, [73, 1, 1, 1], id="batchpad-B4"),
    # BGE-large head count, single sequence, heavy pad.
    pytest.param(16, 64, 512, [40, 512, 511, 1], id="livepad-L512"),
    # MiniLM head size already stick-padded to 64 by _pad_head_dim_to_stick.
    pytest.param(12, 64, 64, [13, 64, 1, 33], id="minilm-padded-D"),
]


@pytest.mark.skipif(not spyre_available(), reason="Spyre device not available")
@pytest.mark.parametrize(("num_heads", "head_size", "aligned_len", "real_lens"), _CASES)
def test_compiled_sdpa_honours_key_pad_mask(num_heads, head_size, aligned_len, real_lens):
    """Compiled Spyre SDPA with an additive key-pad mask matches eager CPU.

    If the mask were dropped, real rows would attend to pad and cosine would fall
    apart (the ~0.46 the encoder docstrings cite). Only real rows are compared:
    padded rows are never read in production.
    """
    set_random_seed(0)
    cpu = torch.device("cpu")
    dtype = torch.float16
    batch = len(real_lens)
    scale = float(head_size) ** -0.5

    shape = (batch, num_heads, aligned_len, head_size)
    q = torch.randn(shape, dtype=dtype, device=cpu)
    k = torch.randn(shape, dtype=dtype, device=cpu)
    v = torch.randn(shape, dtype=dtype, device=cpu)
    mask = _key_pad_mask(real_lens, aligned_len, dtype)

    expected = _sdpa(q, k, v, mask, scale)

    device = torch.device("spyre")
    compiled = torch.compile(_sdpa, dynamic=False)
    actual = compiled(
        convert(q, device),
        convert(k, device),
        convert(v, device),
        convert(mask, device),
        scale,
    )
    actual = convert(actual, "cpu")

    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all(), "compiled SDPA produced NaN/Inf"

    # Real rows only, per sequence: a dropped mask shows up here.
    for seq, length in enumerate(real_lens):
        cos = _row_cosine(actual[seq, :, :length, :], expected[seq, :, :length, :])
        assert cos.min() > 0.99, (
            f"seq {seq} (len {length}) min row cosine {cos.min():.4f} -- "
            "compiled SDPA is not honouring attn_mask"
        )


@pytest.mark.skipif(not spyre_available(), reason="Spyre device not available")
def test_masked_sdpa_beats_an_unmasked_reference():
    """Guards the test above from being vacuously true.

    If the pad content happened not to matter, the assertion above would pass even
    with a dropped mask. So check the masked and unmasked references actually
    differ at these shapes: that difference is what the test can detect.
    """
    set_random_seed(0)
    cpu = torch.device("cpu")
    dtype = torch.float16
    real_lens = [100, 128, 37, 64]
    num_heads, head_size, aligned_len = 12, 64, 128
    scale = float(head_size) ** -0.5

    shape = (len(real_lens), num_heads, aligned_len, head_size)
    q = torch.randn(shape, dtype=dtype, device=cpu)
    k = torch.randn(shape, dtype=dtype, device=cpu)
    v = torch.randn(shape, dtype=dtype, device=cpu)
    mask = _key_pad_mask(real_lens, aligned_len, dtype)

    masked = _sdpa(q, k, v, mask, scale)
    unmasked = F.scaled_dot_product_attention(q, k, v, attn_mask=None, scale=scale)

    # Sequence 1 is full length, so masking is a no-op there; 2 is mostly pad.
    cos = _row_cosine(masked[2, :, : real_lens[2], :], unmasked[2, :, : real_lens[2], :])
    assert cos.mean() < 0.9, (
        "masked and unmasked attention agree at these shapes, so the test above "
        "cannot distinguish a dropped mask -- pick a harder case"
    )
