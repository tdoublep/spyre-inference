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

"""Dense-grid encoder attention: reshape, one SDPA call, reshape back.

``forward`` receives exactly ``B * L`` rows with sequence ``s`` at rows ``s*L``, so these
tests feed that layout directly. ``test_masked_sdpa_spyre.py`` covers the kernel itself;
this file covers grid choice, mask construction, per-step caching and the write-back.
"""

from unittest.mock import Mock

import pytest
import torch
from spyre_testing_plugin.attn_helpers import assert_close_outliers
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec, EncoderOnlyAttentionSpec

from spyre_inference import envs
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionMetadataBuilder,
    SpyrePagedKVCache,
)
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    SpyreEncoderAttentionImpl,
    build_key_pad_mask,
)

# extra `encoder_attention` mark so CI can split this into its own job
# because these tests are pretty slow.
pytestmark = [pytest.mark.attention, pytest.mark.encoder_attention]

# Small declared shapes so the tests stay fast. The ladders cross, and each gains its
# own limit, so these give (64|128) x (2|4).
_TEST_MAX_MODEL_LEN = 128
_TEST_MAX_NUM_SEQS = 4


@pytest.fixture(autouse=True)
def declared_shapes(monkeypatch):
    """Pin the declared ``(L, B)`` ladders; the impl reads them at construction."""
    monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "64")
    monkeypatch.setenv("SPYRE_ATTN_NUM_SEQS_BUCKETS", "2")
    envs.clear_env_cache()
    yield
    envs.clear_env_cache()


@pytest.fixture()
def configure_device(request):
    """The spyre card check is lazy so it does not claim the device at import."""
    device_mode = request.param
    if device_mode == "spyre" and not spyre_available():
        pytest.skip("Spyre device not available")
    return device_mode


@pytest.fixture()
def configure_compilation(request):
    from vllm.config import get_cached_compilation_config
    from vllm.config.compilation import CompilationMode

    torch._dynamo.reset()
    cfg = get_cached_compilation_config()
    original_mode = cfg.mode
    original_limit = torch._dynamo.config.accumulated_recompile_limit
    cfg.mode = getattr(CompilationMode, request.param)
    torch._dynamo.config.accumulated_recompile_limit = 1024
    yield request.param
    cfg.mode = original_mode
    torch._dynamo.config.accumulated_recompile_limit = original_limit
    torch._dynamo.reset()


def _build_metadata(
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    spec_cls: type[AttentionSpec] = EncoderOnlyAttentionSpec,
):
    """Use the real SpyreAttentionMetadataBuilder to construct metadata.

    ``seq_lens`` / ``query_start_loc`` carry the *real* ragged lengths: that is what
    production hands attention, and what the key-pad mask is derived from.
    """
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    vllm_config.model_config.get_num_attention_heads = Mock(return_value=num_query_heads)
    vllm_config.model_config.get_num_kv_heads = Mock(return_value=num_kv_heads)
    vllm_config.cache_config.block_size = block_size

    kv_cache_spec = spec_cls(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
    )
    builder = SpyreAttentionMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layers.0.self_attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    num_seqs = len(seq_lens)
    num_actual_tokens = int(query_start_loc[-1].item())
    common_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        num_reqs=num_seqs,
        num_actual_tokens=num_actual_tokens,
        max_query_len=int((query_start_loc[1:] - query_start_loc[:-1]).max().item()),
        max_seq_len=int(seq_lens.max().item()),
        block_table_tensor=torch.zeros(num_seqs, 1, dtype=torch.int32),
        slot_mapping=torch.zeros(num_actual_tokens, dtype=torch.int64),
        causal=False,
    )
    return builder.build(common_prefix_len=0, common_attn_metadata=common_metadata)


def _make_impl(num_query_heads: int, num_kv_heads: int, head_size: int):
    from vllm.config import get_current_vllm_config

    # The shapes come from the engine limits crossed with the two ladders above, so the
    # limits have to be the test's, not the default model's.
    cfg = get_current_vllm_config()
    cfg.model_config.max_model_len = _TEST_MAX_MODEL_LEN
    cfg.scheduler_config.max_num_seqs = _TEST_MAX_NUM_SEQS
    cfg.scheduler_config.max_num_batched_tokens = _TEST_MAX_MODEL_LEN * _TEST_MAX_NUM_SEQS
    return SpyreEncoderAttentionImpl(
        num_heads=num_query_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
    )


def _dense_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    real_lens: list[int],
    aligned_len: int,
    scale: float,
) -> list[torch.Tensor]:
    """Per-sequence bidirectional attention over real rows only.

    Returns one ``[n_s, H, D]`` tensor per sequence. Pad rows are excluded rather
    than compared: nothing downstream reads them.
    """
    out: list[torch.Tensor] = []
    for seq, length in enumerate(real_lens):
        base = seq * aligned_len
        q = query[base : base + length] * scale
        k = key[base : base + length]
        v = value[base : base + length]
        if q.shape[1] != k.shape[1]:
            repeat = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, repeat, dim=1)
            v = torch.repeat_interleave(v, repeat, dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out.append(torch.einsum("hqk,khd->qhd", attn, v))
    return out


class TestKeyPadMask:
    def test_shape_is_broadcastable_over_head_and_query(self):
        """``[B, 1, 1, L]``: SDPA broadcasts the singleton axes itself."""
        mask = build_key_pad_mask(3, 64, [10, 64, 1], dtype=torch.float16)
        assert mask.shape == (3, 1, 1, 64)
        assert mask.is_contiguous()

    def test_zero_on_real_keys_and_negative_on_pad(self):
        mask = build_key_pad_mask(2, 64, [10, 64], dtype=torch.float16)
        assert torch.equal(mask[0, 0, 0, :10], torch.zeros(10, dtype=torch.float16))
        assert (mask[0, 0, 0, 10:] < 0).all()
        # Full-length sequence is unmasked everywhere.
        assert torch.equal(mask[1, 0, 0], torch.zeros(64, dtype=torch.float16))

    def test_pad_fill_is_finite(self):
        """``-inf`` would make a fully masked row NaN inside SDPA's online softmax."""
        mask = build_key_pad_mask(1, 64, [8], dtype=torch.float16)
        assert torch.isfinite(mask).all()
        assert mask.min() <= torch.finfo(torch.float16).min / 2

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="num_seqs"):
            build_key_pad_mask(3, 64, [10, 20], dtype=torch.float16)


@pytest.mark.parametrize(
    "configure_device",
    [pytest.param("cpu", id="device_cpu"), pytest.param("spyre", id="device_spyre")],
    indirect=True,
)
@pytest.mark.parametrize(
    "configure_compilation",
    [
        pytest.param("NONE", id="compilation_NONE"),
        pytest.param("STOCK_TORCH_COMPILE", id="compilation_STOCK"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    ("real_lens", "aligned_len", "batch"),
    [
        pytest.param([64, 64], 64, 2, id="full_B2"),
        pytest.param([37, 64], 64, 2, id="livepad_B2"),
        pytest.param([9, 70, 5], 128, 4, id="livepad_batchpad_B4"),
        pytest.param([100], 128, 4, id="single_seq_batchpad"),
    ],
)
@pytest.mark.parametrize(
    "num_heads", [pytest.param((16, 4), id="GQA"), pytest.param((16, 16), id="MHA")]
)
@torch.inference_mode()
def test_forward_matches_reference(
    default_vllm_config,
    num_heads: tuple[int, int],
    real_lens: list[int],
    aligned_len: int,
    batch: int,
    configure_compilation: str,
    configure_device: str,
) -> None:
    """Dense forward matches a per-sequence bidirectional reference.

    A dropped pad mask shows up here: every real row would attend to the padding
    that fills the rest of its ``L`` columns.
    """
    num_query_heads, num_kv_heads = num_heads
    torch.set_default_device("cpu")
    set_random_seed(0)
    dtype = torch.float16
    head_size = 64
    scale = head_size**-0.5
    padded_tokens = batch * aligned_len

    query = torch.randn(padded_tokens, num_query_heads, head_size, dtype=dtype)
    key = torch.randn(padded_tokens, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(padded_tokens, num_kv_heads, head_size, dtype=dtype)

    attn_metadata = _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=64,
        seq_lens=torch.tensor(real_lens, dtype=torch.int32),
        query_start_loc=torch.tensor([0] + real_lens, dtype=torch.int32).cumsum(
            dim=0, dtype=torch.int32
        ),
    )

    impl = _make_impl(num_query_heads, num_kv_heads, head_size)
    device = torch.device(configure_device)
    output = torch.empty_like(query).to(device)
    impl.forward(
        layer=None,
        query=query,
        key=key,
        value=value,
        kv_cache=SpyrePagedKVCache(k_pages=torch.empty(0), v_pages=torch.empty(0)),
        attn_metadata=attn_metadata,
        output=output,
    )
    assert attn_metadata.encoder_pack_len == aligned_len
    assert attn_metadata.encoder_pack_batch == batch

    actual = output.to("cpu")
    expected = _dense_reference(query, key, value, real_lens, aligned_len, scale)
    for seq, (length, ref) in enumerate(zip(real_lens, expected)):
        base = seq * aligned_len
        assert_close_outliers(
            actual[base : base + length],
            ref,
            max_outliers=5,
            atol=0.3,
            rtol=0.2,
            outlier_atol=0.6,
            outlier_rtol=0.4,
        )


@torch.inference_mode()
def test_grid_and_mask_cached_across_layers(default_vllm_config) -> None:
    """Layer 2+ reuses the step's grid and mask instead of rebuilding them."""
    torch.set_default_device("cpu")
    set_random_seed(0)
    real_lens = [37, 64]
    attn_metadata = _build_metadata(
        num_query_heads=4,
        num_kv_heads=4,
        head_size=64,
        block_size=64,
        seq_lens=torch.tensor(real_lens, dtype=torch.int32),
        query_start_loc=torch.tensor([0] + real_lens, dtype=torch.int32).cumsum(
            dim=0, dtype=torch.int32
        ),
    )
    impl = _make_impl(4, 4, 64)
    query = torch.randn(2 * 64, 4, 64, dtype=torch.float16)
    kv_cache = SpyrePagedKVCache(k_pages=torch.empty(0), v_pages=torch.empty(0))

    impl.forward(
        layer=None,
        query=query,
        key=query,
        value=query,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=torch.empty_like(query),
    )
    first_mask = attn_metadata.encoder_key_pad_mask
    assert first_mask is not None

    impl.forward(
        layer=None,
        query=query,
        key=query,
        value=query,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=torch.empty_like(query),
    )
    assert attn_metadata.encoder_key_pad_mask is first_mask, "mask was rebuilt"


@torch.inference_mode()
def test_forward_rejects_a_body_that_is_not_b_times_l(default_vllm_config) -> None:
    """The grid is pinned to the row count, so a ragged body matches no shape.

    This is what stops the runner's dense expansion and attention's dispatch from
    silently disagreeing: only a shape whose ``B*L`` equals the rows handed in is a
    candidate.
    """
    torch.set_default_device("cpu")
    real_lens = [37, 64]
    attn_metadata = _build_metadata(
        num_query_heads=4,
        num_kv_heads=4,
        head_size=64,
        block_size=64,
        seq_lens=torch.tensor(real_lens, dtype=torch.int32),
        query_start_loc=torch.tensor([0] + real_lens, dtype=torch.int32).cumsum(
            dim=0, dtype=torch.int32
        ),
    )
    impl = _make_impl(4, 4, 64)
    # Chosen shape is (64, 2) = 128 rows; hand it a ragged 101.
    query = torch.randn(101, 4, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="in 101 rows"):
        impl.forward(
            layer=None,
            query=query,
            key=query,
            value=query,
            kv_cache=SpyrePagedKVCache(k_pages=torch.empty(0), v_pages=torch.empty(0)),
            attn_metadata=attn_metadata,
            output=torch.empty_like(query),
        )


@torch.inference_mode()
def test_forward_rejects_a_batch_no_shape_covers(default_vllm_config) -> None:
    """Declared shapes top out at (128, 4); nine sequences cannot be served."""
    torch.set_default_device("cpu")
    real_lens = [8] * 9
    attn_metadata = _build_metadata(
        num_query_heads=4,
        num_kv_heads=4,
        head_size=64,
        block_size=64,
        seq_lens=torch.tensor(real_lens, dtype=torch.int32),
        query_start_loc=torch.tensor([0] + real_lens, dtype=torch.int32).cumsum(
            dim=0, dtype=torch.int32
        ),
    )
    impl = _make_impl(4, 4, 64)
    query = torch.randn(9 * 64, 4, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="No declared encoder shape"):
        impl.forward(
            layer=None,
            query=query,
            key=query,
            value=query,
            kv_cache=SpyrePagedKVCache(k_pages=torch.empty(0), v_pages=torch.empty(0)),
            attn_metadata=attn_metadata,
            output=torch.empty_like(query),
        )


def test_impl_owns_no_kv_cache_machinery(default_vllm_config) -> None:
    """Standalone, not a SpyreAttentionImpl subclass.

    ``attn_layer._can_split`` opts a layer into the traced KV write purely on
    ``do_kv_cache_update`` existing, so an encoder impl that inherited it would
    scatter into an unbound cache. Staging buffers were allocated on the same
    isinstance check and never read.
    """
    from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl

    impl = _make_impl(4, 4, 64)
    assert not isinstance(impl, SpyreAttentionImpl)
    assert not hasattr(impl, "do_kv_cache_update")
    assert not hasattr(impl, "staging_buffers")
    assert impl.record_graphs() == 0


@torch.inference_mode()
def test_encoder_build_skips_kv_cache_fields(default_vllm_config) -> None:
    """An ENCODER_ONLY spec gets metadata with no block/page/tile fields."""
    real_lens = [16, 32]
    attn_metadata = _build_metadata(
        num_query_heads=4,
        num_kv_heads=4,
        head_size=64,
        block_size=64,
        seq_lens=torch.tensor(real_lens, dtype=torch.int32),
        query_start_loc=torch.tensor([0] + real_lens, dtype=torch.int32).cumsum(
            dim=0, dtype=torch.int32
        ),
    )
    assert attn_metadata.num_seqs == 2
    assert attn_metadata.page_index_tables_cpu is None
    assert attn_metadata.encoder_pack_batch is None, "grid is built in forward, not build()"
