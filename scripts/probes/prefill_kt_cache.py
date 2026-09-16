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

"""Does storing K transposed pay for the head-major prefill kernel?

``matmul(X, W)`` wants X sticked on its reduction axis and W on its generated axis. The
head-major cache stores a page head_size-innermost, which is what ``probs @ V`` wants but
the transpose of what ``Q @ K^T`` wants, so the K page's last two axes are swapped every
time it is read. This times the same kernel against a K cache that already holds
``[head_size, block_size]`` pages, so no swap is needed, and checks both against SDPA.

    Q_LEN=512 SEQ_LEN=2048 LAYOUT_SOLVER=greedy python scripts/probes/prefill_kt_cache.py

Env: Q_LEN, SEQ_LEN, BLOCK_SIZE, KV_HEADS, QPK, HEAD_SIZE, ITERS, ARMS
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("SPYRE_NUM_CPUS", "8")

import torch  # noqa: E402
import torch_spyre  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

torch_spyre._autoload()
torch.spyre.set_device(0)
torch.zeros(1, dtype=torch.float16).to("spyre")

from spyre_inference.v1.attention.ops.layout import head_major_kv_layout  # noqa: E402


def _int(name, default):
    return int(os.environ.get(name, default))


KV = _int("KV_HEADS", 8)
QPK = _int("QPK", 4)
D = _int("HEAD_SIZE", 128)
B = _int("BLOCK_SIZE", 128)
Q_LEN = _int("Q_LEN", 512)
SEQ_LEN = _int("SEQ_LEN", 2048)
ITERS = _int("ITERS", 8)
NUM_HEADS = KV * QPK
NUM_BLOCKS = (SEQ_LEN + B - 1) // B
NUM_PAGES = max(NUM_BLOCKS, 8)
CTX = SEQ_LEN - Q_LEN
SCALE = D**-0.5
FP16_MIN = torch.finfo(torch.float16).min
ARMS = os.environ.get("ARMS", "kt,baseline").split(",")
# Q=1 production runs 908's folded kernel, not the batched one.
FOLD = Q_LEN == 1 and _int("FOLD", 1) == 1

print(
    f"config: Q_LEN={Q_LEN} SEQ_LEN={SEQ_LEN} NUM_BLOCKS={NUM_BLOCKS} BLOCK_SIZE={B} "
    f"KV={KV} QPK={QPK} D={D} iters={ITERS} arms={ARMS} fold={FOLD}"
)


def decode_kernel(query, row_index, k_pages, v_pages, tables, mask_tiles, scale, num_blocks,
                  q_len, num_heads, num_kv_heads, head_size, block_size, k_is_transposed):
    """908's folded decode kernel, optionally over a pre-transposed K cache.

    Production's decode path, so the Q=1 arm measures what it would actually save.
    """
    g = num_heads // num_kv_heads
    row = row_index[:1]
    q = query.index_select(0, row).reshape(num_kv_heads, g, head_size)

    tile_max = None
    tile_sum = None
    tile_out = None
    for i in range(num_blocks):
        kv_rows = tables[i]
        if k_is_transposed:
            kt = k_pages[kv_rows].reshape(num_kv_heads, head_size, block_size)
        else:
            kt = k_pages[kv_rows].reshape(num_kv_heads, block_size, head_size).permute(0, 2, 1)
        v_page = v_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)
        scores = torch.matmul(q, kt) * scale
        scores = scores + mask_tiles[i]
        scores_max = torch.amax(scores, dim=-1, keepdim=True)
        if i == 0:
            tile_max = scores_max
            probs = torch.exp(scores - scores_max)
            tile_out = torch.matmul(probs, v_page)
            tile_sum = probs.sum(dim=-1, keepdim=True)
        else:
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_out = tile_out * rescale
            tile_sum = tile_sum * rescale
            probs = torch.exp(scores - new_max)
            tile_out = tile_out + torch.matmul(probs, v_page)
            tile_sum = tile_sum + probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    return (tile_out / tile_sum).reshape(1, num_heads, head_size)


def kernel(query, row_index, k_pages, v_pages, tables, mask_tiles, scale, num_blocks,
           q_len, num_heads, num_kv_heads, head_size, block_size, k_is_transposed):
    """915's batched GQA prefill kernel, optionally over a pre-transposed K cache."""
    g = num_heads // num_kv_heads
    q_rows = query.index_select(0, row_index[:q_len])
    q = q_rows.unsqueeze(0).transpose(1, 2).reshape(num_kv_heads, g, q_len, head_size)

    tile_max = None
    tile_sum = None
    tile_out = None
    for i in range(num_blocks):
        idx = tables[i]
        k_page = k_pages.index_select(0, idx).squeeze(0).unsqueeze(1)
        v_page = v_pages.index_select(0, idx).squeeze(0).unsqueeze(1)
        kt = k_page if k_is_transposed else k_page.transpose(-2, -1)
        scores = torch.matmul(q, kt) * scale
        scores = scores + mask_tiles[i]
        scores_max = torch.amax(scores, dim=-1, keepdim=True)
        if i == 0:
            tile_max = scores_max
            probs = torch.exp(scores - tile_max)
            tile_out = torch.matmul(probs, v_page)
            tile_sum = probs.sum(dim=-1, keepdim=True)
        else:
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_out = tile_out * rescale
            tile_sum = tile_sum * rescale
            probs = torch.exp(scores - new_max)
            tile_out = tile_out + torch.matmul(probs, v_page)
            tile_sum = tile_sum + probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    attn = tile_out / tile_sum
    attn = attn.reshape(1, num_heads, q_len, head_size).transpose(1, 2)
    return attn.reshape(q_len, num_heads, head_size)


k_host = torch.randn(NUM_PAGES, KV, B, D, dtype=torch.float16)
v_host = torch.randn(NUM_PAGES, KV, B, D, dtype=torch.float16)
query = torch.randn(Q_LEN, NUM_HEADS, D, dtype=torch.float16)
row_index = torch.arange(Q_LEN, dtype=torch.int32).to("spyre")
pages_used = list(range(NUM_BLOCKS))

q_abs = CTX + torch.arange(Q_LEN).unsqueeze(1)
masks = []
for i in range(NUM_BLOCKS):
    p = i * B + torch.arange(B).unsqueeze(0)
    allow = (p <= q_abs) & (p < SEQ_LEN)
    tile = torch.zeros(Q_LEN, B, dtype=torch.float16)
    tile[~allow] = FP16_MIN
    masks.append(tile.contiguous().to("spyre"))
if FOLD:
    heads_col = torch.arange(KV, dtype=torch.int32).reshape(KV, 1)
    tables = [(pages_used[i] * KV + heads_col).contiguous().to("spyre") for i in range(NUM_BLOCKS)]
else:
    tables = [
        torch.tensor([pages_used[i]], dtype=torch.int32).to("spyre")
        for i in range(NUM_BLOCKS)
    ]
query_dev = query.to("spyre")

# Reference: SDPA over the same pages, in fp32 on the host.
k_ctx = torch.cat([k_host[pages_used[i]] for i in range(NUM_BLOCKS)], dim=1)
v_ctx = torch.cat([v_host[pages_used[i]] for i in range(NUM_BLOCKS)], dim=1)
mask_ctx = torch.cat([m.cpu().float() for m in masks], dim=-1)
q_ref = query.float().reshape(Q_LEN, KV, QPK, D).permute(1, 2, 0, 3)
ref_scores = torch.matmul(q_ref, k_ctx.float().unsqueeze(1).transpose(-2, -1)) * SCALE + mask_ctx
ref = torch.matmul(torch.softmax(ref_scores, dim=-1), v_ctx.float().unsqueeze(1))
ref = ref.reshape(NUM_HEADS, Q_LEN, D).transpose(0, 1)


def swapped_stick_layout(num_pages, block_size, head_size, dtype):
    """Layout for a logical ``[.., block_size, head_size]`` page stored block_size-sticked.

    Same logical shape as the cache has today, but the physical arrangement matmul wants
    for the K side: head_size becomes the outer device axis and block_size lands on the
    stick. If torch-spyre honours this, ``k_page.transpose(-2, -1)`` in the kernel is
    already satisfied by the layout, and the relayout moves to the KV store.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick

    eps = get_elem_in_stick(dtype)
    sticks = (block_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_pages, head_size, sticks, eps],
        stride_map=[block_size * head_size, 1, eps * head_size, head_size],
        device_dtype=get_device_dtype(dtype),
    )


def build(transposed):
    v_layout = head_major_kv_layout(NUM_PAGES * KV, B, D, torch.float16)
    v_dev = v_host.to("spyre", device_layout=v_layout)
    if transposed:
        # [pages, kv, head_size, block_size]: the axis matmul wants sticked is innermost.
        k_layout = head_major_kv_layout(NUM_PAGES * KV, D, B, torch.float16)
        k_dev = k_host.transpose(-2, -1).contiguous().to("spyre", device_layout=k_layout)
        return k_dev, v_dev
    if arm == "ktlayout":
        k_layout = swapped_stick_layout(NUM_PAGES * KV, B, D, torch.float16)
        return k_host.to("spyre", device_layout=k_layout), v_dev
    k_layout = head_major_kv_layout(NUM_PAGES * KV, B, D, torch.float16)
    return k_host.to("spyre", device_layout=k_layout), v_dev


def device_us(prof):
    total = 0.0
    for e in prof.key_averages():
        total += getattr(e, "self_device_time_total", 0.0) or 0.0
    return total


results = {}
for arm in ARMS:
    transposed = arm == "kt"
    k_dev, v_dev = build(transposed)  # noqa: F841 -- `arm` is read inside build
    fn = torch.compile(decode_kernel if FOLD else kernel, dynamic=False)
    if FOLD:
        k_arg = k_dev.view(NUM_PAGES * KV, D, B) if transposed else k_dev.view(NUM_PAGES * KV, B, D)
        v_arg = v_dev.view(NUM_PAGES * KV, B, D)
    else:
        k_arg, v_arg = k_dev, v_dev
    args = (query_dev, row_index, k_arg, v_arg, tables, masks, SCALE, NUM_BLOCKS, Q_LEN,
            NUM_HEADS, KV, D, B, transposed)
    for _ in range(2):
        got = fn(*args)
    torch.spyre.synchronize()
    err = (got.cpu().float() - ref).abs().max().item()

    per_iter = []
    for _ in range(ITERS):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
            fn(*args)
            torch.spyre.synchronize()
        per_iter.append(device_us(prof))
    per_iter.sort()
    med = per_iter[len(per_iter) // 2]
    results[arm] = med
    print(
        f"{arm:9s} device {med:9.1f}us  (min={per_iter[0]:.1f} max={per_iter[-1]:.1f})  "
        f"max abs err vs SDPA {err:.3e}"
    )

base = results.get("baseline")
for name in ("kt", "ktlayout"):
    if base and name in results:
        d = results[name] - base
        print(f"\n{name.upper()} {base:.1f} -> {results[name]:.1f}us ({100.0 * d / base:+.1f}%)")
