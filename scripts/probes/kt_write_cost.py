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

"""What a K-transposed cache costs on the write side.

If K is stored ``[block_size, head_size]``-swapped so ``Q @ K^T`` needs no stick swap, the
KV store has to put the token axis innermost instead. For a block-aligned prefill chunk
that is one permute of the incoming tile plus a page-granular ``index_copy_``; this times
that against the store the head-major backend does today, so the read-side saving can be
netted against it.

    TOKENS=512 python scripts/probes/kt_write_cost.py

Env: TOKENS, BLOCK_SIZE, KV_HEADS, HEAD_SIZE, NUM_PAGES, ITERS
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


T = _int("TOKENS", 512)
B = _int("BLOCK_SIZE", 128)
KV = _int("KV_HEADS", 8)
D = _int("HEAD_SIZE", 128)
NUM_PAGES = _int("NUM_PAGES", 16)
ITERS = _int("ITERS", 8)
NBLK = T // B
assert T % B == 0, "this probe covers the block-aligned chunk case"

print(f"config: TOKENS={T} BLOCK_SIZE={B} KV={KV} D={D} blocks_written={NBLK} iters={ITERS}")

key = torch.randn(T, KV, D, dtype=torch.float16).to("spyre")
block_ids = torch.arange(NBLK, dtype=torch.int32).to("spyre")
# The store the backend does today: one [T] row index per kv head.
rows = [
    (
        (torch.arange(T) // B) * KV * B
        + h * B
        + torch.arange(T) % B
    ).to(torch.int64).to("spyre")
    for h in range(KV)
]

layout = head_major_kv_layout(NUM_PAGES * KV, B, D, torch.float16)
k_tokenmajor = torch.zeros(NUM_PAGES, KV, B, D, dtype=torch.float16).to(
    "spyre", device_layout=layout
)
layout_t = head_major_kv_layout(NUM_PAGES * KV, D, B, torch.float16)
k_swapped = torch.zeros(NUM_PAGES, KV, D, B, dtype=torch.float16).to(
    "spyre", device_layout=layout_t
)


def store_tokenmajor(key, k_rows, row_index):
    key = key * 1.0
    for h, idx in enumerate(row_index):
        k_rows.index_copy_(0, idx, key[:, h])
    return k_rows


def store_swapped(key, k_pages, block_ids, num_blocks, block_size, kv, head_size):
    """Token axis innermost: permute the chunk, then write whole pages."""
    pages = (
        (key * 1.0)
        .reshape(num_blocks, block_size, kv, head_size)
        .permute(0, 2, 3, 1)
        .reshape(num_blocks, kv, head_size, block_size)
    )
    return k_pages.index_copy_(0, block_ids, pages)


def device_us(prof):
    return sum((getattr(e, "self_device_time_total", 0.0) or 0.0) for e in prof.key_averages())


def measure(label, fn, args):
    for _ in range(2):
        fn(*args)
    torch.spyre.synchronize()
    per_iter = []
    for _ in range(ITERS):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
            fn(*args)
            torch.spyre.synchronize()
        per_iter.append(device_us(prof))
    per_iter.sort()
    med = per_iter[len(per_iter) // 2]
    print(f"{label:14s} device {med:8.1f}us  (min={per_iter[0]:.1f} max={per_iter[-1]:.1f})")
    return med


results = {}
try:
    results["token-major"] = measure(
        "token-major",
        torch.compile(store_tokenmajor, dynamic=False),
        (key, k_tokenmajor.view(NUM_PAGES * KV * B, D), rows),
    )
except Exception as e:  # noqa: BLE001
    print(f"token-major   FAILED: {type(e).__name__}: {str(e)[:200]}")
try:
    results["swapped"] = measure(
        "swapped",
        torch.compile(store_swapped, dynamic=False),
        (key, k_swapped, block_ids, NBLK, B, KV, D),
    )
    # Correctness: the swapped store must land the same values, transposed.
    ref = key.cpu().reshape(NBLK, B, KV, D).permute(0, 2, 3, 1)
    got = k_swapped.cpu()[:NBLK]
    print(f"swapped store max abs diff vs source: {(got - ref).abs().max().item():.3e}")
except Exception as e:  # noqa: BLE001
    print(f"swapped       FAILED: {type(e).__name__}: {str(e)[:300]}")

if "token-major" in results and "swapped" in results:
    d = results["swapped"] - results["token-major"]
    print(f"\nWRITE-DELTA {d:+.1f}us per layer ({100.0 * d / results['token-major']:+.1f}%)")
