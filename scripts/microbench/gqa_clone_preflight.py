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

"""Pre-flight for the GQA-clone experiment. Meta tensors only, no device needed.

Prints the ``aten.clone`` traffic each arm's matmul actually leaves in the
post-pass graph — the graph the Spyre lowering sees, after the aten decomposition
and torch-spyre's ``bmm_unflatten_pass``. That is the cost the microbench then
prices, so run this first: if the clones are not there, there is nothing to
measure.

Numerical equivalence is not checked here. ``page_attn_native_bcast_kernel``
falls back to ``torch.matmul`` off the Spyre device, so a CPU comparison would be
vacuous; the microbench's own correctness gate covers the device path.

    PYTHONPATH=. python3 scripts/microbench/gqa_clone_preflight.py
"""

import argparse

import torch
import torch_spyre  # noqa: F401  -- registers spyre::batched_matmul
import torch_spyre._inductor.temp_passes as temp_passes
from torch._decomp import core_aten_decompositions
from torch.fx.experimental.proxy_tensor import make_fx

DTYPE = torch.float16
ARMS = {
    "A_torch_matmul": lambda x, y: torch.matmul(x, y),
    "C_batched_matmul": lambda x, y: torch.ops.spyre.batched_matmul(x, y),
}


def clone_traffic(fn, num_kv_heads, group, query_len, head_size, block_size):
    """Clone shapes and KiB in the QK matmul's post-pass graph."""
    query = torch.empty(
        num_kv_heads, group, query_len, head_size, dtype=DTYPE, device="meta"
    )
    # The size-1 axis is the GQA group slot the query broadcasts against.
    keys = torch.empty(num_kv_heads, 1, head_size, block_size, dtype=DTYPE, device="meta")

    graph = make_fx(fn, decomposition_table=dict(core_aten_decompositions()))(query, keys)
    temp_passes.bmm_unflatten_pass.apply(graph.graph)

    shapes, total = [], 0
    for node in graph.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.clone.default:
            val = node.meta["val"]
            shapes.append(tuple(val.shape))
            total += val.numel() * val.element_size()
    return shapes, total / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--query-lens", type=int, nargs="+", default=[1, 512])
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=128)
    args = ap.parse_args()

    print(
        f"QK matmul [KV,G,q,D] @ [KV,1,D,block], KV={args.num_kv_heads} "
        f"D={args.head_size} block={args.block_size}"
    )
    print(f"{'q':>6} {'G':>3} {'arm':>18} {'KiB':>8}  clone shapes")
    clean = True
    for query_len in args.query_lens:
        for group in args.groups:
            for name, fn in ARMS.items():
                shapes, kib = clone_traffic(
                    fn, args.num_kv_heads, group, query_len, args.head_size, args.block_size
                )
                print(f"{query_len:>6} {group:>3} {name:>18} {kib:>8.0f}  {shapes}")
                if name.startswith("C_") and shapes:
                    clean = False

    print(
        "\nV matmul mirrors QK, so per-page traffic is twice the KiB above.\n"
        f"arm C clone-free: {'PASS' if clean else 'FAIL'}"
    )
    raise SystemExit(0 if clean else 1)


if __name__ == "__main__":
    main()
