"""Per-op device time for one decode variant.

Needs a torch-spyre built with USE_SPYRE_PROFILER=1. The pin in this repo's
pyproject sets it to 0, which compiles the AIUPTI activity provider out, and the
torch profiler then reports no device events at all (every device time reads
0.000) -- so wall clock is the only signal until it is rebuilt.

Answers the question wall clock cannot: of a variant's total, how much is the
page index_select, how much the two matmuls, and how much the online-softmax
pointwise chain.

    LAYOUT_SOLVER=greedy python experimental/profile_decode_variant.py \
        --variant chunked_gather --chunk 8
"""

import argparse
import sys

import torch

sys.path.insert(
    0, "/home/senuser/spyre-inference/.claude/worktrees/batched-decode-design/experimental"
)
from microbench_batched_decode_v2 import build_inputs, make_callables  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-seqs", type=int, default=4)
    p.add_argument("--num-blocks", type=int, default=16)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--num-queries-per-kv", type=int, default=4)
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--staging-rows", type=int, default=513)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--kv-tile", type=int, default=32)
    p.add_argument("--variant", default="chunked_gather")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--top", type=int, default=18)
    a = p.parse_args()

    t = build_inputs(a)
    fn = make_callables(a, t)[a.variant]

    for _ in range(3):
        fn()
    torch.spyre.synchronize()

    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    dev_act = None
    for name in ("PrivateUse1", "SPYRE", "CUDA"):
        if hasattr(ProfilerActivity, name):
            dev_act = getattr(ProfilerActivity, name)
            break
    if dev_act is not None:
        acts.append(dev_act)
    print("activities: %s" % [str(x) for x in acts])

    with profile(activities=acts, record_shapes=True) as prof:
        for _ in range(a.iters):
            fn()
        torch.spyre.synchronize()

    evs = prof.key_averages()
    dev_total = sum((e.self_device_time_total or 0.0) for e in evs)
    cpu_total = sum((e.self_cpu_time_total or 0.0) for e in evs)
    print("\nvariant=%s  iters=%d" % (a.variant, a.iters))
    print("device self total: %9.3f ms/call" % (dev_total / 1000.0 / a.iters))
    print("cpu    self total: %9.3f ms/call" % (cpu_total / 1000.0 / a.iters))

    rows = sorted(evs, key=lambda e: -(e.self_device_time_total or 0.0))
    print("\n%-52s %11s %8s %11s" % ("op", "dev ms/call", "count", "cpu ms/call"))
    for e in rows[: a.top]:
        dev = (e.self_device_time_total or 0.0) / 1000.0 / a.iters
        cpu = (e.self_cpu_time_total or 0.0) / 1000.0 / a.iters
        if dev == 0.0 and cpu < 0.01:
            continue
        print("%-52s %11.3f %8d %11.3f"
              % (str(e.key)[:52], dev, e.count // max(a.iters, 1), cpu))


if __name__ == "__main__":
    main()
