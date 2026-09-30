#!/usr/bin/env python3
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

"""TP=N micro-benchmark: does flattening a rank-2 all_reduce input to 1-D cost anything?

Compares `SpyreCommunicator.all_reduce`'s flatten (`reshape(-1)` -> collective ->
`reshape(orig)`) against handing the collective the [T, H] tensor directly, compiled,
on a bare collective and on the row-parallel-linear shape it serves in the model
(matmul -> all_reduce -> residual add). Both variants are compiled in the same process
and timed in alternating reps, so launch-to-launch drift hits them equally.

    uv run --no-sync python scripts/microbench/all_reduce_flatten_microbench.py \\
        --world-sizes 2,4 --code-dir /tmp/ar_code

`--code-dir` saves each variant's Inductor output code, to diff for extra kernels.

Each (tp, graph) runs in its own set of rank subprocesses; a shape that aborts (the
rank-2 path does for some shapes, spyre-comms#463) is recorded as failed and the
remaining shapes are relaunched.
"""

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time

DEFAULT_SHAPES = (
    "1x4096,3x4096,4x4096,32x4096,128x4096,512x4096,1x5120,32x5120,512x5120"
)
VARIANTS = ("2d", "flat")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch(args):
    shapes_all = args.shapes.split(",")
    results = []
    for tp in (int(x) for x in args.world_sizes.split(",")):
        for graph in args.graphs.split(","):
            pending = list(shapes_all)
            while pending:
                done, stderr_tail = _launch_one(args, tp, graph, pending)
                for r in done:
                    print(f"tp{tp}/{graph} {r['shape']}: 2d {r['2d']['median_us']:.1f} us, "
                          f"flat {r['flat']['median_us']:.1f} us", flush=True)
                results.extend(done)
                finished = {r["shape"] for r in done}
                pending = [s for s in pending if s not in finished]
                if pending:
                    failed = pending.pop(0)
                    print(f"FAILED tp{tp}/{graph} {failed}:\n{stderr_tail}", flush=True)
                    results.append(dict(tp=tp, graph=graph, shape=failed, status="failed"))
                if args.out:
                    with open(args.out, "w") as f:
                        json.dump(results, f, indent=1)
    _summarise(results)


def _launch_one(args, tp, graph, shapes):
    port = _free_port()
    procs = []
    for rank in range(tp):
        env = {
            **os.environ,
            "RANK": str(rank),
            "WORLD_SIZE": str(tp),
            "LOCAL_RANK": str(rank),
            "LOCAL_WORLD_SIZE": str(tp),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "PYTHONUNBUFFERED": "1",
        }
        pin = []
        if args.pin_cpus:
            # A disjoint block per rank, from CPUs local to the cards' NUMA node.
            lo, hi = (int(c) for c in args.pin_cpus.split("-"))
            per = (hi - lo + 1) // tp
            pin = ["taskset", "-c", f"{lo + rank * per}-{lo + (rank + 1) * per - 1}"]
        cmd = pin + [
            sys.executable, __file__, "--worker", "--graph", graph,
            "--shapes", ",".join(shapes), "--iters", str(args.iters),
            "--reps", str(args.reps), "--warmup", str(args.warmup),
        ]
        if args.code_dir:
            cmd += ["--code-dir", args.code_dir]
        procs.append(
            subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, errors="replace",
                             text=True)
        )
    outs = []
    for p in procs:
        try:
            outs.append(p.communicate(timeout=args.timeout))
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            outs.append(p.communicate())
    done = [
        json.loads(line[len("RESULT "):])
        for line in outs[0][0].splitlines()
        if line.startswith("RESULT ")
    ]
    tail = "\n".join(outs[0][1].splitlines()[-15:] + outs[-1][1].splitlines()[-5:])
    return done, tail


def _summarise(results):
    print(f"\n{'tp':>2} {'graph':7} {'shape':>9} {'2d us':>9} {'flat us':>9} {'flat/2d':>8} "
          f"{'paired flat/2d (min..max)':>26}")
    for r in sorted(results, key=lambda r: (r["tp"], r["graph"],
                                            [int(x) for x in r["shape"].split("x")])):
        head = f"{r['tp']:>2} {r['graph']:7} {r['shape']:>9}"
        if r["status"] != "ok":
            print(f"{head} {'FAIL':>9}")
            continue
        a, b = r["2d"]["median_us"], r["flat"]["median_us"]
        ratios = [f / d for f, d in zip(r["flat"]["reps_us"], r["2d"]["reps_us"])]
        print(f"{head} {a:9.1f} {b:9.1f} {b / a:8.3f} "
              f"{statistics.median(ratios):10.3f} ({min(ratios):.3f}..{max(ratios):.3f})")


def worker(args):
    import torch
    import torch.distributed as dist
    from torch._inductor.utils import run_and_get_code

    os.environ["VLLM_PLUGINS"] = "spyre_inference,spyre_inference_ops"
    os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    import torch_spyre

    torch_spyre._autoload()
    torch.spyre.set_device(local_rank)

    from torch.distributed._functional_collectives import _resolve_group_name
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

    load_general_plugins()
    cfg = EngineArgs(
        model="facebook/opt-125m",
        tensor_parallel_size=int(os.environ["WORLD_SIZE"]),
        dtype="float16",
        enforce_eager=True,
        distributed_executor_backend="external_launcher",
    ).create_engine_config()

    with set_current_vllm_config(cfg):
        init_worker_distributed_environment(
            cfg, rank, distributed_init_method="env://", local_rank=local_rank,
            backend=current_platform.dist_backend,
        )
        import vllm.distributed.parallel_state as ps

        gn = _resolve_group_name(ps._TP.device_group)
        cpu_group = ps._TP.cpu_group
        tp = dist.get_world_size(cpu_group)
        device = torch.device(f"spyre:{local_rank}")

        def make(flatten):
            def all_reduce(x):
                orig_shape = x.shape
                if flatten:
                    x = x.reshape(-1)
                out = torch.ops._c10d_functional.all_reduce(x, "sum", gn)
                out = torch.ops._c10d_functional.wait_tensor(out)
                return out.reshape(orig_shape) if flatten else out

            if args.graph == "bare":
                # Not the graph input itself: the compiled collective overwrites it in place.
                return lambda x, w, res: all_reduce(x * 2.0)
            return lambda x, w, res: all_reduce(x @ w) + res

        for shape in args.shapes.split(","):
            t, h = (int(s) for s in shape.split("x"))
            k = h // tp  # per-rank row-parallel input features
            g = torch.Generator().manual_seed(rank)
            x_cpu = (torch.randn(t, k if args.graph == "linear" else h, generator=g) * 0.1).half()
            w_cpu = (torch.randn(k, h, generator=g) * 0.05).half()
            res_cpu = torch.randn(t, h, generator=g).half()
            x, w, res = (a.to(device) for a in (x_cpu, w_cpu, res_cpu))

            steps = {}
            for v in VARIANTS:
                step = torch.compile(make(v == "flat"), dynamic=False)
                out, code = run_and_get_code(step, x, w, res)
                torch.spyre.synchronize(device)
                _check(out, x_cpu, w_cpu, res_cpu, args.graph, cpu_group)
                if args.code_dir and rank == 0:
                    os.makedirs(args.code_dir, exist_ok=True)
                    path = os.path.join(args.code_dir, f"tp{tp}_{args.graph}_{shape}_{v}.py")
                    with open(path, "w") as f:
                        f.write("\n\n# ----\n\n".join(code))
                for _ in range(args.warmup):
                    step(x, w, res)
                torch.spyre.synchronize(device)
                steps[v] = step

            reps = {v: [] for v in VARIANTS}
            for i in range(args.reps):
                for v in VARIANTS if i % 2 == 0 else reversed(VARIANTS):
                    dist.barrier(group=cpu_group)
                    t0 = time.perf_counter()
                    for _ in range(args.iters):
                        steps[v](x, w, res)
                    torch.spyre.synchronize(device)
                    reps[v].append((time.perf_counter() - t0) / args.iters * 1e6)
            if rank == 0:
                print("RESULT " + json.dumps(dict(
                    tp=tp, graph=args.graph, shape=shape, status="ok",
                    **{v: dict(median_us=statistics.median(r), min_us=min(r), reps_us=r)
                       for v, r in reps.items()},
                )), flush=True)

    dist.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _check(out, x_cpu, w_cpu, res_cpu, graph, cpu_group):
    """Compare against a CPU reference: a fast-but-wrong variant is not a result."""
    import torch
    import torch.distributed as dist

    local = (x_cpu.float() @ w_cpu.float()) if graph == "linear" else x_cpu.float() * 2.0
    dist.all_reduce(local, group=cpu_group)
    want = local + res_cpu.float() if graph == "linear" else local
    torch.testing.assert_close(out.cpu().float(), want, atol=5e-2, rtol=2e-2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--graph", default="linear")
    p.add_argument("--world-sizes", default="2,4")
    p.add_argument("--graphs", default="bare,linear")
    p.add_argument("--shapes", default=DEFAULT_SHAPES)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--pin-cpus", help="CPU range (e.g. 0-35) split across ranks via taskset")
    p.add_argument("--code-dir", help="save each variant's Inductor output code here")
    p.add_argument("--out", help="write raw results as JSON")
    args = p.parse_args()
    worker(args) if args.worker else launch(args)


if __name__ == "__main__":
    main()
