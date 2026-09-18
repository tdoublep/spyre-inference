"""Per-step latency at a FIXED shape, in-process. No server, no client.

The claim under test is a launch-count claim: 3 launches per block -> 1, i.e.
~43 -> ~23 launches per step. That is a property of one step at one shape, so
measure exactly that: the same 4 texts (4 slots x extent 512 = the 2048 body
bucket) embedded REPEATS times, wall time per call.

Why not `vllm bench serve`: its warmup phase is sequential, so every warmup
request is a 1-sequence batch, which is a shape warmup never compiled -- one
serving-path compile of ~190 s per new shape. That is a real cost of the
approach but it is not the per-step number, and it swamps it.

Writes b3/out/<B3_TAG>.steps.json, and the last call's embeddings so the perf
arm doubles as a free correctness check.
"""

import json
import os
import statistics
import time

import torch  # noqa: F401  (must precede the torch_spyre backend autoload)

from vllm import LLM

from embed_check import MODEL, TEXTS

OUT = "/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion/b3/out"


def main() -> int:
    tag = os.environ["B3_TAG"]
    repeats = int(os.environ.get("B3_REPEATS", "40"))
    bucket = os.environ.get("B3_BUCKET", "2048")

    llm = LLM(
        model=MODEL,
        runner="pooling",
        max_model_len=512,
        max_num_seqs=4,
        enable_prefix_caching=False,
        **(
            {}
            if bucket == "default"
            else {"compilation_config": {"compile_sizes": [int(bucket)]}}
        ),
    )

    # Settle: the first calls carry any lazy compile the warmup missed, and we
    # want that attributed to itself rather than averaged into the measurement.
    settle = []
    for _ in range(3):
        t0 = time.perf_counter()
        llm.embed(TEXTS)
        settle.append(time.perf_counter() - t0)
    print(f"settle (s): {[f'{x:.4f}' for x in settle]}", flush=True)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        outs = llm.embed(TEXTS)
        times.append(time.perf_counter() - t0)

    times_ms = sorted(x * 1e3 for x in times)
    res = {
        "tag": tag,
        "bucket": bucket,
        "inline": os.environ.get("SPYRE_ATTN_INLINE"),
        "repeats": repeats,
        "settle_ms": [x * 1e3 for x in settle],
        "ms": times_ms,
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "min_ms": times_ms[0],
        "p10_ms": times_ms[max(0, int(0.10 * len(times_ms)) - 1)],
        "p90_ms": times_ms[min(len(times_ms) - 1, int(0.90 * len(times_ms)))],
    }
    with open(f"{OUT}/{tag}.steps.json", "w") as f:
        json.dump(res, f)

    vecs = [list(map(float, o.outputs.embedding)) for o in outs]
    with open(f"{OUT}/{tag}.embeds.json", "w") as f:
        json.dump(
            {
                "tag": tag,
                "dims": [len(v) for v in vecs],
                "norms": [sum(x * x for x in v) ** 0.5 for v in vecs],
                "vectors": vecs,
            },
            f,
        )

    print(
        f"B3_STEPS {tag} inline={res['inline']} bucket={bucket} n={repeats}  "
        f"median={res['median_ms']:.2f} ms  mean={res['mean_ms']:.2f}  "
        f"min={res['min_ms']:.2f}  p10={res['p10_ms']:.2f}  p90={res['p90_ms']:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
