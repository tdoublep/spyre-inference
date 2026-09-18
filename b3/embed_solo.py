"""Solo vs grouped, same process, same compiled artifacts, one arm.

Tests cross-slot contamination directly. `_encoder_slot_kernel` isolates slots by
the SDPA *batch* dim -- the mask only masks intra-slot padding -- so if the batch
dim is not honoured, every query attends every slot's keys. CLS queries are nearly
identical across texts, so contamination collapses their outputs toward one
vector, which is the observed degeneracy.

Prediction if that is the mechanism: SOLO (one request per step, group == 1, no
other slot to contaminate) is correct, GROUPED (all four in one step) is
degenerate. Both are measured here from one engine so the compiled artifacts,
weights and dtype are identical between them.
"""

import json
import os

import torch  # noqa: F401  (must precede the torch_spyre backend autoload)

from vllm import LLM

from embed_check import MODEL, TEXTS

OUT = "/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion/b3/out"


def dump(tag, vecs):
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
    print(f"B3_EMBEDS_WRITTEN {OUT}/{tag}.embeds.json", flush=True)
    for i, v in enumerate(vecs):
        print(f"  {tag} text{i}: head={[round(x, 6) for x in v[:4]]}", flush=True)


def main() -> int:
    tag = os.environ["B3_TAG"]
    llm = LLM(
        model=MODEL,
        runner="pooling",
        max_model_len=512,
        max_num_seqs=4,
        enable_prefix_caching=False,
    )

    # One request per call -> one slot per step -> group == 1.
    solo = [list(map(float, llm.embed([t])[0].outputs.embedding)) for t in TEXTS]
    dump(f"{tag}-solo", solo)

    # All four in one call -> the scheduler groups them -> group > 1.
    grouped = [list(map(float, o.outputs.embedding)) for o in llm.embed(TEXTS)]
    dump(f"{tag}-grouped", grouped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
