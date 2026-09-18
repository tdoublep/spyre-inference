"""Embed fixed texts and dump the vectors, so two arms can be compared bit-for-bit.

Correctness gate for the inlined-attention fusion: a launch-count win is worthless
if the embeddings move. Writes b3/out/<B3_TAG>.embeds.json.
"""

import json
import os

import torch  # noqa: F401  (must precede the torch_spyre backend autoload)

from vllm import LLM

MODEL = "ibm-granite/granite-embedding-278m-multilingual"

# All four must round to extent 512 (i.e. exceed 256 tokens) so that
# num_seqs * extent == 2048 == the body bucket and the slot fast path engages --
# which is the regime the GOAL client produces (random-input-len 512,
# range-ratio 0.5). Shorter texts fall through to the var-len packed path, which is
# data-dependent and cannot be traced when attention is inlined. Lengths still
# differ between them, so per-sequence kv_len varies inside the batch.
TEXTS = [
    "The Spyre accelerator executes fused jobplans on sticks of sixty four elements. " * 18,
    "Encoder models pad every request to a slot extent, which wastes rows but keeps one shape. " * 16,
    "Die Auslastung des Beschleunigers haengt von der Anzahl der Kernelstarts pro Schritt ab. " * 14,
    "El rendimiento depende de la fusion de kernels y del numero de lanzamientos por paso. " * 22,
]


def main() -> int:
    tag = os.environ["B3_TAG"]
    llm = LLM(
        model=MODEL,
        runner="pooling",
        max_model_len=512,
        max_num_seqs=4,
        enable_prefix_caching=False,
        # B3_BUCKET=default leaves compile_sizes alone, i.e. the shipped
        # [256, 512, 1024, 2048] -- needed to tell a harness artefact from a real bug.
        **(
            {}
            if os.environ.get("B3_BUCKET") == "default"
            else {
                "compilation_config": {
                    "compile_sizes": [int(os.environ.get("B3_BUCKET", "2048"))]
                }
            }
        ),
    )
    outs = llm.embed(TEXTS)
    vecs = [list(map(float, o.outputs.embedding)) for o in outs]
    out = {
        "tag": tag,
        "dims": [len(v) for v in vecs],
        "norms": [sum(x * x for x in v) ** 0.5 for v in vecs],
        "vectors": vecs,
    }
    path = f"/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion/b3/out/{tag}.embeds.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"B3_EMBEDS_WRITTEN {path}", flush=True)
    for i, v in enumerate(vecs):
        print(f"  text{i}: dim={len(v)} norm={out['norms'][i]:.6f} head={v[:4]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
