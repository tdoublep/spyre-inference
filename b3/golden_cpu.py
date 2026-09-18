"""Golden embeddings from HuggingFace on CPU. ~15 s, no device, no compile.

This is the reference the Spyre arms should be compared against. Comparing two
Spyre arms to each other costs two multi-minute warmups and cannot tell you which
one is wrong -- which is exactly what happened: the `SPYRE_ATTN_INLINE=0` arm
returned four near-identical vectors (all pairwise cosines > 0.9998) and the
comparison "failed" against a degenerate baseline.

Pooling matches the served config: CLS token, then L2 normalise
(PoolerConfig(seq_pooling_type='CLS', use_activation=True)).
"""

import json
import sys

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/b3")
from embed_check import MODEL, TEXTS  # noqa: E402

OUT = "/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/b3/out/golden-cpu.embeds.json"


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32).eval()

    vecs = []
    with torch.no_grad():
        for t in TEXTS:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=512)
            hidden = model(**enc).last_hidden_state  # [1, T, H]
            cls = hidden[:, 0]  # CLS pooling
            cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
            vecs.append([float(x) for x in cls[0]])

    json.dump(
        {
            "tag": "golden-cpu",
            "dims": [len(v) for v in vecs],
            "norms": [sum(x * x for x in v) ** 0.5 for v in vecs],
            "vectors": vecs,
        },
        open(OUT, "w"),
    )
    print(f"B3_EMBEDS_WRITTEN {OUT}")
    for i, v in enumerate(vecs):
        print(f"  text{i}: dim={len(v)} head={v[:4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
