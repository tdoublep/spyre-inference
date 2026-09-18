"""Smoke test: token-count the probe texts. No device, no compile, ~2 seconds.

The probe needs every text in (256, 512]: >256 so it rounds to extent 512, and
<=512 so max_model_len admits it. Then num_seqs * extent == 2048 == the body
bucket and the slot fast path engages, which is the regime the fusion needs.
"""

import sys

from transformers import AutoTokenizer

sys.path.insert(0, "/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion/b3")
from embed_check import MODEL, TEXTS  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
ok = True
for i, t in enumerate(TEXTS):
    n = len(tok(t)["input_ids"])
    good = 256 < n <= 512
    ok &= good
    print(f"text{i}: {n} tokens  {'OK' if good else '*** OUT OF RANGE (need 257..512)'}")
print("SMOKE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
