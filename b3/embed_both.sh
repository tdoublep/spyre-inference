#!/bin/bash
# Both correctness arms, strictly sequential: the Spyre device takes one process.
WT="/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion"
# Gate: token-count the probe texts first (~2 s, no device). Getting this wrong
# costs two 4-minute compiles, which is exactly how it was got wrong.
if ! /home/senuser/spyre-inference/.venv/bin/python "$WT/b3/smoke_texts.py" 2>&1 | tail -6; then
    echo "SMOKE FAILED - not spending device time"
    exit 1
fi
bash "$WT/b3/embed_arm.sh" ref2-noinline 0
bash "$WT/b3/embed_arm.sh" test2-inline 1
echo "=== COMPARE ==="
/home/senuser/spyre-inference/.venv/bin/python "$WT/b3/embed_diff.py" \
    "$WT/b3/out/ref2-noinline.embeds.json" "$WT/b3/out/test2-inline.embeds.json"
