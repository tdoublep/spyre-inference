#!/bin/bash
# Solo-vs-grouped arm: b3/solo_arm.sh <tag> <0|1 inline>
set -u
WT="/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion"
export PYTHONPATH="$WT:$WT/b3:${PYTHONPATH:-}"
export LAYOUT_SOLVER=greedy
export SPYRE_COMPILE_GRANULARITY=block
export B3_TAG="${1:?tag}"
export SPYRE_ATTN_INLINE="${2:?inline 0|1}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/b3-inductor-solo-$B3_TAG"
rm -rf "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$WT/b3/out"
echo "=== solo arm tag=$B3_TAG inline=$SPYRE_ATTN_INLINE"
/home/senuser/spyre-inference/.venv/bin/python "$WT/b3/embed_solo.py" \
    >"$WT/b3/out/$B3_TAG.solo.log" 2>&1
echo "exit=$?"
grep -E "B3_EMBEDS_WRITTEN|text[0-9]:|slot padding active" "$WT/b3/out/$B3_TAG.solo.log" | tail -20
