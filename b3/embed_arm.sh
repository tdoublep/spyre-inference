#!/bin/bash
# Correctness arm: b3/embed_arm.sh <tag> <0|1 inline>
set -u
WT="/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites"
export PYTHONPATH="$WT:${PYTHONPATH:-}"
export LAYOUT_SOLVER=greedy
export SPYRE_COMPILE_GRANULARITY=block
export TORCH_SENDNN_CACHE_ENABLE=0
export B3_TAG="${1:?tag}"
export SPYRE_ATTN_INLINE="${2:?inline 0|1}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/b3-inductor-embed-$B3_TAG"
rm -rf "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$WT/b3/out"
echo "=== embed arm tag=$B3_TAG SPYRE_ATTN_INLINE=$SPYRE_ATTN_INLINE"
/home/senuser/spyre-inference/.venv/bin/python "$WT/b3/embed_check.py" >"$WT/b3/out/$B3_TAG.embed.log" 2>&1
echo "exit=$?"
grep -E "B3_EMBEDS_WRITTEN|text[0-9]:" "$WT/b3/out/$B3_TAG.embed.log" | tail -6
