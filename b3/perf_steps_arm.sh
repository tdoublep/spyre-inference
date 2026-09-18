#!/bin/bash
# Fixed-shape per-step arm: b3/perf_steps_arm.sh <tag> <0|1 inline> [bucket] [repeats]
# One process at a time -- the accelerator takes a single process.
set -u
WT="/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion"
export PYTHONPATH="$WT:$WT/b3:${PYTHONPATH:-}"
export LAYOUT_SOLVER=greedy
export SPYRE_COMPILE_GRANULARITY=block
export B3_TAG="${1:?tag}"
export SPYRE_ATTN_INLINE="${2:?inline 0|1}"
export B3_BUCKET="${3:-2048}"
export B3_REPEATS="${4:-40}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/b3-inductor-steps-$B3_TAG"
export SPYRE_HOSTPROBE=1
export SPYRE_HOSTPROBE_EVERY=10
export SPYRE_HOSTPROBE_OUT="$WT/b3/out/$B3_TAG.hostprobe.txt"
rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$SPYRE_HOSTPROBE_OUT"
mkdir -p "$WT/b3/out"
echo "=== steps arm tag=$B3_TAG inline=$SPYRE_ATTN_INLINE bucket=$B3_BUCKET repeats=$B3_REPEATS"
/home/senuser/spyre-inference/.venv/bin/python "$WT/b3/perf_steps.py" \
    >"$WT/b3/out/$B3_TAG.steps.log" 2>&1
echo "exit=$?"
grep -E "B3_STEPS|settle|compiled outside warmup" "$WT/b3/out/$B3_TAG.steps.log" | tail -6
