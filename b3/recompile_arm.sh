#!/bin/bash
# Does the fused body graph still recompile per call? b3/recompile_arm.sh <tag> <0|1 inline>
# Single bucket: the comparison only needs the one shape the batch lands on, and
# four default buckets cost ~4x the compile for nothing (learned the slow way).
# TORCH_LOGS=recompiles names the failing guard if it still recompiles, so one arm
# both tests the fix and diagnoses it.
set -u
WT="/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion"
export PYTHONPATH="$WT:$WT/b3:${PYTHONPATH:-}"
export LAYOUT_SOLVER=greedy
export SPYRE_COMPILE_GRANULARITY=block
export B3_TAG="${1:?tag}"
export SPYRE_ATTN_INLINE="${2:?inline 0|1}"
export B3_BUCKET=2048
export B3_REPEATS="${3:-8}"
export TORCH_LOGS=recompiles
export TORCHINDUCTOR_CACHE_DIR="/tmp/b3-inductor-recomp-$B3_TAG"
export SPYRE_HOSTPROBE=1
export SPYRE_HOSTPROBE_EVERY=5
export SPYRE_HOSTPROBE_OUT="$WT/b3/out/$B3_TAG.hostprobe.txt"
rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$SPYRE_HOSTPROBE_OUT"
mkdir -p "$WT/b3/out"
echo "=== recompile arm tag=$B3_TAG inline=$SPYRE_ATTN_INLINE bucket=$B3_BUCKET repeats=$B3_REPEATS"
/home/senuser/spyre-inference/.venv/bin/python "$WT/b3/perf_steps.py" \
    >"$WT/b3/out/$B3_TAG.steps.log" 2>&1
echo "exit=$?"
grep -E "B3_STEPS|settle \(s\)" "$WT/b3/out/$B3_TAG.steps.log" | tail -3
echo "--- recompile reasons (blank is the win):"
grep -E "Recompiling|recompile_reason|triggered by|guard" "$WT/b3/out/$B3_TAG.steps.log" | head -12
