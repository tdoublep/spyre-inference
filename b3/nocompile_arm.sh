#!/bin/bash
# Every lever against per-call recompilation, at once: b3/nocompile_arm.sh <tag> [repeats]
#
#   granularity=model   the layer_name guard re-specialises ONE block code object
#                       per layer (12x). A whole-model graph traces each layer's
#                       attention inline once, so that multiplication disappears.
#   FASTPATH=1          reuses one slot-mapping buffer per shape; a fresh one per
#                       step was the dominant guard failure
#                       (forward_context.slot_mapping size mismatch).
#   plus the in-tree fixes: mask keyed on shape not kv_lens, plan hoisted out of
#   the traced region, and the two instrumentation reads (Logger, dynamo counters)
#   guarded so they stop forcing guards on host state that changes every call.
set -u
WT="/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion"
export SPYRE_ENCODER_FASTPATH=1
export SPYRE_COMPILE_GRANULARITY=model
TAG="${1:?tag}"
REPEATS="${2:-6}"
export PYTHONPATH="$WT:$WT/b3:${PYTHONPATH:-}"
export LAYOUT_SOLVER=greedy
export B3_TAG="$TAG"
export SPYRE_ATTN_INLINE=1
export B3_BUCKET=2048
export B3_REPEATS="$REPEATS"
export TORCH_LOGS=recompiles
export TORCHINDUCTOR_CACHE_DIR="/tmp/b3-inductor-nocomp-$TAG"
export SPYRE_HOSTPROBE=1
export SPYRE_HOSTPROBE_EVERY=5
export SPYRE_HOSTPROBE_OUT="$WT/b3/out/$TAG.hostprobe.txt"
rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$SPYRE_HOSTPROBE_OUT"
mkdir -p "$WT/b3/out"
echo "=== nocompile arm tag=$TAG inline=1 granularity=model fastpath=1 repeats=$REPEATS"
/home/senuser/spyre-inference/.venv/bin/python "$WT/b3/perf_steps.py" \
    >"$WT/b3/out/$TAG.steps.log" 2>&1
echo "exit=$?"
grep -E "B3_STEPS|settle \(s\)" "$WT/b3/out/$TAG.steps.log" | tail -3
echo "--- recompiles (0 is the win):"
grep -c 'Recompiling function' "$WT/b3/out/$TAG.steps.log"
