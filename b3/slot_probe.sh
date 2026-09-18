#!/bin/bash
# Compiled slot kernel vs its CPU reference: b3/slot_probe.sh [tag]
# No vLLM engine, so this is a fast arm. One process at a time all the same.
set -u
WT="/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion"
export PYTHONPATH="$WT:$WT/b3:${PYTHONPATH:-}"
export LAYOUT_SOLVER=greedy
TAG="${1:-slotprobe}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/b3-inductor-$TAG"
rm -rf "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$WT/b3/out"
echo "=== slot kernel probe tag=$TAG"
/home/senuser/spyre-inference/.venv/bin/python \
    \
    "$WT/b3/slot_kernel_probe.py" >"$WT/b3/out/$TAG.log" 2>&1
echo "exit=$?"
grep -E "group=|vs isolated|vs contaminated|-> |SLOT_PROBE|FallbackWarning|Traceback" \
    "$WT/b3/out/$TAG.log" | tail -30
