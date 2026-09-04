#!/usr/bin/env bash
# Re-run of the two nb=128 baseline legs. The first attempt failed instantly because the
# nb128 config had not been copied into the main worktree; the folded leg was unaffected.
# Strictly sequential. Do not edit while running.
set -uo pipefail

WT=/home/senuser/spyre-inference/.claude/worktrees/lx-kpage-land
MAIN="$WT/.claude-scratch/mb/wt-main"
CFG=scripts/microbench/configs/granite33_8b_decode_ctx512plus_nb128.json
OUT="$WT/.claude-scratch/mb/results_nb128"
LOGS="$WT/.claude-scratch/mb/logs_nb128"
mkdir -p "$OUT" "$LOGS"

source ~/spyre-libs/env.sh
VENV=/home/senuser/spyre-inference/.venv

run() {
  local tag="$1" layout="$2"
  echo "############ $tag  $(date +%H:%M:%S) ############"
  (
    export PYTHONPATH="$MAIN"
    export SPYRE_ATTN_PROFILING=1
    export SPYRE_NUM_CPUS=8
    "$VENV/bin/python3" "$MAIN/scripts/microbench/spyre_attn_microbench.py" \
        --config "$MAIN/$CFG" \
        --kv-layout "$layout" \
        --span online_softmax \
        --output-dir "$OUT/$tag"
  ) > "$LOGS/$tag.log" 2>&1
  echo "  $tag exit=$?  $(date +%H:%M:%S)"
  grep -E '^\s+-> |CORRECTNESS' "$LOGS/$tag.log" | tail -6
}

run plain_nb128   plain
run devfill_nb128 slot_major_devfill

echo "############ nb128 baseline re-run done $(date +%H:%M:%S) ############"
