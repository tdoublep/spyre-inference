#!/usr/bin/env bash
# num_blocks control at nb=128, to be read against the nb=64 rows.
#
# Question: does per-page cost depend on the size of the WHOLE cache, or only on the page?
# This decides whether nb=64 numbers extrapolate towards a production cache, and it also
# explains the 10x gap between the tiled and slot-major baselines.
#
#   plain   (default tiled)  -- page index spread across two device dims. If the gather
#                              costs the whole tensor, doubling num_blocks roughly doubles
#                              per-page cost. This is the mechanism hypothesis.
#   devfill (slot-major)     -- production baseline. A page is a contiguous slot slice,
#                              so per-page cost should be flat in num_blocks.
#   folded  (head-major)     -- indexed axis at device position 0; also expected flat.
#
# Only flat legs can be extrapolated to production's much larger cache.
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
  local tag="$1" checkout="$2" layout="$3" lx="$4"
  echo "############ $tag  $(date +%H:%M:%S) ############"
  (
    export PYTHONPATH="$checkout"
    export SPYRE_ATTN_PROFILING=1
    export SPYRE_NUM_CPUS=8
    if [ "$lx" = "1" ]; then
      export SPYRE_LX_KV_LAYOUT=1 SPYRE_LX_RESTICKIFY_RESIDENCY=1
    fi
    "$VENV/bin/python3" "$checkout/scripts/microbench/spyre_attn_microbench.py" \
        --config "$checkout/$CFG" \
        --kv-layout "$layout" \
        --span online_softmax \
        --output-dir "$OUT/$tag"
  ) > "$LOGS/$tag.log" 2>&1
  echo "  $tag exit=$?  $(date +%H:%M:%S)"
  grep -E '^\s+-> |CORRECTNESS' "$LOGS/$tag.log" | tail -6
}

run plain_nb128   "$MAIN" plain              0
run devfill_nb128 "$MAIN" slot_major_devfill 0
run folded_nb128  "$WT"   folded             1

echo "############ nb128 control done $(date +%H:%M:%S) ############"
