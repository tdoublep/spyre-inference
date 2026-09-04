#!/usr/bin/env bash
# Every leg, strictly sequential: one process owns the accelerator at a time.
# Do not edit while running - bash reads the file incrementally.
#
# on-32core is a valid configuration, contrary to what attn_max_cores' docstring reads
# like: SPYRE_ATTN_MAX_CORES=32 at query_len 1 lowers and computes correctly, it is just
# far slower than the 8-core cap (772us vs 125us at 4 pages). It is swept to price the
# cap across context rather than to test whether it runs.
set -uo pipefail

WT=/home/senuser/spyre-inference/.claude/worktrees/lx-kpage-land
MAIN="$WT/.claude-scratch/mb/wt-main"
CFG=scripts/microbench/configs/granite33_8b_decode_ctx512plus.json
OUT="$WT/.claude-scratch/mb/results2"
LOGS="$WT/.claude-scratch/mb/logs2"
mkdir -p "$OUT" "$LOGS"

run() {
  local leg="$1" checkout="$2" span="$3"
  local tag="${leg}_${span}"
  echo "############ $tag  $(date +%H:%M:%S) ############"
  bash "$WT/.claude-scratch/mb/run_leg.sh" \
      "$leg" "$checkout" "$checkout/$CFG" "$span" "$OUT/$tag" \
      > "$LOGS/$tag.log" 2>&1
  local rc=$?
  echo "  $tag exit=$rc  $(date +%H:%M:%S)"
  grep -E '^\s+-> ' "$LOGS/$tag.log" | tail -8
}

for span in online_softmax forward; do
  run main      "$MAIN" "$span"
  run off       "$WT"   "$span"
  run on        "$WT"   "$span"
  run on-nogate "$WT"   "$span"
  run on-32core "$WT"   "$span"
done

echo "############ sweep done $(date +%H:%M:%S) ############"
