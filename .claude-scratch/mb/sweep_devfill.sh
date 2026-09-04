#!/usr/bin/env bash
# THE baseline-validity control.
#
# The main sweep's baseline legs use --kv-layout plain, i.e. a bare cache.to(device) with
# the default tiled device layout. spyre_model_runner.py:1039 shows production does NOT
# do that: the non-folded KV cache is allocated with slot_major_kv_layout, and only the
# folded one with head_major_kv_layout. Under the tiled layout the page index is spread
# across two device dims, which is the condition that makes index_select cost the whole
# tensor -- so `plain` may be an artificially slow baseline rather than a real one.
#
# slot_major_devfill is the layout the argparse help calls "matches the worker": zeroed
# slot-major alloc with history written on device. If it lands near `plain` (~700us/page)
# then `plain` was a fair proxy and the folded speedup stands. If it lands far below,
# most of the measured speedup is an artifact of the baseline's layout, not of LX
# residency, and the headline number has to be restated against this leg instead.
#
# Strictly sequential. Do not edit while running.
set -uo pipefail

WT=/home/senuser/spyre-inference/.claude/worktrees/lx-kpage-land
MAIN="$WT/.claude-scratch/mb/wt-main"
CFG=scripts/microbench/configs/granite33_8b_decode_ctx512plus.json
OUT="$WT/.claude-scratch/mb/results_devfill"
LOGS="$WT/.claude-scratch/mb/logs_devfill"
mkdir -p "$OUT" "$LOGS"

source ~/spyre-libs/env.sh
VENV=/home/senuser/spyre-inference/.venv

run() {
  local leg="$1" checkout="$2"
  local tag="${leg}_devfill"
  echo "############ $tag  $(date +%H:%M:%S) ############"
  (
    export PYTHONPATH="$checkout"
    export SPYRE_ATTN_PROFILING=1
    export SPYRE_NUM_CPUS=8
    # No SPYRE_LX_KV_LAYOUT: this is a baseline leg on the production slot-major frame.
    "$VENV/bin/python3" "$checkout/scripts/microbench/spyre_attn_microbench.py" \
        --config "$checkout/$CFG" \
        --kv-layout slot_major_devfill \
        --span online_softmax \
        --output-dir "$OUT/$tag"
  ) > "$LOGS/$tag.log" 2>&1
  echo "  $tag exit=$?  $(date +%H:%M:%S)"
  grep -E '^\s+-> |CORRECTNESS' "$LOGS/$tag.log" | tail -8
}

run main "$MAIN"
run off  "$WT"

echo "############ devfill control done $(date +%H:%M:%S) ############"
