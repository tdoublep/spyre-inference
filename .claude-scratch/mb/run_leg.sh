#!/usr/bin/env bash
# One micro-benchmark leg, one process. Usage:
#   run_leg.sh <leg> <checkout> <config> <span> <outdir>
#
# <leg> selects the environment:
#   main      | off        -> plain cache, flag unset
#   on        -> SPYRE_LX_KV_LAYOUT=1 SPYRE_LX_RESTICKIFY_RESIDENCY=1, folded cache
#   on-nogate -> SPYRE_LX_KV_LAYOUT=1, restickify gate off (attributes #4153)
#   on-32core -> as `on` plus SPYRE_ATTN_MAX_CORES=32 (prices the 8-core cap)
#
# SPYRE_BUCKETED_DECODE is left unset in every leg: the per-sequence kernel is
# what is under test, and the bucketed path would short-circuit it.
set -euo pipefail

LEG="$1"
CHECKOUT="$2"
CONFIG="$3"
SPAN="$4"
OUTDIR="$5"

source ~/spyre-libs/env.sh

VENV=/home/senuser/spyre-inference/.venv
export PYTHONPATH="$CHECKOUT"
export SPYRE_ATTN_PROFILING=1
export SPYRE_NUM_CPUS=8

KV_LAYOUT=plain
case "$LEG" in
  main|off) ;;
  on)
    export SPYRE_LX_KV_LAYOUT=1 SPYRE_LX_RESTICKIFY_RESIDENCY=1
    KV_LAYOUT=folded
    ;;
  on-nogate)
    export SPYRE_LX_KV_LAYOUT=1
    KV_LAYOUT=folded
    ;;
  on-32core)
    export SPYRE_LX_KV_LAYOUT=1 SPYRE_LX_RESTICKIFY_RESIDENCY=1 SPYRE_ATTN_MAX_CORES=32
    KV_LAYOUT=folded
    ;;
  *) echo "unknown leg: $LEG" >&2; exit 2 ;;
esac

echo "=== leg=$LEG checkout=$CHECKOUT span=$SPAN kv_layout=$KV_LAYOUT ==="
env | grep -E '^SPYRE_' | sort
echo

exec "$VENV/bin/python3" "$CHECKOUT/scripts/microbench/spyre_attn_microbench.py" \
    --config "$CONFIG" \
    --kv-layout "$KV_LAYOUT" \
    --span "$SPAN" \
    --output-dir "$OUTDIR"
