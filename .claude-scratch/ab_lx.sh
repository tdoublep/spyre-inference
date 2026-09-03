#!/usr/bin/env bash
# One leg of the LX-layout A/B. Legs are strictly sequential -- one process owns the
# accelerator, and parallel runs hang or corrupt the compile cache.
#
#   bash .claude-scratch/ab_lx.sh off   # this branch, SPYRE_LX_KV_LAYOUT unset
#   bash .claude-scratch/ab_lx.sh on    # this branch, folded cache + LX residency
#
# Params follow the vllm-bench-latency skill's rules for keeping graph recording out
# of the timed window: input-len a multiple of the block size, and input+output inside
# one further block, so the per-block mask tile count is constant across every decode
# step. The effective block size is 128 (CpuPlatform raises the platform's 64), and
# 3200 is a multiple of both, so 25 pages are gathered per decode step -- a genuinely
# long context, well past the 4 pages the hand-off asks for as a minimum.
set -euo pipefail

LEG="${1:?usage: ab_lx.sh on|off}"
source ~/spyre-libs/env.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(dirname "$HERE")"
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
export PATH=/home/senuser/spyre-inference/.venv/bin:$PATH

MODEL="${MODEL:-/models/granite-3.3-8b-instruct}"
INPUT_LEN="${INPUT_LEN:-3200}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LEN="${MAX_LEN:-3264}"
ITERS="${ITERS:-10}"
WARMUP="${WARMUP:-2}"
CSIZES="${CSIZES:-[1,512]}"

# Required on every leg: unpinned threading makes the numbers wander so legs stop
# being comparable.
export SPYRE_NUM_CPUS=8

if [[ "$LEG" == "on" ]]; then
  export SPYRE_LX_KV_LAYOUT=1
  # Gates the torch-spyre #4153 local-read proof that lets K's permute stay in LX.
  export SPYRE_LX_RESTICKIFY_RESIDENCY=1
else
  unset SPYRE_LX_KV_LAYOUT SPYRE_LX_RESTICKIFY_RESIDENCY || true
fi

OUT="$HERE/ab"
mkdir -p "$OUT"

echo "=== leg $LEG === lx=${SPYRE_LX_KV_LAYOUT:-unset} in=$INPUT_LEN out=$OUTPUT_LEN bs=$BATCH_SIZE max=$MAX_LEN"
vllm bench latency \
  --model "$MODEL" \
  --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" --batch-size "$BATCH_SIZE" \
  --num-iters-warmup "$WARMUP" --num-iters "$ITERS" --max-model-len "$MAX_LEN" \
  -cc.compile_sizes="$CSIZES" --no-enable-prefix-caching \
  --output-json "$OUT/latency_${LEG}.json" 2>&1 | tee "$OUT/latency_${LEG}.log"

echo
cat "$OUT/latency_${LEG}.json"
