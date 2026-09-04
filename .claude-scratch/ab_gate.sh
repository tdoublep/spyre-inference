#!/usr/bin/env bash
# Isolates torch-spyre #4153 end to end: the folded LX KV layout with the restickify
# local-read proof ON vs OFF.
#
# Why this leg never existed: ab_lx.sh exports SPYRE_LX_KV_LAYOUT and
# SPYRE_LX_RESTICKIFY_RESIDENCY together as one "on" leg, so the recorded -7.9% cannot
# say which of the two produced it. The micro-benchmark showed the gate is worth nothing
# on the decode attention READ path, but it never calls do_kv_cache_update and never
# builds a prefill graph, so the store path and prefill were untested.
#
# #4153 is applied to site-packages as a patch (allocator.py.pre4153 is the backup) and
# gated by the env var: with the var unset, _restickify_barrier returns the unconditional
# "read by restickify (cross-frame barrier)" bar, i.e. pre-#4153 behaviour. So the env-var
# A/B is a faithful proxy for having the PR or not.
#
# Both legs run back to back in one process-sequential pass so environment drift cannot
# be mistaken for the effect. Params match ab_lx.sh exactly so the recorded main/off legs
# remain valid anchors.
#
# Never edit this file while a run is in flight: bash reads a script incrementally.
set -uo pipefail

source ~/spyre-libs/env.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(dirname "$HERE")"
export PATH=/home/senuser/spyre-inference/.venv/bin:$PATH

MODEL="${MODEL:-/models/granite-3.3-8b-instruct}"
INPUT_LEN="${INPUT_LEN:-3200}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LEN="${MAX_LEN:-3264}"
ITERS="${ITERS:-10}"
WARMUP="${WARMUP:-2}"
CSIZES="${CSIZES:-[1,512]}"

OUT="$HERE/ab_gate"
mkdir -p "$OUT"

leg() {
  local tag="$1" gate="$2"
  echo "############ leg $tag  gate=$gate  $(date +%H:%M:%S) ############"
  (
    export PYTHONPATH="$WT"
    export SPYRE_NUM_CPUS=8
    export SPYRE_LX_KV_LAYOUT=1
    if [ "$gate" = "1" ]; then
      export SPYRE_LX_RESTICKIFY_RESIDENCY=1
    else
      unset SPYRE_LX_RESTICKIFY_RESIDENCY
    fi
    cd "$WT"
    vllm bench latency \
      --model "$MODEL" \
      --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" --batch-size "$BATCH_SIZE" \
      --num-iters-warmup "$WARMUP" --num-iters "$ITERS" --max-model-len "$MAX_LEN" \
      -cc.compile_sizes="$CSIZES" --no-enable-prefix-caching \
      --output-json "$OUT/latency_${tag}.json"
  ) > "$OUT/${tag}.log" 2>&1
  echo "  $tag exit=$?  $(date +%H:%M:%S)"
  python3 - "$OUT/latency_${tag}.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print("  avg_latency=%.4f  p50=%.4f" % (d["avg_latency"], d["percentiles"]["50"]))
except Exception as e:
    print("  no json:", e)
PY
}

# Gate ON first so a thermal/cache drift trend, if any, works against the claim that the
# gate matters rather than for it.
leg gate_on  1
leg gate_off 0

echo "############ gate A/B done $(date +%H:%M:%S) ############"
