#!/usr/bin/env bash
# Reversed-order repeat of the #4153 gate A/B.
#
# ab_gate.sh ran gate_on first, on the reasoning that drift would then work against the
# gate mattering. That backfired as a control: gate_off came out 0.98% faster, and "the
# second leg is faster" is exactly what drift produces too, so the two explanations are
# not separable from one ordering. Running gate_off first inverts the confound -- if the
# gap follows the gate it is real, if it follows the position it was drift.
#
# Same params as ab_gate.sh so all four legs pool.
#
# Never edit this file while a run is in flight.
set -uo pipefail

source ~/spyre-libs/env.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(dirname "$HERE")"
export PATH=/home/senuser/spyre-inference/.venv/bin:$PATH

MODEL="${MODEL:-/models/granite-3.3-8b-instruct}"
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
      --input-len 3200 --output-len 64 --batch-size 1 \
      --num-iters-warmup 2 --num-iters 10 --max-model-len 3264 \
      -cc.compile_sizes="[1,512]" --no-enable-prefix-caching \
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

leg gate_off_first 0
leg gate_on_second 1

echo "############ reversed gate A/B done $(date +%H:%M:%S) ############"
