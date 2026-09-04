#!/usr/bin/env bash
# Mechanism behind the #4153 gate A/B: count what the local-read proof actually un-bars.
#
# _restickify_barrier (allocator.py) returns one of three things per buffer read by a
# restickify:
#   gate off -> "read by restickify (cross-frame barrier)"     (pre-#4153 behaviour)
#   gate on  -> None                    + debug "local-read proof PASSED"
#   gate on  -> "... (local-read proof failed)" + a debug line naming why
# Those debug lines only emit at DEBUG on logger spyre.inductor.scratchpad.allocator.
#
# If the PASSED count is zero, #4153 cannot matter for this model at all -- a static
# result stronger than any timing, and it covers prefill and the KV store as well as
# decode, because this is a full model compile.
#
# Not timed: VLLM_LOGGING_LEVEL=DEBUG perturbs nothing in the decode window (the
# allocator runs at inductor compile time) but the log volume is large, so iters are
# cut to the minimum that still compiles both graphs.
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
  echo "############ count leg $tag  gate=$gate  $(date +%H:%M:%S) ############"
  (
    export PYTHONPATH="$WT"
    export SPYRE_NUM_CPUS=8
    export SPYRE_LX_KV_LAYOUT=1
    export VLLM_LOGGING_LEVEL=DEBUG
    if [ "$gate" = "1" ]; then
      export SPYRE_LX_RESTICKIFY_RESIDENCY=1
    else
      unset SPYRE_LX_RESTICKIFY_RESIDENCY
    fi
    cd "$WT"
    vllm bench latency \
      --model "$MODEL" \
      --input-len 3200 --output-len 4 --batch-size 1 \
      --num-iters-warmup 1 --num-iters 1 --max-model-len 3264 \
      -cc.compile_sizes="[1,512]" --no-enable-prefix-caching \
      --output-json "$OUT/count_${tag}.json"
  ) > "$OUT/count_${tag}.log" 2>&1
  echo "  $tag exit=$?  $(date +%H:%M:%S)"
  echo "  local-read proof PASSED   : $(grep -c 'local-read proof PASSED'      "$OUT/count_${tag}.log")"
  echo "  proof failed (debug lines): $(grep -c 'restickify proof '            "$OUT/count_${tag}.log")"
  echo "  cross-frame barrier hits  : $(grep -c 'cross-frame barrier'          "$OUT/count_${tag}.log")"
  echo "  spill reasons w/ restickify: $(grep -c 'read by restickify'          "$OUT/count_${tag}.log")"
}

leg count_on  1
leg count_off 0

echo "############ gate count done $(date +%H:%M:%S) ############"
