#!/bin/bash
# GOAL.md protocol end-to-end: b3/goal_arm.sh <tag> <0|1 inline>
#
# Server and client commands are GOAL.md's, with ONE deviation, which GOAL.md
# itself requires: --profile is never passed. Its measurement-trap section shows a
# process that has ever profiled is ~2.8x slower for the rest of its life, and the
# client command as written contains --profile -- which is also why "run the client
# twice" makes run 2 the poisoned one there. Throughput needs a clean process.
#
# GOAL.md also says to clear the inductor cache before starting, so results stay
# consistent; TORCHINDUCTOR_CACHE_DIR is fresh per arm.
#
# inline=1 additionally sets SPYRE_ENCODER_FASTPATH=1: without it a fresh
# slot-mapping tensor per step makes the traced graph recompile every call.
#
# GOAL.md's --random-range-ratio 0.5 is deliberately NOT passed: varying lengths
# produce many (slots, extent) layouts, and each one warmup did not cover costs a
# full fused-body compile (~380 s), which would dominate the run. Fixed 512-token
# prompts give one layout and one compile. The trade is that req/s here is not
# directly comparable to the archive's numbers, which used the varying lengths.
set -u
WT="/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion"
TAG="${1:?tag}"
INLINE="${2:?inline 0|1}"
OUT="$WT/b3/out"
mkdir -p "$OUT"

export PYTHONPATH="$WT:${PYTHONPATH:-}"
export LAYOUT_SOLVER=greedy
export SPYRE_COMPILE_GRANULARITY=block
export SPYRE_ATTN_INLINE="$INLINE"
if [ "$INLINE" = "1" ]; then
    export SPYRE_ENCODER_FASTPATH=1
fi
export TORCHINDUCTOR_CACHE_DIR="/tmp/b3-goal-$TAG"
rm -rf "$TORCHINDUCTOR_CACHE_DIR"

MODEL=ibm-granite/granite-embedding-278m-multilingual
VENV=/home/senuser/spyre-inference/.venv/bin

echo "=== GOAL arm tag=$TAG inline=$INLINE fastpath=${SPYRE_ENCODER_FASTPATH:-0}"
date
# No --compilation-config: GOAL.md does not set compile_sizes, so the shipped
# default buckets apply and the number stays comparable to the archive's.
"$VENV/vllm" serve "$MODEL" \
    --no-enable-prefix-caching \
    --max-model-len 512 \
    --max-num-seqs 4 \
    >"$OUT/$TAG.server.log" 2>&1 &
SRV=$!
trap 'kill -TERM $SRV 2>/dev/null' EXIT

# The fused path compiles once per uncovered layout at ~380 s, so readiness can be
# far out. Poll rather than guess, and give up only after a genuinely long wait.
for _ in $(seq 1 240); do
    curl -s -o /dev/null http://localhost:8000/health 2>/dev/null && break
    if ! kill -0 $SRV 2>/dev/null; then
        echo "SERVER DIED"; tail -30 "$OUT/$TAG.server.log"; exit 1
    fi
    sleep 15
done
if ! curl -s -o /dev/null http://localhost:8000/health 2>/dev/null; then
    echo "SERVER NOT READY"; tail -30 "$OUT/$TAG.server.log"; exit 1
fi
echo "server ready"; date

for run in 1 2; do
    echo "--- client run $run"; date
    "$VENV/vllm" bench serve \
        --model "$MODEL" \
        --base-url http://localhost:8000 \
        --backend openai-embeddings \
        --endpoint /v1/embeddings \
        --dataset-name random \
        --num-prompts 400 \
        --random-input-len 512 \
        --num-warmups 100 \
        >"$OUT/$TAG.client$run.log" 2>&1
    grep -E "Request throughput|Successful requests|Benchmark duration|Mean E2EL" \
        "$OUT/$TAG.client$run.log" || tail -5 "$OUT/$TAG.client$run.log"
done

# Any compile in the timed phase invalidates the number, so surface it either way.
echo "--- late compiles on the server (should be none in the timed phase):"
grep -c 'compiled outside warmup' "$OUT/$TAG.server.log" || true
kill -TERM $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "=== done $TAG"; date
