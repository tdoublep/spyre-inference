#!/bin/bash
# Throughput arm: b3/perf_arm.sh <tag> <0|1 inline>
# GOAL protocol, with --profile deliberately absent: a process that has ever
# profiled is ~2.8x slower for the rest of its life (GOAL measurement trap).
# One server at a time -- the accelerator takes a single process.
set -u
WT="/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion"
TAG="${1:?tag}"
INLINE="${2:?inline 0|1}"
OUT="$WT/b3/out"
mkdir -p "$OUT"

export PYTHONPATH="$WT:${PYTHONPATH:-}"
export LAYOUT_SOLVER=greedy
export SPYRE_COMPILE_GRANULARITY=block
# NOTE: TORCH_SENDNN_CACHE_* is INERT in this stack -- torch_sendnn is not
# installed and torch_spyre only ever sets TORCH_SENDNN_LOG. Do not set it and do
# not attribute anything to it.
export SPYRE_ATTN_INLINE="$INLINE"
export TORCHINDUCTOR_CACHE_DIR="/tmp/b3-inductor-perf-$TAG"

MODEL=ibm-granite/granite-embedding-278m-multilingual

echo "=== perf arm tag=$TAG SPYRE_ATTN_INLINE=$INLINE"
/home/senuser/spyre-inference/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --no-enable-prefix-caching \
    --max-model-len 512 \
    --max-num-seqs 4 \
    --compilation-config '{"compile_sizes":[2048]}' \
    >"$OUT/$TAG.server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

# Warmup is minutes; poll readiness rather than sleeping a guess.
for _ in $(seq 1 120); do
    if curl -s -o /dev/null http://localhost:8000/health 2>/dev/null; then break; fi
    if ! kill -0 $SRV 2>/dev/null; then echo "SERVER DIED"; tail -25 "$OUT/$TAG.server.log"; exit 1; fi
    sleep 10
done
if ! curl -s -o /dev/null http://localhost:8000/health 2>/dev/null; then
    echo "SERVER NOT READY"; tail -25 "$OUT/$TAG.server.log"; exit 1
fi
echo "server ready"

for run in 1 2; do
    echo "--- client run $run"
    # `vllm bench serve` CLI. --num-warmups IS supported (it is absent from --help
    # but present in the parsed namespace) and every archive number was taken with
    # it, so it stays for comparability. --profile is deliberately never passed.
    /home/senuser/spyre-inference/.venv/bin/vllm bench serve \
        --model "$MODEL" \
        --base-url http://localhost:8000 \
        --backend openai-embeddings \
        --endpoint /v1/embeddings \
        --dataset-name random \
        --num-prompts 400 \
        --random-input-len 512 \
        --random-range-ratio 0.5 \
        --num-warmups 100 \
        >"$OUT/$TAG.client$run.log" 2>&1
    grep -E "Request throughput|Successful requests|Benchmark duration" "$OUT/$TAG.client$run.log" || tail -5 "$OUT/$TAG.client$run.log"
done
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "=== done $TAG"
