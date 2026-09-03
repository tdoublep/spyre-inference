#!/usr/bin/env bash
# As ab_lx.sh, but benchmarks whichever checkout SRC points at, for comparing against
# origin/main. Legs are strictly sequential -- one process owns the accelerator.
#
#   SRC=/path/to/main-worktree TAG=main bash .claude-scratch/ab_src.sh
#
# Both cwd and PYTHONPATH are pointed at SRC. Either alone is ambiguous: `python -c`
# puts cwd first on sys.path, while a console script such as `vllm` puts its own bin
# directory there and not cwd, so the two disagree about which checkout wins. Setting
# both removes the question, and the echo below records what actually got imported.
#
# Never edit this file while a run is in flight: bash reads a script incrementally, so
# an edit shifts the offsets under it.
set -euo pipefail

source ~/spyre-libs/env.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:?set SRC to the checkout to benchmark}"
TAG="${TAG:?set TAG for the output file name}"
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export PATH=/home/senuser/spyre-inference/.venv/bin:$PATH

MODEL="${MODEL:-/models/granite-3.3-8b-instruct}"
INPUT_LEN="${INPUT_LEN:-3200}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LEN="${MAX_LEN:-3264}"
ITERS="${ITERS:-10}"
WARMUP="${WARMUP:-2}"
CSIZES="${CSIZES:-[1,512]}"

export SPYRE_NUM_CPUS=8
unset SPYRE_LX_KV_LAYOUT SPYRE_LX_RESTICKIFY_RESIDENCY || true

OUT="$HERE/ab"
mkdir -p "$OUT"

cd "$SRC"

echo "=== leg $TAG === src=$SRC in=$INPUT_LEN out=$OUTPUT_LEN bs=$BATCH_SIZE max=$MAX_LEN"
python -c "import spyre_inference; print('importing', spyre_inference.__file__)"

vllm bench latency \
  --model "$MODEL" \
  --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" --batch-size "$BATCH_SIZE" \
  --num-iters-warmup "$WARMUP" --num-iters "$ITERS" --max-model-len "$MAX_LEN" \
  -cc.compile_sizes="$CSIZES" --no-enable-prefix-caching \
  --output-json "$OUT/latency_${TAG}.json" 2>&1 | tee "$OUT/latency_${TAG}.log"

echo
cat "$OUT/latency_${TAG}.json"
