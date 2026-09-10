#!/bin/bash
# Correctness of the winning design (chunked_ktile) against an eager per-sequence
# reference, at the chunk/kv_tile actually benchmarked and with a partial last block.
export SENTIENT_BASE_INSTALL_DIR=/home/senuser/spyre-libs/opt/ibm/spyre
export LD_LIBRARY_PATH=/home/senuser/spyre-libs/opt/ibm/spyre/senlib/lib:/home/senuser/spyre-libs/opt/ibm/spyre/deeptools/lib:/home/senuser/spyre-libs/opt/ibm/spyre/runtime/lib:/home/senuser/spyre-libs/opt/ibm/spyre/spyre-comms/lib
export PYTHONPATH=/home/senuser/spyre-inference/.claude/worktrees/batched-decode-design:/home/senuser/spyre-inference/.claude/worktrees/batched-decode-design/experimental:/home/senuser/spyre-libs/opt/ibm/spyre/senlib/lib:/home/senuser/spyre-libs/opt/ibm/spyre/runtime/lib
export DEEPTOOLS_PATH=/home/senuser/spyre-libs/opt/ibm/spyre/deeptools/share
export AIU_WORLD_SIZE=4
export LAYOUT_SOLVER=greedy
export SPYRE_LX_PLANNER_RELAYOUT=1
export CO_OPTIMIZING_LX_PLANNING=0
export OMP_NUM_THREADS=8

P=/home/senuser/spyre-inference/.venv/bin/python
C=/home/senuser/spyre-inference/.claude/worktrees/batched-decode-design/experimental/check_batched_decode_candidates.py
CLEAN='^USDT|^INFO|^\[WARNING\]|^\[unspecified\]|^ *ERRR'

echo "### 8 blocks, chunk=8, kv_tile=64, ctx=896 (last block partial) ###"
$P "$C" --num-blocks 8 --chunk 8 --kv-tile 64 --context-len 896 2>&1 | grep -vE "$CLEAN"

echo "### 8 blocks, chunk=4, kv_tile=64, full context ###"
$P "$C" --num-blocks 8 --chunk 4 --kv-tile 64 2>&1 | grep -vE "$CLEAN"

echo "### CHECK WINNER DONE ###"
