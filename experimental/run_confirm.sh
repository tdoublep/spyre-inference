#!/bin/bash
# Confirm the winner: correctness first, then a same-process A/B against the bar,
# then sweep the two knobs around the operating point.
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
D=/home/senuser/spyre-inference/.claude/worktrees/batched-decode-design/experimental
FILT='^  ran|^  FAILED|InductorError|NotImplementedError|Unsupported'

echo "########## CORRECTNESS (random K/V, partial last block) ##########"
$P "$D/check_batched_decode_candidates.py" --num-blocks 4 --chunk 2 --kv-tile 64 \
   --context-len 384 2>&1 | grep -vE "^USDT|^INFO|^\[WARNING\]|^\[unspecified\]|^ *ERRR"

echo "########## A/B same process: bar vs winner ##########"
$P "$D/microbench_batched_decode_v2.py" --iters 30 --chunk 8 --kv-tile 64 \
   --variants unrolled,chunked_ktile 2>&1 | grep -E "$FILT"

echo "########## kv_tile sweep at chunk=8 ##########"
for T in 64 128; do
  echo "--- kv_tile=$T ---"
  $P "$D/microbench_batched_decode_v2.py" --iters 30 --chunk 8 --kv-tile $T \
     --variants chunked_ktile 2>&1 | grep -E "$FILT"
done

echo "########## chunk sweep at kv_tile=64 ##########"
for C in 2 4 16; do
  echo "--- chunk=$C ---"
  $P "$D/microbench_batched_decode_v2.py" --iters 30 --chunk $C --kv-tile 64 \
     --variants chunked_ktile 2>&1 | grep -E "$FILT"
done

echo "########## CONFIRM DONE ##########"
