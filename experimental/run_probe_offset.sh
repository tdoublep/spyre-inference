#!/bin/bash
export SENTIENT_BASE_INSTALL_DIR=/home/senuser/spyre-libs/opt/ibm/spyre
export LD_LIBRARY_PATH=/home/senuser/spyre-libs/opt/ibm/spyre/senlib/lib:/home/senuser/spyre-libs/opt/ibm/spyre/deeptools/lib:/home/senuser/spyre-libs/opt/ibm/spyre/runtime/lib:/home/senuser/spyre-libs/opt/ibm/spyre/spyre-comms/lib
export PYTHONPATH=/home/senuser/spyre-inference/.claude/worktrees/batched-decode-design:/home/senuser/spyre-inference/.claude/worktrees/batched-decode-design/experimental:/home/senuser/spyre-libs/opt/ibm/spyre/senlib/lib:/home/senuser/spyre-libs/opt/ibm/spyre/runtime/lib
export DEEPTOOLS_PATH=/home/senuser/spyre-libs/opt/ibm/spyre/deeptools/share
export AIU_WORLD_SIZE=4
export LAYOUT_SOLVER=greedy
export SPYRE_LX_PLANNER_RELAYOUT=1
export CO_OPTIMIZING_LX_PLANNING=0
export OMP_NUM_THREADS=8

/home/senuser/spyre-inference/.venv/bin/python \
  /home/senuser/spyre-inference/.claude/worktrees/batched-decode-design/experimental/probe_offset_view.py 2>&1 \
  | grep -vE "^USDT|^INFO|^\[WARNING\]|^\[unspecified\]|^ *ERRR"
echo "### PROBE DONE ###"
