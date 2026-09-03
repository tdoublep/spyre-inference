#!/usr/bin/env bash
# Drive repro_lx_residency.py with the pinned Spyre RPMs and a fresh artifact dir.
# A bare /opt/ibm/spyre gives an undefined spyre_comms symbol, hence env.sh.
#
#   bash .claude-scratch/run_lx_residency.sh                  # defaults
#   NUM_BLOCKS=1 bash .claude-scratch/run_lx_residency.sh     # single page
#   WRITE_KV=0 bash .claude-scratch/run_lx_residency.sh       # read side only
set -euo pipefail

source ~/spyre-libs/env.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
# The venv python directly, not `uv run`: uv would re-resolve the pinned
# torch-spyre git rev and clobber a locally installed build.
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=/home/senuser/spyre-inference/.venv/bin/python

OUT_DIR="${OUT_DIR:-$HERE/lx_residency_$(date +%H%M%S)}"
mkdir -p "$OUT_DIR"

OUT_DIR="$OUT_DIR" "$PYTHON" "$HERE/repro_lx_residency.py" 2>&1 | tee "$OUT_DIR/run.log"

echo
echo "log: $OUT_DIR/run.log"
