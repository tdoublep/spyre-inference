#!/usr/bin/env bash
# Run a command under the pinned Spyre RPMs and the SHARED venv at
# /home/senuser/spyre-inference/.venv, with this worktree's sources winning over the
# editable install (its finder is appended to sys.meta_path, so sys.path goes first).
# Never `uv run` here: uv would create a venv in the worktree and re-resolve torch-spyre.
set -euo pipefail
source ~/spyre-libs/env.sh
VENV=/home/senuser/spyre-inference/.venv
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$VENV/bin:$PATH"
exec "$@"
