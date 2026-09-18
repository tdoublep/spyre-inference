#!/bin/bash
# Run the shared venv's python with THIS worktree shadowing the editable install.
# The editable install uses sys.meta_path.append(), so a sys.path entry wins over it.
WT="/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion"
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
exec /home/senuser/spyre-inference/.venv/bin/python "$@"
