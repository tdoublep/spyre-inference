#!/bin/bash
WT="/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion"
export PYTHONPATH="$WT:${PYTHONPATH:-}"
exec /home/senuser/spyre-inference/.venv/bin/python "$WT/b3/smoke_texts.py"
