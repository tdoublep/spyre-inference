#!/bin/bash
# Control: the SHIPPED default granularity (per-block), for a same-pod count comparison
# against granularity=model. H was stopped before running this control.
export SPYRE_COMPILE_GRANULARITY=block
exec bash /home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/b3/count.sh "$@"
