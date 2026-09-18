#!/bin/bash
# Arm: attention not registered as an opaque custom op.
export SPYRE_ATTN_INLINE=1
exec bash /home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/b3/count.sh "$@"
