#!/bin/bash
# Arm: attention not registered as an opaque custom op.
export SPYRE_ATTN_INLINE=1
exec bash /home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion/b3/count.sh "$@"
