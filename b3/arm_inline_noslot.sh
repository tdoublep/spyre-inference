#!/bin/bash
# Diagnostic: inlined attention WITHOUT slot padding. Isolates `out.copy_` in
# _encoder_slot_kernel (a mutation of a traced graph input) as the cause of the
# double free. If this runs, that copy is the culprit, not inlining as such.
export SPYRE_ATTN_INLINE=1
export SPYRE_COMPILE_GRANULARITY=block
export SPYRE_ENCODER_SLOT_PADDING=0
exec bash /home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/b3/count.sh "$@"
