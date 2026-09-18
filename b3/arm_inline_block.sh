#!/bin/bash
# Arm: inlined attention AND per-block compile. Each block should fuse to ~1 kernel
# without building the enormous whole-model graph that segfaulted the allocator.
export SPYRE_ATTN_INLINE=1
export SPYRE_COMPILE_GRANULARITY=block
# NOTE: TORCH_SENDNN_CACHE_* is inert here (torch_sendnn is not installed);
# leaving this unset. It never affected any result.
exec bash /home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/b3/count.sh "$@"
