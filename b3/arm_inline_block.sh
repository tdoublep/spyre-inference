#!/bin/bash
# Arm: inlined attention AND per-block compile. Each block should fuse to ~1 kernel
# without building the enormous whole-model graph that segfaulted the allocator.
export SPYRE_ATTN_INLINE=1
export SPYRE_COMPILE_GRANULARITY=block
# Bypass (never delete - it is on /share and may be shared) the sendnn device-program
# cache, in case a crashed run left an entry that does not match this graph.
export TORCH_SENDNN_CACHE_ENABLE=0
exec bash /home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/b3/count.sh "$@"
