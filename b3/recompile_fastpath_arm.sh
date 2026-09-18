#!/bin/bash
# As recompile_arm.sh, plus SPYRE_ENCODER_FASTPATH=1.
# The dominant per-call recompile trigger is a size mismatch on
# forward_context.slot_mapping[...], and that flag's whole job is to reuse one
# slot-mapping/block-table buffer per shape instead of allocating per step.
set -u
WT="/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion"
export SPYRE_ENCODER_FASTPATH=1
exec bash "$WT/b3/recompile_arm.sh" "$@"
