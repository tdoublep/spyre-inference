#!/usr/bin/env bash
# Run microbench_batched_decode.py with the environment its recorded numbers
# were taken with. Any arguments are forwarded to the script.
#
#   bash experimental/run_microbench_batched_decode.sh
#   bash experimental/run_microbench_batched_decode.sh --num-blocks 8 --block-size 256
#   LAYOUT_SOLVER=cpsat bash experimental/run_microbench_batched_decode.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The pinned RPM tree, if scripts/install-pinned-rpms.sh put one there. Skipped
# when absent, so a host whose /opt/ibm/spyre already matches spyre-rpms.lock
# works untouched. torch-spyre must have been built against whichever set is
# active or it fails at import with an undefined flex::/senlib:: symbol.
SPYRE_LIBS="${SPYRE_LIBS:-${HOME}/spyre-libs}"
if [[ -f "${SPYRE_LIBS}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SPYRE_LIBS}/env.sh"
else
    echo "note: ${SPYRE_LIBS}/env.sh not found; using the system /opt/ibm/spyre" >&2
fi

# greedy, not torch-spyre's cpsat default: this is what the vLLM benchmarks that
# motivated the microbenchmark ran with, so the numbers are comparable to them.
# Override on the command line to compare solvers.
export LAYOUT_SOLVER="${LAYOUT_SOLVER:-greedy}"
export SPYRE_NUM_CPUS="${SPYRE_NUM_CPUS:-8}"

echo "LAYOUT_SOLVER=${LAYOUT_SOLVER} SPYRE_NUM_CPUS=${SPYRE_NUM_CPUS}"
echo "SENTIENT_BASE_INSTALL_DIR=${SENTIENT_BASE_INSTALL_DIR:-<unset>}"

# --no-sync: `uv run` otherwise re-resolves from pyproject.toml and would
# silently replace a locally built torch-spyre.
exec uv run --no-sync python experimental/microbench_batched_decode.py "$@"
