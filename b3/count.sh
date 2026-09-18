#!/bin/bash
# B3 inner loop: fresh-cache compile, then count Inductor launch sites.
#   usage: b3/count.sh <arm-name> [B3_LAYERS]
# Writes b3/out/<arm>.log and b3/out/<arm>.counts, and copies the largest
# generated Inductor module to b3/out/<arm>.module.py so counts stay auditable.
set -u
WT="/home/senuser/spyre-inference/.claude/worktrees/s3c-b3-fusion"
ARM="${1:?arm name required}"
OUT="$WT/b3/out"
mkdir -p "$OUT"

export PYTHONPATH="$WT:${PYTHONPATH:-}"
export LAYOUT_SOLVER=greedy
export SPYRE_COMPILE_GRANULARITY="${SPYRE_COMPILE_GRANULARITY:-model}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/b3-inductor-$ARM"
rm -rf "$TORCHINDUCTOR_CACHE_DIR"
if [ -n "${2:-}" ]; then export B3_LAYERS="$2"; fi
if [ -n "${3:-}" ]; then export B3_INDUCTOR_CFG="$3"; fi
echo "inductor_cfg=${B3_INDUCTOR_CFG:-none}"

echo "=== arm=$ARM granularity=$SPYRE_COMPILE_GRANULARITY layers=${B3_LAYERS:-12} cache=$TORCHINDUCTOR_CACHE_DIR"
/home/senuser/spyre-inference/.venv/bin/python "$WT/b3/compile_only.py" >"$OUT/$ARM.log" 2>&1
rc=$?
echo "exit=$rc  compile: $(grep -o 'B3_COMPILE_SECONDS.*' "$OUT/$ARM.log" | tail -1)"

# Largest generated module is the whole-model graph.
MOD=$(find "$TORCHINDUCTOR_CACHE_DIR" -name '*.py' -printf '%s %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z "$MOD" ]; then
    echo "NO INDUCTOR MODULE FOUND in $TORCHINDUCTOR_CACHE_DIR"
    tail -25 "$OUT/$ARM.log"
    exit 1
fi
cp "$MOD" "$OUT/$ARM.module.py"

{
  echo "arm            $ARM"
  echo "granularity    $SPYRE_COMPILE_GRANULARITY"
  echo "layers         ${B3_LAYERS:-12}"
  echo "module         $MOD"
  echo "module_lines   $(wc -l <"$MOD")"
  echo "sdsc_run_sites $(grep -cE '^[[:space:]]*sdsc_fused[a-z0-9_]*\.run\(' "$MOD")"
  echo "sdsc_distinct  $(grep -oE 'sdsc_fused[a-z0-9_]*' "$MOD" | sort -u | wc -l)"
  echo "modules_total  $(find "$TORCHINDUCTOR_CACHE_DIR" -name '*.py' | wc -l)"
  echo "unified_attn   $(grep -c 'unified_attention' "$MOD")"
  echo "fallback       $(grep -c '_run_compiled_fallback' "$MOD")"
  echo "to_dtype_cpu   $(grep -c 'to_dtype_cpu' "$MOD")"
} | tee "$OUT/$ARM.counts"
