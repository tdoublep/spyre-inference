#!/bin/bash
# Per-module launch-site breakdown for a per-block-granularity inductor cache.
# Under granularity=block the 12 identical blocks share one artifact, so the
# static site count must be multiplied by how many times each graph runs.
CACHE="${1:?cache dir}"
for f in $(find "$CACHE" -name '*.py'); do
    n=$(grep -cE '^[[:space:]]*sdsc_fused[a-z0-9_]*\.run\(' "$f")
    [ "$n" -eq 0 ] && continue
    names=$(grep -oE 'sdsc_fused[a-z0-9_]*\.run\(' "$f" | sed 's/\.run(//' | tr '\n' ' ')
    ua=$(grep -c 'unified_attention' "$f")
    echo "sites=$n unified_attention=$ua  $(basename "$f")"
    echo "    $names"
done
