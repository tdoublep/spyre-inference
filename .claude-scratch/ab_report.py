"""Side-by-side of the LX-layout A/B legs, straight from the benchmark JSON."""

import json
import statistics

LEGS = {
    "main": "origin/main (8f0936c)",
    "off": "branch, flag off",
    "off2": "branch, flag off (rerun)",
    "on": "branch, flag ON (LX folded)",
}

d = {t: json.load(open(f".claude-scratch/ab/latency_{t}.json")) for t in LEGS}

print(f"{'leg':32s}{'avg (s)':>10s}{'p50':>9s}{'p90':>9s}{'p99':>9s}{'spread':>9s}  gate")
for t, label in LEGS.items():
    j = d[t]
    L = j["latencies"]
    sp = max(L) / min(L)
    p = j["percentiles"]
    gate = "LEAK" if sp > 1.1 else "clean"
    print(
        f"{label:32s}{j['avg_latency']:10.4f}{p['50']:9.4f}"
        f"{p['90']:9.4f}{p['99']:9.4f}{sp:9.4f}  {gate}"
    )

base = [d[t]["avg_latency"] for t in ("main", "off", "off2")]
bm = statistics.mean(base)
on = d["on"]["avg_latency"]
print(
    f"\nbaseline (main + 2x flag-off): mean {bm:.4f}s, "
    f"range {min(base):.4f}-{max(base):.4f} ({(max(base) - min(base)) / bm * 100:.2f}%)"
)
print(f"LX folded layout:             {on:.4f}s  {on - bm:+.4f}s  ({(on - bm) / bm * 100:+.2f}%)")
print(
    f"decode throughput:            {64 / bm:.3f} -> {64 / on:.3f} tok/s "
    f"({(64 / on) / (64 / bm) * 100 - 100:+.2f}%)"
)
