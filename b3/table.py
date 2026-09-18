"""Compact table: every embeds dump vs the CPU golden, plus its own spread.

Two numbers decide an arm. Against the golden: is it computing the right thing.
Its own within-run spread: four texts in four languages must be distinguishable,
so a run whose own vectors sit at cosine ~1.0 is degenerate whatever it scores
against anything else.
"""

import glob
import json
import os
import sys

OUT = os.path.dirname(os.path.abspath(__file__)) + "/out"
GOLDEN = f"{OUT}/golden-cpu.embeds.json"


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


def main() -> int:
    g = json.load(open(GOLDEN))["vectors"]
    paths = sys.argv[1:] or sorted(glob.glob(f"{OUT}/*.embeds.json"))

    print(f"{'arm':<28} {'vs golden':>10} {'max|d|':>9} {'own spread (min..max)':>24}  verdict")
    print("-" * 88)
    for p in paths:
        d = json.load(open(p))
        if d["tag"] == "golden-cpu":
            continue
        v = d["vectors"]
        worst = min(cos(gi, vi) for gi, vi in zip(g, v))
        mx = max(max(abs(x - y) for x, y in zip(gi, vi)) for gi, vi in zip(g, v))
        pair = [cos(v[i], v[j]) for i in range(len(v)) for j in range(i + 1, len(v))]
        # Golden's own spread is the yardstick for "distinguishable".
        gp = [cos(g[i], g[j]) for i in range(len(g)) for j in range(i + 1, len(g))]
        verdict = "PASS" if worst > 0.9995 else "FAIL"
        if min(pair) > 0.999:
            verdict += " (DEGENERATE)"
        print(
            f"{d['tag']:<28} {worst:>10.6f} {mx:>9.6f} "
            f"{min(pair):>10.6f}..{max(pair):<10.6f}  {verdict}"
        )
    print("-" * 88)
    print(f"{'golden-cpu (reference)':<28} {1.0:>10.6f} {0.0:>9.6f} "
          f"{min(gp):>10.6f}..{max(gp):<10.6f}  --")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
