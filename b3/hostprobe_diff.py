"""Steady-state host span costs: difference the last two cumulative dumps.

_hostprobe writes cumulative totals every SPYRE_HOSTPROBE_EVERY steps, so the
first dump carries the lazy compiles of the settle phase. Differencing two later
dumps removes them, which is what the module's own docstring asks for.
"""

import sys


def parse(path):
    dumps, cur = [], None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("==="):
            cur = {"_steps": int(line.split("steps=")[1].split()[0])}
            dumps.append(cur)
        elif line.startswith("#") or not line.strip() or cur is None:
            continue
        else:
            parts = line.split("\t")
            if len(parts) == 3:
                cur[parts[0]] = (float(parts[1]), int(parts[2]))
    return dumps


def main() -> int:
    for path in sys.argv[1:]:
        dumps = parse(path)
        if len(dumps) < 2:
            print(f"{path}: need two dumps, have {len(dumps)}")
            continue
        a, b = dumps[-2], dumps[-1]
        steps = b["_steps"] - a["_steps"]
        print(f"\n{path.split('/')[-1]}   steady-state over {steps} steps")
        print(f"  {'span':<42} {'ms/call':>9} {'calls':>7} {'total ms':>9}")
        rows = []
        for k, v in b.items():
            if k == "_steps" or k not in a:
                continue
            t, n = v
            dt = (t - a[k][0]) * 1e3
            dn = n - a[k][1]
            if dn > 0 and dt > 0.05:
                rows.append((dt, k, dt / dn, dn))
        for dt, k, per, dn in sorted(rows, reverse=True):
            print(f"  {k:<42} {per:>9.3f} {dn:>7} {dt:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
