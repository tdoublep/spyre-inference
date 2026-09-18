"""Within-run pairwise cosines. Free: reads the dumps, no device.

Four texts in four languages should be clearly distinguishable. A run whose own
vectors are near-identical is degenerate, whichever side of a comparison it is on.
"""

import json
import sys

for path in sys.argv[1:]:
    d = json.load(open(path))
    v = d["vectors"]
    print(f"\n{d['tag']}  ({path.split('/')[-1]})")
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            dot = sum(a * b for a, b in zip(v[i], v[j]))
            ni = sum(a * a for a in v[i]) ** 0.5
            nj = sum(b * b for b in v[j]) ** 0.5
            print(f"   cos(text{i}, text{j}) = {dot / (ni * nj):.6f}")
