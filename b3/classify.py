"""Classify every sdsc launch site in an Inductor wrapper module.

Splits the 29 into real compute vs pure data movement by reading each kernel's
async_compile OpSpec list, and reports the graph fragment that produced it.
"""

import re
import sys
from collections import Counter

src = open(sys.argv[1]).read()

# Each kernel definition: preceding comment block, then `name = async_compile.sdsc(`
defs = {}
for m in re.finditer(r"^(\w+) = async_compile\.sdsc\(", src, re.M):
    name = m.group(1)
    # body = up to the matching close before the next top-level def
    nxt = src.find("\nasync_compile.wait", m.end())
    end = src.find("\n\n\n", m.end())
    body = src[m.end() : end if end > 0 else nxt]
    # comment block above
    head_start = src.rfind("\n\n\n", 0, m.start())
    head = src[head_start : m.start()]
    defs[name] = (head, body)

order = [m.group(1) for m in re.finditer(r"^\s*(sdsc_fused[a-z0-9_]*)\.run\(", src, re.M)]

print(f"launch sites: {len(order)}   distinct kernels defined: {len(defs)}\n")

kinds = Counter()
rows = []
for i, name in enumerate(order):
    head, body = defs.get(name, ("", ""))
    ops = re.findall(r"op='([a-z_0-9]+)'", body)
    aten = re.search(r"Original ATen: \[([^\]]*)\]", head)
    aten_s = aten.group(1) if aten else ""
    nusers0 = "num_users=0" in head
    allocs = re.findall(r"allocation=\{'(\w+)'", body)
    movement_only = set(ops) <= {"identity", "copy", "clone"}
    kind = "MOVE" if movement_only else "COMPUTE"
    kinds[kind] += 1
    rows.append((i, name, kind, ",".join(sorted(set(ops))), ",".join(sorted(set(allocs))), aten_s[:60], nusers0))

print(f"{'#':>3} {'kernel':<46} {'kind':<8} {'ops':<22} {'alloc':<10} aten")
for i, name, kind, ops, allocs, aten_s, nusers0 in rows:
    print(f"{i:>3} {name:<46} {kind:<8} {ops:<22} {allocs:<10} {aten_s}")

print()
for k, v in kinds.items():
    print(f"{k:<8} {v}")
