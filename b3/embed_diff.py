"""Compare two embed_check dumps: cosine similarity and max abs delta per text."""

import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))

print(f"{a['tag']}  vs  {b['tag']}")
worst_cos = 1.0
worst_abs = 0.0
for i, (va, vb) in enumerate(zip(a["vectors"], b["vectors"])):
    dot = sum(x * y for x, y in zip(va, vb))
    na = sum(x * x for x in va) ** 0.5
    nb = sum(y * y for y in vb) ** 0.5
    cos = dot / (na * nb)
    mx = max(abs(x - y) for x, y in zip(va, vb))
    worst_cos = min(worst_cos, cos)
    worst_abs = max(worst_abs, mx)
    print(f"  text{i}: cosine={cos:.8f}  max_abs_delta={mx:.6f}")

print(f"\nworst cosine   {worst_cos:.8f}")
print(f"worst abs delta {worst_abs:.6f}")
# fp16 accumulation order differs between a fused and an unfused kernel, so exact
# equality is not the bar; a real numerical break shows up far below this.
print("VERDICT:", "PASS" if worst_cos > 0.9995 else "FAIL")
