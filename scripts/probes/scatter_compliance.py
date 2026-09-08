"""Probe which scatter destinations avoid a whole-destination relayout copy.

A scatter's indirect access must land on the outermost device dims or
`scatter_destination_check` copies the destination in and back for every scatter.
Only a genuinely 2-D destination with an explicit row-outermost layout qualifies: a
2-D view of a 3-D buffer inherits its parent's decomposition, and the default tiled
layout spreads the row index over two device dims and stores to the wrong rows
(torch-spyre#3705).

    for v in tiled rows_heads flat3d flat2d flat2d_view flat2d_default; do
        VARIANT=$v python scripts/probes/scatter_compliance.py
    done
"""

import os
import re
import tempfile
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR") or tempfile.mkdtemp(prefix="scatter_"))
OUT.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(OUT / "cache")
os.environ.setdefault("SPYRE_INDUCTOR_LOG", "1")
os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "DEBUG")
LOG = OUT / "planner.log"
os.environ.setdefault("SPYRE_LOG_FILE", str(LOG))

import torch  # noqa: E402
import torch_spyre  # noqa: E402

torch_spyre._autoload()
torch.spyre.set_device(0)
torch.zeros(1, dtype=torch.float16).to("spyre")

from torch_spyre._C import (  # noqa: E402
    SpyreTensorLayout,
    get_device_dtype,
    get_elem_in_stick,
)

ROWS, KV, QPK, D = 8, 8, 4, 128
H = KV * QPK
VARIANT = os.environ.get("VARIANT", "tiled")


def layout(device_size, stride_map):
    return SpyreTensorLayout(
        device_size=device_size, stride_map=stride_map, device_dtype=get_device_dtype(torch.float16)
    )


eps = get_elem_in_stick(torch.float16)
sticks = (D + eps - 1) // eps

# Each variant differs only in the destination's shape and device layout.
if VARIANT == "tiled":  # default device layout
    dest = torch.zeros(ROWS, H, D, dtype=torch.float16).to("spyre")
elif VARIANT == "rows_heads":  # slot_major_kv_layout: [rows, heads, sticks, eps]
    dest = torch.zeros(ROWS, H, D, dtype=torch.float16).to(
        "spyre", device_layout=layout([ROWS, H, sticks, eps], [H * D, D, eps, 1])
    )
elif VARIANT == "flat3d":  # 3-dim layout, 3-D logical tensor
    dest = torch.zeros(ROWS, H, D, dtype=torch.float16).to(
        "spyre", device_layout=layout([ROWS * H, sticks, eps], [D, eps, 1])
    )
elif VARIANT == "flat2d":  # genuinely 2-D logical tensor, 3-dim layout
    dest = torch.zeros(ROWS * H, D, dtype=torch.float16).to(
        "spyre", device_layout=layout([ROWS * H, sticks, eps], [D, eps, 1])
    )
elif VARIANT == "flat2d_view":
    # What the impl needs: a 2-D base the kernel scatters into, plus a 3-D view of the
    # same storage for the caller's copy-back, so no second buffer and no extra copy.
    base = torch.zeros(ROWS * H, D, dtype=torch.float16).to(
        "spyre", device_layout=layout([ROWS * H, sticks, eps], [D, eps, 1])
    )
    dest = base
    VIEW3D = base.view(ROWS, H, D)
elif VARIANT == "flat2d_default":  # 2-D logical, default layout
    dest = torch.zeros(ROWS * H, D, dtype=torch.float16).to("spyre")
else:
    raise SystemExit(f"unknown VARIANT {VARIANT}")

tables = [
    torch.tensor([kv * QPK + g for kv in range(KV)], dtype=torch.int32).to("spyre")
    for g in range(QPK)
]
srcs = [torch.randn(KV, 1, D, dtype=torch.float16).to("spyre") for _ in range(QPK)]


def store(out, tables, srcs):
    flat = out if out.dim() == 2 else out.view(-1, D)
    for g in range(QPK):
        flat.index_copy_(0, tables[g], srcs[g].squeeze(1))
    return out


torch.compile(store, dynamic=False)(dest, tables, srcs)

# Numerics: group g owns heads kv*QPK+g of row 0.
# For flat2d_view, read back through the 3-D view to prove that aliasing works.
got = VIEW3D.cpu()[0] if VARIANT == "flat2d_view" else dest.cpu().reshape(-1, H, D)[0]
ok = all(
    torch.allclose(got[kv * QPK + g], srcs[g].cpu()[kv, 0], atol=1e-3)
    for g in range(QPK)
    for kv in range(KV)
)

text = LOG.read_text(errors="replace") if LOG.is_file() else ""
verdicts = re.findall(r"scatter_destination_check: (\w+) (indirect_device_pos=[^\n]+)", text)
copies = len(re.findall(r"inserting mutation relayout copy", text))
kernels = sorted(p.name for p in (OUT / "cache" / "inductor-spyre").glob("*sdsc*"))

print(f"\n=== VARIANT={VARIANT} ===")
print(f"numerics: {'ok' if ok else 'WRONG'}")
for buf, v in verdicts:
    print(f"  {buf}: {v}")
print(f"relayout copies inserted: {copies}")
print(f"kernels: {len(kernels)} {[k.split('_sdsc_')[-1][:44] for k in kernels]}")
raise SystemExit(0 if ok and copies == 0 else 1)
