"""Kernel launches per attention call: legacy kernel vs the LX kernel's tails.

The unrolled group loop has to recombine its num_queries_per_kv results, and a
torch.stack does not fuse into the attention kernel -- it becomes a second launch,
which at ~130 us of fixed cost per launch would swamp the LX win. This counts SDSC
directories (one per launch) for the legacy kernel and for each candidate tail, so
the comparison is measured rather than assumed.

Each variant compiles in its own inductor cache dir so the counts do not pool.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(tempfile.mkdtemp(prefix="kcount_"))
os.environ.setdefault("TORCH_LOGS", "+torch_spyre.inductor")

KV, QPK, D, B, NUM_BLOCKS, Q = 8, 4, 128, 64, 7, 1
NUM_HEADS = KV * QPK
SCALE = D**-0.5
FP16_MIN = -65504.0


def count(tag: str) -> None:
    """Compile one variant in a fresh cache dir and print its launch count."""
    cache = ROOT / tag
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    import torch

    sdsc = cache / "inductor-spyre"
    names = sorted(d.name for d in sdsc.glob("*sdsc*")) if sdsc.is_dir() else []
    short = [n.split("_sdsc_")[-1][:52] for n in names]
    print(f"{tag:16s} launches {len(names)}: {short}", flush=True)
    del torch


VARIANT = sys.argv[1] if len(sys.argv) > 1 else None
if VARIANT is None:
    # Each variant needs a clean process: TORCHINDUCTOR_CACHE_DIR is read at import.
    import subprocess

    for tag in ("legacy", "stack", "scatter"):
        subprocess.run([sys.executable, __file__, tag, str(ROOT)], check=False)
    shutil.rmtree(ROOT, ignore_errors=True)
    raise SystemExit(0)

ROOT = Path(sys.argv[2])
CACHE = ROOT / VARIANT
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE)

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

EPS = get_elem_in_stick(torch.float16)
DT = get_device_dtype(torch.float16)


def rows3(rows, mid, inner):
    return SpyreTensorLayout(
        device_size=[rows, mid, (inner + EPS - 1) // EPS, EPS],
        stride_map=[mid * inner, inner, EPS, 1],
        device_dtype=DT,
    )


def slot_major(slots):
    return SpyreTensorLayout(
        device_size=[slots, KV, (D + EPS - 1) // EPS, EPS],
        stride_map=[KV * D, D, EPS, 1],
        device_dtype=DT,
    )


NUM_PAGES = 32
query = torch.randn(max(2 * Q, 8), NUM_HEADS, D, dtype=torch.float16).to("spyre")
row_index = torch.arange(Q, dtype=torch.int32).to("spyre")
masks = []
for i in range(NUM_BLOCKS):
    m = torch.zeros(Q, B, dtype=torch.float16)
    if i == NUM_BLOCKS - 1:
        m[:, B // 2 :] = FP16_MIN
    masks.append(m.to("spyre"))
out = torch.zeros(Q, NUM_HEADS, D, dtype=torch.float16).to("spyre")

if VARIANT == "legacy":
    from spyre_inference.v1.attention.backends.spyre_attn import _create_compilable_page_attn

    k = torch.randn(NUM_PAGES, B, KV, D, dtype=torch.float16).to(
        "spyre", device_layout=slot_major(NUM_PAGES * B)
    )
    v = torch.randn(NUM_PAGES, B, KV, D, dtype=torch.float16).to(
        "spyre", device_layout=slot_major(NUM_PAGES * B)
    )
    table = torch.zeros(NUM_BLOCKS, 32, dtype=torch.int32)
    table[:, 0] = torch.arange(NUM_BLOCKS, dtype=torch.int32)
    fn = _create_compilable_page_attn(
        NUM_BLOCKS, Q, NUM_HEADS, KV, D, store_mode="copy", needs_gather=True
    )
    torch.compile(fn, dynamic=False)(
        query, row_index, k, v, table.to("spyre"), masks, SCALE, out=out
    )
else:
    from spyre_inference.v1.attention.backends.spyre_attn import (
        _capped_attn_cores,
        _create_compilable_lx_page_attn,
    )

    layout = rows3(NUM_PAGES * KV, B, D)
    k = torch.randn(NUM_PAGES * KV, B, D, dtype=torch.float16).to("spyre", device_layout=layout)
    v = torch.randn(NUM_PAGES * KV, B, D, dtype=torch.float16).to("spyre", device_layout=layout)
    heads = torch.arange(KV, dtype=torch.int32).reshape(KV, 1)
    kv_tables = [(i * KV + heads).contiguous().to("spyre") for i in range(NUM_BLOCKS)]
    head_tables = [
        torch.tensor([kv * QPK + g for kv in range(KV)], dtype=torch.int32).to("spyre")
        for g in range(QPK)
    ]
    fn = _create_compilable_lx_page_attn(
        NUM_BLOCKS,
        Q,
        NUM_HEADS,
        KV,
        D,
        B,
        store_mode="copy",
        needs_gather=True,
        combine="scatter" if VARIANT == "scatter" else "stack",
    )
    with _capped_attn_cores():
        torch.compile(fn, dynamic=False)(
            query, row_index, k, v, kv_tables, head_tables, masks, SCALE, out=out
        )

sdsc = CACHE / "inductor-spyre"
names = sorted(d.name for d in sdsc.glob("*sdsc*")) if sdsc.is_dir() else []
short = [n.split("_sdsc_")[-1][:56] for n in names]
print(f"\nRESULT {VARIANT:10s} launches {len(names)}: {short}", flush=True)
