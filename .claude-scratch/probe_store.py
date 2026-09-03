"""Which part of the folded store torch-spyre can lower.

The 2-D destination and 2-D source were both measured working in
bench_kv_store_frames.py frame B. This isolates what breaks when the destination is
a view of the 3-D gather source and the source is flattened from [T, KV, D].
"""

import os
import tempfile

OUT = tempfile.mkdtemp(prefix="probe_store_")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(OUT, "cache")
os.environ.setdefault("TORCH_LOGS", "+torch_spyre.inductor")
os.environ["SPYRE_INDUCTOR_LOG"] = "1"
os.environ["SPYRE_INDUCTOR_LOG_LEVEL"] = "DEBUG"
LOG = os.path.join(OUT, "planner.log")
os.environ["SPYRE_LOG_FILE"] = LOG

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

P, B, KV, D, T = 32, 64, 8, 128, 4
EPS = get_elem_in_stick(torch.float16)
DT = get_device_dtype(torch.float16)


def rows3(rows, mid, inner):
    return SpyreTensorLayout(
        device_size=[rows, mid, (inner + EPS - 1) // EPS, EPS],
        stride_map=[mid * inner, inner, EPS, 1],
        device_dtype=DT,
    )


def rows2(rows, width):
    return SpyreTensorLayout(
        device_size=[rows, (width + EPS - 1) // EPS, EPS],
        stride_map=[width, EPS, 1],
        device_dtype=DT,
    )


def store_flat(dst, idx, src):
    dst.index_copy_(0, idx, src)


def store_flatten(dst, idx, src):
    dst.index_copy_(0, idx, src.flatten(0, 1))


slots = torch.tensor([5 * B + 9, 5 * B + 10, 6 * B + 0, 63], dtype=torch.int64)
pages = torch.div(slots, B, rounding_mode="floor")
offs = slots - pages * B
heads = torch.arange(KV, dtype=torch.int64)
rows = ((pages.unsqueeze(1) * KV + heads) * B + offs.unsqueeze(1)).reshape(-1)
key3 = torch.randn(T, KV, D, dtype=torch.float16)
key2 = key3.reshape(T * KV, D).contiguous()
host = torch.randn(P * KV * B, D, dtype=torch.float16)


def run(label, alloc, dest, src_kind, fn):
    base = host.clone()
    if alloc == "2d":
        dev = base.to("spyre", device_layout=rows2(P * KV * B, D))
    else:
        dev = base.reshape(P * KV, B, D).to("spyre", device_layout=rows3(P * KV, B, D))
    try:
        dst = dev if dest == "as-is" else dev.view(-1, D)
    except Exception as e:  # noqa: BLE001
        print(f"{label:40s} VIEW FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
        return
    src = {"2d": key2, "3d": key3, "3d-flat-out": key3.view(-1, D)}[src_kind]
    try:
        torch.compile(fn, dynamic=False)(dst, rows.to("spyre"), src.to("spyre"))
    except Exception as e:  # noqa: BLE001
        print(f"{label:40s} COMPILE FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
        return
    want = base.clone()
    want.index_copy_(0, rows, key2)
    got = dev.cpu().reshape(-1, D)
    err = (got.float() - want.float()).abs().max().item()
    n = 0
    if os.path.isfile(LOG):
        n = open(LOG, errors="replace").read().count("mutation relayout copy")
    print(f"{label:40s} err {err:.3e}  cumulative relayout copies {n}", flush=True)


# Control: the frame the store benchmark measured working.
run("S1 alloc2d / dest as-is / src 2d", "2d", "as-is", "2d", store_flat)
# Source flattened inside the graph (what failed).
run("S2 alloc2d / dest as-is / src flatten", "2d", "as-is", "3d", store_flatten)
# Source flattened outside the graph.
run("S3 alloc2d / dest as-is / src view-out", "2d", "as-is", "3d-flat-out", store_flat)
# Destination is a view of the 3-D gather source.
run("S4 alloc3d / dest view / src 2d", "3d", "view", "2d", store_flat)
run("S5 alloc3d / dest view / src view-out", "3d", "view", "3d-flat-out", store_flat)
run("S6 alloc3d / dest view / src flatten", "3d", "view", "3d", store_flatten)
print(f"\nartifacts: {OUT}")
