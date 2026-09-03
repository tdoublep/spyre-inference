"""How to hand the folded store a [T*KV, head_size] source that is already on device.

probe_store.py showed the folded destination is fine and an in-graph flatten is not.
But its working source was reshaped on the host before transfer; the real `key` is
produced on device by the QKV projection, so the reshape has to happen there. This
probes the ways of doing that.
"""

import os
import tempfile

OUT = tempfile.mkdtemp(prefix="probe_store2_")
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


slots = torch.tensor([5 * B + 9, 5 * B + 10, 6 * B + 0, 63], dtype=torch.int64)
pages = torch.div(slots, B, rounding_mode="floor")
offs = slots - pages * B
heads = torch.arange(KV, dtype=torch.int64)
rows = ((pages.unsqueeze(1) * KV + heads) * B + offs.unsqueeze(1)).reshape(-1)
rows_dev = rows.to("spyre")
key3 = torch.randn(T, KV, D, dtype=torch.float16)
host = torch.randn(P * KV, B, D, dtype=torch.float16)


def fresh():
    dev = host.to("spyre", device_layout=rows3(P * KV, B, D))
    return dev, dev.cpu().reshape(-1, D)


def report(label, dev, before, key_dev):
    want = before.clone()
    want.index_copy_(0, rows, key_dev.cpu().reshape(-1, D))
    got = dev.cpu().reshape(-1, D)
    err = (got.float() - want.float()).abs().max().item()
    moved = (got.index_select(0, rows) - before.index_select(0, rows)).abs().max().item()
    n = open(LOG, errors="replace").read().count("mutation relayout copy") if os.path.isfile(LOG) else 0
    print(
        f"{label:44s} err {err:.3e}  wrote {moved:.3e}  cumulative relayouts {n}",
        flush=True,
    )


def attempt(label, body):
    dev, before = fresh()
    key_dev = key3.to("spyre")
    try:
        body(dev, key_dev)
    except Exception as e:  # noqa: BLE001
        print(f"{label:44s} FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
        return
    report(label, dev, before, key_dev)


def plain(dst, idx, src):
    dst.index_copy_(0, idx, src)


def in_graph_flatten_contig(dst, idx, src):
    dst.index_copy_(0, idx, src.flatten(0, 1).contiguous())


def in_graph_clone(dst, idx, src):
    dst.index_copy_(0, idx, src.reshape(-1, src.shape[-1]).clone())


def per_head(dst, idx_list, src):
    for h, idx in enumerate(idx_list):
        dst.index_copy_(0, idx, src.select(1, h))


# P1: device-side view of the [T, KV, D] source (what the impl does now).
attempt(
    "P1 src device view(-1, D)",
    lambda dev, k: torch.compile(plain, dynamic=False)(dev.view(-1, D), rows_dev, k.view(-1, D)),
)
# P2: eager reshape on device, which may materialise rather than view.
attempt(
    "P2 src device reshape(-1, D)",
    lambda dev, k: torch.compile(plain, dynamic=False)(
        dev.view(-1, D), rows_dev, k.reshape(-1, D)
    ),
)
# P3: flatten then contiguous inside the graph.
attempt(
    "P3 in-graph flatten().contiguous()",
    lambda dev, k: torch.compile(in_graph_flatten_contig, dynamic=False)(
        dev.view(-1, D), rows_dev, k
    ),
)
# P4: reshape then clone inside the graph.
attempt(
    "P4 in-graph reshape().clone()",
    lambda dev, k: torch.compile(in_graph_clone, dynamic=False)(dev.view(-1, D), rows_dev, k),
)
# P5: keep both sides 3-D, [N, 1, D] destination against a [T*KV, 1, D] source.
attempt(
    "P5 dest [N,1,D] / src [T*KV,1,D]",
    lambda dev, k: torch.compile(plain, dynamic=False)(
        dev.view(-1, 1, D), rows_dev, k.view(-1, 1, D)
    ),
)
# P6: one index_copy_ per kv head, source sliced on the head axis in-graph.
idx_per_head = [
    ((pages * KV + h) * B + offs).to("spyre") for h in range(KV)
]
attempt(
    "P6 per-head index_copy_ (KV ops)",
    lambda dev, k: torch.compile(per_head, dynamic=False)(dev.view(-1, D), idx_per_head, k),
)
print(f"\nartifacts: {OUT}")
