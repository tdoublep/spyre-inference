"""Check that the folded (page, kv_head) KV cache keeps both gathered pages in LX.

Drives the real `spyre_attn` kernels, so it needs SPYRE_LX_KV_LAYOUT's code path and
therefore a build that has it. Verifies the store, the attention numerics against an
independent SDPA reference, the launch count, and the LX residency of every gather.

    NUM_BLOCKS=7 python scripts/probes/lx_kv_residency.py
"""

import os
import re
import tempfile
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR") or tempfile.mkdtemp(prefix="verify_lx_"))
OUT.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(OUT / "inductor-cache")
os.environ.setdefault("TORCH_LOGS", "+torch_spyre.inductor")
os.environ.setdefault("SPYRE_INDUCTOR_LOG", "1")
os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "DEBUG")
PLANNER_LOG = OUT / "planner.log"
os.environ.setdefault("SPYRE_LOG_FILE", str(PLANNER_LOG))
os.environ["SPYRE_LX_KV_LAYOUT"] = "1"

import torch  # noqa: E402
import torch_spyre  # noqa: E402
import torch_spyre._inductor.pass_utils  # noqa: E402

torch_spyre._autoload()
torch.spyre.set_device(0)
torch.zeros(1, dtype=torch.float16).to("spyre")

from spyre_inference.v1.attention.backends.spyre_attn import (  # noqa: E402
    _capped_attn_cores,
    _create_compilable_lx_page_attn,
    _create_folded_reshape_and_cache,
    flat_row_layout,
    head_major_kv_layout,
)

NUM_PAGES, B, KV, D = 32, 64, 8, 128
QPK = 4
NUM_HEADS = KV * QPK
NUM_BLOCKS = int(os.environ.get("NUM_BLOCKS", "7"))  # pages gathered per step
Q_LEN = 1
SCALE = D**-0.5
FP16_MIN = torch.finfo(torch.float16).min

fails = []


def check(name, got, want, tol):
    err = (got.float() - want.float()).abs().max().item()
    ok = err <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {err:.3e} (tol {tol:.0e})")
    if not ok:
        fails.append(name)


# Same expansion SlotMapping.slots_for does: a token becomes KV rows.
k_host = torch.randn(NUM_PAGES * KV, B, D, dtype=torch.float16)
v_host = torch.randn(NUM_PAGES * KV, B, D, dtype=torch.float16)
layout = head_major_kv_layout(NUM_PAGES * KV, B, D, torch.float16)
k_dev = k_host.to("spyre", device_layout=layout)
v_dev = v_host.to("spyre", device_layout=layout)
k_before = k_dev.cpu().view(-1, D)

NUM_TOKENS = 4
slots = torch.tensor([5 * B + 9, 5 * B + 10, 6 * B + 0, 0 * B + 63], dtype=torch.int64)
pages = torch.div(slots, B, rounding_mode="floor")
offs = slots - pages * B
# Per head, as SlotMapping.slots_for builds it.
per_head = [(pages * KV + h) * B + offs for h in range(KV)]
heads = torch.arange(KV, dtype=torch.int64)
rows = ((pages.unsqueeze(1) * KV + heads) * B + offs.unsqueeze(1)).reshape(-1)

key = torch.randn(NUM_TOKENS, KV, D, dtype=torch.float16)
value = torch.randn(NUM_TOKENS, KV, D, dtype=torch.float16)
key_dev = key.to("spyre")
value_dev = value.to("spyre")

# Referenced against what the device already holds: Spyre stores fp16 as SEN169,
# which costs a mantissa bit on transfer, so a pure host reference would not match.
k_ref = k_dev.cpu()
v_ref = v_dev.cpu()
k_ref.view(-1, D).index_copy_(0, rows, key_dev.cpu().reshape(-1, D))
v_ref.view(-1, D).index_copy_(0, rows, value_dev.cpu().reshape(-1, D))

torch.compile(_create_folded_reshape_and_cache(KV), dynamic=False)(
    key_dev,
    value_dev,
    k_dev.view(-1, D),
    v_dev.view(-1, D),
    [t.to("spyre") for t in per_head],
)
check("store K", k_dev.cpu(), k_ref, 0.0)
check("store V", v_dev.cpu(), v_ref, 0.0)
# index_copy_ writes nothing to a single-row destination (torch-spyre#4007), which
# a diff-against-expected cannot catch on its own.
moved = (k_dev.cpu().view(-1, D).index_select(0, rows) - k_before.index_select(0, rows)).abs()
print(
    f"{'ok  ' if moved.max() > 0 else 'FAIL'} store touched its rows: max delta {moved.max():.3e}"
)
if moved.max() == 0:
    fails.append("store wrote nothing")

query = torch.randn(max(2 * Q_LEN, 8), NUM_HEADS, D, dtype=torch.float16)
row_index = torch.arange(Q_LEN, dtype=torch.int32)
pages_used = torch.arange(NUM_BLOCKS, dtype=torch.int32)
head_ids = torch.arange(KV, dtype=torch.int32).reshape(KV, 1)
kv_tables = [(int(pages_used[i]) * KV + head_ids).contiguous() for i in range(NUM_BLOCKS)]
head_tables = [
    torch.tensor([kv * QPK + g for kv in range(KV)], dtype=torch.int32) for g in range(QPK)
]
masks = [torch.zeros(Q_LEN, B, dtype=torch.float16) for _ in range(NUM_BLOCKS)]
masks[-1][:, B // 2 :] = FP16_MIN  # a real tail block

# CPU reference only, never compiled: a second compiled variant would make every
# per-kernel count below ambiguous between the two.
kernel = _create_compilable_lx_page_attn(NUM_BLOCKS, Q_LEN, NUM_HEADS, KV, D, B)
dev_args = (
    query.to("spyre"),
    row_index.to("spyre"),
    k_dev,
    v_dev,
    [t.to("spyre") for t in kv_tables],
    [t.to("spyre") for t in head_tables],
    [m.to("spyre") for m in masks],
    SCALE,
)

cpu_args = (
    query.float(),
    row_index,
    k_dev.cpu().float(),
    v_dev.cpu().float(),
    kv_tables,
    head_tables,
    [m.float() for m in masks],
    SCALE,
)
want_cpu = kernel(*cpu_args)

STAGING_ROWS = query.shape[0]
scatter_kernel = _create_compilable_lx_page_attn(
    NUM_BLOCKS, Q_LEN, NUM_HEADS, KV, D, B, store_out=True
)
out_flat = torch.zeros(STAGING_ROWS * NUM_HEADS, D, dtype=torch.float16).to(
    "spyre", device_layout=flat_row_layout(STAGING_ROWS * NUM_HEADS, D, torch.float16)
)
out_dev = out_flat.view(STAGING_ROWS, NUM_HEADS, D)
out_row_tables = [
    torch.tensor([kv * QPK + g for kv in range(KV)], dtype=torch.int32).to("spyre")
    for g in range(QPK)
]
with _capped_attn_cores(KV * Q_LEN):
    torch.compile(scatter_kernel, dynamic=False)(
        *dev_args, out=out_flat, out_row_tables=out_row_tables
    )
check("attention vs CPU", out_dev.cpu()[:Q_LEN], want_cpu, 5e-3)

# Independent reference: plain SDPA over the same pages, so a restructuring bug
# in the kernel itself cannot hide inside a device-vs-CPU comparison of one closure.
k_flat = k_dev.cpu().float().reshape(NUM_PAGES, KV, B, D)
v_flat = v_dev.cpu().float().reshape(NUM_PAGES, KV, B, D)
k_ctx = torch.cat([k_flat[int(pages_used[i])] for i in range(NUM_BLOCKS)], dim=1)
v_ctx = torch.cat([v_flat[int(pages_used[i])] for i in range(NUM_BLOCKS)], dim=1)
mask_ctx = torch.cat([m.float() for m in masks], dim=-1)  # [Q_LEN, NUM_BLOCKS*B]
q = query[:Q_LEN].float().reshape(Q_LEN, KV, QPK, D).permute(1, 2, 0, 3)
scores = torch.matmul(q, k_ctx.unsqueeze(1).transpose(-2, -1)) * SCALE + mask_ctx
probs = torch.softmax(scores, dim=-1)
want = torch.matmul(probs, v_ctx.unsqueeze(1))
want = want.reshape(NUM_HEADS, Q_LEN, D).transpose(0, 1)
check("attention vs SDPA", out_dev.cpu()[:Q_LEN], want, 5e-3)

text = PLANNER_LOG.read_text(errors="replace") if PLANNER_LOG.is_file() else ""
pinned = re.findall(r"lx_pinning: (\w+) \(index\) . lx\b", text)
print(f"\ngathers pinned LX: {len(pinned)} {pinned}")
refused = re.findall(r"lx_pinning: (\w+) \(index\) . ([^\n]+)", text)
reasons: dict[str, int] = {}
for _buf, why in refused:
    if why.strip() != "lx":
        reasons[why.strip()] = reasons.get(why.strip(), 0) + 1
for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
    print(f"  refused x{n}: {why}")
restickify_bar = text.count("read by restickify")
print(f"gathers barred by the restickify cross-frame barrier: {restickify_bar}")
relayout = text.count("mutation relayout copy")
print(f"mutation relayout copies: {relayout}")

pools = []
root = OUT / "inductor-cache" / "inductor-spyre"
kernels = [d.name for d in sorted(root.glob("*sdsc*"))] if root.is_dir() else []
# One SDSC directory is one launch, classified by the ops in its name.
store_kernels = [k for k in kernels if "index_copy" in k and "bmm" not in k]
scatter_attn = [k for k in kernels if "bmm" in k and "index_copy" in k]
print(f"launches -- store {len(store_kernels)}, attention {len(scatter_attn)}")
if len(store_kernels) != 1:
    fails.append(f"store took {len(store_kernels)} launches, want 1")
if len(scatter_attn) != 1:
    fails.append(f"fused store tail took {len(scatter_attn)} launches, want 1")
for d in sorted(root.glob("*")) if root.is_dir() else []:
    bundle = d / "bundle.mlir"
    if bundle.is_file():
        m = re.search(r"device_mem_allocate (\d+) bytes", bundle.read_text())
        if m:
            pools.append((d.name[:48], int(m.group(1)) / 1024))
for name, kb in pools:
    print(f"  HBM pool {kb:8.1f} KB  {name}")
attn_pools = sorted(kb for name, kb in pools if "bmm" in name)
print(f"POOLSWEEP blocks={NUM_BLOCKS} attn_pools_kb={attn_pools}")

# Full K+V residency, not lowered to suit the stack: a run that cannot reach it is
# reporting that the feature does not work here. K needs torch-spyre#4153's proof
# that a restickify's read is core-local, or its gather is barred outright.
proof = "per_core_views_equal" in (
    Path(torch_spyre._inductor.pass_utils.__file__).read_text(errors="replace")
)
want_pinned = 2 * NUM_BLOCKS
if len(pinned) < want_pinned:
    reason = (
        "restickify proof absent -- K cannot reach LX without torch-spyre#4153"
        if not proof
        else "proof present, so this is a regression"
    )
    fails.append(f"only {len(pinned)}/{want_pinned} gathers in LX ({reason})")
if restickify_bar:
    fails.append(f"{restickify_bar} gathers barred by the restickify barrier")
if relayout:
    fails.append(f"{relayout} relayout copies")

print("\nFAILURES: " + (", ".join(fails) if fails else "none"))
print(f"artifacts: {OUT}")
raise SystemExit(1 if fails else 0)
