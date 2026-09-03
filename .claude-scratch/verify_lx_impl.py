"""Device check for the folded (page, kv_head) KV cache as spyre_inference builds it.

Covers the two things the hand-off left unmeasured: that the decode store works in
the real frame (open question 3), and that the real attention kernel -- not the
standalone reproducer -- still gets both gathered pages into LX.

Run through .claude-scratch/venv.sh so the pinned RPMs and this worktree win.
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

torch_spyre._autoload()
torch.spyre.set_device(0)
torch.zeros(1, dtype=torch.float16).to("spyre")

from spyre_inference.v1.attention.backends.spyre_attn import (  # noqa: E402
    _capped_attn_cores,
    _create_compilable_lx_page_attn,
    _create_folded_cache_store,
    head_major_kv_layout,
)

NUM_PAGES, B, KV, D = 32, 64, 8, 128
QPK = 4
NUM_HEADS = KV * QPK
NUM_BLOCKS = 7  # >= 448 tokens of context, several pages per step
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


# --- 1. the folded store ---------------------------------------------------
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
# Flat (token, head) order, for the host reference only.
heads = torch.arange(KV, dtype=torch.int64)
rows = ((pages.unsqueeze(1) * KV + heads) * B + offs.unsqueeze(1)).reshape(-1)

key = torch.randn(NUM_TOKENS, KV, D, dtype=torch.float16)
value = torch.randn(NUM_TOKENS, KV, D, dtype=torch.float16)
key_dev = key.to("spyre")
value_dev = value.to("spyre")

# The reference is built from what the device already holds, so this checks the store
# -- which rows it wrote and with what -- and not the fp16 format the transfer used.
# Spyre stores fp16 as SEN169, which costs a mantissa bit on the way in.
k_ref = k_dev.cpu()
v_ref = v_dev.cpu()
k_ref.view(-1, D).index_copy_(0, rows, key_dev.cpu().reshape(-1, D))
v_ref.view(-1, D).index_copy_(0, rows, value_dev.cpu().reshape(-1, D))

torch.compile(_create_folded_cache_store(KV), dynamic=False)(
    key_dev,
    value_dev,
    k_dev.view(-1, D),
    v_dev.view(-1, D),
    [t.to("spyre") for t in per_head],
)
check("store K", k_dev.cpu(), k_ref, 0.0)
check("store V", v_dev.cpu(), v_ref, 0.0)
# A positive control: the store must actually have changed those rows, or a no-op
# store would pass the check above.
moved = (k_dev.cpu().view(-1, D).index_select(0, rows) - k_before.index_select(0, rows)).abs()
print(f"{'ok  ' if moved.max() > 0 else 'FAIL'} store touched its rows: max delta {moved.max():.3e}")
if moved.max() == 0:
    fails.append("store wrote nothing")

# --- 2. the real attention kernel -----------------------------------------
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
with _capped_attn_cores():
    got = torch.compile(kernel, dynamic=False)(*dev_args)

# Reference: the folded cache read back, run through the same closure on CPU.
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
check("attention (stack tail)", got.cpu(), kernel(*cpu_args), 5e-3)

# The scatter tail is what batch-1 decode uses, and it recombines the groups by
# writing each into its own head positions rather than stacking -- a different tail,
# so it needs its own numerics check.
scatter_kernel = _create_compilable_lx_page_attn(
    NUM_BLOCKS, Q_LEN, NUM_HEADS, KV, D, B, store_mode="copy"
)
out_dev = torch.zeros(Q_LEN, NUM_HEADS, D, dtype=torch.float16).to("spyre")
with _capped_attn_cores():
    torch.compile(scatter_kernel, dynamic=False)(*dev_args, out=out_dev)
check("attention (scatter tail)", out_dev.cpu(), kernel(*cpu_args), 5e-3)

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
check("attention vs SDPA", got.cpu(), want, 5e-3)
check("scatter tail vs SDPA", out_dev.cpu(), want, 5e-3)

# --- 3. residency ----------------------------------------------------------
text = PLANNER_LOG.read_text(errors="replace") if PLANNER_LOG.is_file() else ""
pinned = re.findall(r"lx_pinning: (\w+) \(index\) . lx\b", text)
print(f"\ngathers pinned LX: {len(pinned)} {pinned}")
relayout = text.count("mutation relayout copy")
print(f"mutation relayout copies: {relayout}")

pools = []
root = OUT / "inductor-cache" / "inductor-spyre"
kernels = [d.name for d in sorted(root.glob("*sdsc*"))] if root.is_dir() else []
# One SDSC directory is one launch. Classify by the ops in each name: the store is
# the index_copy/select one with no bmm, the scatter tail's attention carries both a
# bmm and an index_copy, and fused_stack is the stack tail's extra launch.
store_kernels = [k for k in kernels if "index_copy" in k and "bmm" not in k]
scatter_attn = [k for k in kernels if "bmm" in k and "index_copy" in k]
stack_extra = [k for k in kernels if "stack" in k]
print(
    f"launches -- store {len(store_kernels)}, scatter-tail attention "
    f"{len(scatter_attn)}, stack-tail extra {len(stack_extra)}"
)
if len(store_kernels) != 1:
    fails.append(f"store took {len(store_kernels)} launches, want 1")
if len(scatter_attn) != 1:
    fails.append(f"scatter tail took {len(scatter_attn)} launches, want 1")
for d in sorted(root.glob("*")) if root.is_dir() else []:
    bundle = d / "bundle.mlir"
    if bundle.is_file():
        m = re.search(r"device_mem_allocate (\d+) bytes", bundle.read_text())
        if m:
            pools.append((d.name[:48], int(m.group(1)) / 1024))
for name, kb in pools:
    print(f"  HBM pool {kb:8.1f} KB  {name}")

want_pinned = 2 * NUM_BLOCKS
if len(pinned) < want_pinned:
    fails.append(f"only {len(pinned)}/{want_pinned} gathers in LX")
if relayout:
    fails.append(f"{relayout} relayout copies")

print("\nFAILURES: " + (", ".join(fails) if fails else "none"))
print(f"artifacts: {OUT}")
raise SystemExit(1 if fails else 0)
