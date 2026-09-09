"""Bisect which structural feature the LX pin depends on.

`lx_transposed_attn_probe.py` at MAX_CORES=8 pins all its K gathers.
`repro_gather_view_width.py`, a hand-built minimal version, pins nothing at the same core
count and the same chosen split. This walks from the minimal shape to the full one, adding
one feature per variant, and reports the first variant that pins -- which names the feature
the pin actually depends on.

    python scripts/probes/repro_reduce_pin.py            # all variants at E=8, cores=8
    VARIANT=3 python scripts/probes/repro_reduce_pin.py  # just one

Variants (cumulative):
    0  minimal: 2 blocks, 1 query group, per-entry mask, V consumed raw
    1  + V permuted, out = v_t @ probs  (the working kernel's dataflow)
    2  + 4 query groups sharing each gathered page
    3  + mask broadcast over the entry axis ([block, query] instead of [E, block, query])
    4  + 8 blocks
    5  + the full output tail (stack over groups, transpose to [query, heads, head_size])

Env: VARIANT, ENTRIES, CORES, Q_LEN, BLOCK_SIZE, HEAD_SIZE, OUT_DIR
"""

import os
import re
import tempfile
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR") or tempfile.mkdtemp(prefix="reduce_pin_"))
OUT.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(OUT / "inductor-cache")
os.environ.setdefault("SPYRE_INDUCTOR_LOG", "1")
os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "DEBUG")
PLANNER_LOG = OUT / "planner.log"
os.environ.setdefault("SPYRE_LOG_FILE", str(PLANNER_LOG))
os.environ["SPYRE_LX_KV_LAYOUT"] = "1"

import torch  # noqa: E402
import torch_spyre  # noqa: E402
from torch_spyre._C import (  # noqa: E402
    SpyreTensorLayout,
    get_device_dtype,
    get_elem_in_stick,
)
from torch_spyre._inductor import config as ts_config  # noqa: E402

torch_spyre._autoload()
torch.spyre.set_device(0)
torch.zeros(1, dtype=torch.float16).to("spyre")

DTYPE = torch.float16
D = int(os.environ.get("HEAD_SIZE", 128))
B = int(os.environ.get("BLOCK_SIZE", 128))
Q = int(os.environ.get("Q_LEN", 512))
E = int(os.environ.get("ENTRIES", 8))
CORES = int(os.environ.get("CORES", 8))
SCALE = D**-0.5

FEATURES = {
    0: dict(permute_v=False, ngroups=1, mask2d=False, nblocks=2, tail=False),
    1: dict(permute_v=True, ngroups=1, mask2d=False, nblocks=2, tail=False),
    2: dict(permute_v=True, ngroups=4, mask2d=False, nblocks=2, tail=False),
    3: dict(permute_v=True, ngroups=4, mask2d=True, nblocks=2, tail=False),
    4: dict(permute_v=True, ngroups=4, mask2d=True, nblocks=8, tail=False),
    5: dict(permute_v=True, ngroups=4, mask2d=True, nblocks=8, tail=True),
}


def rows_outermost_layout(num_rows, block_size, head_size, dtype):
    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_rows, block_size, sticks, eps],
        stride_map=[block_size * head_size, head_size, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


def kernel(table_k, table_v, indices, q_groups, masks, permute_v, tail, entries, ngroups):
    acc_o: list = []
    acc_s: list = []
    acc_m: list = []
    for blk in range(len(indices)):
        page_k = table_k[indices[blk]].reshape(entries, B, D)
        page_v = table_v[indices[blk]].reshape(entries, B, D)
        v_t = page_v.permute(0, 2, 1) if permute_v else None
        mask = masks[blk]
        for g in range(ngroups):
            scores = torch.matmul(page_k, q_groups[g]) * SCALE
            scores = scores + mask
            m = torch.amax(scores, dim=1, keepdim=True)
            probs = torch.exp(scores - m)
            if permute_v:
                o = torch.matmul(v_t, probs)  # [E, D, Q]
                s = probs.sum(dim=1, keepdim=True)  # [E, 1, Q]
                mm = m
            else:
                o = torch.matmul(probs.transpose(1, 2), page_v)  # [E, Q, D]
                s = probs.sum(dim=1, keepdim=True).transpose(1, 2)
                mm = m.transpose(1, 2)
            if blk == 0:
                acc_o.append(o)
                acc_s.append(s)
                acc_m.append(mm)
            else:
                new_max = torch.maximum(acc_m[g], mm)
                r_old = torch.exp(acc_m[g] - new_max)
                r_new = torch.exp(mm - new_max)
                acc_o[g] = acc_o[g] * r_old + o * r_new
                acc_s[g] = acc_s[g] * r_old + s * r_new
                acc_m[g] = new_max

    groups = [acc_o[g] / acc_s[g] for g in range(ngroups)]
    if not tail:
        return groups
    attn = torch.stack(groups, dim=1)
    if permute_v:
        attn = attn.reshape(entries * ngroups, D, Q)
        return attn.permute(2, 0, 1)
    attn = attn.reshape(entries * ngroups, Q, D)
    return attn.transpose(0, 1)


def run(variant):
    f = FEATURES[variant]
    nblocks, ngroups, permute_v, mask2d, tail = (
        f["nblocks"],
        f["ngroups"],
        f["permute_v"],
        f["mask2d"],
        f["tail"],
    )
    num_rows = E * nblocks
    log_before = PLANNER_LOG.stat().st_size if PLANNER_LOG.is_file() else 0
    layout = rows_outermost_layout(num_rows, B, D, DTYPE)

    k_host = torch.randn(num_rows, B, D, dtype=DTYPE)
    v_host = torch.randn(num_rows, B, D, dtype=DTYPE)
    idx_host = [
        (torch.arange(E, dtype=torch.int32) + blk * E).reshape(E, 1) for blk in range(nblocks)
    ]
    q_host = [torch.randn(E, D, Q, dtype=DTYPE) for _ in range(ngroups)]
    mshape = (B, Q) if mask2d else (E, B, Q)
    mask_host = [torch.zeros(*mshape, dtype=DTYPE) for _ in range(nblocks)]

    prev = ts_config.sencores
    ts_config.sencores = CORES
    try:
        got = torch.compile(kernel, dynamic=False)(
            k_host.to("spyre", device_layout=layout),
            v_host.to("spyre", device_layout=layout),
            [t.to("spyre") for t in idx_host],
            [t.to("spyre") for t in q_host],
            [m.to("spyre") for m in mask_host],
            permute_v,
            tail,
            E,
            ngroups,
        )
    finally:
        ts_config.sencores = prev

    # Reference: full softmax over the concatenated token axis, per query group.
    k_cat = torch.cat([k_host[i.reshape(-1).long()] for i in idx_host], dim=1).float()
    v_cat = torch.cat([v_host[i.reshape(-1).long()] for i in idx_host], dim=1).float()
    err = 0.0
    for g in range(ngroups):
        sc = torch.matmul(k_cat, q_host[g].float()) * SCALE
        p = torch.softmax(sc, dim=1)
        ref = torch.matmul(p.transpose(1, 2), v_cat)  # [E, Q, D]
        if tail:
            mine = got.cpu().float()[:, [e * ngroups + g for e in range(E)], :]
            mine = mine.transpose(0, 1)  # [E, Q, D]
        else:
            mine = got[g].cpu().float()
            if permute_v:
                mine = mine.transpose(1, 2)  # [E, D, Q] -> [E, Q, D]
        err = max(err, (mine - ref).abs().max().item())

    text = PLANNER_LOG.read_text(errors="replace")[log_before:]
    verdicts = re.findall(r"lx_pinning: (\S+) \(index\)\s*.\s*([^\n]+)", text)
    pinned = sum(1 for _b, why in verdicts if why.strip() == "lx")
    splits = sorted(set(re.findall(r"work_slice_dims=(\(\([^)]*\)(?:, \([^)]*\))*\))", text)))
    covers = sorted(set(re.findall(r"view covers (\d+) cores but op runs (\d+)", text)))
    tag = ",".join(k for k, v in [("permV", permute_v), ("mask2d", mask2d), ("tail", tail)] if v)
    print(
        f"\n--- V{variant}  blocks={nblocks} groups={ngroups} [{tag or 'none'}]"
        f"   numerics {err:.2e}"
    )
    print(f"    pinned {pinned}/{len(verdicts)} gathers   splits={splits[:3]}")
    if covers:
        print(f"    view-vs-op: {covers}")
    return variant, pinned, len(verdicts), err


sel = os.environ.get("VARIANT")
order = [int(sel)] if sel else sorted(FEATURES)
rows = [run(v) for v in order]

print(f"\n=== SUMMARY (E={E}, cores={CORES})")
for variant, pinned, total, err in rows:
    state = "PINS" if pinned else "HBM "
    print(f"  V{variant}  {state}  {pinned}/{total}  numerics={err:.1e}")
print(
    "\nThe first variant that reads PINS names the feature the pin depends on.\n"
    "Measured 2026-09-09 (E=8, cores=8): V0 pins 0/4, V1 pins 2/4 -- the single\n"
    "difference is that V1 permutes V instead of consuming it raw. With both pages raw\n"
    "the two matmuls impose competing core divisions and NEITHER gather pins.\n"
    "V5 reproduces the full kernel exactly (8/16, split ((0,4),(1,2))).\n\n"
    "VARIANT=1 is the filable reproducer. Sweeping it:\n"
    "    E= 8 cores= 8 -> 2/4 PINS      E= 8 cores=32 -> 0/4 HBM\n"
    "    E=16 cores=16 -> 2/4 PINS      E=32 cores=32 -> 2/4 PINS\n"
    "i.e. the pin holds whenever the gather's entry count is at least the consumer's\n"
    "core count, and fails when entries < cores. That is the bug to report: a gather\n"
    "with 8 entries cannot be pinned for a consumer running on 32 cores, even though\n"
    "nothing is broadcast and the index is 2-D."
)
print(f"artifacts: {OUT}")
