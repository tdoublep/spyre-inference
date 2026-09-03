#!/usr/bin/env python3
"""LX vs HBM residency of the per-sequence paged-attention kernel.

Standalone: torch and torch_spyre only. The kernel, the KV-cache store and the
page-cache device layout are inlined below, so nothing is imported from
spyre-inference (or vLLM) and the script can be handed to torch-spyre as-is.

Reports, for every buffer the kernel allocates:

  * LX or HBM per allocation, from the generated SDSC.
  * the layout planner's per-op verdict and reason, from
    `lx_pinning: <buf> (<kind>) -> <reason>` (scratchpad.allocator at DEBUG).
  * the per-arg layouts the cost model predicted (SPYRE_DUMP_COST).
  * the HBM spill pool the kernel asks for (bundle.mlir device_mem_allocate).

The result is checked against a CPU run of the same closure.

Buffers of interest, in the kernel's own terms:
    k_page / v_page   the gathered KV page   (index_select -> permute)
    scores            q @ k_page^T
    tile_probs        exp(scores - max)      = the "P" operand of the second matmul
    tile_output       tile_probs @ v_page

The KV write is included by default (WRITE_KV=1): `index_copy_` into slot-major
views of the pages, compiled as its own graph, so one run covers the store as
well as the read side. It appears as a second kernel in the report.

Defaults mimic a granite-3.3-8b decode step: 32 query heads over 8 KV heads,
head_size 128, block_size 64, one query row, four active pages.

Env knobs:
  NUM_BLOCKS=4  Q_LEN=1  KV_HEADS=8  QPK=4  HEAD_SIZE=128  BLOCK_SIZE=64
  NUM_PAGES=32                      cache pages; sets the gather's source size,
                                    which does not move the LX verdicts
  WRITE_KV=0                        skip the KV store, read side only
  LAYOUT_SOLVER=cpsat|greedy        passed through to torch-spyre
  OUT_DIR=<dir>                     artifacts land here (default: fresh tmp dir)
  VERBOSE=1                         per-op verdicts and unclassified buffers

Each run gets a fresh TORCHINDUCTOR_CACHE_DIR: a warm cache skips compilation and
emits neither SDSC nor planner logs.
"""

import json
import os
import re
import tempfile
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR") or tempfile.mkdtemp(prefix="lx_residency_"))
OUT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT / "inductor-cache"
PLANNER_LOG = OUT / "planner.log"
COST_FILE = OUT / "cost.txt"

# Must all precede the torch import.
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_DIR)
os.environ.setdefault("TORCH_LOGS", "+torch_spyre.inductor")
os.environ.setdefault("SPYRE_INDUCTOR_LOG", "1")
os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "DEBUG")
os.environ.setdefault("SPYRE_LOG_FILE", str(PLANNER_LOG))
os.environ.setdefault("SPYRE_DUMP_COST", "1")
os.environ.setdefault("SPYRE_DUMP_COST_FILE", str(COST_FILE))

import torch  # noqa: E402
import torch_spyre  # noqa: E402


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


VERBOSE = os.environ.get("VERBOSE") == "1"
WRITE_KV = os.environ.get("WRITE_KV", "1") == "1"

KV_HEADS = _int("KV_HEADS", 8)
QPK = _int("QPK", 4)
HEAD_SIZE = _int("HEAD_SIZE", 128)
BLOCK_SIZE = _int("BLOCK_SIZE", 64)
NUM_BLOCKS = _int("NUM_BLOCKS", 4)
NUM_PAGES = _int("NUM_PAGES", 32)
Q_LEN = _int("Q_LEN", 1)

# Fold the page's row axis into the gather's indexed axis: instead of one index
# entry selecting BLOCK_SIZE rows, use FOLD_ENTRIES entries of ROWS_PER_ENTRY
# rows each, so the entry dim has enough units to be split across cores.
# FOLD=none reproduces the original one-entry-per-page gather.
FOLD = os.environ.get("FOLD", "none")
# One index entry per slot by default: the folded source is then the natural
# slot-major view [NUM_PAGES*BLOCK_SIZE, KV_HEADS, HEAD_SIZE] and the reshape
# back to the page frame is a squeeze. Fewer entries flatten rows into the head
# axis, which makes the following permute an interleaved index that
# insert_restickify_padding rejects.
ENTRIES = _int("FOLD_ENTRIES", BLOCK_SIZE)
INDEX_2D = os.environ.get("INDEX_2D", "1") == "1"
FOLD_SRC_RANK = _int("FOLD_SRC_RANK", 3)
# GQA handling. "broadcast": one matmul over a [KV_HEADS, QPK, ...] query, so
# inductor clones the page out to the group axis -- that clone runs on
# kv x group cores while the page has no group axis, which is what bars the
# page from LX. "group": unroll the groups, so each matmul reads the page
# directly and no op reads it across an axis it does not have.
GQA = os.environ.get("GQA", "broadcast")
assert GQA in ("broadcast", "group"), GQA
assert FOLD in ("none", "k", "kv"), FOLD
assert BLOCK_SIZE % ENTRIES == 0, (BLOCK_SIZE, ENTRIES)
ROWS_PER_ENTRY = BLOCK_SIZE // ENTRIES

NUM_HEADS = KV_HEADS * QPK
SCALE = HEAD_SIZE**-0.5
FP16_MIN = torch.finfo(torch.float16).min
INT32_ELEMS_PER_STICK = 32  # 128-byte stick / 4 bytes

torch_spyre._autoload()
torch.spyre.set_device(0)
torch.zeros(1, dtype=torch.float16).to("spyre")

from torch_spyre._C import (  # noqa: E402
    SpyreTensorLayout,
    get_device_dtype,
    get_elem_in_stick,
)


def fetch_page_folded(pages, entry_index):
    """One page, gathered as ENTRIES row-blocks instead of one whole-page entry.

    `pages` is viewed as [NUM_PAGES * ENTRIES, ROWS_PER_ENTRY, KV_HEADS, HEAD_SIZE]
    so the indexed axis carries ENTRIES units per page. With INDEX_2D the index is
    [ENTRIES, 1] and gathered by advanced indexing, which keeps the entry variable
    off the index's stick axis; otherwise a 1-D `index_select` is used and the
    entry dim only splits in whole 32-entry int32 sticks.

    Returns the page in its original [BLOCK_SIZE, KV_HEADS, HEAD_SIZE] frame.
    """
    if FOLD_SRC_RANK == 3:
        # Rows and heads flattened together, so a 2-D index gives a rank-4 gather
        # output. A rank-4 source (rank-5 output) makes the relayout that reads it
        # fail in dxp: "Could not find any suitable dimension mapping".
        folded = pages.view(NUM_PAGES * ENTRIES, ROWS_PER_ENTRY * KV_HEADS, HEAD_SIZE)
    else:
        folded = pages.view(NUM_PAGES * ENTRIES, ROWS_PER_ENTRY, KV_HEADS, HEAD_SIZE)
    if INDEX_2D:
        rows = folded[entry_index]
    else:
        rows = folded.index_select(0, entry_index.reshape(-1))
    return rows.reshape(BLOCK_SIZE, KV_HEADS, HEAD_SIZE)


def paged_attn_kernel(
    query, query_row_index, k_pages, v_pages, page_index_table, mask_tiles, entry_index_table
):
    """Online-softmax attention over NUM_BLOCKS pages for one sequence.

    Shapes:
        query             [num_tokens, NUM_HEADS, HEAD_SIZE], the whole batch's query
        query_row_index   int32; its first Q_LEN entries are this sequence's rows
        k_pages, v_pages  [NUM_PAGES, BLOCK_SIZE, KV_HEADS, HEAD_SIZE]
        page_index_table  [NUM_BLOCKS, INT32_ELEMS_PER_STICK] int32, page index in col 0
        mask_tiles        NUM_BLOCKS additive tiles of [Q_LEN, BLOCK_SIZE]
        entry_index_table [NUM_BLOCKS, ENTRIES, 1] int32 row-block indices, for FOLD

    NUM_BLOCKS and Q_LEN are module-level constants, so Dynamo unrolls the loop.
    """
    # Gathered, not sliced: a compiled region reads a view from offset 0.
    q_rows = query.index_select(0, query_row_index[:Q_LEN])
    q = q_rows.unsqueeze(0).transpose(1, 2).reshape(KV_HEADS, QPK, Q_LEN, HEAD_SIZE)

    tile_max = None
    tile_sum = None
    tile_output = None

    for i in range(NUM_BLOCKS):
        # index_select, not `k_pages[page_idx]`: subscripting lowers to aten.index,
        # which upcasts the int32 index to int64.
        page_idx = page_index_table[i, 0:1]
        if FOLD in ("k", "kv"):
            k_page = fetch_page_folded(k_pages, entry_index_table[i])
        else:
            k_page = k_pages.index_select(0, page_idx).squeeze(0)
        if FOLD == "kv":
            v_page = fetch_page_folded(v_pages, entry_index_table[i])
        else:
            v_page = v_pages.index_select(0, page_idx).squeeze(0)
        # Token-major page to head-major for the matmuls; permutes on device.
        k_page_4d = k_page.permute(1, 0, 2).unsqueeze(1)
        v_page_4d = v_page.permute(1, 0, 2).unsqueeze(1)

        scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * SCALE
        scores = scores + mask_tiles[i]
        scores_max = torch.amax(scores, dim=-1, keepdim=True)

        if i == 0:
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = torch.matmul(tile_probs, v_page_4d)
            tile_sum = tile_probs.sum(dim=-1, keepdim=True)
        else:
            assert tile_max is not None and tile_sum is not None and tile_output is not None
            new_max = torch.maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_output = tile_output * rescale
            tile_sum = tile_sum * rescale
            tile_probs = torch.exp(scores - new_max)
            tile_output += torch.matmul(tile_probs, v_page_4d)
            tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    assert tile_output is not None and tile_sum is not None
    attn = tile_output / tile_sum
    attn = attn.reshape(1, NUM_HEADS, Q_LEN, HEAD_SIZE).transpose(1, 2)
    return attn.reshape(Q_LEN, NUM_HEADS, HEAD_SIZE)


def paged_attn_kernel_group_loop(
    query, query_row_index, k_pages, v_pages, page_index_table, mask_tiles, entry_index_table
):
    """As :func:`paged_attn_kernel`, with the query groups unrolled.

    Each group runs its own pair of matmuls against the page as gathered, so the
    page is never expanded to a group axis and no consumer reads it across one.
    The online-softmax state becomes one entry per group.
    """
    q_rows = query.index_select(0, query_row_index[:Q_LEN])
    q = q_rows.unsqueeze(0).transpose(1, 2).reshape(KV_HEADS, QPK, Q_LEN, HEAD_SIZE)

    tile_max: list = [None] * QPK
    tile_sum: list = [None] * QPK
    tile_out: list = [None] * QPK

    for i in range(NUM_BLOCKS):
        page_idx = page_index_table[i, 0:1]
        if FOLD in ("k", "kv"):
            k_page = fetch_page_folded(k_pages, entry_index_table[i])
        else:
            k_page = k_pages.index_select(0, page_idx).squeeze(0)
        if FOLD == "kv":
            v_page = fetch_page_folded(v_pages, entry_index_table[i])
        else:
            v_page = v_pages.index_select(0, page_idx).squeeze(0)
        k_t = k_page.permute(1, 2, 0)  # [KV_HEADS, HEAD_SIZE, BLOCK_SIZE]
        v_3d = v_page.permute(1, 0, 2)  # [KV_HEADS, BLOCK_SIZE, HEAD_SIZE]

        for g in range(QPK):
            q_g = q[:, g]  # [KV_HEADS, Q_LEN, HEAD_SIZE]
            scores = torch.matmul(q_g, k_t) * SCALE
            scores = scores + mask_tiles[i]
            scores_max = torch.amax(scores, dim=-1, keepdim=True)
            if i == 0:
                tile_max[g] = scores_max
                probs = torch.exp(scores - scores_max)
                tile_out[g] = torch.matmul(probs, v_3d)
                tile_sum[g] = probs.sum(dim=-1, keepdim=True)
            else:
                new_max = torch.maximum(tile_max[g], scores_max)
                rescale = torch.exp(tile_max[g] - new_max)
                tile_out[g] = tile_out[g] * rescale
                tile_sum[g] = tile_sum[g] * rescale
                probs = torch.exp(scores - new_max)
                tile_out[g] = tile_out[g] + torch.matmul(probs, v_3d)
                tile_sum[g] = tile_sum[g] + probs.sum(dim=-1, keepdim=True)
                tile_max[g] = new_max

    attn = torch.stack([tile_out[g] / tile_sum[g] for g in range(QPK)], dim=1)
    attn = attn.reshape(1, NUM_HEADS, Q_LEN, HEAD_SIZE).transpose(1, 2)
    return attn.reshape(Q_LEN, NUM_HEADS, HEAD_SIZE)


ATTN_KERNEL = paged_attn_kernel_group_loop if GQA == "group" else paged_attn_kernel


def reshape_and_cache_kernel(key, value, k_slots, v_slots, slot_mapping):
    k_slots.index_copy_(0, slot_mapping, key)
    v_slots.index_copy_(0, slot_mapping, value)


def slot_major_layout(num_slots: int):
    """Slot-axis-outermost page layout, so the slot index stays on one device dim."""
    eps = get_elem_in_stick(torch.float16)
    sticks = (HEAD_SIZE + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_slots, KV_HEADS, sticks, eps],
        stride_map=[KV_HEADS * sticks * eps, sticks * eps, eps, 1],
        device_dtype=get_device_dtype(torch.float16),
    )


def make_pages() -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(NUM_PAGES, BLOCK_SIZE, KV_HEADS, HEAD_SIZE, dtype=torch.float16)
    v = torch.randn(NUM_PAGES, BLOCK_SIZE, KV_HEADS, HEAD_SIZE, dtype=torch.float16)
    layout = slot_major_layout(NUM_PAGES * BLOCK_SIZE)
    return k.to("spyre", device_layout=layout), v.to("spyre", device_layout=layout)


def write_kv(k_dev: torch.Tensor, v_dev: torch.Tensor) -> None:
    """The indirect store into slot-major views, as its own compiled graph. Writes
    Q_LEN tokens at the head of page 0, which the attention kernel then reads."""
    k_slots = k_dev.view(-1, KV_HEADS, HEAD_SIZE)
    v_slots = v_dev.view(-1, KV_HEADS, HEAD_SIZE)
    key = torch.randn(Q_LEN, KV_HEADS, HEAD_SIZE, dtype=torch.float16)
    value = torch.randn(Q_LEN, KV_HEADS, HEAD_SIZE, dtype=torch.float16)
    slots = torch.arange(Q_LEN, dtype=torch.int32)

    torch.compile(reshape_and_cache_kernel, dynamic=False)(
        key.to("spyre"), value.to("spyre"), k_slots, v_slots, slots.to("spyre")
    )

    k_err = k_dev.cpu().view(-1, KV_HEADS, HEAD_SIZE)[:Q_LEN].float() - key.float()
    v_err = v_dev.cpu().view(-1, KV_HEADS, HEAD_SIZE)[:Q_LEN].float() - value.float()
    print(
        f"KV store: k err {k_err.abs().max().item():.3e}  v err {v_err.abs().max().item():.3e}",
        flush=True,
    )


def build_attn_args(k_dev: torch.Tensor, v_dev: torch.Tensor):
    """(device args, cpu reference args) for paged_attn_kernel."""
    # More query rows than this sequence uses, so the in-graph gather selects a
    # strict subset rather than its whole source.
    num_tokens = max(2 * Q_LEN, 8)
    query = torch.randn(num_tokens, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
    row_index = torch.arange(Q_LEN, dtype=torch.int32)
    page_table = torch.zeros(NUM_BLOCKS, INT32_ELEMS_PER_STICK, dtype=torch.int32)
    page_table[:, 0] = torch.arange(NUM_BLOCKS, dtype=torch.int32)
    masks = [torch.zeros(Q_LEN, BLOCK_SIZE, dtype=torch.float16) for _ in range(NUM_BLOCKS)]
    # Half of the last block masked off, as a real tail block would be.
    masks[-1][:, BLOCK_SIZE // 2 :] = FP16_MIN
    # Row-block indices for the folded gather, built on the host: entry e of
    # block i is row-block page_table[i] * ENTRIES + e. Trailing size-1 axis so
    # the entry variable is not the index's stick axis.
    entry_table = (
        page_table[:, 0:1] * ENTRIES + torch.arange(ENTRIES, dtype=torch.int32)
    ).reshape(NUM_BLOCKS, ENTRIES, 1)

    dev_args = [
        query.to("spyre"),
        row_index.to("spyre"),
        k_dev,
        v_dev,
        page_table.to("spyre"),
        [m.to("spyre") for m in masks],
        entry_table.to("spyre"),
    ]
    cpu_args = [
        query.float(),
        row_index,
        k_dev.cpu().float(),
        v_dev.cpu().float(),
        page_table,
        [m.float() for m in masks],
        entry_table,
    ]
    return dev_args, cpu_args


def walk_allocations(node, acc: list) -> None:
    """Every (name, component) pair from `allocate` schedule-tree nodes."""
    if isinstance(node, dict):
        if node.get("nodeType_") == "allocate":
            acc.append((node.get("name_", "?"), node.get("component_", "?")))
        for v in node.values():
            walk_allocations(v, acc)
    elif isinstance(node, list):
        for v in node:
            walk_allocations(v, acc)


def sdsc_report() -> None:
    """Per-SDSC allocation verdicts. One line per op; the SDSC's own top-level key
    names the op, since the Tensor<N> labels are per-kernel and anonymous."""
    sdsc_root = CACHE_DIR / "inductor-spyre"
    dirs = sorted(sdsc_root.glob("*_sdsc_*")) if sdsc_root.is_dir() else []
    print(f"\n=== SDSC allocations — {len(dirs)} kernel(s) ===")
    if not dirs:
        print("no SDSC emitted: nothing compiled (a warm inductor cache?)")
        return
    for d in dirs:
        pool = ""
        bundle = d / "bundle.mlir"
        if bundle.is_file():
            m = re.search(r"device_mem_allocate (\d+) bytes", bundle.read_text())
            if m:
                pool = f"  HBM spill pool {int(m.group(1)) / 1024:.1f} KB"
        print(f"\n{d.name.split('_sdsc_')[1][:70]}{pool}")
        totals = {"lx": 0, "hbm": 0}
        for js in sorted(d.glob("sdsc_*.json"), key=lambda p: int(p.stem.split("_")[1])):
            blob = json.loads(js.read_text())
            op_name = next(iter(blob), "?")
            allocs: list = []
            walk_allocations(blob, allocs)
            lx = [n for n, c in allocs if c == "lx" or n.endswith("_lx")]
            hbm = [n for n, c in allocs if not (c == "lx" or n.endswith("_lx"))]
            totals["lx"] += len(lx)
            totals["hbm"] += len(hbm)
            spill = " HBM: " + ",".join(n.removeprefix("allocate-") for n in hbm) if hbm else ""
            print(f"  {op_name:<28} LX={len(lx)} HBM={len(hbm)}{spill}")
        print(f"  {'TOTAL':<28} LX={totals['lx']} HBM={totals['hbm']}")


ROLE_BY_KIND = {
    "index": "page gather (k_page / v_page)",
    "clone": "GQA broadcast of the page",
    "expand": "GQA broadcast of the page",
    "batched_matmul": "q @ k_page^T  and  tile_probs @ v_page",
    "exp": "tile_probs (the P operand)",
    "amax": "row max",
    "sum": "row sum",
    "index_put_": "KV cache store",
    "restickify": "relayout copy",
}


def parse_verdicts() -> list[list[tuple[str, str, str]]]:
    """Per-graph blocks of (buf, kind, verdict) from the planner log.

    `lx_pinning` fires once per op at the end of layout planning, so the lines
    arrive in contiguous runs, one run per compiled graph.
    """
    if not PLANNER_LOG.is_file():
        return []
    pat = re.compile(r"lx_pinning: (\S+) \((\S+)\) . (.*)")
    blocks: list[list[tuple[str, str, str]]] = []
    current: list[tuple[str, str, str]] = []
    for ln in PLANNER_LOG.read_text(errors="replace").splitlines():
        m = pat.search(ln)
        if m:
            current.append((m.group(1), m.group(2), m.group(3).strip()))
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def verdict_report() -> None:
    """LX vs not, aggregated by op kind, with the planner's refusal reasons."""
    blocks = parse_verdicts()
    if not blocks:
        print("\n(no planner verdicts: scratchpad.allocator did not log at DEBUG)")
        return
    for block in blocks:
        store = any(k == "index_put_" for _, k, _ in block)
        label = "KV store graph" if store else "attention graph"
        print(f"\n=== LX verdicts by op kind — {label} ({len(block)} ops) ===")
        kinds: dict[str, dict] = {}
        for _, kind, verdict in block:
            e = kinds.setdefault(kind, {"lx": 0, "no": 0, "why": []})
            if verdict == "lx":
                e["lx"] += 1
            else:
                e["no"] += 1
                short = verdict.split(":")[0] if "PerCoreView" in verdict else verdict
                if short not in e["why"]:
                    e["why"].append(short)
        for kind, e in sorted(kinds.items(), key=lambda kv: -kv[1]["no"]):
            role = ROLE_BY_KIND.get(kind, "")
            why = ("  <- " + "; ".join(e["why"])) if e["why"] else ""
            print(f"  {kind:<16} LX {e['lx']:>2}   not-LX {e['no']:>2}   {role}{why}")
        if VERBOSE:
            for buf, kind, verdict in block:
                print(f"    {buf:<8} {kind:<16} {verdict[:150]}")


def classify(shape: list[int]) -> str:
    """Name the kernel-level role of a buffer from its torch shape."""
    q, b, d, kv, h = Q_LEN, BLOCK_SIZE, HEAD_SIZE, KV_HEADS, NUM_HEADS
    roles = {
        (NUM_PAGES, b, kv, d): "page cache (arg)",
        (NUM_PAGES * b, kv, d): "page cache, slot-major view",
        (1, b, kv, d): "k_page / v_page  <- the gather",
        (ENTRIES, 1, ROWS_PER_ENTRY, kv, d): "k_page / v_page  <- the folded gather",
        (ENTRIES, ROWS_PER_ENTRY, kv, d): "k_page / v_page  <- the folded gather",
        (ENTRIES, 1, ROWS_PER_ENTRY * kv, d): "k_page / v_page  <- the folded gather",
        (ENTRIES, ROWS_PER_ENTRY * kv, d): "k_page / v_page  <- the folded gather",
        (b, kv, d): "k_page / v_page, row-major",
        (kv, QPK, d, b): "K clone, GQA-expanded + transposed for q @ K^T",
        (kv, QPK, b, d): "V clone, GQA-expanded",
        (kv, 1, b, d): "page, head-major",
        (q, h, d): "query rows",
        (1, h, d): "query rows",
        (kv, QPK, q, b): "scores / tile_probs  (the P operand)",
        (kv, QPK, q, d): "tile_output",
        (kv, QPK, q, 1): "tile_max / tile_sum",
        (kv, d, b): "k_page, head-major transposed for q @ K^T",
        (kv, b, d): "v_page, head-major",
        (kv, q, b): "scores / tile_probs  (the P operand)",
        (kv, q, d): "tile_output",
        (kv, q, 1): "tile_max / tile_sum",
    }
    role = roles.get(tuple(shape), "")
    if role and b == d and tuple(shape) == (kv, QPK, q, d):
        # BLOCK_SIZE == HEAD_SIZE makes the scores and tile_output shapes identical.
        return "scores / tile_probs / tile_output (shapes collide at head_size == block_size)"
    return role


def role_report() -> None:
    """Residency by kernel-level role, joined from the cost dump's per-op lines."""
    if not COST_FILE.is_file():
        return
    pat = re.compile(r"output\s+(\S+)\s+torch \[([\d, ]+)\] -> device \[[\d, ]+\] in (LX|HBM)")
    rows = []
    for ln in COST_FILE.read_text(errors="replace").splitlines():
        m = pat.search(ln)
        if m:
            shape = [int(x) for x in m.group(2).split(",")]
            rows.append((m.group(1), shape, m.group(3), classify(shape)))
    print(f"\n=== residency by role ({len(rows)} produced buffers) ===")
    for op, shape, where, role in rows:
        if role or VERBOSE:
            print(f"  {where:<4} {op:<8} {str(shape):<22} {role}")
    named = [r for r in rows if r[3]]
    print(f"\n  LX  roles: {', '.join(sorted({r[3] for r in named if r[2] == 'LX'})) or '-'}")
    print(f"  HBM roles: {', '.join(sorted({r[3] for r in named if r[2] == 'HBM'})) or '-'}")


def store_report() -> None:
    """Mutation relayout copies torch-spyre inserted for the store's destination."""
    if not PLANNER_LOG.is_file():
        return
    copies = [
        ln
        for ln in PLANNER_LOG.read_text(errors="replace").splitlines()
        if "mutation relayout copy" in ln
    ]
    print(f"\n=== KV store relayout copies ({len(copies)}) ===")
    for ln in copies:
        print("  " + ln.split("]", 2)[-1].strip())


print(
    f"blocks={NUM_BLOCKS} q_len={Q_LEN} kv_heads={KV_HEADS} qpk={QPK} "
    f"head_size={HEAD_SIZE} block_size={BLOCK_SIZE} "
    f"write_kv={WRITE_KV} solver={os.environ.get('LAYOUT_SOLVER', 'default')} "
    f"fold={FOLD} entries={ENTRIES} rows_per_entry={ROWS_PER_ENTRY} index_2d={INDEX_2D} "
    f"fold_src_rank={FOLD_SRC_RANK} gqa={GQA} "
    f"fix4258={os.environ.get('SPYRE_FIX_4258', '0')}",
    flush=True,
)
print(f"artifacts: {OUT}", flush=True)

k_dev, v_dev = make_pages()
if WRITE_KV:
    write_kv(k_dev, v_dev)

dev_args, cpu_args = build_attn_args(k_dev, v_dev)
# Independent check that the two kernel shapes agree on CPU, so a restructuring
# bug cannot hide inside the device-vs-CPU comparison of a single closure.
xcheck = (paged_attn_kernel(*cpu_args) - paged_attn_kernel_group_loop(*cpu_args)).abs()
print(f"cpu cross-check broadcast vs group-loop: max abs diff {xcheck.max().item():.3e}")

got = torch.compile(ATTN_KERNEL, dynamic=False)(*dev_args).cpu().float()
ref = ATTN_KERNEL(*cpu_args)

diff = (got - ref).abs().max().item()
denom = ref.abs().max().item() or 1.0
print(f"\nattention: max abs diff vs CPU {diff:.3e}  (rel {diff / denom:.3e})")
print("NUMERICS OK" if diff / denom < 2e-2 else "NUMERICS SUSPECT — do not trust the layout")

sdsc_report()
verdict_report()
role_report()
store_report()
print(f"\nartifacts: {OUT}")
