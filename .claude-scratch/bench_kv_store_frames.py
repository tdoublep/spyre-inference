"""Decode KV store cost for the three cache frames, measured not argued.

A: slot-major   [P*B, KV, D]        index_copy_ on 1 slot row      (today's frame)
B: V head-major [P*KV*B, D]         index_copy_ on KV rows         (V's folded frame)
C: K transposed [P*KV, D, B]        column write via index_put_    (K's folded frame)

Reports, per variant: compiles, numerics vs CPU, the mutation relayout copies
torch-spyre inserted, and the kernel's HBM pool from bundle.mlir.
"""

import os
import re
import tempfile

OUT = tempfile.mkdtemp(prefix="store_")
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

P, B, KV, D = 32, 64, 8, 128
EPS = get_elem_in_stick(torch.float16)
PAGE, TOK = 5, 9  # write token TOK of page PAGE


def rows_outermost(rows, mid, inner):
    return SpyreTensorLayout(
        device_size=[rows, mid, (inner + EPS - 1) // EPS, EPS],
        stride_map=[mid * inner, inner, EPS, 1],
        device_dtype=get_device_dtype(torch.float16),
    )


def two_d_rows(rows, width):
    return SpyreTensorLayout(
        device_size=[rows, (width + EPS - 1) // EPS, EPS],
        stride_map=[width, EPS, 1],
        device_dtype=get_device_dtype(torch.float16),
    )


def store_a(cache, idx, src):  # [P*B, KV, D], src [1, KV, D]
    cache.index_copy_(0, idx, src)


def store_b(cache, idx, src):  # [P*KV*B, D], src [KV, D]
    cache.index_copy_(0, idx, src)


def store_c(cache, rows, col, src):  # [P*KV, D, B], src [KV, D]
    cache[rows, :, col] = src


def store_c2(cache, rows, d_idx, col, src):  # explicit 3-index index_put_
    cache.index_put_((rows.reshape(-1, 1), d_idx.reshape(1, -1), col.reshape(-1, 1)), src)


def store_c3(cache, rows, col, src):  # read-modify-write whole tiles, then row copy
    tile = cache.index_select(0, rows)          # [KV, D, B]
    tile = tile.index_copy(2, col, src.unsqueeze(-1))
    cache.index_copy_(0, rows, tile)


def pool_bytes():
    root = os.path.join(OUT, "cache", "inductor-spyre")
    best = 0
    for d in os.listdir(root) if os.path.isdir(root) else []:
        f = os.path.join(root, d, "bundle.mlir")
        if os.path.isfile(f):
            m = re.search(r"device_mem_allocate (\d+) bytes", open(f).read())
            if m:
                best = max(best, int(m.group(1)))
    return best


def copies_since(mark):
    if not os.path.isfile(LOG):
        return 0, mark
    text = open(LOG, errors="replace").read()
    n = text[mark:].count("mutation relayout copy")
    return n, len(text)


mark = 0
key = torch.randn(KV, D, dtype=torch.float16)

# ---- A: slot-major, one row per token -------------------------------------
base = torch.randn(P * B, KV, D, dtype=torch.float16)
a_dev = base.to("spyre", device_layout=rows_outermost(P * B, KV, D))
idx_a = torch.tensor([PAGE * B + TOK], dtype=torch.int32)
src_a = key.reshape(1, KV, D)
try:
    torch.compile(store_a, dynamic=False)(a_dev, idx_a.to("spyre"), src_a.to("spyre"))
    got = a_dev.cpu()[PAGE * B + TOK].float()
    err = (got - key.float()).abs().max().item()
    n, mark = copies_since(mark)
    print(f"A slot-major   err {err:.3e}  relayout copies {n}  pool {pool_bytes()/1024:.1f} KB")
except Exception as e:
    n, mark = copies_since(mark)
    print(f"A slot-major   FAILED {type(e).__name__}: {str(e)[:110]}")

# ---- B: V head-major, KV contiguous rows ---------------------------------
vb = torch.randn(P * KV * B, D, dtype=torch.float16)
b_dev = vb.to("spyre", device_layout=two_d_rows(P * KV * B, D))
rows_b = torch.tensor([(PAGE * KV + kv) * B + TOK for kv in range(KV)], dtype=torch.int32)
try:
    torch.compile(store_b, dynamic=False)(b_dev, rows_b.to("spyre"), key.to("spyre"))
    got = b_dev.cpu().index_select(0, rows_b.long()).float()
    err = (got - key.float()).abs().max().item()
    n, mark = copies_since(mark)
    print(f"B V head-major err {err:.3e}  relayout copies {n}  pool {pool_bytes()/1024:.1f} KB")
except Exception as e:
    n, mark = copies_since(mark)
    print(f"B V head-major FAILED {type(e).__name__}: {str(e)[:110]}")

# ---- C: K transposed, column write ---------------------------------------
kc = torch.randn(P * KV, D, B, dtype=torch.float16)
c_dev = kc.to("spyre", device_layout=rows_outermost(P * KV, D, B))
rows_c = torch.tensor([PAGE * KV + kv for kv in range(KV)], dtype=torch.int32)
col_c = torch.tensor([TOK], dtype=torch.int32)
d_idx = torch.arange(D, dtype=torch.int32)
for label, fn in (("C1 advanced-idx", store_c), ("C2 index_put_", store_c2),
                  ("C3 tile RMW", store_c3)):
    fresh = kc.clone().to("spyre", device_layout=rows_outermost(P * KV, D, B))
    args = (
        (fresh, rows_c.to("spyre"), d_idx.to("spyre"), col_c.to("spyre"), key.to("spyre"))
        if fn is store_c2
        else (fresh, rows_c.to("spyre"), col_c.to("spyre"), key.to("spyre"))
    )
    try:
        torch.compile(fn, dynamic=False)(*args)
        got = fresh.cpu()[rows_c.long()][:, :, TOK].float()
        err = (got - key.float()).abs().max().item()
        n, mark = copies_since(mark)
        ok = "OK " if err < 1e-2 else "BAD"
        print(f"{label:<16} {ok} err {err:.3e}  relayout copies {n}  pool {pool_bytes()/1024:.1f} KB")
    except Exception as e:
        n, mark = copies_since(mark)
        print(f"{label:<16} FAILED {type(e).__name__}: {str(e)[:100]}")
