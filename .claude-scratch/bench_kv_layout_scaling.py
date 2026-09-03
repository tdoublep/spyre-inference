"""Does gather/scatter cost scale with total cache size, or with elements moved?

The failure mode we are testing for: on Spyre a gather whose indexed axis is not at
device position 0 costs O(whole source tensor), so its time is flat in index count and
linear in cache size. Same worry for the scatter destination.

Sweep NUM_PAGES (total cache bytes) with the moved-element count held FIXED, and
compare each variant's time at the largest cache against its time at the smallest.
Flat => cost tracks elements moved. Rising ~linearly => the pathology.

  G1  gather, slot-major   [P, B, KV, D]      index_select(0, page)     one page
  G2  gather, head-major   [P*KV, B, D]       c[kv_idx_2d]              one page
  S1  scatter, slot-major  [P*B, KV, D]       index_copy_(0, slot)      one token
  S2  scatter, head-major  [P*KV*B, D]        index_copy_(0, rows8)     one token

There is no torch.spyre.synchronize, so each variant queues N launches and then forces
the queue to drain with a small readback; the readback is the same size at every P, so
it is a constant offset and cannot manufacture a slope.
"""

import os
import statistics
import tempfile
import time

os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="scal_")

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

B, KV, D = 64, 8, 128
EPS = get_elem_in_stick(torch.float16)
PAGES = [int(x) for x in os.environ.get("PAGES", "8,64,512").split(",")]
N = int(os.environ.get("N", 50))
REPS = int(os.environ.get("REPS", 3))


def layout(dims):
    """Rows-outermost stick layout: indexed axis stays at device position 0."""
    strides, acc = [], 1
    for d in reversed(dims):
        strides.append(acc)
        acc *= d
    strides.reverse()  # contiguous host strides
    dev = list(dims[:-1]) + [(dims[-1] + EPS - 1) // EPS, EPS]
    return SpyreTensorLayout(
        device_size=dev,
        stride_map=strides[:-1] + [EPS, 1],
        device_dtype=get_device_dtype(torch.float16),
    )


def timed(fn, args, probe, name=None):
    """Median over REPS of the mean per-launch wall time of N queued launches."""
    if name is not None and name not in VARIANTS:
        return float("nan")
    # fullgraph=True so a graph break raises instead of silently running eager.
    compiled = torch.compile(fn, dynamic=False, fullgraph=True)
    for _ in range(3):
        compiled(*args)
    probe().cpu()
    # Differential timing: (time for 2N launches) - (time for N launches), divided
    # by N. Any constant per-measurement overhead cancels -- notably the sync probe,
    # whose .cpu() on a slice of the destination may materialise the whole cache and
    # would otherwise masquerade as cache-size scaling.
    def batch(count):
        t0 = time.perf_counter()
        for _ in range(count):
            compiled(*args)
        probe().cpu()
        return time.perf_counter() - t0

    out = []
    for _ in range(REPS):
        t1 = batch(N)
        t2 = batch(2 * N)
        out.append((t2 - t1) / N)
    return statistics.median(out)


def g1(cache, idx):
    return cache.index_select(0, idx)


def g2(cache, idx1d):
    return cache.index_select(0, idx1d)


def s1(cache, idx, src):
    cache.index_copy_(0, idx, src)


def s2(cache, idx, src):
    cache.index_copy_(0, idx, src)


_ALL = ("G0", "G1", "G2", "S1", "S2")
VARIANTS = tuple(
    v for v in _ALL if v in os.environ.get("ONLY", ",".join(_ALL)).split(",")
)
results: dict[str, dict[int, float]] = {k: {} for k in _ALL}

for P in PAGES:
    mb = P * B * KV * D * 2 / 2**20
    print(f"\n=== NUM_PAGES={P}  cache {mb:.0f} MB per tensor ===", flush=True)
    page = P // 2

    # G0: POSITIVE CONTROL -- default stickification, which relocates the indexed
    # axis away from device position 0. If the benchmark cannot show a slope here,
    # it cannot be trusted to show its absence elsewhere.
    c0 = torch.randn(P * B, KV, D, dtype=torch.float16).to("spyre")
    idx0 = torch.tensor([page * B], dtype=torch.int32).to("spyre")
    results["G0"][P] = timed(g1, (c0, idx0), lambda: g1(c0, idx0), name="G0")

    # G1: slot-major page gather
    c = torch.randn(P, B, KV, D, dtype=torch.float16).to(
        "spyre", device_layout=layout([P * B, KV, D])
    )
    idx = torch.tensor([page], dtype=torch.int32).to("spyre")
    results["G1"][P] = timed(g1, (c, idx), lambda: g1(c, idx), name="G1")

    # G2: head-major page gather, (page, kv) entries
    ch = torch.randn(P * KV, B, D, dtype=torch.float16).to(
        "spyre", device_layout=layout([P * KV, B, D])
    )
    i2 = (page * KV + torch.arange(KV, dtype=torch.int32)).to("spyre")
    results["G2"][P] = timed(g2, (ch, i2), lambda: g2(ch, i2), name="G2")

    # S1: slot-major token scatter
    cs = torch.randn(P * B, KV, D, dtype=torch.float16).to(
        "spyre", device_layout=layout([P * B, KV, D])
    )
    si = torch.tensor([page * B + 3], dtype=torch.int32).to("spyre")
    src1 = torch.randn(1, KV, D, dtype=torch.float16).to("spyre")
    results["S1"][P] = timed(s1, (cs, si, src1), lambda: cs[0:1], name="S1")

    # S2: head-major token scatter
    cs2 = torch.randn(P * KV * B, D, dtype=torch.float16).to(
        "spyre", device_layout=layout([P * KV * B, D])
    )
    rows = torch.tensor(
        [(page * KV + kv) * B + 3 for kv in range(KV)], dtype=torch.int32
    ).to("spyre")
    src2 = torch.randn(KV, D, dtype=torch.float16).to("spyre")
    results["S2"][P] = timed(s2, (cs2, rows, src2), lambda: cs2[0:1], name="S2")

    for k in VARIANTS:
        print(f"  {k}  {results[k][P] * 1e6:9.1f} us", flush=True)

def sdsc_kernels():
    root = os.path.join(os.environ["TORCHINDUCTOR_CACHE_DIR"], "inductor-spyre")
    return len([d for d in os.listdir(root) if "_sdsc_" in d]) if os.path.isdir(root) else 0


print(f"\nSDSC kernels emitted: {sdsc_kernels()}  (0 would mean nothing compiled)")

print("\n=== scaling: us per launch, moved bytes held constant ===")
hdr = "  var  " + "".join(f"{P:>12}" for P in PAGES) + "     ratio(max/min)"
print(hdr)
def mb(P):
    return P * B * KV * D * 2 / 2**20


for k in VARIANTS:
    row = "".join(f"{results[k][P] * 1e6:11.1f}u" for P in PAGES)
    # Least-squares fit of time = fixed + slope * cache_MB. The ~130 us per-kernel
    # launch cost floors every measurement, so the slope is the real signal.
    xs = [mb(P) for P in PAGES]
    ys = [results[k][P] * 1e6 for P in PAGES]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
    fixed = my - slope * mx
    print(f"  {k}  {row}   fixed {fixed:6.0f}us  slope {slope:7.2f} us/MB")
print(f"\ncache spans {mb(PAGES[0]):.0f}-{mb(PAGES[-1]):.0f} MB with moved bytes fixed;"
      f" slope ~0 means cost tracks elements moved, not cache size")
