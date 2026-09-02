"""Record every LX-residency decision torch-spyre makes for a compiled graph.

Residency is decided in two places, and a buffer must survive both to actually
live in scratchpad:

1. ``ScratchpadAllocator.plan_allocation`` grants an LX address or records a
   spill reason. Reasons are either *declared* (a structural veto such as
   "index tensor or indirectly accessed" or "read by restickify (cross-frame
   barrier)") or *capacity* ("no room on scratchpad (t=a-b, size=N KB)"). The
   distinction picks the fix: a declared veto needs a lowering/layout change, a
   capacity loss needs a shorter live range or a smaller working set.
2. ``demote_incoherent_lx_buffers`` runs post-fusion and can revoke an address
   the allocator already granted, when a producer's loop order disagrees with
   its consumers'. A buffer demoted here looks pinned at stage 1 and still ends
   up in HBM.

The report pairs those verdicts with a structural K/V lane classification so
the paged-attention question ("do the K and V page tiles stay resident?") is
answered per lane rather than per anonymous buffer name.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
from collections import defaultdict

_MATMUL_NAMES = ("matmul", "bmm", "mm", "batchmatmul")
_DEMOTE_RE = re.compile(r"demoted (\S+) out of LX: (.*)")

_graphs: list[dict] = []
_demotions: list[dict] = []
_pending_allocation: list = []
_out_path: str | None = None


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


class _DemotionCollector(logging.Handler):
    def emit(self, record):
        m = _DEMOTE_RE.search(record.getMessage())
        if m:
            _demotions.append({"buffer": m.group(1), "reason": m.group(2)})


def install(out_path: str) -> None:
    """Patch the allocator and start collecting. Idempotent per process."""
    global _out_path
    if _out_path is not None:
        return
    _out_path = out_path

    from torch_spyre._inductor.scratchpad import allocator as A

    base = A.ScratchpadAllocator

    # _get_spill_reasons is where the solved allocation is still in hand; stash it
    # so _log_lx_pinning (which only receives graph + reasons) can report sizes
    # and lifetimes. Subclasses override it, so patch each definition.
    for cls in [base, *_subclasses(base)]:
        orig = cls.__dict__.get("_get_spill_reasons")
        if orig is None:
            continue

        def make(orig):
            def patched(self, solver, allocation):
                reasons = orig(self, solver, allocation)
                _pending_allocation[:] = list(allocation)
                return reasons

            return patched

        cls._get_spill_reasons = make(orig)

    orig_log = base._log_lx_pinning

    def log_patched(self, graph, reasons):
        try:
            _graphs.append(_build_report(graph, reasons, list(_pending_allocation)))
        except Exception as exc:  # never break a compile over instrumentation
            _graphs.append({"error": f"{type(exc).__name__}: {exc}"})
        _pending_allocation.clear()
        return orig_log(self, graph, reasons)

    base._log_lx_pinning = log_patched

    handler = _DemotionCollector()
    handler.setLevel(logging.INFO)
    sched_logger = logging.getLogger("spyre.inductor.scheduler")
    sched_logger.addHandler(handler)
    if sched_logger.level > logging.INFO or sched_logger.level == 0:
        sched_logger.setLevel(logging.INFO)

    atexit.register(write_report)


def write_report() -> None:
    if _out_path is None:
        return
    for g in _graphs:
        by_name = {b["name"]: b for b in g.get("buffers", [])}
        for d in _demotions:
            if d["buffer"] in by_name:
                by_name[d["buffer"]]["demoted"] = d["reason"]
    payload = {"graphs": _graphs, "demotions": _demotions}
    os.makedirs(os.path.dirname(os.path.abspath(_out_path)) or ".", exist_ok=True)
    with open(_out_path, "w") as f:
        json.dump(payload, f, indent=2)


def _build_report(graph, reasons: dict, allocation: list) -> dict:
    from torch_spyre._inductor.pass_utils import op_read_writes, op_short_name
    from torch_spyre._inductor.scratchpad.utils import calculate_liveness

    inputs = dict(graph.graph_inputs)
    liveness = calculate_liveness(graph)
    alloc_by_name = {b.name: b for b in allocation}

    order = {}
    reads: dict[str, list[str]] = {}
    shorts: dict[str, str] = {}
    for i, op in enumerate(graph.operations):
        name = op.name
        order[name] = i
        shorts[name] = op_short_name(op)
        try:
            reads[name] = sorted({d.name for d in op_read_writes(op).reads})
        except Exception:
            reads[name] = []

    deriv_cache: dict[str, frozenset] = {}

    def deriv(name: str) -> frozenset:
        """Transitive read-closure of ``name``, including ``name`` itself."""
        if name in deriv_cache:
            return deriv_cache[name]
        deriv_cache[name] = frozenset()  # cycle guard
        acc = {name}
        for r in reads.get(name, []):
            acc |= deriv(r)
        out = frozenset(acc)
        deriv_cache[name] = out
        return out

    def is_matmul(name: str) -> bool:
        s = shorts.get(name, "")
        return any(k in s for k in _MATMUL_NAMES)

    exp_bufs = {n for n, s in shorts.items() if s.startswith("exp")}

    # The KV caches are the largest identically-sized *pair* of graph inputs.
    # Deliberately not "inputs read by a gather": the cache is reached through a
    # restickify first, so the gather's read dep names that copy, not the input.
    by_size: dict[int, list[str]] = defaultdict(list)
    for n, b in inputs.items():
        sz = _input_size(b)
        if sz:
            by_size[sz].append(n)
    kv_roots: set[str] = set()
    for sz in sorted(by_size, reverse=True):
        if len(by_size[sz]) == 2:
            kv_roots = set(by_size[sz])
            break

    lanes: dict[str, dict] = {}
    for name in sorted(order, key=order.get):
        if not is_matmul(name):
            continue
        operands = [r for r in reads.get(name, []) if r in order or r in inputs]
        if len(operands) < 2:
            continue
        # P@V is the matmul whose other operand comes out of the softmax exp.
        via_exp = [o for o in operands if deriv(o) & exp_bufs]
        page_side = [o for o in operands if o not in via_exp and (deriv(o) & kv_roots)]
        if not page_side:
            continue
        lane = "V" if via_exp else "K"
        for operand in page_side:
            roots = deriv(operand) & kv_roots
            lanes.setdefault(lane, {"matmuls": [], "roots": set(), "operands": []})
            lanes[lane]["matmuls"].append(name)
            lanes[lane]["roots"] |= roots
            lanes[lane]["operands"].append(operand)

    # Cross-check: the two lanes must name different cache inputs. If the exp
    # test mislabelled anything, the roots collide and the report says so.
    k_roots = lanes.get("K", {}).get("roots", set())
    v_roots = lanes.get("V", {}).get("roots", set())
    lane_conflict = bool(k_roots & v_roots) or not (k_roots and v_roots)

    def slice_back(operand: str) -> list[str]:
        return sorted(
            (b for b in deriv(operand) if b in order),
            key=lambda b: order[b],
        )

    def verdict(name: str) -> tuple[str, str]:
        reason = reasons.get(name)
        if reason is None or reason == "lx":
            return "lx", ""
        return "hbm", reason

    # How each user slices each buffer across cores. A buffer is refused LX when
    # its users disagree here, so printing both sides names the fix directly.
    divisions: dict[str, list[dict]] = defaultdict(list)
    try:
        from torch_spyre._inductor.pass_utils import _per_core_view_on_buf
        from torch_spyre._inductor.scratchpad.utils import _get_buffer_user_deps

        cache: dict = {}
        for buf_name, users in _get_buffer_user_deps(graph).items():
            for op, dep in users:
                try:
                    view, partial, representable = _per_core_view_on_buf(
                        op, dep, buf_name, cache
                    )
                except Exception as exc:
                    divisions[buf_name].append({"op": op.get_name(), "error": repr(exc)})
                    continue
                is_write = dep in op_read_writes(op).writes
                divisions[buf_name].append(
                    {
                        "op": op.get_name(),
                        "role": "write" if is_write else "read",
                        "work_slice_dims": [
                            [int(d), int(f)] for d, f in view.work_slice_dims
                        ],
                        "partial_reduction": bool(partial),
                        "representable": bool(representable),
                    }
                )
    except Exception as exc:
        divisions["__error__"].append({"error": f"{type(exc).__name__}: {exc}"})

    buffers = []
    for name in sorted(order, key=order.get):
        v, reason = verdict(name)
        b = alloc_by_name.get(name)
        buffers.append(
            {
                "name": name,
                "op": shorts.get(name, "?"),
                "verdict": v,
                "reason": reason,
                "size_bytes": getattr(b, "size", None),
                "address": getattr(b, "address", None),
                "lifetime": liveness.get(name, []),
                "reads": reads.get(name, []),
                "divisions": divisions.get(name, []),
                "shape": _op_shape(graph, name),
            }
        )

    lane_out = {}
    for lane, info in lanes.items():
        chains = []
        for operand in info["operands"]:
            chains.append({"operand": operand, "chain": slice_back(operand)})
        lane_out[lane] = {
            "roots": sorted(info["roots"]),
            "num_tiles": len(info["operands"]),
            "chains": chains,
        }

    return {
        "num_ops": len(graph.operations),
        "graph_inputs": {
            n: {"size": _input_size(b)} for n, b in inputs.items()
        },
        "kv_roots": sorted(kv_roots),
        "lane_conflict": lane_conflict,
        "op_kinds": sorted(set(shorts.values())),
        "lanes": lane_out,
        "buffers": buffers,
    }


def _op_shape(graph, name: str) -> list[int] | None:
    """The buffer's device shape -- the axes the ``dimN/f`` splits refer to."""
    try:
        buf = graph.try_get_buffer(name)
        layout = buf.get_layout()
        dev = getattr(layout, "device_layout", None)
        size = getattr(dev, "size", None) or layout.size
        return [int(s) for s in size]
    except Exception:
        return None


def _input_size(buf) -> int | None:
    try:
        return int(buf.get_layout().storage_size())
    except Exception:
        try:
            n = 1
            for d in buf.get_size():
                n *= int(d)
            return n
        except Exception:
            return None
