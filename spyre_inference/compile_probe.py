# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attribute torch.compile work to a server lifecycle phase.

A compile that lands after warmup is a latency bug: the request that triggers it
waits out a full Inductor build. Nothing here fails on one -- it records what
compiled and how long it took, so a ``vllm serve`` run can be checked after.

Dynamo already counts its own work; this only splits those counters at the
warmup/serving boundary, which Dynamo has no concept of. ``counters`` and
``cumulative_time_spent_ns`` are the ground truth, differenced per phase --
notably ``inductor.fxgraph_cache_miss``, which separates a re-trace that hit the
Inductor cache from a full build. ``on_compile_start``/``on_compile_end`` are
used only to raise the per-event alarm, since a counter delta alone can't say
which call site paid for it.

For the same question without a code change, ``TORCH_LOGS=recompiles_verbose``
names the guard that failed and ``TORCH_TRACE=<dir>`` records every compile for
``tlparse``. Both are noisy about *when*, which is what the phase split adds.
"""

from __future__ import annotations

import functools
import json
import os
import threading
import time
import traceback
from typing import Any

from vllm.logger import init_logger

from spyre_inference import envs

logger = init_logger(__name__)

STARTUP = "startup"
WARMUP = "warmup"
SERVING = "serving"

# The dynamo counters worth reporting; the rest are noise for this question.
_COUNTERS = (
    ("frames", "total"),
    ("frames", "ok"),
    ("stats", "unique_graphs"),
    ("inductor", "fxgraph_cache_miss"),
    ("inductor", "fxgraph_cache_hit"),
    ("aot_autograd", "autograd_cache_miss"),
)
_TIMERS = ("entire_frame_compile", "backend_compile", "inductor_compile", "code_gen")


@functools.cache
def _opaque_dirs() -> tuple[str, ...]:
    """Directories whose frames are compiler machinery rather than a call site.

    Dynamo raises the callback from deep inside itself, via a @contextmanager, so
    the innermost interesting frame is many frames out past torch and contextlib.
    """
    import contextlib as _contextlib

    import torch

    return (
        os.path.dirname(torch.__file__),
        os.path.dirname(_contextlib.__file__),
        os.path.dirname(__file__),
    )


def _call_site() -> str:
    """Innermost frame that is neither compiler machinery nor this module."""
    for frame in reversed(traceback.extract_stack()):
        if frame.filename.startswith(_opaque_dirs()):
            continue
        return f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}"
    return "<unknown>"


def _snapshot() -> dict[str, float]:
    from torch._dynamo.utils import counters, cumulative_time_spent_ns

    snap = {f"{group}.{name}": counters[group][name] for group, name in _COUNTERS}
    snap.update({f"{name}_s": cumulative_time_spent_ns.get(name, 0.0) / 1e9 for name in _TIMERS})
    return snap


class _CompileProbe:
    def __init__(self, out_path: str | None) -> None:
        self.out_path = out_path
        self.phase = STARTUP
        self.phase_start = _snapshot()
        self.per_phase: dict[str, dict[str, float]] = {}
        # Serving-phase events are kept in full; warmup's hundreds are only counted.
        self.serving_events: list[dict[str, Any]] = []
        self.misses: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._started_site = "<unknown>"

    def on_start(self, args) -> None:
        self._started_at = time.perf_counter()
        self._started_site = _call_site()

    def on_end(self, args) -> None:
        elapsed = time.perf_counter() - self._started_at if self._started_at else 0.0
        with self._lock:
            if self.phase != SERVING:
                return
            event = {
                "compile_id": str(getattr(args, "compile_id", "?")),
                "trigger": getattr(getattr(args, "callback_trigger", None), "name", "?"),
                "site": self._started_site,
                "seconds": round(elapsed, 3),
                "thread": threading.current_thread().name,
            }
            self.serving_events.append(event)
            logger.warning(
                "COMPILE PROBE: compile #%d in the SERVING phase took %.3fs "
                "(compile_id=%s, trigger=%s, from %s)",
                len(self.serving_events),
                elapsed,
                event["compile_id"],
                event["trigger"],
                event["site"],
            )
            self._flush()

    def record_miss(self, kind: str, key: Any) -> None:
        """A kernel-cache lookup that had to build its variant.

        Dynamo's counters see the resulting compile; this records *which* variant
        escaped the recorded set, which is the actionable half.
        """
        with self._lock:
            self.misses.append({"phase": self.phase, "kind": kind, "key": repr(key)})
            if self.phase == SERVING:
                logger.warning("COMPILE PROBE: %s kernel-cache MISS in serving: %r", kind, key)
                self._flush()

    def mark_phase(self, phase: str) -> None:
        with self._lock:
            self._close_phase()
            logger.info(
                "COMPILE PROBE: phase %s -> %s; %s so far: %s",
                self.phase,
                phase,
                self.phase,
                self.per_phase[self.phase],
            )
            self.phase = phase
            self.phase_start = _snapshot()
            self._flush()

    def _close_phase(self) -> None:
        now = _snapshot()
        self.per_phase[self.phase] = {
            k: round(now[k] - self.phase_start[k], 3) for k in now if now[k] != self.phase_start[k]
        }

    def report(self) -> dict[str, Any]:
        self._close_phase()
        serving = self.per_phase.get(SERVING, {})
        return {
            "pid": os.getpid(),
            "phase": self.phase,
            # The headline: any nonzero entry here is compilation the bench paid for.
            "serving_phase_dynamo_deltas": serving,
            "verdict": (
                "CLEAN: no compilation in the serving phase"
                if not serving and not self.serving_events
                else "COMPILED IN SERVING PHASE"
            ),
            "per_phase_dynamo_deltas": self.per_phase,
            "serving_compile_events": len(self.serving_events),
            "serving_events": self.serving_events,
            "kernel_misses_by_phase": {
                p: sum(1 for m in self.misses if m["phase"] == p)
                for p in sorted({m["phase"] for m in self.misses})
            },
            "serving_kernel_misses": [m for m in self.misses if m["phase"] == SERVING],
        }

    def start_periodic_flush(self, interval: float = 5.0) -> None:
        if not self.out_path:
            return

        def loop() -> None:
            while True:
                time.sleep(interval)
                with self._lock:
                    self._flush()

        threading.Thread(target=loop, name="compile-probe-flush", daemon=True).start()

    def _flush(self) -> None:
        if not self.out_path:
            return
        path = f"{self.out_path}.{os.getpid()}.json"
        try:
            with open(path, "w") as fh:
                json.dump(self.report(), fh, indent=2)
        except OSError as exc:  # a probe must never take the server down
            logger.warning("COMPILE PROBE: could not write %s: %s", path, exc)


_probe: _CompileProbe | None = None


def install() -> None:
    """Register the dynamo callbacks. No-op unless ``SPYRE_COMPILE_PROBE`` is set."""
    global _probe
    if _probe is not None or not envs.SPYRE_COMPILE_PROBE:
        return
    import torch._dynamo.callback as dynamo_callback

    _probe = _CompileProbe(out_path=envs.SPYRE_COMPILE_PROBE_OUT)
    dynamo_callback.on_compile_start(_probe.on_start)
    dynamo_callback.on_compile_end(_probe.on_end)
    # A clean run raises no event, so nothing would rewrite the report after
    # warmup and "no compiles" would be indistinguishable from "never flushed".
    _probe.start_periodic_flush()
    logger.info(
        "COMPILE PROBE: installed in pid %d, report -> %s",
        os.getpid(),
        f"{_probe.out_path}.{os.getpid()}.json" if _probe.out_path else "(log only)",
    )


def mark_phase(phase: str) -> None:
    if _probe is not None:
        _probe.mark_phase(phase)


def record_miss(kind: str, key: Any) -> None:
    if _probe is not None:
        _probe.record_miss(kind, key)


def dump() -> dict[str, Any] | None:
    """Force a report write; safe to call from a signal handler or at teardown."""
    if _probe is None:
        return None
    with _probe._lock:
        _probe._flush()
        return _probe.report()
