"""Env-gated perf_counter attribution of the worker's host step path.

Enable with SPYRE_HOSTPROBE=1.  Cumulative per-span totals are appended to
SPYRE_HOSTPROBE_OUT (default /tmp/hostprobe.txt) every SPYRE_HOSTPROBE_EVERY
steps (default 100).  Difference two dumps to get steady state, otherwise the
lazy warmup compilation is averaged into every step.

perf_counter spans are ~free (swarm-01 measured 121.19 req/s with probes vs
120.77/121.29 without); kineto is not, because it inflates exactly the host
side being measured.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

_t: dict[str, float] = defaultdict(float)
_n: dict[str, int] = defaultdict(int)
_steps = 0
_out = os.environ.get("SPYRE_HOSTPROBE_OUT", "/tmp/hostprobe.txt")
_every = int(os.environ.get("SPYRE_HOSTPROBE_EVERY", "100"))


def enabled() -> bool:
    return os.environ.get("SPYRE_HOSTPROBE", "").lower() in ("1", "true", "t", "yes", "y")


class span:
    __slots__ = ("name", "t0")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _t[self.name] += time.perf_counter() - self.t0
        _n[self.name] += 1
        return False


def _wrap(obj, name: str, label: str | None = None) -> bool:
    fn = getattr(obj, name, None)
    if fn is None:
        return False
    key = label or name

    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            _t[key] += time.perf_counter() - t0
            _n[key] += 1

    try:
        setattr(obj, name, wrapper)
    except (AttributeError, TypeError):
        return False
    return True


def dump(tag: str = "") -> None:
    with open(_out, "a") as f:
        f.write(f"=== steps={_steps} tag={tag} t={time.time():.3f}\n")
        for k in sorted(_t, key=lambda k: -_t[k]):
            f.write(f"{k}\t{_t[k]:.6f}\t{_n[k]}\n")
        f.flush()


def tick() -> None:
    global _steps
    _steps += 1
    if _steps % _every == 0:
        dump()


# --- cProfile mode: complete function-level map of the worker step ---------
# Inflates absolute Python time, so it is for ATTRIBUTION ONLY; confirm the
# candidates it names with the free perf_counter spans above.
_prof = None
_prof_lo = int(os.environ.get("SPYRE_HOSTPROBE_CPROFILE_FROM", "200"))
_prof_hi = int(os.environ.get("SPYRE_HOSTPROBE_CPROFILE_TO", "400"))
_prof_done = False


def cprofile_enabled() -> bool:
    return os.environ.get("SPYRE_HOSTPROBE_CPROFILE", "").lower() in (
        "1", "true", "t", "yes", "y"
    )


def _cprofile_step():
    """Accumulate a cProfile over steps [_prof_lo, _prof_hi), then dump once."""
    global _prof, _prof_done
    if _prof_done:
        return None
    if _steps == _prof_lo and _prof is None:
        import cProfile

        _prof = cProfile.Profile()
    if _prof is None:
        return None
    if _steps >= _prof_hi:
        import pstats

        _prof.dump_stats(_out + ".prof")
        with open(_out + ".cprofile.txt", "w") as f:
            st = pstats.Stats(_prof, stream=f)
            st.sort_stats("cumulative").print_stats(120)
            st.sort_stats("tottime").print_stats(120)
        _prof_done = True
        _prof = None
        return None
    return _prof


_installed = False


def install(runner) -> None:
    """Wrap the host step path of a TorchSpyreModelRunner instance."""
    global _installed
    if _installed or not enabled():
        return
    _installed = True

    ok, missing = [], []
    for name in (
        "_update_states",
        "_prepare_inputs",
        "_get_slot_mappings",
        "_build_attention_metadata",
        "_determine_batch_execution_and_padding",
        "_preprocess",
        "_pool",
        "_sync_device",
        "_extract_encoder_inputs",
        "_execute_mm_encoder",
        "_compute_cascade_attn_prefix_lens",
        "_maybe_add_model_args",
        "_gather_mm_embeddings",
        "_model_forward",
        "_allow_microbatching",
        "_get_cumsum_and_arange",
    ):
        (ok if _wrap(runner, name) else missing).append(name)

    # the model wrapper: separates our device-facing call from vLLM bookkeeping
    model = getattr(runner, "model", None)
    if model is not None:
        for name in ("_plan_slots", "embed_input_ids", "embed_multimodal"):
            (ok if _wrap(model, name, f"model.{name}") else missing).append(name)

    ib = getattr(runner, "input_batch", None)
    if ib is not None:
        for name in (
            "add_request",
            "remove_request",
            "condense",
            "refresh_metadata",
            "_register_add_request",
            "make_pooling_metadata",
            "make_sampling_metadata",
        ):
            (ok if _wrap(ib, name, f"input_batch.{name}") else missing).append(name)
        bt = getattr(ib, "block_table", None)
        if bt is not None:
            for name in ("commit_block_table", "append_row", "add_row", "compute_slot_mapping"):
                (ok if _wrap(bt, name, f"block_table.{name}") else missing).append(name)

    pooler = getattr(getattr(runner, "model", None), "pooler", None)
    if pooler is not None:
        _wrap(pooler, "__call__", "pooler.__call__")

    inner = getattr(runner, "execute_model", None)
    if inner is not None:

        use_cprofile = cprofile_enabled()
        last_exit = [None]

        def execute_model(*a, **kw):
            if last_exit[0] is not None:
                _t["OUTSIDE_execute_model(engine+ipc)"] += time.perf_counter() - last_exit[0]
                _n["OUTSIDE_execute_model(engine+ipc)"] += 1
            pr = _cprofile_step() if use_cprofile else None
            t0 = time.perf_counter()
            if pr is not None:
                pr.enable()
            try:
                return inner(*a, **kw)
            finally:
                if pr is not None:
                    pr.disable()
                now = time.perf_counter()
                _t["execute_model"] += now - t0
                _n["execute_model"] += 1
                last_exit[0] = now
                tick()

        runner.execute_model = execute_model
        ok.append("execute_model")

    with open(_out, "a") as f:
        f.write(f"# hostprobe installed: wrapped={ok} missing={missing}\n")
