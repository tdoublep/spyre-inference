"""Reproducer: RAS ComputeHardwareError 0x7b1b from a single-row in-graph gather.

Run the configuration that faults (~20s, no model weights, no vLLM engine):

    uv run --no-sync python examples/experimental/spyre_gather_hw_error_repro.py \
        --tokens 1 --qlen 1 --scatter no --mutating no

    [unspecified] ERRR ras_base.hpp:74 {"code":"0x7b1b", ...
      "name":"RAS::RUNTIMESCHEDULER::ComputeHardwareError", ...}
    RuntimeError: StreamInErrorState

Observed 3/3 in fresh processes. Requires the in-graph query gather, so it needs
this branch's kernel; it cannot run against main.

Shape of the test: an outer torch.compile'd "block" does a fused-QKV matmul,
then calls an opaque custom op that invokes a second compiled graph --
spyre-inference's real paged-attention kernel -- which index_selects its query
rows out of the outer graph's buffer. With tokens == padded_query_len == 1 that
gather selects the whole 1-row source and the kernel miscompiles into bad
device addresses.

What each variable does, measured one at a time from the config above:

    tokens=1  qlen=1   gather selects whole 1-row buffer ..... FAULT
    tokens=2  qlen=1   gather selects a strict subset ........ passes
    tokens=64 qlen=64  whole buffer but 64 rows .............. passes
    --mutating yes     mutating custom op .................... FAULT (no effect)
    --scatter yes      reshape_and_cache before attention .... passes (MASKS it)
    --clean-v          v materialized, not a column narrow ... FAULT (no effect)

So the trigger needs a gather that selects its entire source AND a source of
exactly one row. The production fix is to skip a gather that selects everything
(see needs_gather in spyre_attn.py), which is why bs=1 decode was the only
failing case: any multi-sequence batch selects a strict subset.

DO NOT SIMPLIFY WITHOUT RE-CHECKING IT STILL FAULTS. The trigger is fragile and
depends on the surrounding graph, not just the gather. All of these stopped
reproducing: a hand-written kernel performing the same ops; a plain (non
fused-QKV) outer matmul; dropping the unused slot_mapping argument from the
custom op; replacing the trailing F.pad. The ingredient list above is necessary
but the sufficient set has not been isolated.

Environment: torch-spyre a3128985, vllm v0.27.1.
"""

import argparse
import torch
import torch_spyre  # noqa: F401

from spyre_inference.v1.attention.backends.spyre_attn import (
    INT32_ELEMS_PER_STICK,
    _create_compilable_page_attn,
    _reshape_and_cache_kernel,
)

H, D, NKV = 32, 128, 8
NQ = H // NKV
QKV_COLS = H * D + 2 * NKV * D


def build(tokens, qlen, nb, bs, total_pages, dev, scatter, mutating):
    kernel = _create_compilable_page_attn(
        nb, qlen, H, D, has_alibi=False, logits_soft_cap=0.0, num_kv_heads=NKV
    )
    attn_c = torch.compile(kernel, dynamic=False)
    cache_c = torch.compile(_reshape_and_cache_kernel, dynamic=False)

    masks = [torch.zeros(qlen, bs, dtype=torch.float16).to(dev) for _ in range(nb)]
    scale = D**-0.5
    slots = (-1, NKV, D)

    def body(q, k, v, qri, kp, vp, pit, slot_mapping):
        if scatter:
            cache_c(k, v, kp.view(slots), vp.view(slots), slot_mapping)
        return attn_c(q, qri, kp, vp, pit, masks, scale)

    tag = f"{tokens}_{qlen}_{nb}_{int(scatter)}_{int(mutating)}"
    if mutating:
        name = f"attn_mut_{tag}"

        @torch.library.custom_op(f"repro3::{name}", mutates_args=("out",))
        def op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qri: torch.Tensor,
               kp: torch.Tensor, vp: torch.Tensor, pit: torch.Tensor,
               slot_mapping: torch.Tensor, out: torch.Tensor) -> None:
            res = body(q, k, v, qri, kp, vp, pit, slot_mapping)
            out[0:qlen] = res[:qlen]

        @op.register_fake
        def _(q, k, v, qri, kp, vp, pit, slot_mapping, out) -> None:
            return
    else:
        name = f"attn_ret_{tag}"

        @torch.library.custom_op(f"repro3::{name}", mutates_args=())
        def op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qri: torch.Tensor,
               kp: torch.Tensor, vp: torch.Tensor, pit: torch.Tensor,
               slot_mapping: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            return body(q, k, v, qri, kp, vp, pit, slot_mapping)

        @op.register_fake
        def _(q, k, v, qri, kp, vp, pit, slot_mapping, out):
            return q.new_empty((qlen, H, D))

    return getattr(torch.ops.repro3, name), mutating


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--qlen", type=int, default=1)
    ap.add_argument("--num-blocks", type=int, default=1, dest="nb")
    ap.add_argument("--block-size", type=int, default=128, dest="bs")
    ap.add_argument("--total-pages", type=int, default=5, dest="pages")
    ap.add_argument("--scatter", default="yes", choices=["yes", "no"])
    ap.add_argument("--mutating", default="yes", choices=["yes", "no"])
    ap.add_argument("--layers", type=int, default=1, help="attention calls per fwd")
    ap.add_argument("--warm-prefill", default="no", choices=["yes", "no"], dest="warm")
    ap.add_argument("--clean-v", action="store_true", dest="clean_v",
                    help="materialize v instead of passing a column-narrow view")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    tokens, qlen, nb, bs = args.tokens, args.qlen, args.nb, args.bs
    dev = torch.device("spyre")
    torch.manual_seed(0)

    kp = torch.randn(args.pages, bs, NKV, D, dtype=torch.float16).mul_(0.1).to(dev)
    vp = torch.randn(args.pages, bs, NKV, D, dtype=torch.float16).mul_(0.1).to(dev)

    def run_shape(tokens, qlen, label):
        op, mutating = build(tokens, qlen, nb, bs, args.pages, dev, args.scatter == "yes",
                             args.mutating == "yes")
        hid = torch.randn(tokens, H * D, dtype=torch.float16).to(dev)
        wqkv = torch.randn(H * D, QKV_COLS, dtype=torch.float16).mul_(0.02).to(dev)

        pit = torch.zeros(nb, INT32_ELEMS_PER_STICK, dtype=torch.int32)
        for i in range(nb):
            pit[i, 0] = i
        pit = pit.to(dev)

        idx_len = (qlen + INT32_ELEMS_PER_STICK - 1) // INT32_ELEMS_PER_STICK * INT32_ELEMS_PER_STICK
        qri = torch.zeros(idx_len, dtype=torch.int32)
        qri[:qlen] = torch.arange(qlen, dtype=torch.int32).clamp_max(tokens - 1)
        qri = qri.to(dev)
        slot_mapping = torch.arange(tokens, dtype=torch.int32).to(dev)

        def outer(hid, wqkv, qri, kp, vp, pit, slot_mapping):
            qkv = torch.matmul(hid, wqkv)
            # q, k as fresh buffers (rope outputs in the real model); v a
            # strided slice of the fused buffer at a nonzero storage_offset.
            q = (qkv[:, : H * D] * 1.0).reshape(tokens, H, D)
            k = (qkv[:, H * D : H * D + NKV * D] * 1.0).reshape(tokens, NKV, D)
            v = qkv[:, H * D + NKV * D :]
            if args.clean_v:
                v = v * 1.0
            v = v.reshape(tokens, NKV, D)
            out = torch.zeros(tokens, H, D, dtype=torch.float16, device=q.device)
            for _ in range(args.layers):
                r = op(q, k, v, qri, kp, vp, pit, slot_mapping, out)
                if not mutating:
                    out = out + torch.nn.functional.pad(r, (0, 0, 0, 0, 0, tokens - qlen))
            return out * 1.0

        run = torch.compile(outer, dynamic=False)
        for i in range(args.iters):
            o = run(hid, wqkv, qri, kp, vp, pit, slot_mapping)
            host = o.to("cpu")
            print(f"  [{label}] iter {i}: ok finite={bool(torch.isfinite(host).all())}")

    print(f"tokens={tokens} qlen={qlen} nb={nb} scatter={args.scatter} "
          f"mutating={args.mutating} layers={args.layers} warm_prefill={args.warm}")
    if args.warm == "yes":
        run_shape(64, 64, "prefill")
    run_shape(tokens, qlen, "target")
    print("PASS")


if __name__ == "__main__":
    main()
