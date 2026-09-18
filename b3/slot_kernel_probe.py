"""Does the compiled slot kernel match its own CPU reference? Per group size.

No vLLM engine, no eager compute on device: operands are built on the host and
`convert`ed, which is the ordinary path, then the COMPILED kernel runs. The
reference is the same Python function on CPU.

`_encoder_slot_kernel` isolates slots by the SDPA batch dim -- the mask only
masks intra-slot padding -- so if that dim is not honoured, every query attends
every slot's keys. Two references distinguish the cases:

  isolated:     each slot attends only its own real keys (what we want)
  contaminated: every row attends all `group * extent` keys (batch dim collapsed)
  nomask:       each slot attends its own whole `extent`, padding included (the
                additive mask dropped -- torch-spyre#4526's failure mode)

The measured embeddings are degenerate at group == 1 as well as group == 4, which
rules contamination out and puts `nomask` first: it is group-independent, and
every sequence's padding rows are identical, so attending them drags all outputs
toward one vector.
"""

import os
import sys
import warnings

import torch

import torch.nn.functional as F  # noqa: E402

# A silent CPU fallback changes the numerical path (fp16 accumulation order) and
# could fake any verdict here. `-W` cannot do this: it resolves the category's
# module at interpreter startup, before torch is imported, and torch_spyre
# refuses to import first.
from torch_spyre.ops.fallbacks import FallbackWarning  # noqa: E402

warnings.simplefilter("error", FallbackWarning)

from spyre_inference.custom_ops.utils import convert, register  # noqa: E402
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (  # noqa: E402
    _encoder_slot_compiled,
    _encoder_slot_kernel,
    encoder_mask,
)

H = KV = 12
D = 64
EXTENT = 512
DTYPE = torch.float16
SCALE = D**-0.5
# Mirrors the probe texts: same extent, different real kv_len per slot.
KV_LENS = [380, 386, 352, 398]


def contaminated_ref(q, k, v, group, kv_lens):
    """Every row attends every slot's real keys: the batch dim collapsed."""
    rows = group * EXTENT
    keep = torch.zeros(1, 1, 1, rows, dtype=DTYPE)
    keep[:] = torch.finfo(DTYPE).min
    for s in range(group):
        keep[..., s * EXTENT : s * EXTENT + kv_lens[s]] = 0
    qq = q.reshape(1, rows, H, D).transpose(1, 2)
    kk = k.reshape(1, rows, KV, D).transpose(1, 2)
    vv = v.reshape(1, rows, KV, D).transpose(1, 2)
    attn = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=keep, scale=SCALE)
    return attn.transpose(1, 2).reshape(rows, H, D)


def nomask_ref(q, k, v, group):
    """Each slot attends its own whole extent, padding included: mask dropped."""
    qq = q.reshape(group, EXTENT, H, D).transpose(1, 2)
    kk = k.reshape(group, EXTENT, KV, D).transpose(1, 2)
    vv = v.reshape(group, EXTENT, KV, D).transpose(1, 2)
    attn = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=None, scale=SCALE)
    return attn.transpose(1, 2).reshape(group * EXTENT, H, D)


def rel(a, b):
    d = (a.float() - b.float()).abs().max().item()
    s = b.float().abs().max().item()
    return d, d / s if s else float("inf")


def main() -> int:
    # No engine here, so nothing else has registered spyre_convert yet.
    register()
    torch.set_default_device("cpu")
    torch.manual_seed(0)
    dev = torch.device(os.environ.get("B3_DEVICE", "spyre"))

    print(f"device={dev}  extent={EXTENT} heads={H} head_size={D} dtype={DTYPE}")
    bad = 0
    for group in (1, 2, 4):
        kv_lens = KV_LENS[:group]
        rows = group * EXTENT
        q = torch.randn(rows, H, D, dtype=DTYPE)
        k = torch.randn(rows, KV, D, dtype=DTYPE)
        v = torch.randn(rows, KV, D, dtype=DTYPE)
        mask = torch.cat([encoder_mask(EXTENT, kv, DTYPE) for kv in kv_lens], dim=0)

        isolated = _encoder_slot_kernel(
            torch.zeros_like(q), q, k, v, mask, SCALE, group, H, KV, D
        )
        contaminated = contaminated_ref(q, k, v, group, kv_lens)
        nomask = nomask_ref(q, k, v, group)

        out = convert(torch.zeros_like(q), dev)
        _encoder_slot_compiled(
            out,
            convert(q, dev),
            convert(k, dev),
            convert(v, dev),
            convert(mask, dev),
            SCALE,
            group,
            H,
            KV,
            D,
        )
        got = convert(out, "cpu")

        d_iso, r_iso = rel(got, isolated)
        d_con, r_con = rel(got, contaminated)
        # Only the real rows matter; the pooler never reads a slot's padding.
        real = torch.cat(
            [
                torch.arange(s * EXTENT, s * EXTENT + kv_lens[s])
                for s in range(group)
            ]
        )
        d_iso_r, r_iso_r = rel(got[real], isolated[real])
        d_con_r, r_con_r = rel(got[real], contaminated[real])
        d_nom_r, r_nom_r = rel(got[real], nomask[real])

        # Isolated first: at group == 1 the contaminated reference IS the isolated
        # one (there is no second slot), so a tie must not read as contamination.
        if r_iso_r < 0.05:
            verdict = "MATCHES ISOLATED (correct)"
        elif r_con_r < 0.05:
            verdict = "MATCHES CONTAMINATED (batch dim collapsed)"
        elif r_nom_r < 0.05:
            verdict = "MATCHES NOMASK (additive mask dropped)"
        else:
            verdict = "MATCHES NONE OF THE THREE"
        if r_iso_r >= 0.05:
            bad += 1
        print(
            f"\ngroup={group} rows={rows}\n"
            f"  vs isolated      max|d|={d_iso:.5f} rel={r_iso:.4f}   "
            f"(real rows: {d_iso_r:.5f} rel={r_iso_r:.4f})\n"
            f"  vs contaminated  max|d|={d_con:.5f} rel={r_con:.4f}   "
            f"(real rows: {d_con_r:.5f} rel={r_con_r:.4f})\n"
            f"  vs nomask                                  "
            f"(real rows: {d_nom_r:.5f} rel={r_nom_r:.4f})\n"
            f"  -> {verdict}",
            flush=True,
        )

    print(f"\nSLOT_PROBE {'FAIL' if bad else 'PASS'} ({bad} group size(s) wrong)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
