"""Can ONE 4-D KV allocation serve all three readers the folded plan needs?

  4-D  [pages, KV, block, D]        per-sequence read, 1-entry gather
  3-D  view(-1, block, D)          batched decode read      (already proven fast+correct)
  2-D  view(-1, D)                 the KV store's destination

The store is the open one: production's kv_slot_views does view(-1, shape[2]) on a
3-D cache; from 4-D that merges three dims. Verifies placement, not just that it runs.
"""

import torch
from spyre_inference.custom_ops.utils import convert
from spyre_inference.custom_ops.utils import register as register_convert_op
from spyre_inference.v1.attention.backends.spyre_attn import slot_major_kv_layout

register_convert_op()
DEV = torch.device("spyre")
PAGES, KV, BS, D = 9, 8, 128, 128
DT = torch.float16
torch.set_default_device("cpu")
torch.manual_seed(0)

# Exactly PR 855's head_major_kv_layout, spelled via slot_major_kv_layout as the
# harness does: device_size [pages*KV, block, sticks, eps].
layout = slot_major_kv_layout(PAGES * KV, BS, D, DT)
k4 = torch.zeros(PAGES, KV, BS, D, dtype=DT).to(DEV, device_layout=layout)
print(f"4-D alloc ok: {tuple(k4.shape)}")

for name, shape in (("3-D view(-1, block, D)", (-1, BS, D)), ("2-D view(-1, D)", (-1, D))):
    try:
        v = k4.view(*shape)
        print(f"  {name}: ok -> {tuple(v.shape)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {name}: FAILED -> {exc}")

# --- the store, per KV head, into the 2-D view -------------------------------
NT = 4  # decode width
k2 = k4.view(-1, D)
key_h = torch.randn(NT, KV, D).to(DT)
# one token per page, as decode does; row = (page * KV + h) * BS + offset
slots = [(i % PAGES) * BS + (i // PAGES) for i in range(NT)]
rows_per_head = [
    torch.tensor([(s // BS * KV + h) * BS + s % BS for s in slots], dtype=torch.int64)
    for h in range(KV)
]
rows_dev = [convert(r, device=DEV) for r in rows_per_head]
key_dev = convert(key_h, device=DEV)


def store(key, dest, rows):
    for h in range(len(rows)):
        dest.index_copy_(0, rows[h], key.select(1, h))


try:
    torch.compile(store, dynamic=False)(key_dev, k2, rows_dev)
    torch.spyre.synchronize()
    print("  store into 2-D view: compiled and ran")
except Exception as exc:  # noqa: BLE001
    print(f"  store into 2-D view: FAILED -> {exc}")
    raise SystemExit(1)

# --- placement: read the 4-D tensor back and compare -------------------------
got = k4.to("cpu").float()
exp = torch.zeros(PAGES, KV, BS, D)
written = torch.zeros(PAGES, KV, BS, D, dtype=torch.bool)
for i, s in enumerate(slots):
    exp[s // BS, :, s % BS, :] = key_h[i].float()
    written[s // BS, :, s % BS, :] = True

# A misplaced store shows up as error in the cells nobody should have touched.
err_written = (got - exp)[written].abs().max().item()
err_untouched = (got - exp)[~written].abs().max().item()

# Control: does a plain host->device->host round trip of the same fp16 values
# already lose a ULP? If so the store is exact and the readback is what rounds.
rt = convert(key_h, device=DEV).to("cpu").float()
err_roundtrip = (rt - key_h.float()).abs().max().item()

print(f"  placement: written cells max err {err_written:.6f}, untouched cells {err_untouched:.6f}")
print(f"  control:   plain host->device->host round trip max err {err_roundtrip:.6f}")
exact = err_untouched == 0.0 and err_written <= err_roundtrip
print("  VERDICT:", "placed exactly, no worse than a round trip" if exact else "SUSPECT")
