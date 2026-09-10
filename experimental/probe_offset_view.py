"""Does a compiled region read a nonzero-offset slice correctly?

The KV-tiled decode kernels compute the wrong answer, and both of them do it
identically, so the bug is in the tiling. The suspicion is torch-spyre#3770,
which the production kernel already documents as "a compiled region reads a view
from offset 0, ignoring storage_offset". KV tiling slices the slot axis at
lo=kv_tile, which is exactly such a view.

If that is what happens, tile 1 silently re-reads tile 0: the kernel processes
half the KV twice, which is both wrong and cheaper -- so it would explain the
measured "speedup" as an artifact.

This fills a page with a distinct value per slot and asks a compiled function for
the sum of each 64-slot tile. Correct output is one distinct sum per tile; a
storage-offset bug makes every tile report tile 0's sum.
"""

import torch

from spyre_inference.custom_ops.utils import convert
from spyre_inference.custom_ops.utils import register as register_convert_op
from spyre_inference.v1.attention.backends.spyre_attn import slot_major_kv_layout

DEV = torch.device("spyre")
BLOCK, KV, D, TILE = 128, 8, 128, 64


def tile_sums(cache, idx):
    """Sum of each TILE-slot slice of the gathered page, as the kernels slice it."""
    page = cache.index_select(0, idx)  # [1, BLOCK, KV, D]
    outs = []
    for lo in range(0, BLOCK, TILE):
        t = page[:, lo : lo + TILE].permute(0, 2, 1, 3)  # [1, KV, TILE, D]
        outs.append(t.reshape(-1).sum())
    return torch.stack(outs)


def main():
    register_convert_op()
    torch.ones(2, 2, dtype=torch.float16).to(DEV)

    dtype = torch.float16
    num_pages = 2
    host = torch.zeros(num_pages, BLOCK, KV, D, dtype=dtype)
    # Slot s carries the value s/1000 everywhere, so each tile's sum is distinct
    # and a tile that re-reads another tile is unmistakable.
    for s in range(BLOCK):
        host[0, s] = s / 1000.0
    layout = slot_major_kv_layout(num_pages * BLOCK, KV, D, dtype)
    cache = host.to(DEV, device_layout=layout)
    idx = convert(torch.tensor([0], dtype=torch.int32), device=DEV)

    expect = torch.stack([
        host[0, lo : lo + TILE].reshape(-1).float().sum() for lo in range(0, BLOCK, TILE)
    ])

    eager = tile_sums(cache, idx).cpu().float()
    compiled = torch.compile(tile_sums, dynamic=False)(cache, idx).cpu().float()

    print("expected (host):  %s" % [round(v, 1) for v in expect.tolist()])
    print("eager   (device): %s" % [round(v, 1) for v in eager.tolist()])
    print("compiled(device): %s" % [round(v, 1) for v in compiled.tolist()])

    same = abs(compiled[0].item() - compiled[1].item()) < 1.0
    print()
    if same:
        print("BUG CONFIRMED: both tiles report the same sum, so the nonzero-offset")
        print("slice read tile 0 again -- storage_offset ignored (torch-spyre#3770).")
    elif torch.allclose(compiled, expect, rtol=0.02, atol=1.0):
        print("Offset slices read correctly; the tiling bug is elsewhere.")
    else:
        print("Tiles differ but do not match the host; some other mis-read.")


if __name__ == "__main__":
    main()
