"""B3 inner loop: compile the encoder and report Inductor launch-site counts.

Engine init alone triggers ``warming_up_model()``, which is where the compile
happens, so this never runs a request and never needs a client. Restricting
``compile_sizes`` to the single 2048 bucket that 4x512 slot padding actually
lands on cuts the four-bucket warmup down to one.
"""

import json
import os
import sys
import time

import torch  # noqa: F401  (must precede the torch_spyre backend autoload)

from vllm import LLM

MODEL = "ibm-granite/granite-embedding-278m-multilingual"


def main() -> int:
    bucket = int(os.environ.get("B3_BUCKET", "2048"))
    layers = os.environ.get("B3_LAYERS")

    compilation_config: dict = {"compile_sizes": [bucket]}
    inductor_cfg = os.environ.get("B3_INDUCTOR_CFG")
    if inductor_cfg:
        compilation_config["inductor_compile_config"] = json.loads(inductor_cfg)

    kwargs: dict = dict(
        model=MODEL,
        runner="pooling",
        max_model_len=512,
        max_num_seqs=4,
        enable_prefix_caching=False,
        compilation_config=compilation_config,
    )
    if layers:
        # Structural experiments only: fewer blocks compile far faster and the
        # per-block launch-site count is what we are reading off.
        kwargs["hf_overrides"] = {"num_hidden_layers": int(layers)}

    t0 = time.time()
    LLM(**kwargs)
    print(f"B3_COMPILE_SECONDS {time.time() - t0:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
