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

"""One-token generate, to check the K-transposed cache end to end.

`SPYRE_ATTN_KT_CACHE=1` stores K's pages `[head_size, block_size]` and writes them a whole
page at a time, which is exact for a block-aligned prefill chunk. At `max_tokens=1` the
token comes out of the last prefill chunk and no decode step runs, so this is the longest
run whose output the page-granular store fully determines. Run it once per arm and compare
the token: the arms must agree.

    SPYRE_ATTN_KV_LAYOUT=head_major SPYRE_ATTN_KT_CACHE=1 \
      python scripts/probes/kt_cache_e2e_smoke.py

Env: MODEL, INPUT_LEN, MAX_MODEL_LEN
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("VLLM_PLUGINS", "spyre_inference")

import spyre_inference.envs as sp_envs  # noqa: E402

# The worktree trap: a console script puts .venv/bin on sys.path[0], so an editable
# install pointing at the main checkout would silently be the code under test.
print(f"spyre_inference.envs from: {sp_envs.__file__}")
assert hasattr(sp_envs, "SPYRE_ATTN_KT_CACHE"), "not running this worktree's spyre_inference"
print(f"SPYRE_ATTN_KT_CACHE={sp_envs.SPYRE_ATTN_KT_CACHE} "
      f"SPYRE_ATTN_KV_LAYOUT={sp_envs.SPYRE_ATTN_KV_LAYOUT}")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("MODEL", "ibm-granite/granite-3.3-8b-instruct")
INPUT_LEN = int(os.environ.get("INPUT_LEN", "1984"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))


def main():
    llm = LLM(model=MODEL, max_model_len=MAX_MODEL_LEN, block_size=128)
    # A fixed synthetic prompt of exactly INPUT_LEN tokens, so both arms see one input.
    prompt_ids = [(i * 7919) % 20000 + 100 for i in range(INPUT_LEN)]
    out = llm.generate(
        {"prompt_token_ids": prompt_ids},
        SamplingParams(max_tokens=1, temperature=0.0),
    )
    tok = out[0].outputs[0]
    print(f"KT_SMOKE token_ids={list(tok.token_ids)} text={tok.text!r}")


# vLLM spawns its engine core, so the child re-imports this module.
if __name__ == "__main__":
    main()
