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

"""The compile probe's verdict. Backend "eager" keeps this off the accelerator:
the probe reads Dynamo's tracing counters, which move without Inductor."""

import json
import os

import pytest
import torch

from spyre_inference import compile_probe


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """A freshly installed probe, torn down so the next test starts clean."""
    monkeypatch.setattr(compile_probe.envs, "SPYRE_COMPILE_PROBE", True, raising=False)
    monkeypatch.setattr(
        compile_probe.envs, "SPYRE_COMPILE_PROBE_OUT", str(tmp_path / "probe"), raising=False
    )
    monkeypatch.setattr(compile_probe, "_probe", None)
    import torch._dynamo.callback as dynamo_callback

    before = list(dynamo_callback.callback_handler.start_callbacks), list(
        dynamo_callback.callback_handler.end_callbacks
    )
    compile_probe.install()
    yield compile_probe._probe
    dynamo_callback.callback_handler.start_callbacks[:] = before[0]
    dynamo_callback.callback_handler.end_callbacks[:] = before[1]


def test_serving_phase_compile_is_reported(probe):
    compile_probe.mark_phase(compile_probe.WARMUP)
    fn = torch.compile(lambda x: x.relu().sum(), backend="eager", dynamic=False)
    fn(torch.randn(4))
    compile_probe.mark_phase(compile_probe.SERVING)
    fn(torch.randn(8))  # a new shape -> a guard miss -> a serving-phase compile

    report = compile_probe.dump()
    assert report["verdict"] == "COMPILED IN SERVING PHASE"
    assert report["serving_compile_events"] == 1
    # Dynamo's own counters, not the probe's bookkeeping, are the ground truth.
    assert report["serving_phase_dynamo_deltas"]["stats.unique_graphs"] == 1
    assert report["per_phase_dynamo_deltas"][compile_probe.WARMUP]["stats.unique_graphs"] == 1


def test_no_serving_compile_is_clean(probe):
    compile_probe.mark_phase(compile_probe.WARMUP)
    fn = torch.compile(lambda x: x.relu().sum(), backend="eager", dynamic=False)
    fn(torch.randn(4))
    compile_probe.mark_phase(compile_probe.SERVING)
    fn(torch.randn(4))  # same shape -> cache hit -> no compile

    report = compile_probe.dump()
    assert report["verdict"].startswith("CLEAN")
    assert report["serving_phase_dynamo_deltas"] == {}


def test_kernel_miss_is_attributed_to_its_phase(probe):
    compile_probe.mark_phase(compile_probe.WARMUP)
    compile_probe.record_miss("varlen_attn", (1, 512, "index", True))
    compile_probe.mark_phase(compile_probe.SERVING)
    compile_probe.record_miss("varlen_attn", (2, 1, "copy", False))

    report = compile_probe.dump()
    assert report["kernel_misses_by_phase"] == {"serving": 1, "warmup": 1}
    assert report["serving_kernel_misses"] == [
        {"phase": "serving", "kind": "varlen_attn", "key": "(2, 1, 'copy', False)"}
    ]


def test_report_is_written_to_disk(probe):
    compile_probe.mark_phase(compile_probe.SERVING)
    written = json.load(open(f"{probe.out_path}.{os.getpid()}.json"))
    assert written["phase"] == compile_probe.SERVING
