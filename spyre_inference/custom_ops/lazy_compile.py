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

"""Let a layer compile its own kernel when no enclosing graph already covers it.

``STOCK_TORCH_COMPILE`` compiles one transformer block at a time, so layers inside
a block are already part of a graph. Some of the same classes also appear outside
the blocks -- the input embedding, and the norm after the last block -- where
nothing compiles them and they dispatch op by op instead.

A layer cannot know where the model put it, and does not need to: a layer running
while nothing is being traced has no enclosing graph, so it is the outermost thing
and should compile itself. ``torch.compiler.is_compiling()`` is that test, and
Dynamo folds it to ``True`` while tracing, so in-block layers pay nothing for the
check and emit no extra graph.

``SPYRE_COMPILE_GRANULARITY=model`` then falls out for free: under a whole-model
graph every layer is traced, so none of them compiles itself.

A layer opts in by mixing in ``CompileOutermost`` and decorating one method with
``@compile_when_outermost``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TypeVar

import torch

from vllm.config import CompilationMode, get_cached_compilation_config
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

F = TypeVar("F", bound=Callable)


class CompileOutermost:
    """Base for layers with a ``@compile_when_outermost`` kernel (exactly one).

    Samples the compile mode here because construction is the only point where vLLM
    guarantees the config context is live; by the first forward it asserts instead.
    ``enforce_eager`` arrives as mode ``NONE`` (see
    ``platform.apply_config_platform_defaults``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode = get_cached_compilation_config().mode
        self.spyre_compile_enabled = mode is not CompilationMode.NONE
        self.spyre_compiled_kernel: Callable | None = None


def compile_when_outermost(method: F) -> F:
    """Compile ``method`` on its first call that no other graph is already tracing."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if torch.compiler.is_compiling() or not self.spyre_compile_enabled:
            return method(self, *args, **kwargs)
        if self.spyre_compiled_kernel is None:
            logger.info_once(
                "Compiling %s.%s as its own graph: no enclosing graph covers it.",
                type(self).__name__,
                method.__name__,
            )
            # dynamic=False is mandatory: the Spyre backend rejects SymInt shapes.
            self.spyre_compiled_kernel = torch.compile(
                method.__get__(self),
                backend=current_platform.simple_compile_backend,
                fullgraph=True,
                dynamic=False,
            )
        return self.spyre_compiled_kernel(*args, **kwargs)

    return wrapper
