# AOT ID: ['4_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from sympy import sympify
from torch_spyre._inductor.op_spec import TensorArg, TensorWorkDivision, OpSpec, UnimplementedOp, LoopSpec, spyre_constant_tensor, IndirectAccess, DebugHandle, SourceLoc, ProvenanceTransform
from torch_spyre.execution.async_compile import SpyreAsyncCompile
from torch_spyre._C import DataFormats, ElementArrangement, SpyreTensorLayout, spyre_empty_with_layout, set_spyre_tensor_layout
import subprocess

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
from torch_spyre._C import reinterpret_tensor as reinterpret_tensor
from torch_spyre._C import reinterpret_tensor_with_layout
del async_compile
async_compile = SpyreAsyncCompile()


# Topologically Sorted Source Nodes: [mean, norm_mean, layernormnorm], Original ATen: [spyre.exx2, spyre.layernormscale, spyre.layernormnorm]
# Source node to ATen node mapping:
#   layernormnorm => layernormnorm
#   mean => exx2
#   norm_mean => layernormscale
# Graph fragment:
#   %arg2_1 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=arg2_1]
#   %clone : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone]
#   %exx2 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2]
#   %layernormscale : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale]
#   %arg0_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg1_1]
#   %clone : [num_users=2] = call_function[target=torch.ops.aten.clone](args = (%arg2_1,), kwargs = {})
#   %exx2 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%clone, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2, 1e-05), kwargs = {})
#   %layernormnorm : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%clone, %exx2, %layernormscale, %arg0_1, %arg1_1), kwargs = {})
#   return %clone,%exx2,%layernormscale,%layernormnorm
sdsc_fused_exx2_layernormnorm_layernormscale_0 = async_compile.sdsc('sdsc_fused_exx2_layernormnorm_layernormscale_0',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=502502847299613231, source=None, aten_op=None, ir_chain=('clone', 'buf3'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 8192},
                ),
            ]
        ),
        OpSpec(
            op='exx2',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'constants': {'exx2scale': 0.0013020833333333333, 'useZeroMean': False}},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1187314820291675849, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/torch_spyre/_inductor/decompositions.py', start_line=793, start_col=0, end_line=None, end_col=None), aten_op='spyre.exx2.default', ir_chain=('exx2', 'buf0'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 8192},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 0},
                    element_arrangement=ElementArrangement.EXX2,
                ),
            ]
        ),
        OpSpec(
            op='layernormscale',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32)},
            op_info={'constants': {'eps': 1e-05}},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1472935604190438506, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/torch_spyre/_inductor/decompositions.py', start_line=794, start_col=0, end_line=None, end_col=None), aten_op='spyre.layernormscale.default', ir_chain=('layernormscale', 'buf1'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 0},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 106496},
                    element_arrangement=ElementArrangement.EXX2,
                ),
            ]
        ),
        OpSpec(
            op='layernormnorm',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=919735735542596338, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/torch_spyre/_inductor/decompositions.py', start_line=795, start_col=0, end_line=None, end_col=None), aten_op='spyre.layernormnorm.default', ir_chain=('layernormnorm', 'buf2'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 8192},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 0},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 106496},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
            ]
        ),
    ]
)


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        assert_size_stride(arg2_1, (2048, 768), (768, 1), 'input')
        assert_size_stride(arg0_1, (768, ), (1, ), 'input')
        assert_size_stride(arg1_1, (768, ), (1, ), 'input')
        buf2 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_exx2_layernormnorm_layernormscale_0.run(arg2_1, arg0_1, arg1_1, buf2)
        del arg0_1
        del arg1_1
        del arg2_1
        return (buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns
