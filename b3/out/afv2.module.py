# AOT ID: ['0_inference']
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


# Topologically Sorted Source Nodes: [output], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   output => embedding
# Graph fragment:
#   %arg0_1 : Tensor "i64[2048][1]spyre:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f16[250048, 768][768, 1]spyre:0" = PlaceHolder[target=arg1_1]
#   %embedding : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=embedding]
#   %embedding : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %clone_60 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%embedding,), kwargs = {})
#   return %embedding,%clone_60
sdsc_fused_embedding_0 = async_compile.sdsc('sdsc_fused_embedding_0',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5856837397831302851, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/vocab_parallel_embedding.py', start_line=78, start_col=0, end_line=None, end_col=None), aten_op='aten.embedding.default', ir_chain=('embedding', 'buf0'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.IEEE_INT32,
                    device_size=[1, 64, 32],
                    device_coordinates=[sympify('0'), sympify('floor(c0/32)'), sympify('Mod(c0, 32)')],
                    allocation={'hbm': 0},
                    name='arg0_1',
                ),
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[250048, 12, 64],
                    device_coordinates=[IndirectAccess('arg0_1'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6460594415264223131, source=None, aten_op=None, ir_chain=('clone_60', 'buf312'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_60 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_60]
#   %embedding : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=embedding]
#   %full_default : Tensor  = PlaceHolder[target=full_default]
#   %clone_61 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone_60,), kwargs = {})
#   return %clone_61
sdsc_fused_1 = async_compile.sdsc('sdsc_fused_1',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8147865048525988505, source=None, aten_op=None, ir_chain=('clone_61', 'buf313'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [output_1, add], Original ATen: [aten.embedding, aten.add]
# Source node to ATen node mapping:
#   add => add
#   output_1 => embedding_1
# Graph fragment:
#   %full_default : Tensor "i64[2048][1]spyre:0" = PlaceHolder[target=full_default]
#   %arg2_1 : Tensor "f16[64, 768][768, 1]spyre:0" = PlaceHolder[target=arg2_1]
#   %clone_61 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_61]
#   %embedding_1 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=embedding_1]
#   %add : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add]
#   %embedding_1 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %full_default), kwargs = {})
#   %add : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %clone_62 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%add,), kwargs = {})
#   return %embedding_1,%add,%clone_62
sdsc_fused_add_embedding_2 = async_compile.sdsc('sdsc_fused_add_embedding_2',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7398426910028617785, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/vocab_parallel_embedding.py', start_line=78, start_col=0, end_line=None, end_col=None), aten_op='aten.embedding.default', ir_chain=('embedding_1', 'buf3'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.IEEE_INT32,
                    device_size=[1, 64, 32],
                    device_coordinates=[sympify('0'), sympify('floor(c0/32)'), sympify('Mod(c0, 32)')],
                    allocation={'hbm': 0},
                    name='buf2',
                ),
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[64, 12, 64],
                    device_coordinates=[IndirectAccess('buf2'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=527992599741040904, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/models/roberta.py', start_line=76, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add', 'buf4'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8313866838364603182, source=None, aten_op=None, ir_chain=('clone_62', 'buf314'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_62 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_62]
#   %add : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add]
#   %spyre_offset_positions : Tensor  = PlaceHolder[target=spyre_offset_positions]
#   %clone_63 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone_62,), kwargs = {})
#   return %clone_63
sdsc_fused_3 = async_compile.sdsc('sdsc_fused_3',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5184280036554114222, source=None, aten_op=None, ir_chain=('clone_63', 'buf315'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [output_2, embeddings, hidden_states, out, out_1], Original ATen: [aten.embedding, aten.add, aten.layer_norm, aten.mm]
# Source node to ATen node mapping:
#   embeddings => add_1
#   hidden_states => exx2, layernormnorm, layernormscale
#   out => mm
#   out_1 => add_2
#   output_2 => embedding_2
# Graph fragment:
#   %spyre_offset_positions : Tensor "i64[2048][1]spyre:0" = PlaceHolder[target=spyre_offset_positions]
#   %arg4_1 : Tensor "f16[576, 768][768, 1]spyre:0" = PlaceHolder[target=arg4_1]
#   %clone_63 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_63]
#   %embedding_2 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=embedding_2]
#   %add_1 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_1]
#   %exx2 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2]
#   %layernormscale : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale]
#   %arg5_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg6_1]
#   %layernormnorm : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm]
#   %clone : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone]
#   %arg8_1 : Tensor "f16[768, 2304][2304, 1]spyre:0" = PlaceHolder[target=arg8_1]
#   %mm : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=mm]
#   %clone_1 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=clone_1]
#   %arg7_1 : Tensor "f16[2304][1]spyre:0" = PlaceHolder[target=arg7_1]
#   %embedding_2 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %spyre_offset_positions), kwargs = {})
#   %add_1 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %embedding_2), kwargs = {})
#   %exx2 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_1, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2, 1e-05), kwargs = {})
#   %layernormnorm : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=3] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_1, %exx2, %layernormscale, %arg5_1, %arg6_1), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm,), kwargs = {})
#   %mm : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone, %arg8_1), kwargs = {})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm,), kwargs = {})
#   %add_2 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_1, %arg7_1), kwargs = {})
#   return %embedding_2,%add_1,%exx2,%layernormscale,%layernormnorm,%clone,%mm,%clone_1,%add_2
sdsc_fused_add_embedding_layer_norm_mm_4 = async_compile.sdsc('sdsc_fused_add_embedding_layer_norm_mm_4',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8917663811269749035, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/vocab_parallel_embedding.py', start_line=78, start_col=0, end_line=None, end_col=None), aten_op='aten.embedding.default', ir_chain=('embedding_2', 'buf7'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.IEEE_INT32,
                    device_size=[1, 64, 32],
                    device_coordinates=[sympify('0'), sympify('floor(c0/32)'), sympify('Mod(c0, 32)')],
                    allocation={'hbm': 0},
                    name='buf6',
                ),
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[576, 12, 64],
                    device_coordinates=[IndirectAccess('buf6'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4755871796273459158, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/models/roberta.py', start_line=76, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_1', 'buf8'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
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
            debug_handle=DebugHandle(id=321420728071153066, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/models/roberta.py', start_line=82, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2', 'buf9'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=1584223809853715012, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/models/roberta.py', start_line=82, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale', 'buf10'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=1958063469504535276, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/models/roberta.py', start_line=82, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm', 'buf11'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8496362472285676953, source=None, aten_op=None, ir_chain=('clone', 'buf252'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 114688},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 8}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('2304'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=430287890490278656, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm', 'buf12'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 114688},
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1107499551028304405, source=None, aten_op=None, ir_chain=('clone_1', 'buf253'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8547563582301853493, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_2', 'buf13'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 6},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %layernormnorm : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm]
#   %clone_64 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm,), kwargs = {})
#   return %clone_64
sdsc_fused_5 = async_compile.sdsc('sdsc_fused_5',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2701904500764781221, source=None, aten_op=None, ir_chain=('clone_64', 'buf316'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [out_2, out_3, add_4, hidden_states_1, out_4, out_5, hidden_states_2, out_6, out_7, add_7, hidden_states_3, out_8, out_9], Original ATen: [aten.mm, aten.add, aten.layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   add_4 => add_4
#   add_7 => add_7
#   hidden_states_1 => exx2_1, layernormnorm_1, layernormscale_1
#   hidden_states_2 => gelu
#   hidden_states_3 => exx2_2, layernormnorm_2, layernormscale_2
#   out_2 => mm_1
#   out_3 => add_3
#   out_4 => mm_2
#   out_5 => add_5
#   out_6 => mm_3
#   out_7 => add_6
#   out_8 => mm_4
#   out_9 => add_8
# Graph fragment:
#   %clone_64 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_64]
#   %layernormnorm : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm]
#   %clone : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone]
#   %buf15 : Tensor  = PlaceHolder[target=buf15]
#   %buf16 : Tensor  = PlaceHolder[target=buf16]
#   %arg11_1 : Tensor "f16[768, 768][768, 1]spyre:0" = PlaceHolder[target=arg11_1]
#   %mm_1 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_1]
#   %clone_2 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_2]
#   %arg10_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg10_1]
#   %add_3 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_3]
#   %clone_65 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_65]
#   %add_4 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_4]
#   %exx2_1 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_1]
#   %layernormscale_1 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_1]
#   %arg12_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg12_1]
#   %arg13_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg13_1]
#   %layernormnorm_1 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_1]
#   %clone_3 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_3]
#   %arg15_1 : Tensor "f16[768, 3072][3072, 1]spyre:0" = PlaceHolder[target=arg15_1]
#   %mm_2 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=mm_2]
#   %arg14_1 : Tensor "f16[3072][1]spyre:0" = PlaceHolder[target=arg14_1]
#   %add_5 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=add_5]
#   %gelu : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=gelu]
#   %arg17_1 : Tensor "f16[3072, 768][768, 1]spyre:0" = PlaceHolder[target=arg17_1]
#   %mm_3 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_3]
#   %clone_4 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_4]
#   %arg16_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg16_1]
#   %add_6 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_6]
#   %add_7 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_7]
#   %exx2_2 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_2]
#   %layernormscale_2 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_2]
#   %arg18_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg19_1]
#   %layernormnorm_2 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_2]
#   %clone_5 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_5]
#   %arg21_1 : Tensor "f16[768, 2304][2304, 1]spyre:0" = PlaceHolder[target=arg21_1]
#   %mm_4 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=mm_4]
#   %clone_6 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=clone_6]
#   %arg20_1 : Tensor "f16[2304][1]spyre:0" = PlaceHolder[target=arg20_1]
#   %clone_65 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone_64,), kwargs = {})
#   %mm_1 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%empty, %arg11_1), kwargs = {})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_1,), kwargs = {})
#   %add_3 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_2, %arg10_1), kwargs = {})
#   %add_4 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %layernormnorm), kwargs = {})
#   %exx2_1 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_4, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_1 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_1, 1e-05), kwargs = {})
#   %layernormnorm_1 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_4, %exx2_1, %layernormscale_1, %arg12_1, %arg13_1), kwargs = {})
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_1,), kwargs = {})
#   %mm_2 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_3, %arg15_1), kwargs = {})
#   %add_5 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_2, %arg14_1), kwargs = {})
#   %gelu : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%add_5,), kwargs = {})
#   %mm_3 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%gelu, %arg17_1), kwargs = {})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_3,), kwargs = {})
#   %add_6 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_4, %arg16_1), kwargs = {})
#   %add_7 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %layernormnorm_1), kwargs = {})
#   %exx2_2 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_7, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_2 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_2, 1e-05), kwargs = {})
#   %layernormnorm_2 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=3] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_7, %exx2_2, %layernormscale_2, %arg18_1, %arg19_1), kwargs = {})
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_2,), kwargs = {})
#   %mm_4 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_5, %arg21_1), kwargs = {})
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_4,), kwargs = {})
#   %add_8 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_6, %arg20_1), kwargs = {})
#   return %clone_65,%mm_1,%clone_2,%add_3,%add_4,%exx2_1,%layernormscale_1,%layernormnorm_1,%clone_3,%mm_2,%add_5,%gelu,%mm_3,%clone_4,%add_6,%add_7,%exx2_2,%layernormscale_2,%layernormnorm_2,%clone_5,%mm_4,%clone_6,%add_8
sdsc_fused_add_gelu_layer_norm_mm_6 = async_compile.sdsc('sdsc_fused_add_gelu_layer_norm_mm_6',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5291475265399398059, source=None, aten_op=None, ir_chain=('clone_65', 'buf317'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2079926537444466310, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_1', 'buf17'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8976756949028546600, source=None, aten_op=None, ir_chain=('clone_2', 'buf254'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2367199995310006709, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_3', 'buf18'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8907411221441604694, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_4', 'buf19'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=6941978794203223679, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_1', 'buf20'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=1627205672226506830, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_1', 'buf21'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
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
                    allocation={'lx': 393216},
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
            debug_handle=DebugHandle(id=4433366008264883618, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_1', 'buf22'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
                    allocation={'lx': 393216},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1223795052212682774, source=None, aten_op=None, ir_chain=('clone_3', 'buf255'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 401408},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 4}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('3072'), 8), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 4)'), sympify('c1'): sympify('Mod(floor(core_id/4), 8)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=901974512336868189, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_2', 'buf23'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 401408},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8886874049936214862, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_5', 'buf24'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 48, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 7},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='gelufwd',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3076433723360343785, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/activation.py', start_line=372, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu', 'buf25'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1056294828060242111, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_3', 'buf26'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 3072, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 8},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8977811805608500193, source=None, aten_op=None, ir_chain=('clone_4', 'buf256'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7650829536030928210, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_6', 'buf27'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=True, arg_index=9, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 9},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1221650506557275798, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_7', 'buf28'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
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
            debug_handle=DebugHandle(id=7074179379598218243, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_2', 'buf29'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=1321115435312306028, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_2', 'buf30'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=509988247977765752, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_2', 'buf31'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
                    is_input=True, arg_index=10, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 10},
                ),
                TensorArg(
                    is_input=True, arg_index=11, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 11},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8943317561758690238, source=None, aten_op=None, ir_chain=('clone_5', 'buf257'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 114688},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 8}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('2304'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=684601368180192284, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_4', 'buf32'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 114688},
                ),
                TensorArg(
                    is_input=True, arg_index=12, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 12},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1087701785344114456, source=None, aten_op=None, ir_chain=('clone_6', 'buf258'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2043286074886249465, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_8', 'buf33'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                ),
                TensorArg(
                    is_input=True, arg_index=13, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 13},
                ),
                TensorArg(
                    is_input=False, arg_index=14, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 14},
                ),
            ]
        ),
    ]
, pool_size=12582912)


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %layernormnorm_2 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_2]
#   %clone_66 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_2,), kwargs = {})
#   return %clone_66
sdsc_fused_7 = async_compile.sdsc('sdsc_fused_7',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4468347713428936894, source=None, aten_op=None, ir_chain=('clone_66', 'buf318'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [out_10, out_11, add_10, hidden_states_4, out_12, out_13, hidden_states_5, out_14, out_15, add_13, hidden_states_6, out_16, out_17], Original ATen: [aten.mm, aten.add, aten.layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   add_10 => add_10
#   add_13 => add_13
#   hidden_states_4 => exx2_3, layernormnorm_3, layernormscale_3
#   hidden_states_5 => gelu_1
#   hidden_states_6 => exx2_4, layernormnorm_4, layernormscale_4
#   out_10 => mm_5
#   out_11 => add_9
#   out_12 => mm_6
#   out_13 => add_11
#   out_14 => mm_7
#   out_15 => add_12
#   out_16 => mm_8
#   out_17 => add_14
# Graph fragment:
#   %clone_66 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_66]
#   %layernormnorm_2 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_2]
#   %clone_5 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_5]
#   %buf35 : Tensor  = PlaceHolder[target=buf35]
#   %buf36 : Tensor  = PlaceHolder[target=buf36]
#   %arg24_1 : Tensor "f16[768, 768][768, 1]spyre:0" = PlaceHolder[target=arg24_1]
#   %mm_5 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_5]
#   %clone_7 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_7]
#   %arg23_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg23_1]
#   %add_9 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_9]
#   %clone_67 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_67]
#   %add_10 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_10]
#   %exx2_3 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_3]
#   %layernormscale_3 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_3]
#   %arg25_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg25_1]
#   %arg26_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg26_1]
#   %layernormnorm_3 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_3]
#   %clone_8 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_8]
#   %arg28_1 : Tensor "f16[768, 3072][3072, 1]spyre:0" = PlaceHolder[target=arg28_1]
#   %mm_6 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=mm_6]
#   %arg27_1 : Tensor "f16[3072][1]spyre:0" = PlaceHolder[target=arg27_1]
#   %add_11 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=add_11]
#   %gelu_1 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=gelu_1]
#   %arg30_1 : Tensor "f16[3072, 768][768, 1]spyre:0" = PlaceHolder[target=arg30_1]
#   %mm_7 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_7]
#   %clone_9 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_9]
#   %arg29_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg29_1]
#   %add_12 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_12]
#   %add_13 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_13]
#   %exx2_4 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_4]
#   %layernormscale_4 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_4]
#   %arg31_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg31_1]
#   %arg32_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg32_1]
#   %layernormnorm_4 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_4]
#   %clone_10 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_10]
#   %arg34_1 : Tensor "f16[768, 2304][2304, 1]spyre:0" = PlaceHolder[target=arg34_1]
#   %mm_8 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=mm_8]
#   %clone_11 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=clone_11]
#   %arg33_1 : Tensor "f16[2304][1]spyre:0" = PlaceHolder[target=arg33_1]
#   %clone_67 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone_66,), kwargs = {})
#   %mm_5 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%empty_1, %arg24_1), kwargs = {})
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_5,), kwargs = {})
#   %add_9 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_7, %arg23_1), kwargs = {})
#   %add_10 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %layernormnorm_2), kwargs = {})
#   %exx2_3 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_10, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_3 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_3, 1e-05), kwargs = {})
#   %layernormnorm_3 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_10, %exx2_3, %layernormscale_3, %arg25_1, %arg26_1), kwargs = {})
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_3,), kwargs = {})
#   %mm_6 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_8, %arg28_1), kwargs = {})
#   %add_11 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_6, %arg27_1), kwargs = {})
#   %gelu_1 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%add_11,), kwargs = {})
#   %mm_7 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%gelu_1, %arg30_1), kwargs = {})
#   %clone_9 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_7,), kwargs = {})
#   %add_12 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_9, %arg29_1), kwargs = {})
#   %add_13 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %layernormnorm_3), kwargs = {})
#   %exx2_4 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_13, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_4 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_4, 1e-05), kwargs = {})
#   %layernormnorm_4 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=3] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_13, %exx2_4, %layernormscale_4, %arg31_1, %arg32_1), kwargs = {})
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_4,), kwargs = {})
#   %mm_8 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_10, %arg34_1), kwargs = {})
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_8,), kwargs = {})
#   %add_14 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_11, %arg33_1), kwargs = {})
#   return %clone_67,%mm_5,%clone_7,%add_9,%add_10,%exx2_3,%layernormscale_3,%layernormnorm_3,%clone_8,%mm_6,%add_11,%gelu_1,%mm_7,%clone_9,%add_12,%add_13,%exx2_4,%layernormscale_4,%layernormnorm_4,%clone_10,%mm_8,%clone_11,%add_14
sdsc_fused_add_gelu_layer_norm_mm_8 = async_compile.sdsc('sdsc_fused_add_gelu_layer_norm_mm_8',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=446928610847224911, source=None, aten_op=None, ir_chain=('clone_67', 'buf319'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7480162116206994786, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_5', 'buf37'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8445524334202394923, source=None, aten_op=None, ir_chain=('clone_7', 'buf259'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2634238991622358442, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_9', 'buf38'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=799968278693423438, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_10', 'buf39'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=2962231171011173336, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_3', 'buf40'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=3083497755310994524, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_3', 'buf41'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
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
                    allocation={'lx': 393216},
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
            debug_handle=DebugHandle(id=6395772226544634149, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_3', 'buf42'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
                    allocation={'lx': 393216},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5276640537103008626, source=None, aten_op=None, ir_chain=('clone_8', 'buf260'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 401408},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 4}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('3072'), 8), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 4)'), sympify('c1'): sympify('Mod(floor(core_id/4), 8)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8129354367243037308, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_6', 'buf43'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 401408},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=405054642332523263, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_11', 'buf44'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 48, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 7},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='gelufwd',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7542382734565521181, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/activation.py', start_line=372, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu_1', 'buf45'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=924976982876181816, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_7', 'buf46'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 3072, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 8},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5950393607290291749, source=None, aten_op=None, ir_chain=('clone_9', 'buf261'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8286339006543884477, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_12', 'buf47'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=True, arg_index=9, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 9},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4981456203889357182, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_13', 'buf48'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
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
            debug_handle=DebugHandle(id=4650516844512352027, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_4', 'buf49'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=7406981093612509251, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_4', 'buf50'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=3716058376443967195, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_4', 'buf51'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
                    is_input=True, arg_index=10, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 10},
                ),
                TensorArg(
                    is_input=True, arg_index=11, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 11},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5757442350564206553, source=None, aten_op=None, ir_chain=('clone_10', 'buf262'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 114688},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 8}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('2304'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1656002378344356304, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_8', 'buf52'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 114688},
                ),
                TensorArg(
                    is_input=True, arg_index=12, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 12},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4881080511207821979, source=None, aten_op=None, ir_chain=('clone_11', 'buf263'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=891906057767758942, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_14', 'buf53'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                ),
                TensorArg(
                    is_input=True, arg_index=13, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 13},
                ),
                TensorArg(
                    is_input=False, arg_index=14, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 14},
                ),
            ]
        ),
    ]
, pool_size=12582912)


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %layernormnorm_4 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_4]
#   %clone_68 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_4,), kwargs = {})
#   return %clone_68
sdsc_fused_9 = async_compile.sdsc('sdsc_fused_9',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2153298630044304672, source=None, aten_op=None, ir_chain=('clone_68', 'buf320'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [out_18, out_19, add_16, hidden_states_7, out_20, out_21, hidden_states_8, out_22, out_23, add_19, hidden_states_9, out_24, out_25], Original ATen: [aten.mm, aten.add, aten.layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   add_16 => add_16
#   add_19 => add_19
#   hidden_states_7 => exx2_5, layernormnorm_5, layernormscale_5
#   hidden_states_8 => gelu_2
#   hidden_states_9 => exx2_6, layernormnorm_6, layernormscale_6
#   out_18 => mm_9
#   out_19 => add_15
#   out_20 => mm_10
#   out_21 => add_17
#   out_22 => mm_11
#   out_23 => add_18
#   out_24 => mm_12
#   out_25 => add_20
# Graph fragment:
#   %clone_68 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_68]
#   %layernormnorm_4 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_4]
#   %clone_10 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_10]
#   %buf55 : Tensor  = PlaceHolder[target=buf55]
#   %buf56 : Tensor  = PlaceHolder[target=buf56]
#   %arg37_1 : Tensor "f16[768, 768][768, 1]spyre:0" = PlaceHolder[target=arg37_1]
#   %mm_9 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_9]
#   %clone_12 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_12]
#   %arg36_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg36_1]
#   %add_15 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_15]
#   %clone_69 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_69]
#   %add_16 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_16]
#   %exx2_5 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_5]
#   %layernormscale_5 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_5]
#   %arg38_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg38_1]
#   %arg39_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg39_1]
#   %layernormnorm_5 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_5]
#   %clone_13 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_13]
#   %arg41_1 : Tensor "f16[768, 3072][3072, 1]spyre:0" = PlaceHolder[target=arg41_1]
#   %mm_10 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=mm_10]
#   %arg40_1 : Tensor "f16[3072][1]spyre:0" = PlaceHolder[target=arg40_1]
#   %add_17 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=add_17]
#   %gelu_2 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=gelu_2]
#   %arg43_1 : Tensor "f16[3072, 768][768, 1]spyre:0" = PlaceHolder[target=arg43_1]
#   %mm_11 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_11]
#   %clone_14 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_14]
#   %arg42_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg42_1]
#   %add_18 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_18]
#   %add_19 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_19]
#   %exx2_6 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_6]
#   %layernormscale_6 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_6]
#   %arg44_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg44_1]
#   %arg45_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg45_1]
#   %layernormnorm_6 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_6]
#   %clone_15 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_15]
#   %arg47_1 : Tensor "f16[768, 2304][2304, 1]spyre:0" = PlaceHolder[target=arg47_1]
#   %mm_12 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=mm_12]
#   %clone_16 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=clone_16]
#   %arg46_1 : Tensor "f16[2304][1]spyre:0" = PlaceHolder[target=arg46_1]
#   %clone_69 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone_68,), kwargs = {})
#   %mm_9 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%empty_2, %arg37_1), kwargs = {})
#   %clone_12 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_9,), kwargs = {})
#   %add_15 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_12, %arg36_1), kwargs = {})
#   %add_16 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %layernormnorm_4), kwargs = {})
#   %exx2_5 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_16, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_5 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_5, 1e-05), kwargs = {})
#   %layernormnorm_5 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_16, %exx2_5, %layernormscale_5, %arg38_1, %arg39_1), kwargs = {})
#   %clone_13 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_5,), kwargs = {})
#   %mm_10 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_13, %arg41_1), kwargs = {})
#   %add_17 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_10, %arg40_1), kwargs = {})
#   %gelu_2 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%add_17,), kwargs = {})
#   %mm_11 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%gelu_2, %arg43_1), kwargs = {})
#   %clone_14 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_11,), kwargs = {})
#   %add_18 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_14, %arg42_1), kwargs = {})
#   %add_19 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %layernormnorm_5), kwargs = {})
#   %exx2_6 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_19, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_6 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_6, 1e-05), kwargs = {})
#   %layernormnorm_6 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=3] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_19, %exx2_6, %layernormscale_6, %arg44_1, %arg45_1), kwargs = {})
#   %clone_15 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_6,), kwargs = {})
#   %mm_12 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_15, %arg47_1), kwargs = {})
#   %clone_16 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_12,), kwargs = {})
#   %add_20 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_16, %arg46_1), kwargs = {})
#   return %clone_69,%mm_9,%clone_12,%add_15,%add_16,%exx2_5,%layernormscale_5,%layernormnorm_5,%clone_13,%mm_10,%add_17,%gelu_2,%mm_11,%clone_14,%add_18,%add_19,%exx2_6,%layernormscale_6,%layernormnorm_6,%clone_15,%mm_12,%clone_16,%add_20
sdsc_fused_add_gelu_layer_norm_mm_10 = async_compile.sdsc('sdsc_fused_add_gelu_layer_norm_mm_10',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7112975677284757589, source=None, aten_op=None, ir_chain=('clone_69', 'buf321'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1220572866271673743, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_9', 'buf57'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3240358155588956827, source=None, aten_op=None, ir_chain=('clone_12', 'buf264'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4765385575736330571, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_15', 'buf58'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5440327030645324637, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_16', 'buf59'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=59617861397768925, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_5', 'buf60'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=7729562349752628605, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_5', 'buf61'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
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
                    allocation={'lx': 393216},
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
            debug_handle=DebugHandle(id=7775572671224567852, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_5', 'buf62'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
                    allocation={'lx': 393216},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2850288057401970327, source=None, aten_op=None, ir_chain=('clone_13', 'buf265'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 401408},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 4}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('3072'), 8), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 4)'), sympify('c1'): sympify('Mod(floor(core_id/4), 8)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=363306980863509184, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_10', 'buf63'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 401408},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4629429600234980348, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_17', 'buf64'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 48, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 7},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='gelufwd',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3022434909699430858, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/activation.py', start_line=372, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu_2', 'buf65'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5814405026436473945, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_11', 'buf66'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 3072, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 8},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4493525052947549982, source=None, aten_op=None, ir_chain=('clone_14', 'buf266'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7171030278309022216, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_18', 'buf67'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=True, arg_index=9, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 9},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2463908162659167776, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_19', 'buf68'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
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
            debug_handle=DebugHandle(id=1882378407200373857, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_6', 'buf69'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=6190147427258396198, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_6', 'buf70'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=6067651916862504008, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_6', 'buf71'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
                    is_input=True, arg_index=10, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 10},
                ),
                TensorArg(
                    is_input=True, arg_index=11, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 11},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7623668198765284439, source=None, aten_op=None, ir_chain=('clone_15', 'buf267'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 114688},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 8}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('2304'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8187946079773842783, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_12', 'buf72'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 114688},
                ),
                TensorArg(
                    is_input=True, arg_index=12, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 12},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4283361023978855774, source=None, aten_op=None, ir_chain=('clone_16', 'buf268'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4891249233206071874, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_20', 'buf73'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                ),
                TensorArg(
                    is_input=True, arg_index=13, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 13},
                ),
                TensorArg(
                    is_input=False, arg_index=14, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 14},
                ),
            ]
        ),
    ]
, pool_size=12582912)


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %layernormnorm_6 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_6]
#   %clone_70 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_6,), kwargs = {})
#   return %clone_70
sdsc_fused_11 = async_compile.sdsc('sdsc_fused_11',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7099080003835973283, source=None, aten_op=None, ir_chain=('clone_70', 'buf322'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [out_26, out_27, add_22, hidden_states_10, out_28, out_29, hidden_states_11, out_30, out_31, add_25, hidden_states_12, out_32, out_33], Original ATen: [aten.mm, aten.add, aten.layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   add_22 => add_22
#   add_25 => add_25
#   hidden_states_10 => exx2_7, layernormnorm_7, layernormscale_7
#   hidden_states_11 => gelu_3
#   hidden_states_12 => exx2_8, layernormnorm_8, layernormscale_8
#   out_26 => mm_13
#   out_27 => add_21
#   out_28 => mm_14
#   out_29 => add_23
#   out_30 => mm_15
#   out_31 => add_24
#   out_32 => mm_16
#   out_33 => add_26
# Graph fragment:
#   %clone_70 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_70]
#   %layernormnorm_6 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_6]
#   %clone_15 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_15]
#   %buf75 : Tensor  = PlaceHolder[target=buf75]
#   %buf76 : Tensor  = PlaceHolder[target=buf76]
#   %arg50_1 : Tensor "f16[768, 768][768, 1]spyre:0" = PlaceHolder[target=arg50_1]
#   %mm_13 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_13]
#   %clone_17 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_17]
#   %arg49_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg49_1]
#   %add_21 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_21]
#   %clone_71 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_71]
#   %add_22 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_22]
#   %exx2_7 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_7]
#   %layernormscale_7 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_7]
#   %arg51_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg51_1]
#   %arg52_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg52_1]
#   %layernormnorm_7 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_7]
#   %clone_18 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_18]
#   %arg54_1 : Tensor "f16[768, 3072][3072, 1]spyre:0" = PlaceHolder[target=arg54_1]
#   %mm_14 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=mm_14]
#   %arg53_1 : Tensor "f16[3072][1]spyre:0" = PlaceHolder[target=arg53_1]
#   %add_23 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=add_23]
#   %gelu_3 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=gelu_3]
#   %arg56_1 : Tensor "f16[3072, 768][768, 1]spyre:0" = PlaceHolder[target=arg56_1]
#   %mm_15 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_15]
#   %clone_19 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_19]
#   %arg55_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg55_1]
#   %add_24 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_24]
#   %add_25 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_25]
#   %exx2_8 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_8]
#   %layernormscale_8 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_8]
#   %arg57_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg57_1]
#   %arg58_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg58_1]
#   %layernormnorm_8 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_8]
#   %clone_20 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_20]
#   %arg60_1 : Tensor "f16[768, 2304][2304, 1]spyre:0" = PlaceHolder[target=arg60_1]
#   %mm_16 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=mm_16]
#   %clone_21 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=clone_21]
#   %arg59_1 : Tensor "f16[2304][1]spyre:0" = PlaceHolder[target=arg59_1]
#   %clone_71 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone_70,), kwargs = {})
#   %mm_13 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%empty_3, %arg50_1), kwargs = {})
#   %clone_17 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_13,), kwargs = {})
#   %add_21 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_17, %arg49_1), kwargs = {})
#   %add_22 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %layernormnorm_6), kwargs = {})
#   %exx2_7 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_22, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_7 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_7, 1e-05), kwargs = {})
#   %layernormnorm_7 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_22, %exx2_7, %layernormscale_7, %arg51_1, %arg52_1), kwargs = {})
#   %clone_18 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_7,), kwargs = {})
#   %mm_14 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_18, %arg54_1), kwargs = {})
#   %add_23 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_14, %arg53_1), kwargs = {})
#   %gelu_3 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%add_23,), kwargs = {})
#   %mm_15 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%gelu_3, %arg56_1), kwargs = {})
#   %clone_19 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_15,), kwargs = {})
#   %add_24 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_19, %arg55_1), kwargs = {})
#   %add_25 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %layernormnorm_7), kwargs = {})
#   %exx2_8 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_25, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_8 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_8, 1e-05), kwargs = {})
#   %layernormnorm_8 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=3] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_25, %exx2_8, %layernormscale_8, %arg57_1, %arg58_1), kwargs = {})
#   %clone_20 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_8,), kwargs = {})
#   %mm_16 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_20, %arg60_1), kwargs = {})
#   %clone_21 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_16,), kwargs = {})
#   %add_26 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_21, %arg59_1), kwargs = {})
#   return %clone_71,%mm_13,%clone_17,%add_21,%add_22,%exx2_7,%layernormscale_7,%layernormnorm_7,%clone_18,%mm_14,%add_23,%gelu_3,%mm_15,%clone_19,%add_24,%add_25,%exx2_8,%layernormscale_8,%layernormnorm_8,%clone_20,%mm_16,%clone_21,%add_26
sdsc_fused_add_gelu_layer_norm_mm_12 = async_compile.sdsc('sdsc_fused_add_gelu_layer_norm_mm_12',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4716803128186950486, source=None, aten_op=None, ir_chain=('clone_71', 'buf323'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8185316283934129231, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_13', 'buf77'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8798871347769256665, source=None, aten_op=None, ir_chain=('clone_17', 'buf269'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2959215733578917022, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_21', 'buf78'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1000659191303527948, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_22', 'buf79'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=8560818341395493223, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_7', 'buf80'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=3106205740276361650, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_7', 'buf81'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
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
                    allocation={'lx': 393216},
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
            debug_handle=DebugHandle(id=267688163930170608, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_7', 'buf82'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
                    allocation={'lx': 393216},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5614553324580778816, source=None, aten_op=None, ir_chain=('clone_18', 'buf270'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 401408},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 4}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('3072'), 8), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 4)'), sympify('c1'): sympify('Mod(floor(core_id/4), 8)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7678099582199038197, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_14', 'buf83'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 401408},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6453139308136299383, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_23', 'buf84'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 48, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 7},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='gelufwd',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1396393421618931825, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/activation.py', start_line=372, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu_3', 'buf85'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=842613550000810101, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_15', 'buf86'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 3072, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 8},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=130320840314235075, source=None, aten_op=None, ir_chain=('clone_19', 'buf271'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4141405623376103214, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_24', 'buf87'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=True, arg_index=9, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 9},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2885713015502497610, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_25', 'buf88'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
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
            debug_handle=DebugHandle(id=8939941680286577635, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_8', 'buf89'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=1868854443954555134, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_8', 'buf90'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=739942494918558189, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_8', 'buf91'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
                    is_input=True, arg_index=10, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 10},
                ),
                TensorArg(
                    is_input=True, arg_index=11, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 11},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1123322824945349864, source=None, aten_op=None, ir_chain=('clone_20', 'buf272'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 114688},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 8}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('2304'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8298257750208754064, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_16', 'buf92'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 114688},
                ),
                TensorArg(
                    is_input=True, arg_index=12, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 12},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3730998454738775437, source=None, aten_op=None, ir_chain=('clone_21', 'buf273'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=9210222283907998881, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_26', 'buf93'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                ),
                TensorArg(
                    is_input=True, arg_index=13, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 13},
                ),
                TensorArg(
                    is_input=False, arg_index=14, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 14},
                ),
            ]
        ),
    ]
, pool_size=12582912)


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %layernormnorm_8 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_8]
#   %clone_72 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_8,), kwargs = {})
#   return %clone_72
sdsc_fused_13 = async_compile.sdsc('sdsc_fused_13',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8146152874925729440, source=None, aten_op=None, ir_chain=('clone_72', 'buf324'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [out_34, out_35, add_28, hidden_states_13, out_36, out_37, hidden_states_14, out_38, out_39, add_31, hidden_states_15, out_40, out_41], Original ATen: [aten.mm, aten.add, aten.layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   add_28 => add_28
#   add_31 => add_31
#   hidden_states_13 => exx2_9, layernormnorm_9, layernormscale_9
#   hidden_states_14 => gelu_4
#   hidden_states_15 => exx2_10, layernormnorm_10, layernormscale_10
#   out_34 => mm_17
#   out_35 => add_27
#   out_36 => mm_18
#   out_37 => add_29
#   out_38 => mm_19
#   out_39 => add_30
#   out_40 => mm_20
#   out_41 => add_32
# Graph fragment:
#   %clone_72 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_72]
#   %layernormnorm_8 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_8]
#   %clone_20 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_20]
#   %buf95 : Tensor  = PlaceHolder[target=buf95]
#   %buf96 : Tensor  = PlaceHolder[target=buf96]
#   %arg63_1 : Tensor "f16[768, 768][768, 1]spyre:0" = PlaceHolder[target=arg63_1]
#   %mm_17 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_17]
#   %clone_22 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_22]
#   %arg62_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg62_1]
#   %add_27 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_27]
#   %clone_73 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_73]
#   %add_28 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_28]
#   %exx2_9 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_9]
#   %layernormscale_9 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_9]
#   %arg64_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg64_1]
#   %arg65_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg65_1]
#   %layernormnorm_9 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_9]
#   %clone_23 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_23]
#   %arg67_1 : Tensor "f16[768, 3072][3072, 1]spyre:0" = PlaceHolder[target=arg67_1]
#   %mm_18 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=mm_18]
#   %arg66_1 : Tensor "f16[3072][1]spyre:0" = PlaceHolder[target=arg66_1]
#   %add_29 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=add_29]
#   %gelu_4 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=gelu_4]
#   %arg69_1 : Tensor "f16[3072, 768][768, 1]spyre:0" = PlaceHolder[target=arg69_1]
#   %mm_19 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_19]
#   %clone_24 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_24]
#   %arg68_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg68_1]
#   %add_30 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_30]
#   %add_31 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_31]
#   %exx2_10 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_10]
#   %layernormscale_10 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_10]
#   %arg70_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg70_1]
#   %arg71_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg71_1]
#   %layernormnorm_10 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_10]
#   %clone_25 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_25]
#   %arg73_1 : Tensor "f16[768, 2304][2304, 1]spyre:0" = PlaceHolder[target=arg73_1]
#   %mm_20 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=mm_20]
#   %clone_26 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=clone_26]
#   %arg72_1 : Tensor "f16[2304][1]spyre:0" = PlaceHolder[target=arg72_1]
#   %clone_73 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone_72,), kwargs = {})
#   %mm_17 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%empty_4, %arg63_1), kwargs = {})
#   %clone_22 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_17,), kwargs = {})
#   %add_27 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_22, %arg62_1), kwargs = {})
#   %add_28 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %layernormnorm_8), kwargs = {})
#   %exx2_9 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_28, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_9 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_9, 1e-05), kwargs = {})
#   %layernormnorm_9 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_28, %exx2_9, %layernormscale_9, %arg64_1, %arg65_1), kwargs = {})
#   %clone_23 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_9,), kwargs = {})
#   %mm_18 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_23, %arg67_1), kwargs = {})
#   %add_29 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_18, %arg66_1), kwargs = {})
#   %gelu_4 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%add_29,), kwargs = {})
#   %mm_19 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%gelu_4, %arg69_1), kwargs = {})
#   %clone_24 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_19,), kwargs = {})
#   %add_30 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_24, %arg68_1), kwargs = {})
#   %add_31 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_30, %layernormnorm_9), kwargs = {})
#   %exx2_10 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_31, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_10 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_10, 1e-05), kwargs = {})
#   %layernormnorm_10 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=3] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_31, %exx2_10, %layernormscale_10, %arg70_1, %arg71_1), kwargs = {})
#   %clone_25 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_10,), kwargs = {})
#   %mm_20 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_25, %arg73_1), kwargs = {})
#   %clone_26 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_20,), kwargs = {})
#   %add_32 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_26, %arg72_1), kwargs = {})
#   return %clone_73,%mm_17,%clone_22,%add_27,%add_28,%exx2_9,%layernormscale_9,%layernormnorm_9,%clone_23,%mm_18,%add_29,%gelu_4,%mm_19,%clone_24,%add_30,%add_31,%exx2_10,%layernormscale_10,%layernormnorm_10,%clone_25,%mm_20,%clone_26,%add_32
sdsc_fused_add_gelu_layer_norm_mm_14 = async_compile.sdsc('sdsc_fused_add_gelu_layer_norm_mm_14',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8755565645629404882, source=None, aten_op=None, ir_chain=('clone_73', 'buf325'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=87555311679685493, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_17', 'buf97'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=331589132128783534, source=None, aten_op=None, ir_chain=('clone_22', 'buf274'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5512818844186378733, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_27', 'buf98'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2536367189477579964, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_28', 'buf99'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=3748651920625199240, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_9', 'buf100'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=4488053384450716383, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_9', 'buf101'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
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
                    allocation={'lx': 393216},
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
            debug_handle=DebugHandle(id=1744861651971851596, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_9', 'buf102'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
                    allocation={'lx': 393216},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=750851902309520304, source=None, aten_op=None, ir_chain=('clone_23', 'buf275'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 401408},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 4}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('3072'), 8), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 4)'), sympify('c1'): sympify('Mod(floor(core_id/4), 8)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6365487828117417384, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_18', 'buf103'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 401408},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8309824946018563744, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_29', 'buf104'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 48, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 7},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='gelufwd',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1829427347236682941, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/activation.py', start_line=372, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu_4', 'buf105'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1863260837816221528, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_19', 'buf106'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 3072, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 8},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6477724292840153525, source=None, aten_op=None, ir_chain=('clone_24', 'buf276'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3681107406540163874, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_30', 'buf107'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=True, arg_index=9, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 9},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=989071566586552057, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_31', 'buf108'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
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
            debug_handle=DebugHandle(id=5696613188881841372, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_10', 'buf109'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=8984816036712989791, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_10', 'buf110'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=4342169403487962390, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_10', 'buf111'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
                    is_input=True, arg_index=10, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 10},
                ),
                TensorArg(
                    is_input=True, arg_index=11, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 11},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2805391696671365773, source=None, aten_op=None, ir_chain=('clone_25', 'buf277'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 114688},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 8}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('2304'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6949209193651906518, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_20', 'buf112'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 114688},
                ),
                TensorArg(
                    is_input=True, arg_index=12, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 12},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6464858816780727365, source=None, aten_op=None, ir_chain=('clone_26', 'buf278'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=504237182733477744, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_32', 'buf113'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                ),
                TensorArg(
                    is_input=True, arg_index=13, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 13},
                ),
                TensorArg(
                    is_input=False, arg_index=14, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 14},
                ),
            ]
        ),
    ]
, pool_size=12582912)


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %layernormnorm_10 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_10]
#   %clone_74 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_10,), kwargs = {})
#   return %clone_74
sdsc_fused_15 = async_compile.sdsc('sdsc_fused_15',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1067917097173811509, source=None, aten_op=None, ir_chain=('clone_74', 'buf326'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [out_42, out_43, add_34, hidden_states_16, out_44, out_45, hidden_states_17, out_46, out_47, add_37, hidden_states_18, out_48, out_49], Original ATen: [aten.mm, aten.add, aten.layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   add_34 => add_34
#   add_37 => add_37
#   hidden_states_16 => exx2_11, layernormnorm_11, layernormscale_11
#   hidden_states_17 => gelu_5
#   hidden_states_18 => exx2_12, layernormnorm_12, layernormscale_12
#   out_42 => mm_21
#   out_43 => add_33
#   out_44 => mm_22
#   out_45 => add_35
#   out_46 => mm_23
#   out_47 => add_36
#   out_48 => mm_24
#   out_49 => add_38
# Graph fragment:
#   %clone_74 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_74]
#   %layernormnorm_10 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_10]
#   %clone_25 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_25]
#   %buf115 : Tensor  = PlaceHolder[target=buf115]
#   %buf116 : Tensor  = PlaceHolder[target=buf116]
#   %arg76_1 : Tensor "f16[768, 768][768, 1]spyre:0" = PlaceHolder[target=arg76_1]
#   %mm_21 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_21]
#   %clone_27 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_27]
#   %arg75_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg75_1]
#   %add_33 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_33]
#   %clone_75 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_75]
#   %add_34 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_34]
#   %exx2_11 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_11]
#   %layernormscale_11 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_11]
#   %arg77_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg77_1]
#   %arg78_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg78_1]
#   %layernormnorm_11 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_11]
#   %clone_28 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_28]
#   %arg80_1 : Tensor "f16[768, 3072][3072, 1]spyre:0" = PlaceHolder[target=arg80_1]
#   %mm_22 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=mm_22]
#   %arg79_1 : Tensor "f16[3072][1]spyre:0" = PlaceHolder[target=arg79_1]
#   %add_35 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=add_35]
#   %gelu_5 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=gelu_5]
#   %arg82_1 : Tensor "f16[3072, 768][768, 1]spyre:0" = PlaceHolder[target=arg82_1]
#   %mm_23 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_23]
#   %clone_29 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_29]
#   %arg81_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg81_1]
#   %add_36 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_36]
#   %add_37 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_37]
#   %exx2_12 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_12]
#   %layernormscale_12 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_12]
#   %arg83_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg83_1]
#   %arg84_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg84_1]
#   %layernormnorm_12 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_12]
#   %clone_30 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_30]
#   %arg86_1 : Tensor "f16[768, 2304][2304, 1]spyre:0" = PlaceHolder[target=arg86_1]
#   %mm_24 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=mm_24]
#   %clone_31 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=clone_31]
#   %arg85_1 : Tensor "f16[2304][1]spyre:0" = PlaceHolder[target=arg85_1]
#   %clone_75 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone_74,), kwargs = {})
#   %mm_21 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%empty_5, %arg76_1), kwargs = {})
#   %clone_27 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_21,), kwargs = {})
#   %add_33 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_27, %arg75_1), kwargs = {})
#   %add_34 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %layernormnorm_10), kwargs = {})
#   %exx2_11 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_34, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_11 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_11, 1e-05), kwargs = {})
#   %layernormnorm_11 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_34, %exx2_11, %layernormscale_11, %arg77_1, %arg78_1), kwargs = {})
#   %clone_28 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_11,), kwargs = {})
#   %mm_22 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_28, %arg80_1), kwargs = {})
#   %add_35 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_22, %arg79_1), kwargs = {})
#   %gelu_5 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%add_35,), kwargs = {})
#   %mm_23 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%gelu_5, %arg82_1), kwargs = {})
#   %clone_29 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_23,), kwargs = {})
#   %add_36 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_29, %arg81_1), kwargs = {})
#   %add_37 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_36, %layernormnorm_11), kwargs = {})
#   %exx2_12 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_37, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_12 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_12, 1e-05), kwargs = {})
#   %layernormnorm_12 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=3] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_37, %exx2_12, %layernormscale_12, %arg83_1, %arg84_1), kwargs = {})
#   %clone_30 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_12,), kwargs = {})
#   %mm_24 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_30, %arg86_1), kwargs = {})
#   %clone_31 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_24,), kwargs = {})
#   %add_38 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_31, %arg85_1), kwargs = {})
#   return %clone_75,%mm_21,%clone_27,%add_33,%add_34,%exx2_11,%layernormscale_11,%layernormnorm_11,%clone_28,%mm_22,%add_35,%gelu_5,%mm_23,%clone_29,%add_36,%add_37,%exx2_12,%layernormscale_12,%layernormnorm_12,%clone_30,%mm_24,%clone_31,%add_38
sdsc_fused_add_gelu_layer_norm_mm_16 = async_compile.sdsc('sdsc_fused_add_gelu_layer_norm_mm_16',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8604194242275492403, source=None, aten_op=None, ir_chain=('clone_75', 'buf327'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2121178200872283047, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_21', 'buf117'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8482816665820365259, source=None, aten_op=None, ir_chain=('clone_27', 'buf279'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1581243129755039461, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_33', 'buf118'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7992347725261385140, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_34', 'buf119'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=4740820793393474475, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_11', 'buf120'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=8112538438580832693, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_11', 'buf121'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
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
                    allocation={'lx': 393216},
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
            debug_handle=DebugHandle(id=4403729020370580552, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_11', 'buf122'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
                    allocation={'lx': 393216},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=98131141699400652, source=None, aten_op=None, ir_chain=('clone_28', 'buf280'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 401408},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 4}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('3072'), 8), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 4)'), sympify('c1'): sympify('Mod(floor(core_id/4), 8)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1022730687432727677, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_22', 'buf123'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 401408},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3466456230169957609, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_35', 'buf124'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 48, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 7},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='gelufwd',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2664870051774924543, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/activation.py', start_line=372, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu_5', 'buf125'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1107645576610323634, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_23', 'buf126'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 3072, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 8},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4817308103910022579, source=None, aten_op=None, ir_chain=('clone_29', 'buf281'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6533521603765906823, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_36', 'buf127'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=True, arg_index=9, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 9},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5273518494871292421, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_37', 'buf128'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
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
            debug_handle=DebugHandle(id=5586174606825266524, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_12', 'buf129'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=5081057581202594754, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_12', 'buf130'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=2417667220367632553, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_12', 'buf131'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
                    is_input=True, arg_index=10, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 10},
                ),
                TensorArg(
                    is_input=True, arg_index=11, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 11},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8156141735483545204, source=None, aten_op=None, ir_chain=('clone_30', 'buf282'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 114688},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 8}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('2304'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1969224831688701523, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_24', 'buf132'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 114688},
                ),
                TensorArg(
                    is_input=True, arg_index=12, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 12},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8863648713669172537, source=None, aten_op=None, ir_chain=('clone_31', 'buf283'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7897649948250554052, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_38', 'buf133'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                ),
                TensorArg(
                    is_input=True, arg_index=13, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 13},
                ),
                TensorArg(
                    is_input=False, arg_index=14, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 14},
                ),
            ]
        ),
    ]
, pool_size=12582912)


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %layernormnorm_12 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_12]
#   %clone_76 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_12,), kwargs = {})
#   return %clone_76
sdsc_fused_17 = async_compile.sdsc('sdsc_fused_17',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=76349785935681591, source=None, aten_op=None, ir_chain=('clone_76', 'buf328'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [out_50, out_51, add_40, hidden_states_19, out_52, out_53, hidden_states_20, out_54, out_55, add_43, hidden_states_21, out_56, out_57], Original ATen: [aten.mm, aten.add, aten.layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   add_40 => add_40
#   add_43 => add_43
#   hidden_states_19 => exx2_13, layernormnorm_13, layernormscale_13
#   hidden_states_20 => gelu_6
#   hidden_states_21 => exx2_14, layernormnorm_14, layernormscale_14
#   out_50 => mm_25
#   out_51 => add_39
#   out_52 => mm_26
#   out_53 => add_41
#   out_54 => mm_27
#   out_55 => add_42
#   out_56 => mm_28
#   out_57 => add_44
# Graph fragment:
#   %clone_76 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_76]
#   %layernormnorm_12 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_12]
#   %clone_30 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_30]
#   %buf135 : Tensor  = PlaceHolder[target=buf135]
#   %buf136 : Tensor  = PlaceHolder[target=buf136]
#   %arg89_1 : Tensor "f16[768, 768][768, 1]spyre:0" = PlaceHolder[target=arg89_1]
#   %mm_25 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_25]
#   %clone_32 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_32]
#   %arg88_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg88_1]
#   %add_39 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_39]
#   %clone_77 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_77]
#   %add_40 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_40]
#   %exx2_13 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_13]
#   %layernormscale_13 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_13]
#   %arg90_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg90_1]
#   %arg91_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg91_1]
#   %layernormnorm_13 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_13]
#   %clone_33 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_33]
#   %arg93_1 : Tensor "f16[768, 3072][3072, 1]spyre:0" = PlaceHolder[target=arg93_1]
#   %mm_26 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=mm_26]
#   %arg92_1 : Tensor "f16[3072][1]spyre:0" = PlaceHolder[target=arg92_1]
#   %add_41 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=add_41]
#   %gelu_6 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=gelu_6]
#   %arg95_1 : Tensor "f16[3072, 768][768, 1]spyre:0" = PlaceHolder[target=arg95_1]
#   %mm_27 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_27]
#   %clone_34 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_34]
#   %arg94_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg94_1]
#   %add_42 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_42]
#   %add_43 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_43]
#   %exx2_14 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_14]
#   %layernormscale_14 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_14]
#   %arg96_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg96_1]
#   %arg97_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg97_1]
#   %layernormnorm_14 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_14]
#   %clone_35 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_35]
#   %arg99_1 : Tensor "f16[768, 2304][2304, 1]spyre:0" = PlaceHolder[target=arg99_1]
#   %mm_28 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=mm_28]
#   %clone_36 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=clone_36]
#   %arg98_1 : Tensor "f16[2304][1]spyre:0" = PlaceHolder[target=arg98_1]
#   %clone_77 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone_76,), kwargs = {})
#   %mm_25 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%empty_6, %arg89_1), kwargs = {})
#   %clone_32 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_25,), kwargs = {})
#   %add_39 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_32, %arg88_1), kwargs = {})
#   %add_40 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %layernormnorm_12), kwargs = {})
#   %exx2_13 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_40, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_13 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_13, 1e-05), kwargs = {})
#   %layernormnorm_13 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_40, %exx2_13, %layernormscale_13, %arg90_1, %arg91_1), kwargs = {})
#   %clone_33 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_13,), kwargs = {})
#   %mm_26 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_33, %arg93_1), kwargs = {})
#   %add_41 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_26, %arg92_1), kwargs = {})
#   %gelu_6 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%add_41,), kwargs = {})
#   %mm_27 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%gelu_6, %arg95_1), kwargs = {})
#   %clone_34 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_27,), kwargs = {})
#   %add_42 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_34, %arg94_1), kwargs = {})
#   %add_43 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_42, %layernormnorm_13), kwargs = {})
#   %exx2_14 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_43, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_14 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_14, 1e-05), kwargs = {})
#   %layernormnorm_14 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=3] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_43, %exx2_14, %layernormscale_14, %arg96_1, %arg97_1), kwargs = {})
#   %clone_35 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_14,), kwargs = {})
#   %mm_28 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_35, %arg99_1), kwargs = {})
#   %clone_36 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_28,), kwargs = {})
#   %add_44 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_36, %arg98_1), kwargs = {})
#   return %clone_77,%mm_25,%clone_32,%add_39,%add_40,%exx2_13,%layernormscale_13,%layernormnorm_13,%clone_33,%mm_26,%add_41,%gelu_6,%mm_27,%clone_34,%add_42,%add_43,%exx2_14,%layernormscale_14,%layernormnorm_14,%clone_35,%mm_28,%clone_36,%add_44
sdsc_fused_add_gelu_layer_norm_mm_18 = async_compile.sdsc('sdsc_fused_add_gelu_layer_norm_mm_18',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5226543353422723293, source=None, aten_op=None, ir_chain=('clone_77', 'buf329'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5996389493304417768, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_25', 'buf137'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4010204523258133409, source=None, aten_op=None, ir_chain=('clone_32', 'buf284'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8584095527353861389, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_39', 'buf138'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5905839376188586690, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_40', 'buf139'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=7942088828975008992, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_13', 'buf140'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=6147552905763580429, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_13', 'buf141'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
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
                    allocation={'lx': 393216},
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
            debug_handle=DebugHandle(id=2051817251058533347, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_13', 'buf142'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
                    allocation={'lx': 393216},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8632955834796647815, source=None, aten_op=None, ir_chain=('clone_33', 'buf285'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 401408},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 4}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('3072'), 8), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 4)'), sympify('c1'): sympify('Mod(floor(core_id/4), 8)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=9082851284090878207, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_26', 'buf143'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 401408},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1740244696548844448, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_41', 'buf144'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 48, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 7},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='gelufwd',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3175418308725615451, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/activation.py', start_line=372, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu_6', 'buf145'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6897375758880067727, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_27', 'buf146'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 3072, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 8},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7962755715582240254, source=None, aten_op=None, ir_chain=('clone_34', 'buf286'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3587499525396291454, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_42', 'buf147'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=True, arg_index=9, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 9},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6244695271330098738, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_43', 'buf148'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
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
            debug_handle=DebugHandle(id=7077133370645221821, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_14', 'buf149'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=9157500347201319826, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_14', 'buf150'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=1389478972768564155, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_14', 'buf151'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
                    is_input=True, arg_index=10, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 10},
                ),
                TensorArg(
                    is_input=True, arg_index=11, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 11},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7286048805083155019, source=None, aten_op=None, ir_chain=('clone_35', 'buf287'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 114688},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 8}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('2304'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6088148959657353614, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_28', 'buf152'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 114688},
                ),
                TensorArg(
                    is_input=True, arg_index=12, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 12},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8883145682270729689, source=None, aten_op=None, ir_chain=('clone_36', 'buf288'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6004979043900752387, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_44', 'buf153'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                ),
                TensorArg(
                    is_input=True, arg_index=13, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 13},
                ),
                TensorArg(
                    is_input=False, arg_index=14, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 14},
                ),
            ]
        ),
    ]
, pool_size=12582912)


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %layernormnorm_14 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_14]
#   %clone_78 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_14,), kwargs = {})
#   return %clone_78
sdsc_fused_19 = async_compile.sdsc('sdsc_fused_19',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=320131306964535035, source=None, aten_op=None, ir_chain=('clone_78', 'buf330'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [out_58, out_59, add_46, hidden_states_22, out_60, out_61, hidden_states_23, out_62, out_63, add_49, hidden_states_24, out_64, out_65], Original ATen: [aten.mm, aten.add, aten.layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   add_46 => add_46
#   add_49 => add_49
#   hidden_states_22 => exx2_15, layernormnorm_15, layernormscale_15
#   hidden_states_23 => gelu_7
#   hidden_states_24 => exx2_16, layernormnorm_16, layernormscale_16
#   out_58 => mm_29
#   out_59 => add_45
#   out_60 => mm_30
#   out_61 => add_47
#   out_62 => mm_31
#   out_63 => add_48
#   out_64 => mm_32
#   out_65 => add_50
# Graph fragment:
#   %clone_78 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_78]
#   %layernormnorm_14 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_14]
#   %clone_35 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_35]
#   %buf155 : Tensor  = PlaceHolder[target=buf155]
#   %buf156 : Tensor  = PlaceHolder[target=buf156]
#   %arg102_1 : Tensor "f16[768, 768][768, 1]spyre:0" = PlaceHolder[target=arg102_1]
#   %mm_29 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_29]
#   %clone_37 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_37]
#   %arg101_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg101_1]
#   %add_45 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_45]
#   %clone_79 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_79]
#   %add_46 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_46]
#   %exx2_15 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_15]
#   %layernormscale_15 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_15]
#   %arg103_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg103_1]
#   %arg104_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg104_1]
#   %layernormnorm_15 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_15]
#   %clone_38 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_38]
#   %arg106_1 : Tensor "f16[768, 3072][3072, 1]spyre:0" = PlaceHolder[target=arg106_1]
#   %mm_30 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=mm_30]
#   %arg105_1 : Tensor "f16[3072][1]spyre:0" = PlaceHolder[target=arg105_1]
#   %add_47 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=add_47]
#   %gelu_7 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=gelu_7]
#   %arg108_1 : Tensor "f16[3072, 768][768, 1]spyre:0" = PlaceHolder[target=arg108_1]
#   %mm_31 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_31]
#   %clone_39 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_39]
#   %arg107_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg107_1]
#   %add_48 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_48]
#   %add_49 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_49]
#   %exx2_16 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_16]
#   %layernormscale_16 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_16]
#   %arg109_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg109_1]
#   %arg110_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg110_1]
#   %layernormnorm_16 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_16]
#   %clone_40 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_40]
#   %arg112_1 : Tensor "f16[768, 2304][2304, 1]spyre:0" = PlaceHolder[target=arg112_1]
#   %mm_32 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=mm_32]
#   %clone_41 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=clone_41]
#   %arg111_1 : Tensor "f16[2304][1]spyre:0" = PlaceHolder[target=arg111_1]
#   %clone_79 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone_78,), kwargs = {})
#   %mm_29 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%empty_7, %arg102_1), kwargs = {})
#   %clone_37 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_29,), kwargs = {})
#   %add_45 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_37, %arg101_1), kwargs = {})
#   %add_46 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %layernormnorm_14), kwargs = {})
#   %exx2_15 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_46, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_15 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_15, 1e-05), kwargs = {})
#   %layernormnorm_15 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_46, %exx2_15, %layernormscale_15, %arg103_1, %arg104_1), kwargs = {})
#   %clone_38 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_15,), kwargs = {})
#   %mm_30 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_38, %arg106_1), kwargs = {})
#   %add_47 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_30, %arg105_1), kwargs = {})
#   %gelu_7 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%add_47,), kwargs = {})
#   %mm_31 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%gelu_7, %arg108_1), kwargs = {})
#   %clone_39 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_31,), kwargs = {})
#   %add_48 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_39, %arg107_1), kwargs = {})
#   %add_49 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_48, %layernormnorm_15), kwargs = {})
#   %exx2_16 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_49, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_16 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_16, 1e-05), kwargs = {})
#   %layernormnorm_16 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=3] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_49, %exx2_16, %layernormscale_16, %arg109_1, %arg110_1), kwargs = {})
#   %clone_40 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_16,), kwargs = {})
#   %mm_32 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_40, %arg112_1), kwargs = {})
#   %clone_41 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_32,), kwargs = {})
#   %add_50 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_41, %arg111_1), kwargs = {})
#   return %clone_79,%mm_29,%clone_37,%add_45,%add_46,%exx2_15,%layernormscale_15,%layernormnorm_15,%clone_38,%mm_30,%add_47,%gelu_7,%mm_31,%clone_39,%add_48,%add_49,%exx2_16,%layernormscale_16,%layernormnorm_16,%clone_40,%mm_32,%clone_41,%add_50
sdsc_fused_add_gelu_layer_norm_mm_20 = async_compile.sdsc('sdsc_fused_add_gelu_layer_norm_mm_20',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6934022787760232154, source=None, aten_op=None, ir_chain=('clone_79', 'buf331'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1286171502003944573, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_29', 'buf157'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3486826198557576600, source=None, aten_op=None, ir_chain=('clone_37', 'buf289'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8562580624300836973, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_45', 'buf158'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3919640320323541953, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_46', 'buf159'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=8332887196402942241, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_15', 'buf160'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=6505102340348645199, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_15', 'buf161'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
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
                    allocation={'lx': 393216},
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
            debug_handle=DebugHandle(id=7101625245120778361, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_15', 'buf162'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
                    allocation={'lx': 393216},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3928466449638542659, source=None, aten_op=None, ir_chain=('clone_38', 'buf290'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 401408},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 4}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('3072'), 8), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 4)'), sympify('c1'): sympify('Mod(floor(core_id/4), 8)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4262001594035343527, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_30', 'buf163'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 401408},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3540611809055066546, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_47', 'buf164'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 48, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 7},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='gelufwd',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1470228294429669874, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/activation.py', start_line=372, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu_7', 'buf165'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2787979615736823953, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_31', 'buf166'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 3072, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 8},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6572786879321501681, source=None, aten_op=None, ir_chain=('clone_39', 'buf291'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1123339548037843888, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_48', 'buf167'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=True, arg_index=9, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 9},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2417178884529461360, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_49', 'buf168'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
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
            debug_handle=DebugHandle(id=83000318508970664, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_16', 'buf169'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=4933448316563281196, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_16', 'buf170'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=6945907482297010528, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_16', 'buf171'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
                    is_input=True, arg_index=10, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 10},
                ),
                TensorArg(
                    is_input=True, arg_index=11, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 11},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3934122340885986561, source=None, aten_op=None, ir_chain=('clone_40', 'buf292'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 114688},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 8}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('2304'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1027414804887413570, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_32', 'buf172'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 114688},
                ),
                TensorArg(
                    is_input=True, arg_index=12, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 12},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3665731779190550327, source=None, aten_op=None, ir_chain=('clone_41', 'buf293'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4368854144341697048, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_50', 'buf173'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                ),
                TensorArg(
                    is_input=True, arg_index=13, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 13},
                ),
                TensorArg(
                    is_input=False, arg_index=14, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 14},
                ),
            ]
        ),
    ]
, pool_size=12582912)


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %layernormnorm_16 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_16]
#   %clone_80 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_16,), kwargs = {})
#   return %clone_80
sdsc_fused_21 = async_compile.sdsc('sdsc_fused_21',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1273372945385654228, source=None, aten_op=None, ir_chain=('clone_80', 'buf332'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [out_66, out_67, add_52, hidden_states_25, out_68, out_69, hidden_states_26, out_70, out_71, add_55, hidden_states_27, out_72, out_73], Original ATen: [aten.mm, aten.add, aten.layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   add_52 => add_52
#   add_55 => add_55
#   hidden_states_25 => exx2_17, layernormnorm_17, layernormscale_17
#   hidden_states_26 => gelu_8
#   hidden_states_27 => exx2_18, layernormnorm_18, layernormscale_18
#   out_66 => mm_33
#   out_67 => add_51
#   out_68 => mm_34
#   out_69 => add_53
#   out_70 => mm_35
#   out_71 => add_54
#   out_72 => mm_36
#   out_73 => add_56
# Graph fragment:
#   %clone_80 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_80]
#   %layernormnorm_16 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_16]
#   %clone_40 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_40]
#   %buf175 : Tensor  = PlaceHolder[target=buf175]
#   %buf176 : Tensor  = PlaceHolder[target=buf176]
#   %arg115_1 : Tensor "f16[768, 768][768, 1]spyre:0" = PlaceHolder[target=arg115_1]
#   %mm_33 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_33]
#   %clone_42 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_42]
#   %arg114_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg114_1]
#   %add_51 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_51]
#   %clone_81 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_81]
#   %add_52 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_52]
#   %exx2_17 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_17]
#   %layernormscale_17 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_17]
#   %arg116_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg116_1]
#   %arg117_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg117_1]
#   %layernormnorm_17 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_17]
#   %clone_43 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_43]
#   %arg119_1 : Tensor "f16[768, 3072][3072, 1]spyre:0" = PlaceHolder[target=arg119_1]
#   %mm_34 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=mm_34]
#   %arg118_1 : Tensor "f16[3072][1]spyre:0" = PlaceHolder[target=arg118_1]
#   %add_53 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=add_53]
#   %gelu_8 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=gelu_8]
#   %arg121_1 : Tensor "f16[3072, 768][768, 1]spyre:0" = PlaceHolder[target=arg121_1]
#   %mm_35 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_35]
#   %clone_44 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_44]
#   %arg120_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg120_1]
#   %add_54 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_54]
#   %add_55 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_55]
#   %exx2_18 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_18]
#   %layernormscale_18 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_18]
#   %arg122_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg122_1]
#   %arg123_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg123_1]
#   %layernormnorm_18 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_18]
#   %clone_45 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_45]
#   %arg125_1 : Tensor "f16[768, 2304][2304, 1]spyre:0" = PlaceHolder[target=arg125_1]
#   %mm_36 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=mm_36]
#   %clone_46 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=clone_46]
#   %arg124_1 : Tensor "f16[2304][1]spyre:0" = PlaceHolder[target=arg124_1]
#   %clone_81 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone_80,), kwargs = {})
#   %mm_33 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%empty_8, %arg115_1), kwargs = {})
#   %clone_42 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_33,), kwargs = {})
#   %add_51 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_42, %arg114_1), kwargs = {})
#   %add_52 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_51, %layernormnorm_16), kwargs = {})
#   %exx2_17 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_52, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_17 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_17, 1e-05), kwargs = {})
#   %layernormnorm_17 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_52, %exx2_17, %layernormscale_17, %arg116_1, %arg117_1), kwargs = {})
#   %clone_43 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_17,), kwargs = {})
#   %mm_34 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_43, %arg119_1), kwargs = {})
#   %add_53 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_34, %arg118_1), kwargs = {})
#   %gelu_8 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%add_53,), kwargs = {})
#   %mm_35 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%gelu_8, %arg121_1), kwargs = {})
#   %clone_44 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_35,), kwargs = {})
#   %add_54 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_44, %arg120_1), kwargs = {})
#   %add_55 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_54, %layernormnorm_17), kwargs = {})
#   %exx2_18 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_55, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_18 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_18, 1e-05), kwargs = {})
#   %layernormnorm_18 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=3] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_55, %exx2_18, %layernormscale_18, %arg122_1, %arg123_1), kwargs = {})
#   %clone_45 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_18,), kwargs = {})
#   %mm_36 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_45, %arg125_1), kwargs = {})
#   %clone_46 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_36,), kwargs = {})
#   %add_56 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_46, %arg124_1), kwargs = {})
#   return %clone_81,%mm_33,%clone_42,%add_51,%add_52,%exx2_17,%layernormscale_17,%layernormnorm_17,%clone_43,%mm_34,%add_53,%gelu_8,%mm_35,%clone_44,%add_54,%add_55,%exx2_18,%layernormscale_18,%layernormnorm_18,%clone_45,%mm_36,%clone_46,%add_56
sdsc_fused_add_gelu_layer_norm_mm_22 = async_compile.sdsc('sdsc_fused_add_gelu_layer_norm_mm_22',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5371148967959377407, source=None, aten_op=None, ir_chain=('clone_81', 'buf333'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=504202493674633055, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_33', 'buf177'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8467939464398757808, source=None, aten_op=None, ir_chain=('clone_42', 'buf294'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7988042766987304022, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_51', 'buf178'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8172205962418399460, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_52', 'buf179'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=2667754014980937623, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_17', 'buf180'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=4972799195161862730, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_17', 'buf181'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
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
                    allocation={'lx': 393216},
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
            debug_handle=DebugHandle(id=3110657917618958992, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_17', 'buf182'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
                    allocation={'lx': 393216},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5053289773950829856, source=None, aten_op=None, ir_chain=('clone_43', 'buf295'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 401408},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 4}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('3072'), 8), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 4)'), sympify('c1'): sympify('Mod(floor(core_id/4), 8)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3233605952779111132, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_34', 'buf183'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 401408},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=9174655274949029042, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_53', 'buf184'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 48, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 7},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='gelufwd',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7421620089592848301, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/activation.py', start_line=372, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu_8', 'buf185'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7048970228073856000, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_35', 'buf186'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 3072, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 8},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1114017826169316879, source=None, aten_op=None, ir_chain=('clone_44', 'buf296'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7976530777640263477, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_54', 'buf187'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=True, arg_index=9, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 9},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4614065228952488094, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_55', 'buf188'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
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
            debug_handle=DebugHandle(id=2333945151727541806, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_18', 'buf189'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=7700343416820457002, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_18', 'buf190'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=4828736510008524460, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_18', 'buf191'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
                    is_input=True, arg_index=10, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 10},
                ),
                TensorArg(
                    is_input=True, arg_index=11, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 11},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6861299695215295288, source=None, aten_op=None, ir_chain=('clone_45', 'buf297'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 114688},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 8}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('2304'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7090252409512515937, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_36', 'buf192'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 114688},
                ),
                TensorArg(
                    is_input=True, arg_index=12, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 12},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8960029489926218253, source=None, aten_op=None, ir_chain=('clone_46', 'buf298'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=546332298572031579, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_56', 'buf193'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                ),
                TensorArg(
                    is_input=True, arg_index=13, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 13},
                ),
                TensorArg(
                    is_input=False, arg_index=14, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 14},
                ),
            ]
        ),
    ]
, pool_size=12582912)


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %layernormnorm_18 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_18]
#   %clone_82 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_18,), kwargs = {})
#   return %clone_82
sdsc_fused_23 = async_compile.sdsc('sdsc_fused_23',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4953736202031353132, source=None, aten_op=None, ir_chain=('clone_82', 'buf334'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [out_74, out_75, add_58, hidden_states_28, out_76, out_77, hidden_states_29, out_78, out_79, add_61, hidden_states_30, out_80, out_81], Original ATen: [aten.mm, aten.add, aten.layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   add_58 => add_58
#   add_61 => add_61
#   hidden_states_28 => exx2_19, layernormnorm_19, layernormscale_19
#   hidden_states_29 => gelu_9
#   hidden_states_30 => exx2_20, layernormnorm_20, layernormscale_20
#   out_74 => mm_37
#   out_75 => add_57
#   out_76 => mm_38
#   out_77 => add_59
#   out_78 => mm_39
#   out_79 => add_60
#   out_80 => mm_40
#   out_81 => add_62
# Graph fragment:
#   %clone_82 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_82]
#   %layernormnorm_18 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_18]
#   %clone_45 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_45]
#   %buf195 : Tensor  = PlaceHolder[target=buf195]
#   %buf196 : Tensor  = PlaceHolder[target=buf196]
#   %arg128_1 : Tensor "f16[768, 768][768, 1]spyre:0" = PlaceHolder[target=arg128_1]
#   %mm_37 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_37]
#   %clone_47 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_47]
#   %arg127_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg127_1]
#   %add_57 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_57]
#   %clone_83 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_83]
#   %add_58 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_58]
#   %exx2_19 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_19]
#   %layernormscale_19 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_19]
#   %arg129_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg129_1]
#   %arg130_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg130_1]
#   %layernormnorm_19 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_19]
#   %clone_48 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_48]
#   %arg132_1 : Tensor "f16[768, 3072][3072, 1]spyre:0" = PlaceHolder[target=arg132_1]
#   %mm_38 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=mm_38]
#   %arg131_1 : Tensor "f16[3072][1]spyre:0" = PlaceHolder[target=arg131_1]
#   %add_59 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=add_59]
#   %gelu_9 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=gelu_9]
#   %arg134_1 : Tensor "f16[3072, 768][768, 1]spyre:0" = PlaceHolder[target=arg134_1]
#   %mm_39 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_39]
#   %clone_49 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_49]
#   %arg133_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg133_1]
#   %add_60 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_60]
#   %add_61 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_61]
#   %exx2_20 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_20]
#   %layernormscale_20 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_20]
#   %arg135_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg135_1]
#   %arg136_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg136_1]
#   %layernormnorm_20 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_20]
#   %clone_50 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_50]
#   %arg138_1 : Tensor "f16[768, 2304][2304, 1]spyre:0" = PlaceHolder[target=arg138_1]
#   %mm_40 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=mm_40]
#   %clone_51 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=clone_51]
#   %arg137_1 : Tensor "f16[2304][1]spyre:0" = PlaceHolder[target=arg137_1]
#   %clone_83 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone_82,), kwargs = {})
#   %mm_37 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%empty_9, %arg128_1), kwargs = {})
#   %clone_47 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_37,), kwargs = {})
#   %add_57 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_47, %arg127_1), kwargs = {})
#   %add_58 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_57, %layernormnorm_18), kwargs = {})
#   %exx2_19 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_58, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_19 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_19, 1e-05), kwargs = {})
#   %layernormnorm_19 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_58, %exx2_19, %layernormscale_19, %arg129_1, %arg130_1), kwargs = {})
#   %clone_48 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_19,), kwargs = {})
#   %mm_38 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_48, %arg132_1), kwargs = {})
#   %add_59 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_38, %arg131_1), kwargs = {})
#   %gelu_9 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%add_59,), kwargs = {})
#   %mm_39 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%gelu_9, %arg134_1), kwargs = {})
#   %clone_49 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_39,), kwargs = {})
#   %add_60 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_49, %arg133_1), kwargs = {})
#   %add_61 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_60, %layernormnorm_19), kwargs = {})
#   %exx2_20 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_61, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_20 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_20, 1e-05), kwargs = {})
#   %layernormnorm_20 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=3] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_61, %exx2_20, %layernormscale_20, %arg135_1, %arg136_1), kwargs = {})
#   %clone_50 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_20,), kwargs = {})
#   %mm_40 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_50, %arg138_1), kwargs = {})
#   %clone_51 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_40,), kwargs = {})
#   %add_62 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_51, %arg137_1), kwargs = {})
#   return %clone_83,%mm_37,%clone_47,%add_57,%add_58,%exx2_19,%layernormscale_19,%layernormnorm_19,%clone_48,%mm_38,%add_59,%gelu_9,%mm_39,%clone_49,%add_60,%add_61,%exx2_20,%layernormscale_20,%layernormnorm_20,%clone_50,%mm_40,%clone_51,%add_62
sdsc_fused_add_gelu_layer_norm_mm_24 = async_compile.sdsc('sdsc_fused_add_gelu_layer_norm_mm_24',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7368483118900921607, source=None, aten_op=None, ir_chain=('clone_83', 'buf335'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1332053924294873169, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_37', 'buf197'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3955686793381835005, source=None, aten_op=None, ir_chain=('clone_47', 'buf299'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8877718134842135588, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_57', 'buf198'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1868614788934060040, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_58', 'buf199'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=3077865022851566072, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_19', 'buf200'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=2913496000604839790, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_19', 'buf201'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
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
                    allocation={'lx': 393216},
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
            debug_handle=DebugHandle(id=4130356871996479723, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_19', 'buf202'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
                    allocation={'lx': 393216},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4198145103201901691, source=None, aten_op=None, ir_chain=('clone_48', 'buf300'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 401408},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 4}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('3072'), 8), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 4)'), sympify('c1'): sympify('Mod(floor(core_id/4), 8)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1970791627008899640, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_38', 'buf203'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 401408},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=523497249444270253, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_59', 'buf204'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 48, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 7},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='gelufwd',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=162143208716490185, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/activation.py', start_line=372, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu_9', 'buf205'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=43600088444015796, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_39', 'buf206'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 3072, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 8},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7748002697186776999, source=None, aten_op=None, ir_chain=('clone_49', 'buf301'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5345820599873371715, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_60', 'buf207'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=True, arg_index=9, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 9},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4427492096800571895, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_61', 'buf208'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
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
            debug_handle=DebugHandle(id=2885571531879323339, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_20', 'buf209'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=4230265000505556642, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_20', 'buf210'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=7929411550614847959, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_20', 'buf211'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
                    is_input=True, arg_index=10, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 10},
                ),
                TensorArg(
                    is_input=True, arg_index=11, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 11},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4010745993320278241, source=None, aten_op=None, ir_chain=('clone_50', 'buf302'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 114688},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 8}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('2304'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8568123864000051315, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_40', 'buf212'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 114688},
                ),
                TensorArg(
                    is_input=True, arg_index=12, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 12},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6419331841688261304, source=None, aten_op=None, ir_chain=('clone_51', 'buf303'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2695871565826531282, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_62', 'buf213'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                ),
                TensorArg(
                    is_input=True, arg_index=13, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 13},
                ),
                TensorArg(
                    is_input=False, arg_index=14, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 14},
                ),
            ]
        ),
    ]
, pool_size=12582912)


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %layernormnorm_20 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_20]
#   %clone_84 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_20,), kwargs = {})
#   return %clone_84
sdsc_fused_25 = async_compile.sdsc('sdsc_fused_25',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7629350528453260772, source=None, aten_op=None, ir_chain=('clone_84', 'buf336'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [out_82, out_83, add_64, hidden_states_31, out_84, out_85, hidden_states_32, out_86, out_87, add_67, hidden_states_33, out_88, out_89], Original ATen: [aten.mm, aten.add, aten.layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   add_64 => add_64
#   add_67 => add_67
#   hidden_states_31 => exx2_21, layernormnorm_21, layernormscale_21
#   hidden_states_32 => gelu_10
#   hidden_states_33 => exx2_22, layernormnorm_22, layernormscale_22
#   out_82 => mm_41
#   out_83 => add_63
#   out_84 => mm_42
#   out_85 => add_65
#   out_86 => mm_43
#   out_87 => add_66
#   out_88 => mm_44
#   out_89 => add_68
# Graph fragment:
#   %clone_84 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_84]
#   %layernormnorm_20 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_20]
#   %clone_50 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_50]
#   %buf215 : Tensor  = PlaceHolder[target=buf215]
#   %buf216 : Tensor  = PlaceHolder[target=buf216]
#   %arg141_1 : Tensor "f16[768, 768][768, 1]spyre:0" = PlaceHolder[target=arg141_1]
#   %mm_41 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_41]
#   %clone_52 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_52]
#   %arg140_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg140_1]
#   %add_63 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_63]
#   %clone_85 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_85]
#   %add_64 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_64]
#   %exx2_21 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_21]
#   %layernormscale_21 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_21]
#   %arg142_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg142_1]
#   %arg143_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg143_1]
#   %layernormnorm_21 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_21]
#   %clone_53 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_53]
#   %arg145_1 : Tensor "f16[768, 3072][3072, 1]spyre:0" = PlaceHolder[target=arg145_1]
#   %mm_42 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=mm_42]
#   %arg144_1 : Tensor "f16[3072][1]spyre:0" = PlaceHolder[target=arg144_1]
#   %add_65 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=add_65]
#   %gelu_10 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=gelu_10]
#   %arg147_1 : Tensor "f16[3072, 768][768, 1]spyre:0" = PlaceHolder[target=arg147_1]
#   %mm_43 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_43]
#   %clone_54 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_54]
#   %arg146_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg146_1]
#   %add_66 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_66]
#   %add_67 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_67]
#   %exx2_22 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_22]
#   %layernormscale_22 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_22]
#   %arg148_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg148_1]
#   %arg149_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg149_1]
#   %layernormnorm_22 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_22]
#   %clone_55 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_55]
#   %arg151_1 : Tensor "f16[768, 2304][2304, 1]spyre:0" = PlaceHolder[target=arg151_1]
#   %mm_44 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=mm_44]
#   %clone_56 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=clone_56]
#   %arg150_1 : Tensor "f16[2304][1]spyre:0" = PlaceHolder[target=arg150_1]
#   %clone_85 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone_84,), kwargs = {})
#   %mm_41 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%empty_10, %arg141_1), kwargs = {})
#   %clone_52 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_41,), kwargs = {})
#   %add_63 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_52, %arg140_1), kwargs = {})
#   %add_64 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_63, %layernormnorm_20), kwargs = {})
#   %exx2_21 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_64, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_21 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_21, 1e-05), kwargs = {})
#   %layernormnorm_21 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_64, %exx2_21, %layernormscale_21, %arg142_1, %arg143_1), kwargs = {})
#   %clone_53 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_21,), kwargs = {})
#   %mm_42 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_53, %arg145_1), kwargs = {})
#   %add_65 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_42, %arg144_1), kwargs = {})
#   %gelu_10 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%add_65,), kwargs = {})
#   %mm_43 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%gelu_10, %arg147_1), kwargs = {})
#   %clone_54 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_43,), kwargs = {})
#   %add_66 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_54, %arg146_1), kwargs = {})
#   %add_67 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_66, %layernormnorm_21), kwargs = {})
#   %exx2_22 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_67, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_22 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_22, 1e-05), kwargs = {})
#   %layernormnorm_22 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=3] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_67, %exx2_22, %layernormscale_22, %arg148_1, %arg149_1), kwargs = {})
#   %clone_55 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_22,), kwargs = {})
#   %mm_44 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_55, %arg151_1), kwargs = {})
#   %clone_56 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_44,), kwargs = {})
#   %add_68 : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_56, %arg150_1), kwargs = {})
#   return %clone_85,%mm_41,%clone_52,%add_63,%add_64,%exx2_21,%layernormscale_21,%layernormnorm_21,%clone_53,%mm_42,%add_65,%gelu_10,%mm_43,%clone_54,%add_66,%add_67,%exx2_22,%layernormscale_22,%layernormnorm_22,%clone_55,%mm_44,%clone_56,%add_68
sdsc_fused_add_gelu_layer_norm_mm_26 = async_compile.sdsc('sdsc_fused_add_gelu_layer_norm_mm_26',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=747855027357672522, source=None, aten_op=None, ir_chain=('clone_85', 'buf337'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=348562199997171736, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_41', 'buf217'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5385254659584997987, source=None, aten_op=None, ir_chain=('clone_52', 'buf304'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6046811965775418419, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_63', 'buf218'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7614939392646404378, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_64', 'buf219'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=4009997191379590526, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_21', 'buf220'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=6493564214502201914, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_21', 'buf221'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
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
                    allocation={'lx': 393216},
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
            debug_handle=DebugHandle(id=3287128228850842035, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_21', 'buf222'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
                    allocation={'lx': 393216},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3625253874668547039, source=None, aten_op=None, ir_chain=('clone_53', 'buf305'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 401408},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 4}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('3072'), 8), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 4)'), sympify('c1'): sympify('Mod(floor(core_id/4), 8)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=9139039146675827919, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_42', 'buf223'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 401408},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1078776372378466334, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_65', 'buf224'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 48, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 7},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='gelufwd',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=106966195769453007, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/activation.py', start_line=372, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu_10', 'buf225'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7676590988194253643, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_43', 'buf226'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 3072, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 8},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7748764863181582993, source=None, aten_op=None, ir_chain=('clone_54', 'buf306'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7984471170225258098, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_66', 'buf227'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=True, arg_index=9, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 9},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6522115203983035486, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_67', 'buf228'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
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
            debug_handle=DebugHandle(id=4144261612524329201, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_22', 'buf229'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=3134631139597755237, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_22', 'buf230'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=6962173771465902926, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_22', 'buf231'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
                    is_input=True, arg_index=10, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 10},
                ),
                TensorArg(
                    is_input=True, arg_index=11, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 11},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3172082906542433808, source=None, aten_op=None, ir_chain=('clone_55', 'buf307'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 114688},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 8}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('2304'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4712723042601521555, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_44', 'buf232'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 114688},
                ),
                TensorArg(
                    is_input=True, arg_index=12, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 12},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6022592654777243748, source=None, aten_op=None, ir_chain=('clone_56', 'buf308'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 507904},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('2304'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1102589811443663861, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_68', 'buf233'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 802816},
                ),
                TensorArg(
                    is_input=True, arg_index=13, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 13},
                ),
                TensorArg(
                    is_input=False, arg_index=14, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 14},
                ),
            ]
        ),
    ]
, pool_size=12582912)


# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %layernormnorm_22 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_22]
#   %clone_86 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_22,), kwargs = {})
#   return %clone_86
sdsc_fused_27 = async_compile.sdsc('sdsc_fused_27',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2626793052137718321, source=None, aten_op=None, ir_chain=('clone_86', 'buf338'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 0},
                ),
            ]
        ),
    ]
)


# Topologically Sorted Source Nodes: [out_90, out_91, add_70, hidden_states_34, out_92, out_93, hidden_states_35, out_94, out_95, add_73, hidden_states_36], Original ATen: [aten.mm, aten.add, aten.layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   add_70 => add_70
#   add_73 => add_73
#   hidden_states_34 => exx2_23, layernormnorm_23, layernormscale_23
#   hidden_states_35 => gelu_11
#   hidden_states_36 => exx2_24, layernormnorm_24, layernormscale_24
#   out_90 => mm_45
#   out_91 => add_69
#   out_92 => mm_46
#   out_93 => add_71
#   out_94 => mm_47
#   out_95 => add_72
# Graph fragment:
#   %clone_86 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_86]
#   %layernormnorm_22 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_22]
#   %clone_55 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_55]
#   %buf235 : Tensor  = PlaceHolder[target=buf235]
#   %buf236 : Tensor  = PlaceHolder[target=buf236]
#   %arg154_1 : Tensor "f16[768, 768][768, 1]spyre:0" = PlaceHolder[target=arg154_1]
#   %mm_45 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_45]
#   %clone_57 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_57]
#   %arg153_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg153_1]
#   %add_69 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_69]
#   %clone_87 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_87]
#   %add_70 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_70]
#   %exx2_23 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_23]
#   %layernormscale_23 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_23]
#   %arg155_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg155_1]
#   %arg156_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg156_1]
#   %layernormnorm_23 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm_23]
#   %clone_58 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_58]
#   %arg158_1 : Tensor "f16[768, 3072][3072, 1]spyre:0" = PlaceHolder[target=arg158_1]
#   %mm_46 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=mm_46]
#   %arg157_1 : Tensor "f16[3072][1]spyre:0" = PlaceHolder[target=arg157_1]
#   %add_71 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=add_71]
#   %gelu_11 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=gelu_11]
#   %arg160_1 : Tensor "f16[3072, 768][768, 1]spyre:0" = PlaceHolder[target=arg160_1]
#   %mm_47 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_47]
#   %clone_59 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_59]
#   %arg159_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg159_1]
#   %add_72 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_72]
#   %add_73 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_73]
#   %exx2_24 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_24]
#   %layernormscale_24 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_24]
#   %arg161_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg161_1]
#   %arg162_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg162_1]
#   %clone_87 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone_86,), kwargs = {})
#   %mm_45 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%empty_11, %arg154_1), kwargs = {})
#   %clone_57 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_45,), kwargs = {})
#   %add_69 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_57, %arg153_1), kwargs = {})
#   %add_70 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_69, %layernormnorm_22), kwargs = {})
#   %exx2_23 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_70, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_23 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_23, 1e-05), kwargs = {})
#   %layernormnorm_23 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_70, %exx2_23, %layernormscale_23, %arg155_1, %arg156_1), kwargs = {})
#   %clone_58 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm_23,), kwargs = {})
#   %mm_46 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_58, %arg158_1), kwargs = {})
#   %add_71 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_46, %arg157_1), kwargs = {})
#   %gelu_11 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%add_71,), kwargs = {})
#   %mm_47 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%gelu_11, %arg160_1), kwargs = {})
#   %clone_59 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_47,), kwargs = {})
#   %add_72 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_59, %arg159_1), kwargs = {})
#   %add_73 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_72, %layernormnorm_23), kwargs = {})
#   %exx2_24 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_73, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_24 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_24, 1e-05), kwargs = {})
#   %layernormnorm_24 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_73, %exx2_24, %layernormscale_24, %arg161_1, %arg162_1), kwargs = {})
#   return %clone_87,%mm_45,%clone_57,%add_69,%add_70,%exx2_23,%layernormscale_23,%layernormnorm_23,%clone_58,%mm_46,%add_71,%gelu_11,%mm_47,%clone_59,%add_72,%add_73,%exx2_24,%layernormscale_24,%layernormnorm_24
sdsc_fused_add_gelu_layer_norm_mm_28 = async_compile.sdsc('sdsc_fused_add_gelu_layer_norm_mm_28',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=154095852437417762, source=None, aten_op=None, ir_chain=('clone_87', 'buf339'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=124502546467730225, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_45', 'buf237'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5891947049034107038, source=None, aten_op=None, ir_chain=('clone_57', 'buf309'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5871670316247258021, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_69', 'buf238'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6221360268284247297, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_70', 'buf239'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=2774968434018959750, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_23', 'buf240'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=6940022615877890410, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_23', 'buf241'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
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
                    allocation={'lx': 393216},
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
            debug_handle=DebugHandle(id=5446539189764948533, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_23', 'buf242'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
                    allocation={'lx': 393216},
                    element_arrangement=ElementArrangement.EXX2,
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4279910176847611093, source=None, aten_op=None, ir_chain=('clone_58', 'buf310'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 401408},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 4}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('3072'), 8), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 4)'), sympify('c1'): sympify('Mod(floor(core_id/4), 8)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8239857693896263537, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_46', 'buf243'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 401408},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=830985109604159545, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_71', 'buf244'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 48, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 7},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='gelufwd',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8642723297096277676, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/activation.py', start_line=372, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu_11', 'buf245'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('3072'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8088661115013674951, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_47', 'buf246'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 3072, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 8},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4374752363589584380, source=None, aten_op=None, ir_chain=('clone_59', 'buf311'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                    work_division=TensorWorkDivision(work_slices={sympify('c0'): 32}, core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3442968505590175450, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_72', 'buf247'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=True, arg_index=9, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 9},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6294402622714622782, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_73', 'buf248'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
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
            debug_handle=DebugHandle(id=3638280878837640054, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_24', 'buf249'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=5369579656058291905, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_24', 'buf250'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=6114320349393967280, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_24', 'buf251'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 64],
                    device_coordinates=[sympify('0'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
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
                    is_input=True, arg_index=10, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 10},
                ),
                TensorArg(
                    is_input=True, arg_index=11, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 11},
                ),
                TensorArg(
                    is_input=False, arg_index=12, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 12},
                ),
            ]
        ),
    ]
, pool_size=12582912)


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1 = args
        args.clear()
        assert_size_stride(arg0_1, (2048, ), (1, ), 'input')
        assert_size_stride(arg1_1, (250048, 768), (768, 1), 'input')
        buf312 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_embedding_0.run(arg0_1, arg1_1, buf312)
        del arg0_1
        del arg1_1
        # Topologically Sorted Source Nodes: [token_type_ids], Original ATen: [aten.zeros_like]
        buf1 = torch.ops.aten.full.default([2048], 0, dtype=torch.int64, layout=torch.strided, device=device(type='spyre', index=0), pin_memory=False)
        sdsc_fused_1.run(buf312)
        del buf312
        buf2 = buf1
        assert_size_stride(buf2, (2048, ), (1, ), 'torch.ops.aten.full.default')
        assert_alignment(buf2, 16, 'torch.ops.aten.full.default')
        del buf1
        assert_size_stride(arg2_1, (64, 768), (768, 1), 'input')
        buf314 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_embedding_2.run(buf2, arg2_1, buf314)
        del arg2_1
        del buf2
        assert_size_stride(arg3_1, (2048, ), (1, ), 'input')
        # Topologically Sorted Source Nodes: [masked_input], Original ATen: [vllm.spyre_offset_positions]
        buf5 = torch.ops.vllm.spyre_offset_positions.default(arg3_1, 2, device(type='spyre', index=0))
        del arg3_1
        sdsc_fused_3.run(buf314)
        del buf314
        buf6 = buf5
        assert_size_stride(buf6, (2048, ), (1, ), 'torch.ops.vllm.spyre_offset_positions.default')
        # buffer buf6 (op: torch.ops.vllm.spyre_offset_positions.default) is assumed to be not aligned
        del buf5
        assert_size_stride(arg4_1, (576, 768), (768, 1), 'input')
        assert_size_stride(arg5_1, (768, ), (1, ), 'input')
        assert_size_stride(arg6_1, (768, ), (1, ), 'input')
        assert_size_stride(arg8_1, (768, 2304), (2304, 1), 'input')
        assert_size_stride(arg7_1, (2304, ), (1, ), 'input')
        buf13 = spyre_empty_with_layout((2048, 2304), (2304, 1), torch.float16, SpyreTensorLayout(device_size=[36, 2048, 64], stride_map =[64, 2304, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_embedding_layer_norm_mm_4.run(buf6, arg4_1, arg5_1, arg6_1, arg8_1, arg7_1, buf13)
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del arg8_1
        del buf6
        buf14 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        buf316 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_5.run(buf316)
        # Topologically Sorted Source Nodes: [split, query, key, value, unified_attention_with_output], Original ATen: [aten.split_with_sizes, aten.view, aten.as_strided, vllm.unified_attention_with_output]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf13, (2048, 12, 64), (2304, 64, 1), 0), reinterpret_tensor(buf13, (2048, 12, 64), (2304, 64, 1), 768), reinterpret_tensor(buf13, (2048, 12, 64), (2304, 64, 1), 1536), reinterpret_tensor(buf14, (2048, 12, 64), (768, 64, 1), 0), arg9_1)
        del arg9_1
        del buf13
        assert_size_stride(arg11_1, (768, 768), (768, 1), 'input')
        assert_size_stride(arg10_1, (768, ), (1, ), 'input')
        assert_size_stride(arg12_1, (768, ), (1, ), 'input')
        assert_size_stride(arg13_1, (768, ), (1, ), 'input')
        assert_size_stride(arg15_1, (768, 3072), (3072, 1), 'input')
        assert_size_stride(arg14_1, (3072, ), (1, ), 'input')
        assert_size_stride(arg17_1, (3072, 768), (768, 1), 'input')
        assert_size_stride(arg16_1, (768, ), (1, ), 'input')
        assert_size_stride(arg18_1, (768, ), (1, ), 'input')
        assert_size_stride(arg19_1, (768, ), (1, ), 'input')
        assert_size_stride(arg21_1, (768, 2304), (2304, 1), 'input')
        assert_size_stride(arg20_1, (2304, ), (1, ), 'input')
        buf33 = spyre_empty_with_layout((2048, 2304), (2304, 1), torch.float16, SpyreTensorLayout(device_size=[36, 2048, 64], stride_map =[64, 2304, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_gelu_layer_norm_mm_6.run(buf316, buf14, arg11_1, arg10_1, arg12_1, arg13_1, arg15_1, arg14_1, arg17_1, arg16_1, arg18_1, arg19_1, arg21_1, arg20_1, buf33)
        del arg10_1
        del arg11_1
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del arg16_1
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del arg21_1
        del buf14
        del buf316
        buf34 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        buf318 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_7.run(buf318)
        # Topologically Sorted Source Nodes: [split_1, query_1, key_1, value_1, unified_attention_with_output_1], Original ATen: [aten.split_with_sizes, aten.view, aten.as_strided, vllm.unified_attention_with_output]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf33, (2048, 12, 64), (2304, 64, 1), 0), reinterpret_tensor(buf33, (2048, 12, 64), (2304, 64, 1), 768), reinterpret_tensor(buf33, (2048, 12, 64), (2304, 64, 1), 1536), reinterpret_tensor(buf34, (2048, 12, 64), (768, 64, 1), 0), arg22_1)
        del arg22_1
        del buf33
        assert_size_stride(arg24_1, (768, 768), (768, 1), 'input')
        assert_size_stride(arg23_1, (768, ), (1, ), 'input')
        assert_size_stride(arg25_1, (768, ), (1, ), 'input')
        assert_size_stride(arg26_1, (768, ), (1, ), 'input')
        assert_size_stride(arg28_1, (768, 3072), (3072, 1), 'input')
        assert_size_stride(arg27_1, (3072, ), (1, ), 'input')
        assert_size_stride(arg30_1, (3072, 768), (768, 1), 'input')
        assert_size_stride(arg29_1, (768, ), (1, ), 'input')
        assert_size_stride(arg31_1, (768, ), (1, ), 'input')
        assert_size_stride(arg32_1, (768, ), (1, ), 'input')
        assert_size_stride(arg34_1, (768, 2304), (2304, 1), 'input')
        assert_size_stride(arg33_1, (2304, ), (1, ), 'input')
        buf53 = spyre_empty_with_layout((2048, 2304), (2304, 1), torch.float16, SpyreTensorLayout(device_size=[36, 2048, 64], stride_map =[64, 2304, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_gelu_layer_norm_mm_8.run(buf318, buf34, arg24_1, arg23_1, arg25_1, arg26_1, arg28_1, arg27_1, arg30_1, arg29_1, arg31_1, arg32_1, arg34_1, arg33_1, buf53)
        del arg23_1
        del arg24_1
        del arg25_1
        del arg26_1
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        del arg31_1
        del arg32_1
        del arg33_1
        del arg34_1
        del buf318
        del buf34
        buf54 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        buf320 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_9.run(buf320)
        # Topologically Sorted Source Nodes: [split_2, query_2, key_2, value_2, unified_attention_with_output_2], Original ATen: [aten.split_with_sizes, aten.view, aten.as_strided, vllm.unified_attention_with_output]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf53, (2048, 12, 64), (2304, 64, 1), 0), reinterpret_tensor(buf53, (2048, 12, 64), (2304, 64, 1), 768), reinterpret_tensor(buf53, (2048, 12, 64), (2304, 64, 1), 1536), reinterpret_tensor(buf54, (2048, 12, 64), (768, 64, 1), 0), arg35_1)
        del arg35_1
        del buf53
        assert_size_stride(arg37_1, (768, 768), (768, 1), 'input')
        assert_size_stride(arg36_1, (768, ), (1, ), 'input')
        assert_size_stride(arg38_1, (768, ), (1, ), 'input')
        assert_size_stride(arg39_1, (768, ), (1, ), 'input')
        assert_size_stride(arg41_1, (768, 3072), (3072, 1), 'input')
        assert_size_stride(arg40_1, (3072, ), (1, ), 'input')
        assert_size_stride(arg43_1, (3072, 768), (768, 1), 'input')
        assert_size_stride(arg42_1, (768, ), (1, ), 'input')
        assert_size_stride(arg44_1, (768, ), (1, ), 'input')
        assert_size_stride(arg45_1, (768, ), (1, ), 'input')
        assert_size_stride(arg47_1, (768, 2304), (2304, 1), 'input')
        assert_size_stride(arg46_1, (2304, ), (1, ), 'input')
        buf73 = spyre_empty_with_layout((2048, 2304), (2304, 1), torch.float16, SpyreTensorLayout(device_size=[36, 2048, 64], stride_map =[64, 2304, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_gelu_layer_norm_mm_10.run(buf320, buf54, arg37_1, arg36_1, arg38_1, arg39_1, arg41_1, arg40_1, arg43_1, arg42_1, arg44_1, arg45_1, arg47_1, arg46_1, buf73)
        del arg36_1
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        del arg41_1
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        del arg46_1
        del arg47_1
        del buf320
        del buf54
        buf74 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        buf322 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_11.run(buf322)
        # Topologically Sorted Source Nodes: [split_3, query_3, key_3, value_3, unified_attention_with_output_3], Original ATen: [aten.split_with_sizes, aten.view, aten.as_strided, vllm.unified_attention_with_output]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf73, (2048, 12, 64), (2304, 64, 1), 0), reinterpret_tensor(buf73, (2048, 12, 64), (2304, 64, 1), 768), reinterpret_tensor(buf73, (2048, 12, 64), (2304, 64, 1), 1536), reinterpret_tensor(buf74, (2048, 12, 64), (768, 64, 1), 0), arg48_1)
        del arg48_1
        del buf73
        assert_size_stride(arg50_1, (768, 768), (768, 1), 'input')
        assert_size_stride(arg49_1, (768, ), (1, ), 'input')
        assert_size_stride(arg51_1, (768, ), (1, ), 'input')
        assert_size_stride(arg52_1, (768, ), (1, ), 'input')
        assert_size_stride(arg54_1, (768, 3072), (3072, 1), 'input')
        assert_size_stride(arg53_1, (3072, ), (1, ), 'input')
        assert_size_stride(arg56_1, (3072, 768), (768, 1), 'input')
        assert_size_stride(arg55_1, (768, ), (1, ), 'input')
        assert_size_stride(arg57_1, (768, ), (1, ), 'input')
        assert_size_stride(arg58_1, (768, ), (1, ), 'input')
        assert_size_stride(arg60_1, (768, 2304), (2304, 1), 'input')
        assert_size_stride(arg59_1, (2304, ), (1, ), 'input')
        buf93 = spyre_empty_with_layout((2048, 2304), (2304, 1), torch.float16, SpyreTensorLayout(device_size=[36, 2048, 64], stride_map =[64, 2304, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_gelu_layer_norm_mm_12.run(buf322, buf74, arg50_1, arg49_1, arg51_1, arg52_1, arg54_1, arg53_1, arg56_1, arg55_1, arg57_1, arg58_1, arg60_1, arg59_1, buf93)
        del arg49_1
        del arg50_1
        del arg51_1
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        del arg56_1
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del buf322
        del buf74
        buf94 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        buf324 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_13.run(buf324)
        # Topologically Sorted Source Nodes: [split_4, query_4, key_4, value_4, unified_attention_with_output_4], Original ATen: [aten.split_with_sizes, aten.view, aten.as_strided, vllm.unified_attention_with_output]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf93, (2048, 12, 64), (2304, 64, 1), 0), reinterpret_tensor(buf93, (2048, 12, 64), (2304, 64, 1), 768), reinterpret_tensor(buf93, (2048, 12, 64), (2304, 64, 1), 1536), reinterpret_tensor(buf94, (2048, 12, 64), (768, 64, 1), 0), arg61_1)
        del arg61_1
        del buf93
        assert_size_stride(arg63_1, (768, 768), (768, 1), 'input')
        assert_size_stride(arg62_1, (768, ), (1, ), 'input')
        assert_size_stride(arg64_1, (768, ), (1, ), 'input')
        assert_size_stride(arg65_1, (768, ), (1, ), 'input')
        assert_size_stride(arg67_1, (768, 3072), (3072, 1), 'input')
        assert_size_stride(arg66_1, (3072, ), (1, ), 'input')
        assert_size_stride(arg69_1, (3072, 768), (768, 1), 'input')
        assert_size_stride(arg68_1, (768, ), (1, ), 'input')
        assert_size_stride(arg70_1, (768, ), (1, ), 'input')
        assert_size_stride(arg71_1, (768, ), (1, ), 'input')
        assert_size_stride(arg73_1, (768, 2304), (2304, 1), 'input')
        assert_size_stride(arg72_1, (2304, ), (1, ), 'input')
        buf113 = spyre_empty_with_layout((2048, 2304), (2304, 1), torch.float16, SpyreTensorLayout(device_size=[36, 2048, 64], stride_map =[64, 2304, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_gelu_layer_norm_mm_14.run(buf324, buf94, arg63_1, arg62_1, arg64_1, arg65_1, arg67_1, arg66_1, arg69_1, arg68_1, arg70_1, arg71_1, arg73_1, arg72_1, buf113)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        del arg66_1
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        del arg71_1
        del arg72_1
        del arg73_1
        del buf324
        del buf94
        buf114 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        buf326 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_15.run(buf326)
        # Topologically Sorted Source Nodes: [split_5, query_5, key_5, value_5, unified_attention_with_output_5], Original ATen: [aten.split_with_sizes, aten.view, aten.as_strided, vllm.unified_attention_with_output]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf113, (2048, 12, 64), (2304, 64, 1), 0), reinterpret_tensor(buf113, (2048, 12, 64), (2304, 64, 1), 768), reinterpret_tensor(buf113, (2048, 12, 64), (2304, 64, 1), 1536), reinterpret_tensor(buf114, (2048, 12, 64), (768, 64, 1), 0), arg74_1)
        del arg74_1
        del buf113
        assert_size_stride(arg76_1, (768, 768), (768, 1), 'input')
        assert_size_stride(arg75_1, (768, ), (1, ), 'input')
        assert_size_stride(arg77_1, (768, ), (1, ), 'input')
        assert_size_stride(arg78_1, (768, ), (1, ), 'input')
        assert_size_stride(arg80_1, (768, 3072), (3072, 1), 'input')
        assert_size_stride(arg79_1, (3072, ), (1, ), 'input')
        assert_size_stride(arg82_1, (3072, 768), (768, 1), 'input')
        assert_size_stride(arg81_1, (768, ), (1, ), 'input')
        assert_size_stride(arg83_1, (768, ), (1, ), 'input')
        assert_size_stride(arg84_1, (768, ), (1, ), 'input')
        assert_size_stride(arg86_1, (768, 2304), (2304, 1), 'input')
        assert_size_stride(arg85_1, (2304, ), (1, ), 'input')
        buf133 = spyre_empty_with_layout((2048, 2304), (2304, 1), torch.float16, SpyreTensorLayout(device_size=[36, 2048, 64], stride_map =[64, 2304, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_gelu_layer_norm_mm_16.run(buf326, buf114, arg76_1, arg75_1, arg77_1, arg78_1, arg80_1, arg79_1, arg82_1, arg81_1, arg83_1, arg84_1, arg86_1, arg85_1, buf133)
        del arg75_1
        del arg76_1
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        del arg81_1
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        del arg86_1
        del buf114
        del buf326
        buf134 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        buf328 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_17.run(buf328)
        # Topologically Sorted Source Nodes: [split_6, query_6, key_6, value_6, unified_attention_with_output_6], Original ATen: [aten.split_with_sizes, aten.view, aten.as_strided, vllm.unified_attention_with_output]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf133, (2048, 12, 64), (2304, 64, 1), 0), reinterpret_tensor(buf133, (2048, 12, 64), (2304, 64, 1), 768), reinterpret_tensor(buf133, (2048, 12, 64), (2304, 64, 1), 1536), reinterpret_tensor(buf134, (2048, 12, 64), (768, 64, 1), 0), arg87_1)
        del arg87_1
        del buf133
        assert_size_stride(arg89_1, (768, 768), (768, 1), 'input')
        assert_size_stride(arg88_1, (768, ), (1, ), 'input')
        assert_size_stride(arg90_1, (768, ), (1, ), 'input')
        assert_size_stride(arg91_1, (768, ), (1, ), 'input')
        assert_size_stride(arg93_1, (768, 3072), (3072, 1), 'input')
        assert_size_stride(arg92_1, (3072, ), (1, ), 'input')
        assert_size_stride(arg95_1, (3072, 768), (768, 1), 'input')
        assert_size_stride(arg94_1, (768, ), (1, ), 'input')
        assert_size_stride(arg96_1, (768, ), (1, ), 'input')
        assert_size_stride(arg97_1, (768, ), (1, ), 'input')
        assert_size_stride(arg99_1, (768, 2304), (2304, 1), 'input')
        assert_size_stride(arg98_1, (2304, ), (1, ), 'input')
        buf153 = spyre_empty_with_layout((2048, 2304), (2304, 1), torch.float16, SpyreTensorLayout(device_size=[36, 2048, 64], stride_map =[64, 2304, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_gelu_layer_norm_mm_18.run(buf328, buf134, arg89_1, arg88_1, arg90_1, arg91_1, arg93_1, arg92_1, arg95_1, arg94_1, arg96_1, arg97_1, arg99_1, arg98_1, buf153)
        del arg88_1
        del arg89_1
        del arg90_1
        del arg91_1
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del arg96_1
        del arg97_1
        del arg98_1
        del arg99_1
        del buf134
        del buf328
        buf154 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        buf330 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_19.run(buf330)
        # Topologically Sorted Source Nodes: [split_7, query_7, key_7, value_7, unified_attention_with_output_7], Original ATen: [aten.split_with_sizes, aten.view, aten.as_strided, vllm.unified_attention_with_output]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf153, (2048, 12, 64), (2304, 64, 1), 0), reinterpret_tensor(buf153, (2048, 12, 64), (2304, 64, 1), 768), reinterpret_tensor(buf153, (2048, 12, 64), (2304, 64, 1), 1536), reinterpret_tensor(buf154, (2048, 12, 64), (768, 64, 1), 0), arg100_1)
        del arg100_1
        del buf153
        assert_size_stride(arg102_1, (768, 768), (768, 1), 'input')
        assert_size_stride(arg101_1, (768, ), (1, ), 'input')
        assert_size_stride(arg103_1, (768, ), (1, ), 'input')
        assert_size_stride(arg104_1, (768, ), (1, ), 'input')
        assert_size_stride(arg106_1, (768, 3072), (3072, 1), 'input')
        assert_size_stride(arg105_1, (3072, ), (1, ), 'input')
        assert_size_stride(arg108_1, (3072, 768), (768, 1), 'input')
        assert_size_stride(arg107_1, (768, ), (1, ), 'input')
        assert_size_stride(arg109_1, (768, ), (1, ), 'input')
        assert_size_stride(arg110_1, (768, ), (1, ), 'input')
        assert_size_stride(arg112_1, (768, 2304), (2304, 1), 'input')
        assert_size_stride(arg111_1, (2304, ), (1, ), 'input')
        buf173 = spyre_empty_with_layout((2048, 2304), (2304, 1), torch.float16, SpyreTensorLayout(device_size=[36, 2048, 64], stride_map =[64, 2304, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_gelu_layer_norm_mm_20.run(buf330, buf154, arg102_1, arg101_1, arg103_1, arg104_1, arg106_1, arg105_1, arg108_1, arg107_1, arg109_1, arg110_1, arg112_1, arg111_1, buf173)
        del arg101_1
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        del arg106_1
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        del arg111_1
        del arg112_1
        del buf154
        del buf330
        buf174 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        buf332 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_21.run(buf332)
        # Topologically Sorted Source Nodes: [split_8, query_8, key_8, value_8, unified_attention_with_output_8], Original ATen: [aten.split_with_sizes, aten.view, aten.as_strided, vllm.unified_attention_with_output]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf173, (2048, 12, 64), (2304, 64, 1), 0), reinterpret_tensor(buf173, (2048, 12, 64), (2304, 64, 1), 768), reinterpret_tensor(buf173, (2048, 12, 64), (2304, 64, 1), 1536), reinterpret_tensor(buf174, (2048, 12, 64), (768, 64, 1), 0), arg113_1)
        del arg113_1
        del buf173
        assert_size_stride(arg115_1, (768, 768), (768, 1), 'input')
        assert_size_stride(arg114_1, (768, ), (1, ), 'input')
        assert_size_stride(arg116_1, (768, ), (1, ), 'input')
        assert_size_stride(arg117_1, (768, ), (1, ), 'input')
        assert_size_stride(arg119_1, (768, 3072), (3072, 1), 'input')
        assert_size_stride(arg118_1, (3072, ), (1, ), 'input')
        assert_size_stride(arg121_1, (3072, 768), (768, 1), 'input')
        assert_size_stride(arg120_1, (768, ), (1, ), 'input')
        assert_size_stride(arg122_1, (768, ), (1, ), 'input')
        assert_size_stride(arg123_1, (768, ), (1, ), 'input')
        assert_size_stride(arg125_1, (768, 2304), (2304, 1), 'input')
        assert_size_stride(arg124_1, (2304, ), (1, ), 'input')
        buf193 = spyre_empty_with_layout((2048, 2304), (2304, 1), torch.float16, SpyreTensorLayout(device_size=[36, 2048, 64], stride_map =[64, 2304, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_gelu_layer_norm_mm_22.run(buf332, buf174, arg115_1, arg114_1, arg116_1, arg117_1, arg119_1, arg118_1, arg121_1, arg120_1, arg122_1, arg123_1, arg125_1, arg124_1, buf193)
        del arg114_1
        del arg115_1
        del arg116_1
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        del arg121_1
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        del buf174
        del buf332
        buf194 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        buf334 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_23.run(buf334)
        # Topologically Sorted Source Nodes: [split_9, query_9, key_9, value_9, unified_attention_with_output_9], Original ATen: [aten.split_with_sizes, aten.view, aten.as_strided, vllm.unified_attention_with_output]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf193, (2048, 12, 64), (2304, 64, 1), 0), reinterpret_tensor(buf193, (2048, 12, 64), (2304, 64, 1), 768), reinterpret_tensor(buf193, (2048, 12, 64), (2304, 64, 1), 1536), reinterpret_tensor(buf194, (2048, 12, 64), (768, 64, 1), 0), arg126_1)
        del arg126_1
        del buf193
        assert_size_stride(arg128_1, (768, 768), (768, 1), 'input')
        assert_size_stride(arg127_1, (768, ), (1, ), 'input')
        assert_size_stride(arg129_1, (768, ), (1, ), 'input')
        assert_size_stride(arg130_1, (768, ), (1, ), 'input')
        assert_size_stride(arg132_1, (768, 3072), (3072, 1), 'input')
        assert_size_stride(arg131_1, (3072, ), (1, ), 'input')
        assert_size_stride(arg134_1, (3072, 768), (768, 1), 'input')
        assert_size_stride(arg133_1, (768, ), (1, ), 'input')
        assert_size_stride(arg135_1, (768, ), (1, ), 'input')
        assert_size_stride(arg136_1, (768, ), (1, ), 'input')
        assert_size_stride(arg138_1, (768, 2304), (2304, 1), 'input')
        assert_size_stride(arg137_1, (2304, ), (1, ), 'input')
        buf213 = spyre_empty_with_layout((2048, 2304), (2304, 1), torch.float16, SpyreTensorLayout(device_size=[36, 2048, 64], stride_map =[64, 2304, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_gelu_layer_norm_mm_24.run(buf334, buf194, arg128_1, arg127_1, arg129_1, arg130_1, arg132_1, arg131_1, arg134_1, arg133_1, arg135_1, arg136_1, arg138_1, arg137_1, buf213)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        del arg131_1
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        del arg136_1
        del arg137_1
        del arg138_1
        del buf194
        del buf334
        buf214 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        buf336 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_25.run(buf336)
        # Topologically Sorted Source Nodes: [split_10, query_10, key_10, value_10, unified_attention_with_output_10], Original ATen: [aten.split_with_sizes, aten.view, aten.as_strided, vllm.unified_attention_with_output]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf213, (2048, 12, 64), (2304, 64, 1), 0), reinterpret_tensor(buf213, (2048, 12, 64), (2304, 64, 1), 768), reinterpret_tensor(buf213, (2048, 12, 64), (2304, 64, 1), 1536), reinterpret_tensor(buf214, (2048, 12, 64), (768, 64, 1), 0), arg139_1)
        del arg139_1
        del buf213
        assert_size_stride(arg141_1, (768, 768), (768, 1), 'input')
        assert_size_stride(arg140_1, (768, ), (1, ), 'input')
        assert_size_stride(arg142_1, (768, ), (1, ), 'input')
        assert_size_stride(arg143_1, (768, ), (1, ), 'input')
        assert_size_stride(arg145_1, (768, 3072), (3072, 1), 'input')
        assert_size_stride(arg144_1, (3072, ), (1, ), 'input')
        assert_size_stride(arg147_1, (3072, 768), (768, 1), 'input')
        assert_size_stride(arg146_1, (768, ), (1, ), 'input')
        assert_size_stride(arg148_1, (768, ), (1, ), 'input')
        assert_size_stride(arg149_1, (768, ), (1, ), 'input')
        assert_size_stride(arg151_1, (768, 2304), (2304, 1), 'input')
        assert_size_stride(arg150_1, (2304, ), (1, ), 'input')
        buf233 = spyre_empty_with_layout((2048, 2304), (2304, 1), torch.float16, SpyreTensorLayout(device_size=[36, 2048, 64], stride_map =[64, 2304, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_gelu_layer_norm_mm_26.run(buf336, buf214, arg141_1, arg140_1, arg142_1, arg143_1, arg145_1, arg144_1, arg147_1, arg146_1, arg148_1, arg149_1, arg151_1, arg150_1, buf233)
        del arg140_1
        del arg141_1
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        del arg146_1
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        del arg151_1
        del buf214
        del buf336
        buf234 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        buf338 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_27.run(buf338)
        # Topologically Sorted Source Nodes: [split_11, query_11, key_11, value_11, unified_attention_with_output_11], Original ATen: [aten.split_with_sizes, aten.view, aten.as_strided, vllm.unified_attention_with_output]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf233, (2048, 12, 64), (2304, 64, 1), 0), reinterpret_tensor(buf233, (2048, 12, 64), (2304, 64, 1), 768), reinterpret_tensor(buf233, (2048, 12, 64), (2304, 64, 1), 1536), reinterpret_tensor(buf234, (2048, 12, 64), (768, 64, 1), 0), arg152_1)
        del arg152_1
        del buf233
        assert_size_stride(arg154_1, (768, 768), (768, 1), 'input')
        assert_size_stride(arg153_1, (768, ), (1, ), 'input')
        assert_size_stride(arg155_1, (768, ), (1, ), 'input')
        assert_size_stride(arg156_1, (768, ), (1, ), 'input')
        assert_size_stride(arg158_1, (768, 3072), (3072, 1), 'input')
        assert_size_stride(arg157_1, (3072, ), (1, ), 'input')
        assert_size_stride(arg160_1, (3072, 768), (768, 1), 'input')
        assert_size_stride(arg159_1, (768, ), (1, ), 'input')
        assert_size_stride(arg161_1, (768, ), (1, ), 'input')
        assert_size_stride(arg162_1, (768, ), (1, ), 'input')
        buf251 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_gelu_layer_norm_mm_28.run(buf338, buf234, arg154_1, arg153_1, arg155_1, arg156_1, arg158_1, arg157_1, arg160_1, arg159_1, arg161_1, arg162_1, buf251)
        del arg153_1
        del arg154_1
        del arg155_1
        del arg156_1
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        del arg161_1
        del arg162_1
        return (buf251, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns
