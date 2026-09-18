# AOT ID: ['19_inference']
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


# Topologically Sorted Source Nodes: [q_rows, k_rows, v_rows, reshape, q, reshape_1, k, reshape_2, v, attn, attn_1, index_copy_], Original ATen: [aten.index_select, aten.view, aten.transpose, aten._scaled_dot_product_fused_attention_overrideable, aten.index_copy]
# Source node to ATen node mapping:
#   attn => add, add_1, add_2, add_3, add_4, add_5, amax, amax_1, amax_2, amax_3, clone, clone_1, clone_2, clone_3, clone_4, clone_5, div, exp, exp_1, exp_2, exp_3, full_default, full_default_1, full_default_2, maximum, maximum_1, mul, mul_1, mul_2, mul_3, mul_4, mul_5, mul_6, permute_3, permute_4, permute_5, slice_1, slice_2, slice_3, slice_4, slice_5, slice_6, sub, sub_1, sub_2, sub_3, sum_1, sum_2, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4
#   attn_1 => view_15
#   index_copy_ => index_put
#   k => permute_1
#   k_rows => index_1
#   q => permute
#   q_rows => index
#   reshape => view
#   reshape_1 => view_1
#   reshape_2 => view_2
#   v => permute_2
#   v_rows => index_2
# Graph fragment:
#   %arg0_1 : Tensor "f16[2048, 12, 64][2304, 64, 1]spyre:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "i32[2048][1]spyre:0" = PlaceHolder[target=arg1_1]
#   %restickify_default : Tensor "f16[2048, 12, 64][2304, 64, 1]spyre:0" = PlaceHolder[target=restickify_default]
#   %arg2_1 : Tensor "f16[2048, 12, 64][2304, 64, 1]spyre:0" = PlaceHolder[target=arg2_1]
#   %restickify_default_1 : Tensor "f16[2048, 12, 64][2304, 64, 1]spyre:0" = PlaceHolder[target=restickify_default_1]
#   %arg3_1 : Tensor "f16[2048, 12, 64][2304, 64, 1]spyre:0" = PlaceHolder[target=arg3_1]
#   %restickify_default_2 : Tensor "f16[2048, 12, 64][2304, 64, 1]spyre:0" = PlaceHolder[target=restickify_default_2]
#   %index : Tensor "f16[2048, 12, 64][768, 64, 1]spyre:0" = PlaceHolder[target=index]
#   %buf4 : Tensor "f16[][]spyre:0" = PlaceHolder[target=buf4]
#   %buf6 : Tensor "f16[][]spyre:0" = PlaceHolder[target=buf6]
#   %full_default_1 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=full_default_1]
#   %full_default_2 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=full_default_2]
#   %clone : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=clone]
#   %clone_6 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=clone_6]
#   %buf50 : Tensor "f16[][]spyre:0" = PlaceHolder[target=buf50]
#   %index_1 : Tensor "f16[2048, 12, 64][768, 64, 1]spyre:0" = PlaceHolder[target=index_1]
#   %index_2 : Tensor "f16[2048, 12, 64][768, 64, 1]spyre:0" = PlaceHolder[target=index_2]
#   %mul_1 : Tensor "f16[4, 12, 256, 64][196608, 64, 768, 1]spyre:0" = PlaceHolder[target=mul_1]
#   %expand_4 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=expand_4]
#   %expand_1 : Tensor "f16[4, 12, 64, 256][196608, 16384, 256, 1]spyre:0" = PlaceHolder[target=expand_1]
#   %batched_matmul_default_3 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=batched_matmul_default_3]
#   %arg4_1 : Tensor "f16[4, 1, 1, 512][512, 512, 512, 1]spyre:0" = PlaceHolder[target=arg4_1]
#   %add : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=add]
#   %amax : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=amax]
#   %clone_8 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=clone_8]
#   %amax_2 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=amax_2]
#   %maximum : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=maximum]
#   %sub : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=sub]
#   %sub_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=sub_1]
#   %amax_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=amax_1]
#   %clone_9 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=clone_9]
#   %exp_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=exp_1]
#   %expand_2 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=expand_2]
#   %mul_2 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=mul_2]
#   %sum_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=sum_1]
#   %expand_3 : Tensor "f16[4, 12, 256, 64][196608, 16384, 64, 1]spyre:0" = PlaceHolder[target=expand_3]
#   %full_default : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=full_default]
#   %clone_7 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=clone_7]
#   %mul_3 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=mul_3]
#   %batched_matmul_default_2 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=batched_matmul_default_2]
#   %mul_4 : Tensor "f16[4, 12, 256, 64][196608, 64, 768, 1]spyre:0" = PlaceHolder[target=mul_4]
#   %expand_5 : Tensor "f16[4, 12, 64, 256][196608, 16384, 256, 1]spyre:0" = PlaceHolder[target=expand_5]
#   %batched_matmul_default_1 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=batched_matmul_default_1]
#   %add_3 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=add_3]
#   %amax_3 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=amax_3]
#   %maximum_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=maximum_1]
#   %sub_2 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=sub_2]
#   %sub_3 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=sub_3]
#   %add_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=add_1]
#   %exp_3 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=exp_3]
#   %expand_6 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=expand_6]
#   %mul_5 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=mul_5]
#   %sum_2 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=sum_2]
#   %expand_7 : Tensor "f16[4, 12, 256, 64][196608, 16384, 64, 1]spyre:0" = PlaceHolder[target=expand_7]
#   %add_2 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=add_2]
#   %mul_6 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=mul_6]
#   %batched_matmul_default : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=batched_matmul_default]
#   %add_5 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=add_5]
#   %add_4 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=add_4]
#   %div : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=div]
#   %clone_10 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=clone_10]
#   %index_put : Tensor "f16[2048, 12, 64][768, 64, 1]spyre:0" = PlaceHolder[target=index_put]
#   %clone_5 : Tensor "f16[4, 512, 12, 64][393216, 768, 64, 1]spyre:0" = PlaceHolder[target=clone_5]
#   %restickify_default_3 : Tensor "f16[2048, 12, 64][768, 64, 1]spyre:0" = PlaceHolder[target=restickify_default_3]
#   %buf49 : Tensor "f16[2048, 12, 64][768, 64, 1]spyre:0" = PlaceHolder[target=buf49]
#   %restickify_default_4 : [num_users=0] = call_function[target=torch.ops.spyre.restickify.default](args = (%restickify_default_3,), kwargs = {})
#   %restickify_default_3 : [num_users=1] = call_function[target=torch.ops.spyre.restickify.default](args = (%index_put,), kwargs = {})
#   %restickify_default_2 : [num_users=0] = call_function[target=torch.ops.spyre.restickify.default](args = (%arg3_1,), kwargs = {})
#   %restickify_default_1 : [num_users=0] = call_function[target=torch.ops.spyre.restickify.default](args = (%arg2_1,), kwargs = {})
#   %restickify_default : [num_users=0] = call_function[target=torch.ops.spyre.restickify.default](args = (%arg0_1,), kwargs = {})
#   %index : Tensor "f16[2048, 12, 64][768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%arg1_1]), kwargs = {})
#   %index_1 : Tensor "f16[2048, 12, 64][768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg2_1, [%arg1_1]), kwargs = {})
#   %index_2 : Tensor "f16[2048, 12, 64][768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg3_1, [%arg1_1]), kwargs = {})
#   %view : Tensor "f16[4, 512, 12, 64][393216, 768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index, [4, 512, 12, 64]), kwargs = {})
#   %permute : Tensor "f16[4, 12, 512, 64][393216, 64, 768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1, 3]), kwargs = {})
#   %view_1 : Tensor "f16[4, 512, 12, 64][393216, 768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_1, [4, 512, 12, 64]), kwargs = {})
#   %permute_1 : Tensor "f16[4, 12, 512, 64][393216, 64, 768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 2, 1, 3]), kwargs = {})
#   %view_2 : Tensor "f16[4, 512, 12, 64][393216, 768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_2, [4, 512, 12, 64]), kwargs = {})
#   %permute_2 : Tensor "f16[4, 12, 512, 64][393216, 64, 768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_6 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone,), kwargs = {})
#   %full_default : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 12, 512, 64], 0), kwargs = {dtype: torch.float16, layout: torch.strided, device: spyre:0, pin_memory: False})
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%full_default,), kwargs = {})
#   %full_default_1 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 12, 512, 64], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: spyre:0, pin_memory: False})
#   %amax : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%full_default_1, [-1]), kwargs = {})
#   %clone_8 : [num_users=2] = call_function[target=torch.ops.aten.clone](args = (%amax,), kwargs = {})
#   %full_default_2 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 12, 512, 64], 0), kwargs = {dtype: torch.float16, layout: torch.strided, device: spyre:0, pin_memory: False})
#   %amax_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%full_default_2, [-1]), kwargs = {})
#   %clone_9 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%amax_1,), kwargs = {})
#   %mul : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone, 0.3535533905932738), kwargs = {})
#   %slice_1 : Tensor "f16[4, 12, 256, 64][393216, 64, 768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_1, 2, 0, 256), kwargs = {})
#   %slice_2 : Tensor "f16[4, 12, 256, 64][393216, 64, 768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 2, 0, 256), kwargs = {})
#   %mul_1 : Tensor "f16[4, 12, 256, 64][196608, 64, 768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_1, 0.3535533905932738), kwargs = {})
#   %clone_1 : Tensor "f16[4, 12, 256, 64][196608, 16384, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_3 : Tensor "f16[4, 12, 64, 256][196608, 64, 1, 768]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_2 : Tensor "f16[4, 12, 64, 256][196608, 16384, 256, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %batched_matmul_default_3 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.batched_matmul.default](args = (%expand, %expand_1), kwargs = {})
#   %slice_3 : Tensor "f16[4, 1, 1, 256][512, 512, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg4_1, 3, 0, 256), kwargs = {})
#   %add : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%batched_matmul_default_3, %slice_3), kwargs = {})
#   %amax_2 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add, [-1]), kwargs = {})
#   %maximum : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=4] = call_function[target=torch.ops.aten.maximum.default](args = (%clone_8, %amax_2), kwargs = {})
#   %unsqueeze : Tensor "f16[4, 12, 512, 1][6144, 512, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%maximum, -1), kwargs = {})
#   %sub : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %unsqueeze), kwargs = {})
#   %exp : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sub_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_8, %maximum), kwargs = {})
#   %exp_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %mul_2 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone_9, %exp_1), kwargs = {})
#   %sum_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1]), kwargs = {})
#   %add_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %sum_1), kwargs = {})
#   %batched_matmul_default_2 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.batched_matmul.default](args = (%expand_2, %expand_3), kwargs = {})
#   %unsqueeze_1 : Tensor "f16[4, 12, 512, 1][6144, 512, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp_1, -1), kwargs = {})
#   %mul_3 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone_7, %unsqueeze_1), kwargs = {})
#   %add_2 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %batched_matmul_default_2), kwargs = {})
#   %slice_4 : Tensor "f16[4, 12, 256, 64][393216, 64, 768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_1, 2, 256, 512), kwargs = {})
#   %slice_5 : Tensor "f16[4, 12, 256, 64][393216, 64, 768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 2, 256, 512), kwargs = {})
#   %mul_4 : Tensor "f16[4, 12, 256, 64][196608, 64, 768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_4, 0.3535533905932738), kwargs = {})
#   %clone_3 : Tensor "f16[4, 12, 256, 64][196608, 16384, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_5,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_4 : Tensor "f16[4, 12, 64, 256][196608, 64, 1, 768]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_4, [0, 1, 3, 2]), kwargs = {})
#   %clone_4 : Tensor "f16[4, 12, 64, 256][196608, 16384, 256, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   %batched_matmul_default_1 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.batched_matmul.default](args = (%expand_4, %expand_5), kwargs = {})
#   %slice_6 : Tensor "f16[4, 1, 1, 256][512, 512, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg4_1, 3, 256, 512), kwargs = {})
#   %add_3 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%batched_matmul_default_1, %slice_6), kwargs = {})
#   %amax_3 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_3, [-1]), kwargs = {})
#   %maximum_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.maximum.default](args = (%maximum, %amax_3), kwargs = {})
#   %unsqueeze_2 : Tensor "f16[4, 12, 512, 1][6144, 512, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%maximum_1, -1), kwargs = {})
#   %sub_2 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %unsqueeze_2), kwargs = {})
#   %exp_2 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sub_3 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%maximum, %maximum_1), kwargs = {})
#   %exp_3 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_3,), kwargs = {})
#   %mul_5 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %exp_3), kwargs = {})
#   %sum_2 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [-1]), kwargs = {})
#   %add_4 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %sum_2), kwargs = {})
#   %batched_matmul_default : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.batched_matmul.default](args = (%expand_6, %expand_7), kwargs = {})
#   %unsqueeze_3 : Tensor "f16[4, 12, 512, 1][6144, 512, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp_3, -1), kwargs = {})
#   %mul_6 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %unsqueeze_3), kwargs = {})
#   %add_5 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %batched_matmul_default), kwargs = {})
#   %unsqueeze_4 : Tensor "f16[4, 12, 512, 1][6144, 512, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_4, -1), kwargs = {})
#   %div : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_5, %unsqueeze_4), kwargs = {})
#   %clone_10 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%div,), kwargs = {})
#   %permute_5 : Tensor "f16[4, 512, 12, 64][393216, 64, 32768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div, [0, 2, 1, 3]), kwargs = {})
#   %clone_5 : Tensor "f16[4, 512, 12, 64][393216, 768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_15 : Tensor "f16[2048, 12, 64][768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_5, [2048, 12, 64]), kwargs = {})
#   %index_put : Tensor "f16[2048, 12, 64][768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%arg5_1, [%arg1_1], %view_15), kwargs = {})
#   return %restickify_default,%index,%restickify_default_1,%index_1,%restickify_default_2,%index_2,%clone,%full_default,%full_default_1,%amax,%full_default_2,%amax_1,%clone_6,%expand_4,%mul_1,%expand_3,%expand_1,%batched_matmul_default_3,%add,%amax_2,%clone_8,%maximum,%sub,%expand_2,%sub_1,%exp_1,%clone_9,%mul_2,%sum_1,%add_1,%batched_matmul_default_2,%clone_7,%mul_3,%add_2,%mul_4,%expand_7,%expand_5,%batched_matmul_default_1,%add_3,%amax_3,%maximum_1,%sub_2,%expand_6,%sub_3,%exp_3,%mul_5,%sum_2,%add_4,%batched_matmul_default,%mul_6,%add_5,%div,%clone_10,%clone_5,%restickify_default_3,%buf49,%restickify_default_4
sdsc_fused__scaled_dot_product_fused_attention_overrideable_index_copy_index_select_transpose_view_0 = async_compile.sdsc('sdsc_fused__scaled_dot_product_fused_attention_overrideable_index_copy_index_select_transpose_view_0',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=918311540789157568, source=None, aten_op=None, ir_chain=('restickify_default', 'buf53'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c1'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 36, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('c1'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8346667133486725190, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=99, start_col=0, end_line=None, end_col=None), aten_op='aten.index_select.default', ir_chain=('index', 'buf0'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='insert_restickify', reason='redirect consumer to restickified input'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.IEEE_INT32,
                    device_size=[1, 1, 64, 32],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('floor(c0/32)'), sympify('Mod(c0, 32)')],
                    allocation={'hbm': 1},
                    name='arg1_1',
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 36, 64],
                    device_coordinates=[sympify('floor(c2/64)'), IndirectAccess('arg1_1'), sympify('c1'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c1'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 9437184},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8651520208525719955, source=None, aten_op=None, ir_chain=('restickify_default_1', 'buf54'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c1 + 12'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 36, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('c1 + 12'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7816771498333148075, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=100, start_col=0, end_line=None, end_col=None), aten_op='aten.index_select.default', ir_chain=('index_1', 'buf1'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='insert_restickify', reason='redirect consumer to restickified input'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.IEEE_INT32,
                    device_size=[1, 1, 64, 32],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('floor(c0/32)'), sympify('Mod(c0, 32)')],
                    allocation={'hbm': 1},
                    name='arg1_1',
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 36, 64],
                    device_coordinates=[sympify('floor(c2/64)'), IndirectAccess('arg1_1'), sympify('c1 + 12'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c1'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 12582912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=9063738291450937306, source=None, aten_op=None, ir_chain=('restickify_default_2', 'buf55'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c1 + 24'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 3},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 36, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('c1 + 24'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6638142833581113712, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=101, start_col=0, end_line=None, end_col=None), aten_op='aten.index_select.default', ir_chain=('index_2', 'buf2'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='insert_restickify', reason='redirect consumer to restickified input'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.IEEE_INT32,
                    device_size=[1, 1, 64, 32],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('floor(c0/32)'), sympify('Mod(c0, 32)')],
                    allocation={'hbm': 1},
                    name='arg1_1',
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 36, 64],
                    device_coordinates=[sympify('floor(c2/64)'), IndirectAccess('arg1_1'), sympify('c1 + 24'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c1'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 15728640},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('512'), 32), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('0'), sympify('c2'): sympify('Mod(core_id, 32)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=384740451836655075, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone', 'permute', 'view', 'buf3'), fused_from=(DebugHandle(id=1807768127788485654, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone',), fused_from=(), transform_history=()), DebugHandle(id=5867719102715137985, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=140, start_col=0, end_line=None, end_col=None), aten_op='aten.transpose.int', ir_chain=('permute',), fused_from=(), transform_history=()), DebugHandle(id=8580950343354786085, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=140, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 4, 512, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c0'), sympify('c2'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 9437184},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('512'), 32), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('0'), sympify('c2'): sympify('Mod(core_id, 32)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4623453487515922078, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('full_default', 'buf5'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 196608},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('512'), 32), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('0'), sympify('c2'): sympify('Mod(core_id, 32)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3290348333837216587, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('full_default_1', 'buf7'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='max',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('512'), 32), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('0'), sympify('c2'): sympify('Mod(core_id, 32)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7911995733255025487, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('amax', 'buf8'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 491520},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('512'), 32), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('0'), sympify('c2'): sympify('Mod(core_id, 32)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8276563753563360026, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('full_default_2', 'buf10'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 4},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 688128},
                ),
            ]
        ),
        OpSpec(
            op='max',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('512'), 32), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('0'), sympify('c2'): sympify('Mod(core_id, 32)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=9206347928299320266, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('amax_1', 'buf11'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 688128},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 786432},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('64'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(core_id, 4)), 4)'), sympify('c2'): sympify('Mod(floor(Mod(floor(core_id/4), 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7969214884449307410, source=None, aten_op=None, ir_chain=('clone_6', 'buf58'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c2'): 32}, core_id_to_work_slice={sympify('c2'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 98304},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c2'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(core_id, 4)), 4)'), sympify('c2'): sympify('Mod(floor(Mod(floor(core_id/4), 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='mul',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1935828838745218449, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul', 'buf12'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 98304},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='mul',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('256'), 1), sympify('c2'): (sympify('12'), 4), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('0'), sympify('c2'): sympify('Mod(core_id, 4)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6885383028838157095, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_1', 'permute_1', 'slice_1', 'view_1', 'buf13'), fused_from=(DebugHandle(id=8143240155156378101, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_1',), fused_from=(), transform_history=()), DebugHandle(id=963564299322220934, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=141, start_col=0, end_line=None, end_col=None), aten_op='aten.transpose.int', ir_chain=('permute_1',), fused_from=(), transform_history=()), DebugHandle(id=3730399082067380294, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('slice_1',), fused_from=(), transform_history=()), DebugHandle(id=8775588297274009122, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=141, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_1',), fused_from=(), transform_history=())), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 4, 512, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c2'), sympify('c0'), sympify('c1'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 12582912},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 256, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c2'), sympify('c1'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 983040},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('256'), 1), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('0'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5503549003862147530, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_1', 'permute_2', 'slice_2', 'view_2', 'buf14'), fused_from=(DebugHandle(id=7748820657194833146, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_1',), fused_from=(), transform_history=()), DebugHandle(id=353813194746067337, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=142, start_col=0, end_line=None, end_col=None), aten_op='aten.transpose.int', ir_chain=('permute_2',), fused_from=(), transform_history=()), DebugHandle(id=1132747713129513798, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('slice_2',), fused_from=(), transform_history=()), DebugHandle(id=4547031481231723753, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=142, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_2',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 4, 512, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c0'), sympify('c2'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 15728640},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 256, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 9437184},
                ),
            ]
        ),
        OpSpec(
            op='ReStickifyOpHBM',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('64'), 1), sympify('c3'): (sympify('256'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('0'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5870524461789763571, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_2', 'permute_3', 'buf15'), fused_from=(DebugHandle(id=6834276789776821074, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_2',), fused_from=(), transform_history=()), DebugHandle(id=8348530548129573606, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('permute_3',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 256, 4, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c1'), sympify('c3'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 983040},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 64, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 0},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('256'), 1), sympify('c4'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0'), sympify('c4'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1146477753423887580, source=None, aten_op=None, ir_chain=('batched_matmul_default_3', 'buf16'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c4/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c4, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 64, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c4'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 983040},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('256'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4983301553800056920, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add', 'slice_3', 'buf17'), fused_from=(DebugHandle(id=4687122253134995631, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add',), fused_from=(), transform_history=()), DebugHandle(id=488127811832428933, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('slice_3',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 983040},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 8, 4, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm': 7},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 983040},
                ),
            ]
        ),
        OpSpec(
            op='max',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('256'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5735663547655570489, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('amax_2', 'buf18'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 983040},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1376256},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(core_id, 4)), 4)'), sympify('c2'): sympify('Mod(floor(Mod(floor(core_id/4), 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8381486156206718047, source=None, aten_op=None, ir_chain=('clone_8', 'buf60'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 491520},
                    work_division=TensorWorkDivision(work_slices={sympify('c2'): 32}, core_id_to_work_slice={sympify('c2'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 589824},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c2'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(core_id, 4)), 4)'), sympify('c2'): sympify('Mod(floor(Mod(floor(core_id/4), 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='maximum',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1890077361064605398, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('maximum', 'buf19'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 589824},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1376256},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1376256},
                ),
            ]
        ),
        OpSpec(
            op='sub',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('256'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4663212582333777483, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub', 'unsqueeze', 'buf20'), fused_from=(DebugHandle(id=8403910893866952374, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub',), fused_from=(), transform_history=()), DebugHandle(id=2810091853436499222, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('unsqueeze',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 983040},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1376256},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 983040},
                ),
            ]
        ),
        OpSpec(
            op='exp',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('256'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2747003284634485694, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('exp', 'buf21'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 983040},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 983040},
                ),
            ]
        ),
        OpSpec(
            op='sub',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1603264990576388312, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub_1', 'buf22'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 589824},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1376256},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1474560},
                ),
            ]
        ),
        OpSpec(
            op='exp',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5677443678029016304, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('exp_1', 'buf23'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1474560},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1474560},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(core_id, 4)), 4)'), sympify('c2'): sympify('Mod(floor(Mod(floor(core_id/4), 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3102195545038915432, source=None, aten_op=None, ir_chain=('clone_9', 'buf61'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 786432},
                    work_division=TensorWorkDivision(work_slices={sympify('c2'): 32}, core_id_to_work_slice={sympify('c2'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 884736},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c2'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(core_id, 4)), 4)'), sympify('c2'): sympify('Mod(floor(Mod(floor(core_id/4), 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='mul',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=3987921893808716437, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_2', 'buf24'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 884736},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1474560},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='sum',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('256'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6963398749091962773, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sum_1', 'buf25'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 983040},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1204866810652909229, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_1', 'buf26'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('64'), 1), sympify('c4'): (sympify('256'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0'), sympify('c4'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8291846595621657398, source=None, aten_op=None, ir_chain=('batched_matmul_default_2', 'buf27'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c4/64)'), sympify('c0'), sympify('Mod(c4, 64)')],
                    allocation={'lx': 983040},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 256, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c4'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 9437184},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 393216},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('64'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(core_id, 4)), 4)'), sympify('c2'): sympify('Mod(floor(Mod(floor(core_id/4), 8)), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7861081363920250262, source=None, aten_op=None, ir_chain=('clone_7', 'buf59'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 196608},
                    work_division=TensorWorkDivision(work_slices={sympify('c2'): 32}, core_id_to_work_slice={sympify('c2'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 294912},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c2'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(core_id, 4)), 4)'), sympify('c2'): sympify('Mod(floor(Mod(floor(core_id/4), 8)), 8)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='mul',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2747983291087774305, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_3', 'unsqueeze_1', 'buf28'), fused_from=(DebugHandle(id=7583862333572157710, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_3',), fused_from=(), transform_history=()), DebugHandle(id=3491802023200749811, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('unsqueeze_1',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1474560},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 196608},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4994768580266409571, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_2', 'buf29'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 393216},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 196608},
                ),
            ]
        ),
        OpSpec(
            op='mul',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('256'), 1), sympify('c2'): (sympify('12'), 4), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('0'), sympify('c2'): sympify('Mod(core_id, 4)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=351150742692536192, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_4', 'permute_1', 'slice_4', 'view_1', 'buf30'), fused_from=(DebugHandle(id=4803139810492552794, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_4',), fused_from=(), transform_history=()), DebugHandle(id=963564299322220934, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=141, start_col=0, end_line=None, end_col=None), aten_op='aten.transpose.int', ir_chain=('permute_1',), fused_from=(), transform_history=()), DebugHandle(id=6436308577920815192, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('slice_4',), fused_from=(), transform_history=()), DebugHandle(id=8775588297274009122, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=141, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_1',), fused_from=(), transform_history=())), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 4, 512, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c2'), sympify('c0'), sympify('c1 + 256'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 12582912},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 6},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 256, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c2'), sympify('c1'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('256'), 1), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('0'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1880352242005809296, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_3', 'permute_2', 'slice_5', 'view_2', 'buf31'), fused_from=(DebugHandle(id=6068268152231260879, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_3',), fused_from=(), transform_history=()), DebugHandle(id=353813194746067337, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=142, start_col=0, end_line=None, end_col=None), aten_op='aten.transpose.int', ir_chain=('permute_2',), fused_from=(), transform_history=()), DebugHandle(id=8825569736954863068, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('slice_5',), fused_from=(), transform_history=()), DebugHandle(id=4547031481231723753, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=142, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_2',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 4, 512, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c0'), sympify('c2 + 256'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 15728640},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 256, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 11010048},
                ),
            ]
        ),
        OpSpec(
            op='ReStickifyOpHBM',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('64'), 1), sympify('c3'): (sympify('256'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('0'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6858217778425570015, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_4', 'permute_4', 'buf32'), fused_from=(DebugHandle(id=6367171389009255905, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_4',), fused_from=(), transform_history=()), DebugHandle(id=7633893598496844985, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('permute_4',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 256, 4, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c1'), sympify('c3'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 64, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 1572864},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('256'), 1), sympify('c4'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0'), sympify('c4'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=1169423378526771053, source=None, aten_op=None, ir_chain=('batched_matmul_default_1', 'buf33'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c4/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c4, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 64, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c4'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 1572864},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('256'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5581583362759917789, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_3', 'slice_6', 'buf34'), fused_from=(DebugHandle(id=6960212548286426052, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_3',), fused_from=(), transform_history=()), DebugHandle(id=1311853468763056234, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('slice_6',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 8, 4, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('floor(c3/64) + 4'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm': 7},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='max',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('256'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4003707071191472759, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('amax_3', 'buf35'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='maximum',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4898112130311491972, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('maximum_1', 'buf36'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1376256},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='sub',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('256'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=8723221108208571999, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub_2', 'unsqueeze_2', 'buf37'), fused_from=(DebugHandle(id=8454414257327551235, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub_2',), fused_from=(), transform_history=()), DebugHandle(id=7949979451535169542, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('unsqueeze_2',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='exp',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('256'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=767310833141823864, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('exp_2', 'buf38'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 294912},
                ),
            ]
        ),
        OpSpec(
            op='sub',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=2107097306393774669, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub_3', 'buf39'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1376256},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1376256},
                ),
            ]
        ),
        OpSpec(
            op='exp',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6563945360920749870, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('exp_3', 'buf40'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1376256},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1376256},
                ),
            ]
        ),
        OpSpec(
            op='mul',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4620450070549194153, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_5', 'buf41'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1376256},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='sum',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('256'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=848615925407118749, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sum_2', 'buf42'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5387375842815564873, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_4', 'buf43'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('64'), 1), sympify('c4'): (sympify('256'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0'), sympify('c4'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4605877516473058399, source=None, aten_op=None, ir_chain=('batched_matmul_default', 'buf44'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c4/64)'), sympify('c0'), sympify('Mod(c4, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 256, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c4'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 11010048},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 0},
                ),
            ]
        ),
        OpSpec(
            op='mul',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4421519691004350300, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_6', 'unsqueeze_3', 'buf45'), fused_from=(DebugHandle(id=2662877099685315493, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_6',), fused_from=(), transform_history=()), DebugHandle(id=8843320404560722877, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('unsqueeze_3',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 1376256},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 196608},
                ),
            ]
        ),
        OpSpec(
            op='add',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5018487244208532710, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_5', 'buf46'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 196608},
                ),
            ]
        ),
        OpSpec(
            op='realdiv',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 4), sympify('c2'): (sympify('512'), 8), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 4)'), sympify('c2'): sympify('Mod(floor(core_id/4), 8)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6269707778668916758, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('div', 'unsqueeze_4', 'buf47'), fused_from=(DebugHandle(id=5006824474438891622, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('div',), fused_from=(), transform_history=()), DebugHandle(id=962675954221522044, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('unsqueeze_4',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('0'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('0')],
                    allocation={'lx': 98304},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 196608},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('512'), 32), sympify('c3'): (sympify('64'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c2'): sympify('Mod(floor(Mod(core_id, 32)), 32)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=7994239021761521957, source=None, aten_op=None, ir_chain=('clone_10', 'buf62'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 196608},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c2'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(core_id, 4)), 4)'), sympify('c2'): sympify('Mod(floor(Mod(floor(core_id/4), 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c2'): 32}, core_id_to_work_slice={sympify('c2'): sympify('Mod(floor(Mod(core_id, 32)), 32)')}, num_cores=32),
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('512'), 32), sympify('c2'): (sympify('12'), 1), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('Mod(core_id, 32)'), sympify('c2'): sympify('0'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=5758865698726321084, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_5', 'permute_5', 'buf48'), fused_from=(DebugHandle(id=8233100920762403970, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_5',), fused_from=(), transform_history=()), DebugHandle(id=3709087211791702779, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=143, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('permute_5',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c2'), sympify('c1'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 512, 12, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 12582912},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=311310280395918320, source=None, aten_op=None, ir_chain=('restickify_default_3', 'buf56'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c1'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 8},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 12, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('c1'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 15728640},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('512'), 8), sympify('z0'): (sympify('4'), 4), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('z0'): sympify('Mod(floor(core_id/8), 4)'), sympify('c1'): sympify('0'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6878246752543034760, source=None, aten_op=None, ir_chain=('index_put', 'view_15', 'buf49'), fused_from=(DebugHandle(id=3348002677201244519, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=205, start_col=0, end_line=None, end_col=None), aten_op='aten.index_copy.default', ir_chain=('index_put',), fused_from=(), transform_history=()), DebugHandle(id=7189895595612834646, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=152, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_15',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.IEEE_INT32,
                    device_size=[1, 1, 4, 16, 32],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('z0'), sympify('floor(c0/32)'), sympify('Mod(c0, 32)')],
                    allocation={'hbm': 1},
                    name='arg1_1',
                ),
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 512, 12, 4, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('c1'), sympify('z0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 12582912},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 2048, 12, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('0'), IndirectAccess('arg1_1'), sympify('c1'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 15728640},
                ),
            ]
        ),
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 32), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 32)'), sympify('c1'): sympify('0'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4582154189224422974, source=None, aten_op=None, ir_chain=('restickify_default_4', 'buf57'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 2048, 12, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('c1'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 15728640},
                ),
                TensorArg(
                    is_input=False, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c1'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 8},
                ),
            ]
        ),
    ]
, pool_size=18874368)


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
        args.clear()
        buf4 = spyre_constant_tensor(0.0, torch.device("spyre:0"), torch.float16)
        buf6 = spyre_constant_tensor(-inf, torch.device("spyre:0"), torch.float16)
        buf50 = spyre_constant_tensor(0.3535533905932738, torch.device("spyre:0"), torch.float16)
        assert_size_stride(arg0_1, (2048, 12, 64), (2304, 64, 1), 'input')
        assert_size_stride(arg1_1, (2048, ), (1, ), 'input')
        assert_size_stride(arg2_1, (2048, 12, 64), (2304, 64, 1), 'input')
        assert_size_stride(arg3_1, (2048, 12, 64), (2304, 64, 1), 'input')
        assert_size_stride(arg4_1, (4, 1, 1, 512), (512, 512, 512, 1), 'input')
        assert_size_stride(arg5_1, (2048, 12, 64), (768, 64, 1), 'input')
        sdsc_fused__scaled_dot_product_fused_attention_overrideable_index_copy_index_select_transpose_view_0.run(arg0_1, arg1_1, arg2_1, arg3_1, buf4, buf6, buf50, arg4_1, arg5_1)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        return (arg5_1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns
