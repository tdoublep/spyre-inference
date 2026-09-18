# AOT ID: ['5_inference']
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


# Topologically Sorted Source Nodes: [out, out_1], Original ATen: [aten.mm, aten.add]
# Source node to ATen node mapping:
#   out => mm
#   out_1 => add
# Graph fragment:
#   %arg2_1 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=arg2_1]
#   %arg1_1 : Tensor "f16[768, 2304][2304, 1]spyre:0" = PlaceHolder[target=arg1_1]
#   %mm : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=mm]
#   %clone_6 : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=clone_6]
#   %arg0_1 : Tensor "f16[2304][1]spyre:0" = PlaceHolder[target=arg0_1]
#   %mm : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%arg2_1, %arg1_1), kwargs = {})
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm,), kwargs = {})
#   %add : Tensor "f16[2048, 2304][2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_6, %arg0_1), kwargs = {})
#   return %mm,%clone_6,%add
sdsc_fused_add_mm_0 = async_compile.sdsc('sdsc_fused_add_mm_0',
    [
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('2048'), 8), sympify('c1'): (sympify('2304'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 8)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6938513630776324539, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm', 'buf0'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 1},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
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
            debug_handle=DebugHandle(id=4782515705933174787, source=None, aten_op=None, ir_chain=('clone_6', 'buf94'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                    work_division=TensorWorkDivision(work_slices={sympify('c1'): 4, sympify('c0'): 8}, core_id_to_work_slice={sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/8), 4)), 4)'), sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')}, num_cores=32),
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
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
            debug_handle=DebugHandle(id=576147082821142320, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add', 'buf1'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 2},
                ),
                TensorArg(
                    is_input=False, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[36, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 3},
                ),
            ]
        ),
    ]
)


cpp_fused_cat_fill_lift_fresh_1 = async_compile.cpp_pybinding(['at::Half*', 'at::Half*', 'at::Half*', 'at::Half*', 'at::Half*', 'at::Half*', 'at::Half*', 'at::Half*', 'at::Half*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(at::Half* in_out_ptr0,
                       at::Half* out_ptr0,
                       at::Half* out_ptr1,
                       at::Half* out_ptr2,
                       at::Half* out_ptr3,
                       at::Half* out_ptr4,
                       at::Half* out_ptr5,
                       at::Half* out_ptr6,
                       at::Half* out_ptr7)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    auto in_ptr0 = in_out_ptr0;
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                out_ptr0[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(32L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512L)))
                    {
                        auto tmp0 = out_ptr0[static_cast<int64_t>(0L)];
                        auto tmp1 = at::vec::Vectorized<at::Half>(tmp0);
                        tmp1.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(32));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    {
                        auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                        out_ptr2[static_cast<int64_t>(0L)] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(32L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512L)))
                    {
                        auto tmp0 = out_ptr2[static_cast<int64_t>(0L)];
                        auto tmp1 = at::vec::Vectorized<at::Half>(tmp0);
                        tmp1.store(out_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(32));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    {
                        auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                        out_ptr4[static_cast<int64_t>(0L)] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(32L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512L)))
                    {
                        auto tmp0 = out_ptr4[static_cast<int64_t>(0L)];
                        auto tmp1 = at::vec::Vectorized<at::Half>(tmp0);
                        tmp1.store(out_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(32));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    {
                        auto tmp0 = in_out_ptr0[static_cast<int64_t>(0L)];
                        in_out_ptr0[static_cast<int64_t>(0L)] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(32L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512L)))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<int64_t>(0L)];
                        auto tmp1 = at::vec::Vectorized<at::Half>(tmp0);
                        tmp1.store(out_ptr6 + static_cast<int64_t>(x0), static_cast<int64_t>(32));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(32L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512L)))
                    {
                        auto tmp0 = at::vec::Vectorized<at::Half>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(32));
                        tmp0.store(out_ptr7 + static_cast<int64_t>(x0), static_cast<int64_t>(32));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(32L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512L)))
                    {
                        auto tmp0 = at::vec::Vectorized<at::Half>::loadu(out_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(32));
                        tmp0.store(out_ptr7 + static_cast<int64_t>(512L + x0), static_cast<int64_t>(32));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(32L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512L)))
                    {
                        auto tmp0 = at::vec::Vectorized<at::Half>::loadu(out_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(32));
                        tmp0.store(out_ptr7 + static_cast<int64_t>(1024L + x0), static_cast<int64_t>(32));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(32L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512L)))
                    {
                        auto tmp0 = at::vec::Vectorized<at::Half>::loadu(out_ptr6 + static_cast<int64_t>(x0), static_cast<int64_t>(32));
                        tmp0.store(out_ptr7 + static_cast<int64_t>(1536L + x0), static_cast<int64_t>(32));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


# Topologically Sorted Source Nodes: [split, query, key, value, view_4, q_1, view_5, k_1, view_6, v_1, attn, reshape, copy_, out_2, out_3, add_2, hidden_states, out_4, out_5, hidden_states_1, out_6, out_7, add_5, hidden_states_2], Original ATen: [aten.split_with_sizes, aten.view, aten.transpose, aten._scaled_dot_product_fused_attention_overrideable, aten.mm, aten.add, aten.layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   add_2 => add_8
#   add_5 => add_11
#   attn => add_1, add_2, add_3, add_4, add_5, add_6, amax, amax_1, amax_2, amax_3, clone, clone_1, clone_2, clone_3, clone_4, clone_5, div, exp, exp_1, exp_2, exp_3, full_default_10, full_default_8, full_default_9, maximum, maximum_1, mul, mul_1, mul_2, mul_3, mul_4, mul_5, mul_6, permute_3, permute_4, permute_5, slice_1, slice_2, slice_3, slice_4, slice_5, slice_6, sub, sub_1, sub_2, sub_3, sum_1, sum_2, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4
#   copy_ => view_20
#   hidden_states => exx2, layernormnorm, layernormscale
#   hidden_states_1 => gelu
#   hidden_states_2 => exx2_1, layernormnorm_1, layernormscale_1
#   k_1 => permute_1
#   key => view_2
#   out_2 => mm_1
#   out_3 => add_7
#   out_4 => mm_2
#   out_5 => add_9
#   out_6 => mm_3
#   out_7 => add_10
#   q_1 => permute
#   query => view
#   reshape => view_19
#   split => split_with_sizes
#   v_1 => permute_2
#   value => view_3
#   view_4 => view_4
#   view_5 => view_5
#   view_6 => view_6
# Graph fragment:
#   %add : Tensor "f16[2048, 2304][2304, 1]spyre:0" = PlaceHolder[target=add]
#   %buf31 : Tensor "f16[][]spyre:0" = PlaceHolder[target=buf31]
#   %buf33 : Tensor "f16[][]spyre:0" = PlaceHolder[target=buf33]
#   %full_default_9 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=full_default_9]
#   %full_default_10 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=full_default_10]
#   %clone : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=clone]
#   %clone_7 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=clone_7]
#   %buf91 : Tensor "f16[][]spyre:0" = PlaceHolder[target=buf91]
#   %mul_1 : Tensor "f16[4, 12, 256, 64][196608, 64, 768, 1]spyre:0" = PlaceHolder[target=mul_1]
#   %expand_4 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=expand_4]
#   %expand_1 : Tensor "f16[4, 12, 64, 256][196608, 16384, 256, 1]spyre:0" = PlaceHolder[target=expand_1]
#   %batched_matmul_default_3 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=batched_matmul_default_3]
#   %spyre_convert : Tensor "f16[4, 1, 1, 512][512, 512, 512, 1]spyre:0" = PlaceHolder[target=spyre_convert]
#   %add_1 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=add_1]
#   %amax : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=amax]
#   %clone_9 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=clone_9]
#   %amax_2 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=amax_2]
#   %maximum : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=maximum]
#   %sub : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=sub]
#   %sub_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=sub_1]
#   %amax_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=amax_1]
#   %clone_10 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=clone_10]
#   %exp_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=exp_1]
#   %expand_2 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=expand_2]
#   %mul_2 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=mul_2]
#   %sum_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=sum_1]
#   %expand_3 : Tensor "f16[4, 12, 256, 64][196608, 16384, 64, 1]spyre:0" = PlaceHolder[target=expand_3]
#   %full_default_8 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=full_default_8]
#   %clone_8 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=clone_8]
#   %mul_3 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=mul_3]
#   %batched_matmul_default_2 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=batched_matmul_default_2]
#   %mul_4 : Tensor "f16[4, 12, 256, 64][196608, 64, 768, 1]spyre:0" = PlaceHolder[target=mul_4]
#   %expand_5 : Tensor "f16[4, 12, 64, 256][196608, 16384, 256, 1]spyre:0" = PlaceHolder[target=expand_5]
#   %batched_matmul_default_1 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=batched_matmul_default_1]
#   %add_4 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=add_4]
#   %amax_3 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=amax_3]
#   %maximum_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=maximum_1]
#   %sub_2 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=sub_2]
#   %sub_3 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=sub_3]
#   %add_2 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=add_2]
#   %exp_3 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=exp_3]
#   %expand_6 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0" = PlaceHolder[target=expand_6]
#   %mul_5 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=mul_5]
#   %sum_2 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=sum_2]
#   %expand_7 : Tensor "f16[4, 12, 256, 64][196608, 16384, 64, 1]spyre:0" = PlaceHolder[target=expand_7]
#   %add_3 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=add_3]
#   %mul_6 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=mul_6]
#   %batched_matmul_default : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=batched_matmul_default]
#   %add_6 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=add_6]
#   %add_5 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0" = PlaceHolder[target=add_5]
#   %div : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=div]
#   %clone_11 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0" = PlaceHolder[target=clone_11]
#   %clone_5 : Tensor "f16[4, 512, 12, 64][393216, 768, 64, 1]spyre:0" = PlaceHolder[target=clone_5]
#   %arg4_1 : Tensor "f16[768, 768][768, 1]spyre:0" = PlaceHolder[target=arg4_1]
#   %mm_1 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_1]
#   %clone_12 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_12]
#   %arg3_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg3_1]
#   %add_7 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_7]
#   %arg2_1 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=arg2_1]
#   %add_8 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_8]
#   %exx2 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2]
#   %layernormscale : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale]
#   %arg5_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg6_1]
#   %layernormnorm : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=layernormnorm]
#   %clone_13 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_13]
#   %arg8_1 : Tensor "f16[768, 3072][3072, 1]spyre:0" = PlaceHolder[target=arg8_1]
#   %mm_2 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=mm_2]
#   %arg7_1 : Tensor "f16[3072][1]spyre:0" = PlaceHolder[target=arg7_1]
#   %add_9 : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=add_9]
#   %gelu : Tensor "f16[2048, 3072][3072, 1]spyre:0" = PlaceHolder[target=gelu]
#   %arg10_1 : Tensor "f16[3072, 768][768, 1]spyre:0" = PlaceHolder[target=arg10_1]
#   %mm_3 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=mm_3]
#   %clone_14 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=clone_14]
#   %arg9_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg9_1]
#   %add_10 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_10]
#   %add_11 : Tensor "f16[2048, 768][768, 1]spyre:0" = PlaceHolder[target=add_11]
#   %exx2_1 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=exx2_1]
#   %layernormscale_1 : Tensor "f16[2048, 1][1, 2048]spyre:0" = PlaceHolder[target=layernormscale_1]
#   %arg11_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg11_1]
#   %arg12_1 : Tensor "f16[768][1]spyre:0" = PlaceHolder[target=arg12_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%add, [768, 768, 768], -1), kwargs = {})
#   %view : Tensor "f16[2048, 12, 64][2304, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [-1, 12, 64]), kwargs = {})
#   %view_2 : Tensor "f16[2048, 12, 64][2304, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [-1, 12, 64]), kwargs = {})
#   %view_3 : Tensor "f16[2048, 12, 64][2304, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [-1, 12, 64]), kwargs = {})
#   %view_4 : Tensor "f16[4, 512, 12, 64][1179648, 2304, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view, [4, 512, 12, 64]), kwargs = {})
#   %permute : Tensor "f16[4, 12, 512, 64][1179648, 64, 2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 2, 1, 3]), kwargs = {})
#   %view_5 : Tensor "f16[4, 512, 12, 64][1179648, 2304, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_2, [4, 512, 12, 64]), kwargs = {})
#   %permute_1 : Tensor "f16[4, 12, 512, 64][1179648, 64, 2304, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %view_6 : Tensor "f16[4, 512, 12, 64][1179648, 2304, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_3, [4, 512, 12, 64]), kwargs = {})
#   %permute_2 : Tensor "f16[4, 12, 512, 64][1179648, 64, 2304, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_7 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%clone,), kwargs = {})
#   %full_default_8 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 12, 512, 64], 0), kwargs = {dtype: torch.float16, layout: torch.strided, device: spyre:0, pin_memory: False})
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%full_default_8,), kwargs = {})
#   %full_default_9 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 12, 512, 64], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: spyre:0, pin_memory: False})
#   %amax : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%full_default_9, [-1]), kwargs = {})
#   %clone_9 : [num_users=2] = call_function[target=torch.ops.aten.clone](args = (%amax,), kwargs = {})
#   %full_default_10 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 12, 512, 64], 0), kwargs = {dtype: torch.float16, layout: torch.strided, device: spyre:0, pin_memory: False})
#   %amax_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%full_default_10, [-1]), kwargs = {})
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%amax_1,), kwargs = {})
#   %mul : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone, 0.3535533905932738), kwargs = {})
#   %slice_1 : Tensor "f16[4, 12, 256, 64][1179648, 64, 2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_1, 2, 0, 256), kwargs = {})
#   %slice_2 : Tensor "f16[4, 12, 256, 64][1179648, 64, 2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 2, 0, 256), kwargs = {})
#   %mul_1 : Tensor "f16[4, 12, 256, 64][196608, 64, 768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_1, 0.3535533905932738), kwargs = {})
#   %clone_1 : Tensor "f16[4, 12, 256, 64][196608, 16384, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_3 : Tensor "f16[4, 12, 64, 256][196608, 64, 1, 768]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_2 : Tensor "f16[4, 12, 64, 256][196608, 16384, 256, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %batched_matmul_default_3 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.batched_matmul.default](args = (%expand, %expand_1), kwargs = {})
#   %slice_3 : Tensor "f16[4, 1, 1, 256][512, 512, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%spyre_convert, 3, 0, 256), kwargs = {})
#   %add_1 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%batched_matmul_default_3, %slice_3), kwargs = {})
#   %amax_2 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_1, [-1]), kwargs = {})
#   %maximum : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=4] = call_function[target=torch.ops.aten.maximum.default](args = (%clone_9, %amax_2), kwargs = {})
#   %unsqueeze : Tensor "f16[4, 12, 512, 1][6144, 512, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%maximum, -1), kwargs = {})
#   %sub : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %unsqueeze), kwargs = {})
#   %exp : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sub_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_9, %maximum), kwargs = {})
#   %exp_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %mul_2 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone_10, %exp_1), kwargs = {})
#   %sum_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1]), kwargs = {})
#   %add_2 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %sum_1), kwargs = {})
#   %batched_matmul_default_2 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.batched_matmul.default](args = (%expand_2, %expand_3), kwargs = {})
#   %unsqueeze_1 : Tensor "f16[4, 12, 512, 1][6144, 512, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp_1, -1), kwargs = {})
#   %mul_3 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone_8, %unsqueeze_1), kwargs = {})
#   %add_3 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %batched_matmul_default_2), kwargs = {})
#   %slice_4 : Tensor "f16[4, 12, 256, 64][1179648, 64, 2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_1, 2, 256, 512), kwargs = {})
#   %slice_5 : Tensor "f16[4, 12, 256, 64][1179648, 64, 2304, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 2, 256, 512), kwargs = {})
#   %mul_4 : Tensor "f16[4, 12, 256, 64][196608, 64, 768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_4, 0.3535533905932738), kwargs = {})
#   %clone_3 : Tensor "f16[4, 12, 256, 64][196608, 16384, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_5,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_4 : Tensor "f16[4, 12, 64, 256][196608, 64, 1, 768]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_4, [0, 1, 3, 2]), kwargs = {})
#   %clone_4 : Tensor "f16[4, 12, 64, 256][196608, 16384, 256, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   %batched_matmul_default_1 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.batched_matmul.default](args = (%expand_4, %expand_5), kwargs = {})
#   %slice_6 : Tensor "f16[4, 1, 1, 256][512, 512, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%spyre_convert, 3, 256, 512), kwargs = {})
#   %add_4 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%batched_matmul_default_1, %slice_6), kwargs = {})
#   %amax_3 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_4, [-1]), kwargs = {})
#   %maximum_1 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.maximum.default](args = (%maximum, %amax_3), kwargs = {})
#   %unsqueeze_2 : Tensor "f16[4, 12, 512, 1][6144, 512, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%maximum_1, -1), kwargs = {})
#   %sub_2 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %unsqueeze_2), kwargs = {})
#   %exp_2 : Tensor "f16[4, 12, 512, 256][1572864, 131072, 256, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sub_3 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%maximum, %maximum_1), kwargs = {})
#   %exp_3 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_3,), kwargs = {})
#   %mul_5 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %exp_3), kwargs = {})
#   %sum_2 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [-1]), kwargs = {})
#   %add_5 : Tensor "f16[4, 12, 512][6144, 512, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %sum_2), kwargs = {})
#   %batched_matmul_default : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.batched_matmul.default](args = (%expand_6, %expand_7), kwargs = {})
#   %unsqueeze_3 : Tensor "f16[4, 12, 512, 1][6144, 512, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp_3, -1), kwargs = {})
#   %mul_6 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %unsqueeze_3), kwargs = {})
#   %add_6 : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %batched_matmul_default), kwargs = {})
#   %unsqueeze_4 : Tensor "f16[4, 12, 512, 1][6144, 512, 1, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_5, -1), kwargs = {})
#   %div : Tensor "f16[4, 12, 512, 64][393216, 32768, 64, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_6, %unsqueeze_4), kwargs = {})
#   %clone_11 : [num_users=0] = call_function[target=torch.ops.aten.clone](args = (%div,), kwargs = {})
#   %permute_5 : Tensor "f16[4, 512, 12, 64][393216, 64, 32768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%div, [0, 2, 1, 3]), kwargs = {})
#   %clone_5 : Tensor "f16[4, 512, 12, 64][393216, 768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_19 : Tensor "f16[2048, 12, 64][768, 64, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_5, [2048, 12, 64]), kwargs = {})
#   %view_20 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_19, [2048, 768]), kwargs = {})
#   %mm_1 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_20, %arg4_1), kwargs = {})
#   %clone_12 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_1,), kwargs = {})
#   %add_7 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_12, %arg3_1), kwargs = {})
#   %add_8 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %arg2_1), kwargs = {})
#   %exx2 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_8, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2, 1e-05), kwargs = {})
#   %layernormnorm : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_8, %exx2, %layernormscale, %arg5_1, %arg6_1), kwargs = {})
#   %clone_13 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%layernormnorm,), kwargs = {})
#   %mm_2 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%clone_13, %arg8_1), kwargs = {})
#   %add_9 : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_2, %arg7_1), kwargs = {})
#   %gelu : Tensor "f16[2048, 3072][3072, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.gelu.default](args = (%add_9,), kwargs = {})
#   %mm_3 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%gelu, %arg10_1), kwargs = {})
#   %clone_14 : [num_users=1] = call_function[target=torch.ops.aten.clone](args = (%mm_3,), kwargs = {})
#   %add_10 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone_14, %arg9_1), kwargs = {})
#   %add_11 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %layernormnorm), kwargs = {})
#   %exx2_1 : Tensor "f16[2048][1]spyre:0"[num_users=2] = call_function[target=torch.ops.spyre.exx2.default](args = (%add_11, 0.0013020833333333333, False), kwargs = {})
#   %layernormscale_1 : Tensor "f16[2048][1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormscale.default](args = (%exx2_1, 1e-05), kwargs = {})
#   %layernormnorm_1 : Tensor "f16[2048, 768][768, 1]spyre:0"[num_users=1] = call_function[target=torch.ops.spyre.layernormnorm.default](args = (%add_11, %exx2_1, %layernormscale_1, %arg11_1, %arg12_1), kwargs = {})
#   return %clone,%full_default_8,%full_default_9,%amax,%full_default_10,%amax_1,%clone_7,%expand_4,%mul_1,%expand_3,%expand_1,%batched_matmul_default_3,%add_1,%amax_2,%clone_9,%maximum,%sub,%expand_2,%sub_1,%exp_1,%clone_10,%mul_2,%sum_1,%add_2,%batched_matmul_default_2,%clone_8,%mul_3,%add_3,%mul_4,%expand_7,%expand_5,%batched_matmul_default_1,%add_4,%amax_3,%maximum_1,%sub_2,%expand_6,%sub_3,%exp_3,%mul_5,%sum_2,%add_5,%batched_matmul_default,%mul_6,%add_6,%div,%clone_11,%clone_5,%mm_1,%clone_12,%add_7,%add_8,%exx2,%layernormscale,%layernormnorm,%clone_13,%mm_2,%add_9,%gelu,%mm_3,%clone_14,%add_10,%add_11,%exx2_1,%layernormscale_1,%layernormnorm_1
sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_gelu_layer_norm_mm_split_with_sizes_transpose_view_2 = async_compile.sdsc('sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_gelu_layer_norm_mm_split_with_sizes_transpose_view_2',
    [
        OpSpec(
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('4'), 1), sympify('c1'): (sympify('12'), 1), sympify('c2'): (sympify('512'), 32), sympify('c3'): (sympify('64'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('0'), sympify('c1'): sympify('0'), sympify('c2'): sympify('Mod(core_id, 32)'), sympify('c3'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=315586294371028648, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone', 'permute', 'split_with_sizes', 'view', 'view_4', 'buf30'), fused_from=(DebugHandle(id=7571308961724228552, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone',), fused_from=(), transform_history=()), DebugHandle(id=125204812885412945, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=222, start_col=0, end_line=None, end_col=None), aten_op='aten.transpose.int', ir_chain=('permute',), fused_from=(), transform_history=()), DebugHandle(id=532880329144244142, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=281, start_col=0, end_line=None, end_col=None), aten_op='aten.split_with_sizes.default', ir_chain=('split_with_sizes',), fused_from=(), transform_history=()), DebugHandle(id=7218287814897079719, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py', start_line=524, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view',), fused_from=(), transform_history=()), DebugHandle(id=7874724732378024118, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=222, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_4',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 4, 512, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c0'), sympify('c2'), sympify('Mod(c3, 64)')],
                    allocation={'hbm': 0},
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
            debug_handle=DebugHandle(id=5589231574156337007, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('full_default_8', 'buf32'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 1},
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
            debug_handle=DebugHandle(id=3465329165565726505, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('full_default_9', 'buf34'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=2, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 2},
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
            debug_handle=DebugHandle(id=7113420240685948076, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('amax', 'buf35'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=3343506774852630117, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('full_default_10', 'buf37'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 1},
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
            debug_handle=DebugHandle(id=2720842137635194629, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('amax_1', 'buf38'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=1975727068591486608, source=None, aten_op=None, ir_chain=('clone_7', 'buf95'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=7712444685998581413, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul', 'buf39'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 512, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 98304},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 3},
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
            debug_handle=DebugHandle(id=4957371889309786807, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_1', 'permute_1', 'slice_1', 'split_with_sizes', 'view_2', 'view_5', 'buf40'), fused_from=(DebugHandle(id=2751036479011035631, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_1',), fused_from=(), transform_history=()), DebugHandle(id=2032097473323888407, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=223, start_col=0, end_line=None, end_col=None), aten_op='aten.transpose.int', ir_chain=('permute_1',), fused_from=(), transform_history=()), DebugHandle(id=3620623304018603640, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('slice_1',), fused_from=(), transform_history=()), DebugHandle(id=532880329144244142, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=281, start_col=0, end_line=None, end_col=None), aten_op='aten.split_with_sizes.default', ir_chain=('split_with_sizes',), fused_from=(), transform_history=()), DebugHandle(id=2145964905626220726, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py', start_line=527, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_2',), fused_from=(), transform_history=()), DebugHandle(id=1948957152519586813, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=223, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_5',), fused_from=(), transform_history=())), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 4, 512, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c2 + 12'), sympify('c0'), sympify('c1'), sympify('Mod(c3, 64)')],
                    allocation={'hbm': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 3},
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
            debug_handle=DebugHandle(id=8017950633470987, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_1', 'permute_2', 'slice_2', 'split_with_sizes', 'view_3', 'view_6', 'buf41'), fused_from=(DebugHandle(id=2392487244006008603, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_1',), fused_from=(), transform_history=()), DebugHandle(id=5049211453498212534, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=224, start_col=0, end_line=None, end_col=None), aten_op='aten.transpose.int', ir_chain=('permute_2',), fused_from=(), transform_history=()), DebugHandle(id=6673039008075634001, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('slice_2',), fused_from=(), transform_history=()), DebugHandle(id=532880329144244142, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=281, start_col=0, end_line=None, end_col=None), aten_op='aten.split_with_sizes.default', ir_chain=('split_with_sizes',), fused_from=(), transform_history=()), DebugHandle(id=4765485281742357021, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py', start_line=529, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_3',), fused_from=(), transform_history=()), DebugHandle(id=1905914418511895031, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=224, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_6',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 4, 512, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1 + 24'), sympify('c0'), sympify('c2'), sympify('Mod(c3, 64)')],
                    allocation={'hbm': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 256, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 0},
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
            debug_handle=DebugHandle(id=5376126390977581476, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_2', 'permute_3', 'buf42'), fused_from=(DebugHandle(id=6508826509545368739, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_2',), fused_from=(), transform_history=()), DebugHandle(id=4154052246658386091, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('permute_3',), fused_from=(), transform_history=())), transform_history=()),
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
            debug_handle=DebugHandle(id=6634581055461874710, source=None, aten_op=None, ir_chain=('batched_matmul_default_3', 'buf43'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=3298759783245543083, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_1', 'slice_3', 'buf44'), fused_from=(DebugHandle(id=7435755766867862223, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_1',), fused_from=(), transform_history=()), DebugHandle(id=973483418626195352, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('slice_3',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 983040},
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 8, 4, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm': 4},
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
            debug_handle=DebugHandle(id=1213339828585352403, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('amax_2', 'buf45'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=7145719651708227281, source=None, aten_op=None, ir_chain=('clone_9', 'buf97'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=2485249226012703874, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('maximum', 'buf46'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=6717936828009262041, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub', 'unsqueeze', 'buf47'), fused_from=(DebugHandle(id=2911669064628575428, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub',), fused_from=(), transform_history=()), DebugHandle(id=8370234104118275517, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('unsqueeze',), fused_from=(), transform_history=())), transform_history=()),
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
            debug_handle=DebugHandle(id=4555432180951697345, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('exp', 'buf48'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=3115605478126334806, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub_1', 'buf49'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=5853675751765404199, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('exp_1', 'buf50'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=461322060859519879, source=None, aten_op=None, ir_chain=('clone_10', 'buf98'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=4912435274835221028, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_2', 'buf51'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=7872045926964743441, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sum_1', 'buf52'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=3284291045510043883, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_2', 'buf53'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=1429278388252351961, source=None, aten_op=None, ir_chain=('batched_matmul_default_2', 'buf54'), fused_from=(), transform_history=()),
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
                    allocation={'hbm_pool': 0},
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
            debug_handle=DebugHandle(id=6694739163542214016, source=None, aten_op=None, ir_chain=('clone_8', 'buf96'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=6883532607992805350, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_3', 'unsqueeze_1', 'buf55'), fused_from=(DebugHandle(id=828073498528793885, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_3',), fused_from=(), transform_history=()), DebugHandle(id=526770419541132762, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('unsqueeze_1',), fused_from=(), transform_history=())), transform_history=()),
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
            debug_handle=DebugHandle(id=8597095831564846728, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_3', 'buf56'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=3313425837713110577, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_4', 'permute_1', 'slice_4', 'split_with_sizes', 'view_2', 'view_5', 'buf57'), fused_from=(DebugHandle(id=751543347369202643, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_4',), fused_from=(), transform_history=()), DebugHandle(id=2032097473323888407, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=223, start_col=0, end_line=None, end_col=None), aten_op='aten.transpose.int', ir_chain=('permute_1',), fused_from=(), transform_history=()), DebugHandle(id=5126013339418157025, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('slice_4',), fused_from=(), transform_history=()), DebugHandle(id=532880329144244142, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=281, start_col=0, end_line=None, end_col=None), aten_op='aten.split_with_sizes.default', ir_chain=('split_with_sizes',), fused_from=(), transform_history=()), DebugHandle(id=2145964905626220726, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py', start_line=527, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_2',), fused_from=(), transform_history=()), DebugHandle(id=1948957152519586813, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=223, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_5',), fused_from=(), transform_history=())), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 4, 512, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c2 + 12'), sympify('c0'), sympify('c1 + 256'), sympify('Mod(c3, 64)')],
                    allocation={'hbm': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=3, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 1, 1, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('0'), sympify('0'), sympify('0')],
                    allocation={'hbm': 3},
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
            debug_handle=DebugHandle(id=8024721129874353707, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_3', 'permute_2', 'slice_5', 'split_with_sizes', 'view_3', 'view_6', 'buf58'), fused_from=(DebugHandle(id=89987922385213495, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_3',), fused_from=(), transform_history=()), DebugHandle(id=5049211453498212534, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=224, start_col=0, end_line=None, end_col=None), aten_op='aten.transpose.int', ir_chain=('permute_2',), fused_from=(), transform_history=()), DebugHandle(id=8962446624378866866, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('slice_5',), fused_from=(), transform_history=()), DebugHandle(id=532880329144244142, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=281, start_col=0, end_line=None, end_col=None), aten_op='aten.split_with_sizes.default', ir_chain=('split_with_sizes',), fused_from=(), transform_history=()), DebugHandle(id=4765485281742357021, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py', start_line=529, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_3',), fused_from=(), transform_history=()), DebugHandle(id=1905914418511895031, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=224, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_6',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=0, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 36, 4, 512, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1 + 24'), sympify('c0'), sympify('c2 + 256'), sympify('Mod(c3, 64)')],
                    allocation={'hbm': 0},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 256, 4, 64],
                    device_coordinates=[sympify('floor(c3/64)'), sympify('c1'), sympify('c2'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm_pool': 0},
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
            debug_handle=DebugHandle(id=5575552407326840376, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_4', 'permute_4', 'buf59'), fused_from=(DebugHandle(id=4189289827818166323, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_4',), fused_from=(), transform_history=()), DebugHandle(id=5351591964731586770, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('permute_4',), fused_from=(), transform_history=())), transform_history=()),
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
            debug_handle=DebugHandle(id=3261972273599326932, source=None, aten_op=None, ir_chain=('batched_matmul_default_1', 'buf60'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=431642243711858708, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_4', 'slice_6', 'buf61'), fused_from=(DebugHandle(id=4503923646736356753, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_4',), fused_from=(), transform_history=()), DebugHandle(id=8873426684525673470, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('slice_6',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 512, 4, 4, 64],
                    device_coordinates=[sympify('c1'), sympify('c2'), sympify('floor(c3/64)'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'lx': 294912},
                ),
                TensorArg(
                    is_input=True, arg_index=4, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 1, 8, 4, 64],
                    device_coordinates=[sympify('0'), sympify('0'), sympify('floor(c3/64) + 4'), sympify('c0'), sympify('Mod(c3, 64)')],
                    allocation={'hbm': 4},
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
            debug_handle=DebugHandle(id=8321529173592841429, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('amax_3', 'buf62'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=4988656519923863936, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('maximum_1', 'buf63'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=2765840124096549923, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub_2', 'unsqueeze_2', 'buf64'), fused_from=(DebugHandle(id=9104609026663197200, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub_2',), fused_from=(), transform_history=()), DebugHandle(id=2351686318514387749, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('unsqueeze_2',), fused_from=(), transform_history=())), transform_history=()),
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
            debug_handle=DebugHandle(id=7826523740286116770, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('exp_2', 'buf65'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=8836568569223079796, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sub_3', 'buf66'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=4276561390638059359, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('exp_3', 'buf67'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=7283124087810712255, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_5', 'buf68'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=8497894442754604558, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('sum_2', 'buf69'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=180544858716056834, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_5', 'buf70'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=775747309736946831, source=None, aten_op=None, ir_chain=('batched_matmul_default', 'buf71'), fused_from=(), transform_history=()),
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
                    allocation={'hbm_pool': 0},
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
            debug_handle=DebugHandle(id=3765090165624257127, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_6', 'unsqueeze_3', 'buf72'), fused_from=(DebugHandle(id=5424531665241960975, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('mul_6',), fused_from=(), transform_history=()), DebugHandle(id=5217120533114696509, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('unsqueeze_3',), fused_from=(), transform_history=())), transform_history=()),
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
            debug_handle=DebugHandle(id=953848613632651238, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('add_6', 'buf73'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=9033234294201021863, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('div', 'unsqueeze_4', 'buf74'), fused_from=(DebugHandle(id=7842793335781778310, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('div',), fused_from=(), transform_history=()), DebugHandle(id=6465702649226844244, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('unsqueeze_4',), fused_from=(), transform_history=())), transform_history=()),
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
            debug_handle=DebugHandle(id=436654012841005448, source=None, aten_op=None, ir_chain=('clone_11', 'buf99'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=3212874122893092905, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_5', 'permute_5', 'buf75'), fused_from=(DebugHandle(id=3879188736327606917, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('clone_5',), fused_from=(), transform_history=()), DebugHandle(id=5211395469549601789, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=225, start_col=0, end_line=None, end_col=None), aten_op='aten._scaled_dot_product_fused_attention_overrideable.default', ir_chain=('permute_5',), fused_from=(), transform_history=())), transform_history=()),
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
                    allocation={'hbm_pool': 3145728},
                ),
            ]
        ),
        OpSpec(
            op='batchmatmul',
            is_reduction=True,
            iteration_space={sympify('c0'): (sympify('512'), 2), sympify('z0'): (sympify('4'), 4), sympify('c1'): (sympify('768'), 4), sympify('c2'): (sympify('768'), 1)},
            op_info={},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(core_id, 2)'), sympify('z0'): sympify('Mod(floor(core_id/2), 4)'), sympify('c1'): sympify('Mod(floor(core_id/8), 4)'), sympify('c2'): sympify('0')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=6306010206062899729, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_1', 'view_19', 'view_20', 'buf76'), fused_from=(DebugHandle(id=4541032506299284014, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_1',), fused_from=(), transform_history=()), DebugHandle(id=571213348475758197, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=234, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_19',), fused_from=(), transform_history=()), DebugHandle(id=8345221725719227009, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/v1/attention/backends/spyre_encoder_attn.py', start_line=234, start_col=0, end_line=None, end_col=None), aten_op='aten.view.default', ir_chain=('view_20',), fused_from=(), transform_history=())), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[512, 12, 4, 64],
                    device_coordinates=[sympify('c0'), sympify('floor(c2/64)'), sympify('z0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 3145728},
                ),
                TensorArg(
                    is_input=True, arg_index=5, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 768, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 5},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 4, 512, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('z0'), sympify('c0'), sympify('Mod(c1, 64)')],
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
            debug_handle=DebugHandle(id=2984650322373645363, source=None, aten_op=None, ir_chain=('clone_12', 'buf100'), fused_from=(), transform_history=()),
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
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=7875824277956007770, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_7', 'buf77'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
                TensorArg(
                    is_input=True, arg_index=6, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 6},
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
            debug_handle=DebugHandle(id=645107518127647076, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_8', 'buf78'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 0},
                ),
                TensorArg(
                    is_input=True, arg_index=7, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 7},
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
            debug_handle=DebugHandle(id=4998745714873028680, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2', 'buf79'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=7110514853607156858, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale', 'buf80'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
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
            debug_handle=DebugHandle(id=6223304343783933478, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=308, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm', 'buf81'), fused_from=(), transform_history=()),
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
                    is_input=True, arg_index=8, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 8},
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
            op='identity',
            is_reduction=False,
            iteration_space={sympify('c0'): (sympify('2048'), 4), sympify('c1'): (sympify('768'), 1)},
            op_info={'lx_relayout_certified': True},
            core_id_to_work_slice={sympify('c0'): sympify('Mod(floor(Mod(core_id, 4)), 4)')},
            symbolic_dim_bounds={},
            debug_handle=DebugHandle(id=4344018005381539540, source=None, aten_op=None, ir_chain=('clone_13', 'buf101'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=3698607255709026186, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_2', 'buf82'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'lx': 114688},
                ),
                TensorArg(
                    is_input=True, arg_index=10, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 768, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 10},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 6291456},
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
            debug_handle=DebugHandle(id=31497657460874637, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_9', 'buf83'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 6291456},
                ),
                TensorArg(
                    is_input=True, arg_index=11, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 48, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 11},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
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
            debug_handle=DebugHandle(id=3077935569395308817, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/layers/activation.py', start_line=372, start_col=0, end_line=None, end_col=None), aten_op='aten.gelu.default', ir_chain=('gelu', 'buf84'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 98304},
                ),
                TensorArg(
                    is_input=False, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm_pool': 6291456},
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
            debug_handle=DebugHandle(id=6353763844960375536, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=61, start_col=0, end_line=None, end_col=None), aten_op='aten.mm.default', ir_chain=('mm_3', 'buf85'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[48, 2048, 64],
                    device_coordinates=[sympify('floor(c2/64)'), sympify('c0'), sympify('Mod(c2, 64)')],
                    allocation={'hbm_pool': 6291456},
                ),
                TensorArg(
                    is_input=True, arg_index=12, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 3072, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c2'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 12},
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
            debug_handle=DebugHandle(id=6185606809446449402, source=None, aten_op=None, ir_chain=('clone_14', 'buf102'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=8313995429730401144, source=SourceLoc(file='/home/senuser/spyre-inference/.claude/worktrees/s3-dev2-b3-launch-sites/spyre_inference/custom_ops/linear.py', start_line=63, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_10', 'buf86'), fused_from=(), transform_history=()),
            args=[
                TensorArg(
                    is_input=True, arg_index=-1, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'lx': 196608},
                ),
                TensorArg(
                    is_input=True, arg_index=13, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 13},
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
            debug_handle=DebugHandle(id=5600839269551904567, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.add.Tensor', ir_chain=('add_11', 'buf87'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=3789316678824338925, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('exx2_1', 'buf88'), fused_from=(), transform_history=()),
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
            debug_handle=DebugHandle(id=1650681472953463743, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormscale_1', 'buf89'), fused_from=(), transform_history=(ProvenanceTransform(kind='rewrite', pass_name='split_multi_ops', reason='rewrite original buffer body'),)),
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
            debug_handle=DebugHandle(id=1107771682839908902, source=SourceLoc(file='/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/model_executor/models/bert.py', start_line=362, start_col=0, end_line=None, end_col=None), aten_op='aten.layer_norm.default', ir_chain=('layernormnorm_1', 'buf90'), fused_from=(), transform_history=()),
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
                    is_input=True, arg_index=14, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 14},
                ),
                TensorArg(
                    is_input=True, arg_index=15, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[1, 12, 64],
                    device_coordinates=[sympify('0'), sympify('floor(c1/64)'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 15},
                ),
                TensorArg(
                    is_input=False, arg_index=16, device_dtype=DataFormats.SEN169_FP16,
                    device_size=[12, 2048, 64],
                    device_coordinates=[sympify('floor(c1/64)'), sympify('c0'), sympify('Mod(c1, 64)')],
                    allocation={'hbm': 16},
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1 = args
        args.clear()
        buf5 = spyre_constant_tensor(0.0, torch.device("cpu"), torch.float16)
        buf31 = spyre_constant_tensor(0.0, torch.device("spyre:0"), torch.float16)
        buf33 = spyre_constant_tensor(-inf, torch.device("spyre:0"), torch.float16)
        buf91 = spyre_constant_tensor(0.3535533905932738, torch.device("spyre:0"), torch.float16)
        assert_size_stride(arg2_1, (2048, 768), (768, 1), 'input')
        assert_size_stride(arg1_1, (768, 2304), (2304, 1), 'input')
        assert_size_stride(arg0_1, (2304, ), (1, ), 'input')
        buf1 = spyre_empty_with_layout((2048, 2304), (2304, 1), torch.float16, SpyreTensorLayout(device_size=[36, 2048, 64], stride_map =[64, 2304, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused_add_mm_0.run(arg2_1, arg1_1, arg0_1, buf1)
        del arg0_1
        del arg1_1
        buf6 = empty_strided_cpu((), (), torch.float16)
        buf7 = empty_strided_cpu((1, 1, 1, 512), (512, 512, 512, 1), torch.float16)
        buf11 = empty_strided_cpu((), (), torch.float16)
        buf12 = empty_strided_cpu((1, 1, 1, 512), (512, 512, 512, 1), torch.float16)
        buf16 = empty_strided_cpu((), (), torch.float16)
        buf17 = empty_strided_cpu((1, 1, 1, 512), (512, 512, 512, 1), torch.float16)
        buf5 = spyre_empty_with_layout((), (), torch.float16, SpyreTensorLayout(device_size=[1, 64], stride_map =[-1, -1], device_dtype=DataFormats.SEN169_FP16))
        buf21 = buf5; del buf5  # reuse
        buf22 = empty_strided_cpu((1, 1, 1, 512), (512, 512, 512, 1), torch.float16)
        buf23 = spyre_empty_with_layout((4, 1, 1, 512), (512, 512, 512, 1), torch.float16, SpyreTensorLayout(device_size=[1, 1, 8, 4, 64], stride_map =[-1, -1, 64, 512, 1], device_dtype=DataFormats.SEN169_FP16))
        buf24 = spyre_empty_with_layout((4, 1, 1, 512), (512, 512, 512, 1), torch.float16, SpyreTensorLayout(device_size=[1, 1, 8, 4, 64], stride_map =[-1, -1, 64, 512, 1], device_dtype=DataFormats.SEN169_FP16))
        buf25 = spyre_empty_with_layout((4, 1, 1, 512), (512, 512, 512, 1), torch.float16, SpyreTensorLayout(device_size=[1, 1, 8, 4, 64], stride_map =[-1, -1, 64, 512, 1], device_dtype=DataFormats.SEN169_FP16))
        buf26 = spyre_empty_with_layout((4, 1, 1, 512), (512, 512, 512, 1), torch.float16, SpyreTensorLayout(device_size=[1, 1, 8, 4, 64], stride_map =[-1, -1, 64, 512, 1], device_dtype=DataFormats.SEN169_FP16))
        buf27 = spyre_empty_with_layout((4, 1, 1, 512), (512, 512, 512, 1), torch.float16, SpyreTensorLayout(device_size=[1, 1, 8, 4, 64], stride_map =[-1, -1, 64, 512, 1], device_dtype=DataFormats.SEN169_FP16))
        cpp_fused_cat_fill_lift_fresh_1(buf21, buf6, buf7, buf11, buf12, buf16, buf17, buf22, buf23)
        del buf11
        del buf12
        del buf16
        del buf17
        del buf21
        del buf22
        del buf6
        del buf7
        # Topologically Sorted Source Nodes: [spyre_convert], Original ATen: [vllm.spyre_convert]
        buf28 = torch.ops.vllm.spyre_convert.default(buf23, device(type='spyre', index=0))
        del buf23
        buf29 = buf28
        assert_size_stride(buf29, (4, 1, 1, 512), (512, 512, 512, 1), 'torch.ops.vllm.spyre_convert.default')
        assert_alignment(buf29, 16, 'torch.ops.vllm.spyre_convert.default')
        del buf28
        assert_size_stride(arg4_1, (768, 768), (768, 1), 'input')
        assert_size_stride(arg3_1, (768, ), (1, ), 'input')
        assert_size_stride(arg5_1, (768, ), (1, ), 'input')
        assert_size_stride(arg6_1, (768, ), (1, ), 'input')
        assert_size_stride(arg8_1, (768, 3072), (3072, 1), 'input')
        assert_size_stride(arg7_1, (3072, ), (1, ), 'input')
        assert_size_stride(arg10_1, (3072, 768), (768, 1), 'input')
        assert_size_stride(arg9_1, (768, ), (1, ), 'input')
        assert_size_stride(arg11_1, (768, ), (1, ), 'input')
        assert_size_stride(arg12_1, (768, ), (1, ), 'input')
        buf90 = spyre_empty_with_layout((2048, 768), (768, 1), torch.float16, SpyreTensorLayout(device_size=[12, 2048, 64], stride_map =[64, 768, 1], device_dtype=DataFormats.SEN169_FP16))
        sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_gelu_layer_norm_mm_split_with_sizes_transpose_view_2.run(buf1, buf31, buf33, buf91, buf29, arg4_1, arg3_1, arg2_1, arg5_1, arg6_1, arg8_1, arg7_1, arg10_1, arg9_1, arg11_1, arg12_1, buf90)
        del arg10_1
        del arg11_1
        del arg12_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del arg8_1
        del arg9_1
        return (buf90, buf29, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns
