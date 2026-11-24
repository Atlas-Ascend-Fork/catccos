// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: %tla_dist_compile %s -o - | %filecheck %s --implicit-check-not=tla_dist.

!tile0 = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!tile1 = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>

module {
  tla.func @comm_lowering(%dst: !tile0, %src: !tile0, %peer: i32) {
    %ub_ptr = tla.alloc_ptr{size_bytes = 512} -> !tla.ptr<f32, ub, 256>
    "tla.vector"() ({
      %scratch = tla.make_tensor_like %ub_ptr like %src layoutTag(<RowMajor>) : !tla.ptr<f32, ub, 256>, !tile0 -> !tile1
      tla_dist.remote_copy %dst, %src, %peer, %peer, %scratch {event_id = 0 : i32, mode = "get", transport = "mte"} : !tile0, !tile0, i32, i32, !tile1
      tla_dist.remote_copy %dst, %src, %peer, %peer, %scratch {event_id = 1 : i32, mode = "get", transport = "mte", atomic_add} : !tile0, !tile0, i32, i32, !tile1
      tla_dist.barrier_all
    }) : () -> ()
    tla.return
  }
}

// The declarations prove that every CATCCOS operation selected its intended
// ACLSHMEM ABI and AIV core type.
// CHECK-DAG: func.func private @shmem_remote_get_gm_row_major_to_gm_row_major_float({{.*}}) attributes {{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @shmem_barrier_all_vec() attributes {{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>

// CHECK-LABEL: func.func @comm_lowering(
// CHECK: call @shmem_remote_get_gm_row_major_to_gm_row_major_float({{[^)]*}}%arg2, {{[^)]*}})
// CHECK-SAME: : ({{.*}}>, {{((i64, ){36})}}i32, i32) -> ()
// CHECK: hivm.hir.set_ctrl true at ctrl[6]
// CHECK: call @shmem_remote_get_gm_row_major_to_gm_row_major_float({{[^)]*}}%arg2, {{[^)]*}})
// CHECK-SAME: : ({{.*}}>, {{((i64, ){36})}}i32, i32) -> ()
// CHECK: hivm.hir.set_ctrl false at ctrl[6]
// CHECK: call @shmem_barrier_all_vec()
// CHECK-NOT: tla_dist.
