// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: %tla_dist_compile %s -o - | %filecheck %s --implicit-check-not=tla_dist.

!gm = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<ui64, gm, 8>>
!ub = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<ui64, ub, 32>>

module {
  tla.func @uint64_mte(%dst: !gm, %src: !gm, %peer: i32) {
    %ub_ptr = tla.alloc_ptr{size_bytes = 1024} -> !tla.ptr<ui64, ub, 32>
    "tla.vector"() ({
      %scratch = tla.make_tensor_like %ub_ptr like %src layoutTag(<RowMajor>) : !tla.ptr<ui64, ub, 32>, !gm -> !ub
      tla_dist.remote_copy %dst, %src, %peer, %peer, %scratch {event_id = 0 : i32, mode = "get", transport = "mte"} : !gm, !gm, i32, i32, !ub
      tla_dist.remote_copy %dst, %src, %peer, %peer, %scratch {event_id = 1 : i32, mode = "put", transport = "mte"} : !gm, !gm, i32, i32, !ub
    }) : () -> ()
    tla.return
  }
}

// CHECK-DAG: func.func private @shmem_remote_get_gm_row_major_to_gm_row_major_uint64_t({{.*}}) attributes {{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @shmem_remote_put_gm_row_major_to_gm_row_major_uint64_t({{.*}}) attributes {{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-LABEL: func.func @uint64_mte(
// CHECK: call @shmem_remote_get_gm_row_major_to_gm_row_major_uint64_t(
// CHECK-SAME: : ({{.*}}>, {{((i64, ){36})}}i32, i32) -> ()
// CHECK: call @shmem_remote_put_gm_row_major_to_gm_row_major_uint64_t(
// CHECK-SAME: : ({{.*}}>, {{((i64, ){36})}}i32, i32) -> ()
