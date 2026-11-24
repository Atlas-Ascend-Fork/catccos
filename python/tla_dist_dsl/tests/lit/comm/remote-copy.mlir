// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: %tla_dist_compile %s -o - | %filecheck %s --implicit-check-not=tla_dist.

!gm = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<32,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!ub = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>

module {
  tla.func @remote_copy_routes(%dst: !gm, %src: !gm, %peer: i32) {
    %ub_ptr = tla.alloc_ptr{size_bytes = 512} -> !tla.ptr<f32, ub, 256>
    "tla.vector"() ({
      %shape = tla.make_shape -> !tla.shape<8,16>
      %stride = tla.make_stride -> !tla.stride<16,1>
      %layout = tla.make_layout %shape, %stride : !tla.shape<8,16>, !tla.stride<16,1> -> !tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>
      %coord = tla.make_coord -> !tla.coord<0,0>
      %scratch = tla.make_tensor %ub_ptr, %layout, %coord : !tla.ptr<f32, ub, 256>, !tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0> -> !ub
      tla_dist.remote_copy %dst, %src, %peer, %peer, %scratch {event_id = 0 : i32, mode = "get", transport = "mte"} : !gm, !gm, i32, i32, !ub
      tla_dist.remote_copy %dst, %src, %peer, %peer, %scratch {event_id = 2 : i32, mode = "put", transport = "mte", atomic_add} : !gm, !gm, i32, i32, !ub
      // GM <-> UB routes require their corresponding explicit IR mode.
      tla_dist.remote_copy %scratch, %src, %peer, %peer {mode = "get", transport = "mte"} : !ub, !gm, i32, i32
      tla_dist.remote_copy %dst, %scratch, %peer, %peer {mode = "put", transport = "mte"} : !gm, !ub, i32, i32
      // Communication reserves IDs 0 and 2 before automatic flag allocation.
      %auto = tla.flag "local_auto" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte3>} -> !tla.flag
      tla.set_flag %auto : !tla.flag
      tla.wait_flag %auto : !tla.flag
    }) : () -> ()
    tla.return
  }
}

// CHECK-DAG: func.func private @shmem_remote_get_gm_row_major_to_gm_row_major_float({{.*}}) attributes {{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-DAG: func.func private @shmem_remote_put_gm_row_major_to_gm_row_major_float({{.*}}) attributes {{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-LABEL: func.func @remote_copy_routes(
// CHECK: call @shmem_remote_get_gm_row_major_to_gm_row_major_float(
// CHECK-SAME: : ({{.*}}>, {{((i64, ){36})}}i32, i32) -> ()
// CHECK: hivm.hir.set_ctrl true at ctrl[6]
// CHECK: call @shmem_remote_put_gm_row_major_to_gm_row_major_float(
// CHECK: hivm.hir.set_ctrl false at ctrl[6]
// CHECK: call @shmem_remote_get_gm_row_major_to_ub_row_major_float(
// CHECK-SAME: : ({{.*}}>, {{((i64, ){24})}}i32) -> ()
// CHECK: call @shmem_remote_put_ub_row_major_to_gm_row_major_float(
// CHECK-SAME: : ({{.*}}>, {{((i64, ){24})}}i32) -> ()
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
