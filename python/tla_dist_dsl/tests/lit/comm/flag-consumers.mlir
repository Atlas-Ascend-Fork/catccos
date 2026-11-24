// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: %tla_dist_compile %s -o - | %filecheck %s --implicit-check-not=tla_dist. --implicit-check-not=tla.flag --implicit-check-not='hivm.hir.{{set|wait}}_flag'
// RUN: %tla_dist_compile %s --mlir-print-ir-before=tla-lower-flag-barrier-to-hivm -o /dev/null 2>&1 | %filecheck %s --check-prefix=RESERVE

!gm = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!ub = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
module {
  tla.func @flag_reservations(%dst: !gm, %src: !gm, %peer: i32) {
    %ptr = tla.alloc_ptr{size_bytes = 512} -> !tla.ptr<f32, ub, 256>
    %scratch = tla.make_tensor_like %ptr like %src layoutTag(<RowMajor>) : !tla.ptr<f32, ub, 256>, !gm -> !ub
    %forward = tla.flag "forward_auto" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte3>} -> !tla.flag
    %reverse = tla.flag "reverse_auto" {src_pipe = #tla.pipe<mte3>, dst_pipe = #tla.pipe<mte2>} -> !tla.flag
    "tla.vector"() ({
      // Reservations also apply to automatic flags used before communication.
      tla.set_flag %forward : !tla.flag
      tla.wait_flag %forward : !tla.flag
      tla.set_flag %reverse : !tla.flag
      tla.wait_flag %reverse : !tla.flag
      tla_dist.remote_copy %dst, %src, %peer, %peer, %scratch {event_id = 3 : i32, mode = "get", transport = "mte"} : !gm, !gm, i32, i32, !ub
      tla_dist.remote_copy %dst, %src, %peer, %peer, %scratch {event_id = 0 : i32, mode = "get", transport = "mte"} : !gm, !gm, i32, i32, !ub
      tla_dist.remote_copy %dst, %src, %peer, %peer, %scratch {event_id = 1 : i32, mode = "get", transport = "mte"} : !gm, !gm, i32, i32, !ub
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @flag_reservations
// CHECK-DAG: %[[ID0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG: %[[ID1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG: %[[ID3:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
// CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
// MTE reserves only the forward direction; reverse ID 0 remains available.
// CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
// CHECK-NEXT: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
// CHECK: call @shmem_remote_get_gm_row_major_to_gm_row_major_float({{.*}}, %[[ID3]])
// CHECK: call @shmem_remote_get_gm_row_major_to_gm_row_major_float({{.*}}, %[[ID0]])
// CHECK: call @shmem_remote_get_gm_row_major_to_gm_row_major_float({{.*}}, %[[ID1]])
// CHECK-NOT: hivm.hir.{{set|wait}}_flag

// RESERVE-COUNT-3: tla.flag "__tla_dist_comm_flag_
// RESERVE-NOT: tla.flag "__tla_dist_comm_flag_
