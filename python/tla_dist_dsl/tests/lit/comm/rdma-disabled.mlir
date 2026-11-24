// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// UNSUPPORTED: rdma
// RUN: sed '/tla_dist.quiet/d' %s | not %tla_dist_compile - -o - 2>&1 | %filecheck %s --check-prefix=COPY
// RUN: sed '/tla_dist.quiet/d; s/mode = "get"/mode = "put"/' %s | not %tla_dist_compile - -o - 2>&1 | %filecheck %s --check-prefix=COPY
// RUN: sed '/tla_dist.remote_copy/d' %s | not %tla_dist_compile - -o - 2>&1 | %filecheck %s --check-prefix=QUIET
// COPY: error: 'tla_dist.remote_copy' op RDMA is not enabled in this build
// QUIET: error: 'tla_dist.quiet' op RDMA is not enabled in this build

!gm = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!ub = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 32>>
module {
  tla.func @rdma_disabled(%dst: !gm, %src: !gm, %peer: i32) {
    "tla.vector"() ({
      %ptr = tla.alloc_ptr{size_bytes = 512} -> !tla.ptr<f32, ub, 32>
      %ub = tla.make_tensor_like %ptr like %src layoutTag(<RowMajor>) : !tla.ptr<f32, ub, 32>, !gm -> !ub
      tla_dist.remote_copy %dst, %src, %peer, %peer, %ub {event_id = 0 : i32, mode = "get", transport = "rdma"} : !gm, !gm, i32, i32, !ub
      tla_dist.quiet %peer, %peer, %ub {event_id = 0 : i32, transport = "rdma"} : i32, i32, !ub
    }) : () -> ()
    tla.return
  }
}
