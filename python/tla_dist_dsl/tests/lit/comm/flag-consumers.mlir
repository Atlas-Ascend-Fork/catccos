// RUN: %tla_dist_compile %s -o - | %filecheck %s --implicit-check-not=tla_dist. --implicit-check-not='tla.flag'

!gm = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!ub = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>
module {
  tla.func @flag_consumers(%dst: !gm, %src: !gm, %peer: i32) {
    %ptr = tla.alloc_ptr{size_bytes = 512} -> !tla.ptr<f32, ub, 256>
    %scratch = tla.make_tensor_like %ptr like %src layoutTag(<RowMajor>) : !tla.ptr<f32, ub, 256>, !gm -> !ub
    %explicit = tla.flag "explicit_submit" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte3>, id = 3 : i64} -> !tla.flag
    %shared = tla.flag "shared_submit" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte3>} -> !tla.flag
    %remote = tla.flag "remote_only" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte3>, id = -1 : i64} -> !tla.flag
    "tla.vector"() ({
      tla.set_flag %shared : !tla.flag
      tla.wait_flag %shared : !tla.flag
      tla_dist.remote_copy %dst, %src, %peer, %scratch, %explicit {mode = "get", transport = "mte"} : !gm, !gm, i32, !ub, !tla.flag
      tla_dist.remote_copy %dst, %src, %peer, %scratch, %shared {mode = "get", transport = "mte"} : !gm, !gm, i32, !ub, !tla.flag
      tla_dist.remote_copy %dst, %src, %peer, %scratch, %remote {mode = "get", transport = "mte"} : !gm, !gm, i32, !ub, !tla.flag
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @flag_consumers
// CHECK-DAG: %[[ID0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG: %[[ID1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG: %[[ID3:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
// CHECK: call @shmem_remote_get_gm_row_major_to_gm_row_major_float({{.*}}, %[[ID3]])
// CHECK: call @shmem_remote_get_gm_row_major_to_gm_row_major_float({{.*}}, %[[ID0]])
// CHECK: call @shmem_remote_get_gm_row_major_to_gm_row_major_float({{.*}}, %[[ID1]])
