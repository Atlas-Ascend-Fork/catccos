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
      %copy_submit_0 = tla.flag "copy_submit_0" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte3>, id = 0 : i64} -> !tla.flag
      tla_dist.remote_copy %dst, %src, %peer, %scratch, %copy_submit_0 {mode = "get", transport = "mte"} : !gm, !gm, i32, !ub, !tla.flag
      %copy_submit_1 = tla.flag "copy_submit_1" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte3>, id = 2 : i64} -> !tla.flag
      tla_dist.remote_copy %dst, %src, %peer, %scratch, %copy_submit_1 {mode = "put", transport = "mte", atomic_add} : !gm, !gm, i32, !ub, !tla.flag
      // GM <-> UB routes require their corresponding explicit IR mode.
      tla_dist.remote_copy %scratch, %src, %peer {mode = "get", transport = "mte"} : !ub, !gm, i32
      tla_dist.remote_copy %dst, %scratch, %peer {mode = "put", transport = "mte"} : !gm, !ub, i32
      // Flag lowering runs before communication lowering and reserves ids of
      // flags that are only consumed by remote_copy.
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
