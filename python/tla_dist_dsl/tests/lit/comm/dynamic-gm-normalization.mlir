// RUN: %tla_dist_compile %s -o - | %filecheck %s
// RUN: sed '/^!tile0 =/s/!tla.stride<16,1>/!tla.stride<16,?>/' %s | %tla_dist_compile - -o - | %filecheck %s

// Runtime tensor arguments use Catlass's rank-4 dynamic ABI. CATCCOS remote
// copies consume the logical rank-2 descriptor ABI, so the communication pass
// must preserve the base offset while normalizing both GM operands.
!tile0 = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, gm, 4>>
!tile1 = !tla.tensor<!tla.layout<!tla.shape<8,16>, !tla.stride<16,1>, !tla.shape<8,16>, RowMajor>, !tla.coord<0,0>, !tla.ptr<f32, ub, 256>>

module {
  tla.func @dynamic_comm(%dst_base: memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<gm>>, %src_base: memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<gm>>, %peer: i32, %runtime_stride: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %dst = tla.tensor_desc %dst_base shape[%c8, %c16, %c1, %c1] stride[%runtime_stride, %c1, %c1, %c1] origin_shape[%c8, %c16] coord[%c0, %c0] : memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<gm>> -> !tile0
    %src = tla.tensor_desc %src_base shape[%c8, %c16, %c1, %c1] stride[%runtime_stride, %c1, %c1, %c1] origin_shape[%c8, %c16] coord[%c0, %c0] : memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<gm>> -> !tile0
    %ub_ptr = tla.alloc_ptr{size_bytes = 512} -> !tla.ptr<f32, ub, 256>
    "tla.vector"() ({
      %scratch = tla.make_tensor_like %ub_ptr like %src layoutTag(<RowMajor>) : !tla.ptr<f32, ub, 256>, !tile0 -> !tile1
      %copy_submit_0 = tla.flag "copy_submit_0" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte3>, id = 0 : i64} -> !tla.flag
      tla_dist.remote_copy %dst, %src, %peer, %scratch, %copy_submit_0 {mode = "get", transport = "mte"} : !tile0, !tile0, i32, !tile1, !tla.flag
      %copy_submit_1 = tla.flag "copy_submit_1" {src_pipe = #tla.pipe<mte2>, dst_pipe = #tla.pipe<mte3>, id = 0 : i64} -> !tla.flag
      tla_dist.remote_copy %dst, %src, %peer, %scratch, %copy_submit_1 {mode = "put", transport = "mte"} : !tile0, !tile0, i32, !tile1, !tla.flag
    }) : () -> ()
    tla.return
  }
}

// CHECK-LABEL: func.func @dynamic_comm(
// CHECK-SAME: %arg0: memref<?x?x?x?xf32
// CHECK-SAME: %arg1: memref<?x?x?x?xf32
// CHECK: %[[DST:[A-Za-z0-9_]+]] = memref.reinterpret_cast %arg0 to offset: [%{{.*}}], sizes: [%{{.*}}, %{{.*}}], strides: [%{{.*}}, %{{.*}}] {{.*}} to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
// CHECK: %[[SRC:[A-Za-z0-9_]+]] = memref.reinterpret_cast %arg1 to offset: [%{{.*}}], sizes: [%{{.*}}, %{{.*}}], strides: [%{{.*}}, %{{.*}}] {{.*}} to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
// CHECK: call @shmem_remote_get_gm_row_major_to_gm_row_major_float(%[[SRC]], %[[DST]], {{.*}}%arg2, {{[^)]*}})
// CHECK: call @shmem_remote_put_gm_row_major_to_gm_row_major_float({{.*}}%arg2, {{[^)]*}})
// CHECK-NOT: tla_dist.
