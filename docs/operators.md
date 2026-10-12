# CATCCOS 算子总览

本文档列出 CATCCOS 当前提供的 C++ 算子样例，按通信/计算模式分为 6 类，并单列 PyTorch binding。Kernel 模板位于 `include/catccos/`，样例位于 `examples/`。

> **说明**
>
> - `aux_ops/`（MoE routing、token unpermute 等）为组合算子内部组件，不单独列入下表，详见 [实现模式 - MoE](operators/implementation_patterns.md#5-moe--groupedmatmul--alltoallv)。
> - 标注 * 的示例暂无独立 README，示例列链接至目录，实现细节见 [实现模式文档](operators/implementation_patterns.md)。
> - `allgather_matmul_rdma` 仅在编译时开启 `RDMA_TRANSPORT` 选项时构建。
> - optional backend 的 HCCL 变体使用各自目录中的构建脚本；SHMEM 变体由顶层 CMake 注册，其中 Ascend950 变体仅在 A5 架构构建。PyTorch binding 需开启 `CATCCOS_TORCH_EXTENSION`。
> - **硬件平台**：`Atlas A2/A3` 表示同时支持 Atlas A2 与 Atlas A3；`Atlas 350` 表示仅支持 Atlas 350 平台。示例目录仍保留 `ascend950_*` 前缀，与代码仓库命名一致。

---

## 1. MatMul + 集合通信

矩阵乘与 AllGather、ReduceScatter、AllReduce 等集合通信的融合算子，面向 Atlas A2/A3 平台。

| 算子名称 | Registry 名 | 通信模式 | 数据类型 | 硬件平台 | Kernel 头文件 | 示例 |
|---------|-------------|---------|---------|---------|--------------|------|
| MatMul + AllReduce | `MatmulAllReduce` | AllReduce | FP16 | Atlas A2/A3 | `dgemm/kernel/matmul_allreduce.hpp` | [matmul_allreduce](../examples/matmul_allreduce/README.md) |
| AllGather + MatMul | `AllGatherMatmul` | AllGather | FP16 | Atlas A2/A3 | `dgemm/kernel/allgather_matmul.hpp` | [allgather_matmul](../examples/allgather_matmul/README.md) |
| MatMul + ReduceScatter | `MatmulReduceScatter` | ReduceScatter | FP16 | Atlas A2/A3 | `dgemm/kernel/matmul_reduce_scatter.hpp` | [matmul_reduce_scatter](../examples/matmul_reduce_scatter/README.md) |
| AllGather + MatMul（输出 Gather 结果） | `AllGatherMatmulWithGatherResult` | AllGather | FP16 | Atlas A2/A3 | `dgemm/kernel/allgather_matmul_with_gather_result.hpp` | [allgather_matmul_with_gather_result](../examples/allgather_matmul_with_gather_result/README.md) |
| AllGather + MatMul（Remote Read） | `AllGatherMatmulRemoteRead` | AllGather（Remote Read） | FP16 | Atlas A2/A3 | `dgemm/kernel/allgather_matmul_with_remote_read.hpp` | [allgather_matmul_remote_read](../examples/allgather_matmul_remote_read/README.md) |
| AllGather + MatMul（RDMA Write） | `AllGatherMatmulRdma` | AllGather（RDMA） | FP16 | Atlas A2/A3 | `dgemm/kernel/allgather_matmul_with_rdma_write.hpp` | [allgather_matmul_rdma](../examples/allgather_matmul_rdma/README.md) |
| AllGather + MatMul（浅融合） | `AllGatherMatmulShallowFusion` | AllGather | FP16 | Atlas A2/A3 | `dgemm/kernel/allgather_matmul_shallow_fusion.hpp` | [allgather_matmul_shallow_fusion](../examples/allgather_matmul_shallow_fusion/main.cpp) |
| MatMul + AllReduce（浅融合） | `MatmulAllReduceShallowFusion` | AllReduce | FP16 | Atlas A2/A3 | `dgemm/kernel/matmul_allreduce_shallow_fusion.hpp` | [matmul_allreduce_shallow_fusion](../examples/matmul_allreduce_shallow_fusion/main.cpp) |
| MatMul + ReduceScatter（浅融合） | `MatmulReduceScatterShallowFusion` | ReduceScatter | FP16 | Atlas A2/A3 | `dgemm/kernel/matmul_reduce_scatter_shallow_fusion.hpp` | [matmul_reduce_scatter_shallow_fusion](../examples/matmul_reduce_scatter_shallow_fusion/main.cpp) |
| AllGather + MatMul（SHMEM 后端） | `AllGatherMatmul` | AllGather | FP16 | Atlas A2/A3 | `dgemm/kernel/allgather_matmul_with_local_optional_backend.hpp` | [allgather_matmul_shmem](../examples/allgather_matmul_optional_backend/allgather_matmul_shmem/main.cpp) |
| AllGather + MatMul（HCCL 后端） | `AllGatherMatmulHccl` | AllGather | FP16 | Atlas A2/A3 | `dgemm/kernel/allgather_matmul_with_local_optional_backend.hpp` | [allgather_matmul_hccl](../examples/allgather_matmul_optional_backend/allgather_matmul_hccl/main.cpp) |

---

## 2. 量化 AllGather-MatMul / MatMul-ReduceScatter

在 MatMul 融合基础上增加 INT8 量化输入与反量化 epilogue。

| 算子名称 | Registry 名 | 通信模式 | 数据类型 | 硬件平台 | Kernel 头文件 | 示例 |
|---------|-------------|---------|---------|---------|--------------|------|
| AllGather + MatMul + Dequant | `AllGatherMatmulDequant` | AllGather | INT8 → FP16 | Atlas A2/A3 | `dgemm/kernel/allgather_matmul_dequant.hpp` | [allgather_matmul_dequant](../examples/allgather_matmul_dequant/main.cpp)* |
| AllGather + MatMul + Dequant + Bias | `AllGatherMatmulDequantBias` | AllGather | INT8 → FP16 | Atlas A2/A3 | `dgemm/kernel/allgather_matmul_dequant_bias.hpp` | [allgather_matmul_dequant_bias](../examples/allgather_matmul_dequant_bias/README.md) |
| MatMul + Dequant + ReduceScatter Write | `MatmulDequantReduceScatterWrite` | ReduceScatter | INT8 → FP16 | Atlas A2/A3 | `dgemm/kernel/matmul_dequant_reduce_scatter_write.hpp` | [matmul_dequant_reduce_scatter_write](../examples/matmul_dequant_reduce_scatter_write/main.cpp)* |
| AllGather + MatMul（W4A4） | `AllGatherMatmulW4A4` | AllGather | W4A4 | Atlas A2/A3 | `dgemm/kernel/allgather_matmul_w4a4.hpp` | [allgather_matmul_w4a4](../examples/allgather_matmul_w4a4/main.cpp) |

---

## 3. MoE / GroupedMatMul + AllToAllV

面向 MoE 场景的分组矩阵乘与 AllToAllV 融合，支持 token 路由、SwiGLU 激活及完整 FFN dispatch/combine 流水线。

| 算子名称 | Registry 名 | 通信模式 | 数据类型 | 硬件平台 | Kernel 头文件 | 示例 |
|---------|-------------|---------|---------|---------|--------------|------|
| GroupedMatMul + AllToAllV | `GroupedMatmulAllToAllV` | AllToAllV | FP16 | Atlas A2/A3 | `dgemm/kernel/grouped_matmul_alltoallv.hpp` | [grouped_matmul_alltoallv](../examples/grouped_matmul_alltoallv/README.md) |
| GroupedMatMul + AllToAllV（TLA） | `GroupedMatmulAllToAllVTla` | AllToAllV | FP16 | Atlas A2/A3 | `dgemm/kernel/grouped_matmul_alltoallv_tla.hpp` | [grouped_matmul_alltoallv_tla](../examples/grouped_matmul_alltoallv_tla/README.md) |
| AllToAllV + GroupedMatMul | `AllToAllVGroupedMatmul` | AllToAllV | FP16 | Atlas A2/A3 | `dgemm/kernel/alltoallv_grouped_matmul.hpp` | [alltoallv_grouped_matmul](../examples/alltoallv_grouped_matmul/README.md) |
| AllToAllV + GMM v2 | `AllToAllVGMMV2` | AllToAllV | FP16 | Atlas A2/A3 | `dgemm/kernel/alltoallv_gmm_v2.hpp` | [alltoallv_gmm_v2](../examples/alltoallv_gmm_v2/README.md) |
| MoE Dispatch GMM | `AllToAllVGMMV2` | AllToAllV | FP16 | Atlas A2/A3 | `dgemm/kernel/alltoallv_gmm_v2.hpp` | [dispatch_gmm](../examples/dispatch_gmm/main.cpp)* |
| MoE Dispatch GMM + SwiGLU | `AllToAllVGMMV2` | AllToAllV | FP16 | Atlas A2/A3 | `dgemm/kernel/alltoallv_gmm_v2.hpp` | [dispatch_gmm_swiglu](../examples/dispatch_gmm_swiglu/main.cpp)* |
| MoE Dispatch GMM + Dequant + SwiGLU | `DispatchGmmDequantSwiglu` | AllToAllV | INT8 → FP16 | Atlas A2/A3 | `dgemm/kernel/alltoallv_gmm_dequant_v2.hpp` | [dispatch_gmm_dequant_swiglu](../examples/dispatch_gmm_dequant_swiglu/main.cpp)* |
| GMM + AllToAllV v2 | `GMMAllToAllVV2` | AllToAllV | FP16 | Atlas A2/A3 | `dgemm/kernel/gmm_alltoallv_v2.hpp` | [gmm_alltoallv_v2](../examples/gmm_alltoallv_v2/README.md) |
| MoE FFN Dispatch + Combine | `DispatchFFNCombine` | AllToAllV | FP16 | Atlas A2/A3 | `alltoallv_gmm_v2.hpp` + `gmm_alltoallv_v2.hpp` | [dispatch_ffn_combine](../examples/dispatch_ffn_combine/README.md) |
| GroupedMatMul + AllToAllV（Small M） | `GroupedMatmulAllToAllvSmallM` | AllToAllV | FP16 | Atlas A2/A3 | `dgemm/kernel/grouped_matmul_alltoallv_small_m.hpp` | [grouped_matmul_alltoallv_small_m](../examples/grouped_matmul_alltoallv_small_m/main.cpp) |

> `AllToAllVGMMV2` 被 `alltoallv_gmm_v2`、`dispatch_gmm`、`dispatch_gmm_swiglu` 三个示例共用 Registry 名，但 Device 实现不同（是否含 SwiGLU 等）。

---

## 4. Atlas 350 通用融合

面向 Atlas 350 平台的 MatMul 与集合通信融合，需开启 `CATCCOS_ENABLE_A5_BUILD` 编译选项。

| 算子名称 | Registry 名 | 通信模式 | 数据类型 | 硬件平台 | Kernel 头文件 | 示例 |
|---------|-------------|---------|---------|---------|--------------|------|
| AllGather + MatMul | `Ascend950AllGatherMatmul` | AllGather | FP16 | Atlas 350 | `dgemm/kernel/ascend950_allgather_matmul.hpp` | [ascend950_allgather_matmul](../examples/ascend950_allgather_matmul/README.md) |
| MatMul + ReduceScatter | `Ascend950MatmulReduceScatter` | ReduceScatter | FP16 | Atlas 350 | `dgemm/kernel/matmul_reduce_scatter_tla.hpp` | [ascend950_matmul_reduce_scatter](../examples/ascend950_matmul_reduce_scatter/README.md) |
| AllToAll + MatMul | `Ascend950AllToAllMatmul` | AllToAll | FP16 | Atlas 350 | `dgemm/kernel/ascend950_alltoall_matmul.hpp` | [ascend950_alltoall_matmul](../examples/ascend950_alltoall_matmul/README.md) |
| MatMul + AllToAll | `Ascend950MatmulAllToAll` | AllToAll | FP16 | Atlas 350 | `dgemm/kernel/ascend950_matmul_alltoall.hpp` | [ascend950_matmul_alltoall](../examples/ascend950_matmul_alltoall/README.md) |
| GroupedMatMul + AllToAllV | `Ascend950GroupedMatmulAllToAllV` | AllToAllV | FP16 | Atlas 350 | `dgemm/kernel/grouped_matmul_alltoallv_tla.hpp` | [ascend950_grouped_matmul_alltoallv](../examples/ascend950_grouped_matmul_alltoallv/README.md) |
| MX-FP8 MatMul + ReduceScatter | `Ascend950MxFp8MatmulReduceScatter` | ReduceScatter | MX-FP8 | Atlas 350 | `dgemm/kernel/matmul_reduce_scatter_mx_tla.hpp` | [ascend950_mxfp8_matmul_reduce_scatter](../examples/ascend950_mxfp8_matmul_reduce_scatter/README.md) |
| AllToAllV + GroupedMatMul | `Ascend950AllToAllVGroupedMatmul` | AllToAllV | FP16 | Atlas 350 | `dgemm/kernel/ascend950_alltoallv_grouped_matmul.hpp` | [ascend950_alltoallv_grouped_matmul](../examples/ascend950_alltoallv_grouped_matmul/main.cpp) |
| AllGather + MatMul（UDMA） | `Ascend950AllGatherMatmulUdma` | AllGather | FP16 | Atlas 350 | `dgemm/kernel/ascend950_allgather_matmul_with_udma.hpp` | [ascend950_allgather_matmul_udma](../examples/ascend950_allgather_matmul_udma/main.cpp) |
| MoE FFN Dispatch + Combine | `Ascend950DispatchFFNCombine` | AllToAllV | BF16 / MX-FP8 | Atlas 350 | `dgemm/kernel/ascend950_alltoall_gmm.hpp` + `ascend950_gmm_alltoall.hpp` | [ascend950_dispatch_ffn_combine](../examples/ascend950_dispatch_ffn_combine/main.cpp) |
| AllGather + MatMul（SHMEM 后端） | `Ascend950AllGatherMatmulShmem` | AllGather | FP16 | Atlas 350 | `dgemm/kernel/ascend950_allgather_matmul_with_local_optional_backend.hpp` | [ascend950_allgather_matmul_shmem](../examples/allgather_matmul_optional_backend/ascend950_allgather_matmul_shmem/main.cpp) |
| AllGather + MatMul（HCCL 后端） | `AllGatherMatmulHccl` | AllGather | FP16 | Atlas 350 | `dgemm/kernel/ascend950_allgather_matmul_with_local_optional_backend.hpp` | [ascend950_allgather_matmul_hccl](../examples/allgather_matmul_optional_backend/ascend950_allgather_matmul_hccl/main.cpp) |

---

## 5. Atlas 350 MX 量化

Atlas 350 平台上的 MX 格式（FP8/FP4）量化融合算子。

| 算子名称 | Registry 名 | 通信模式 | 数据类型 | 硬件平台 | Kernel 头文件 | 示例 |
|---------|-------------|---------|---------|---------|--------------|------|
| MX-FP8 AllGather + MatMul | `Ascend950Fp8MxAllGatherMatmul` | AllGather | MX-FP8 (E4M3) | Atlas 350 | `dgemm/kernel/mx_allgather_matmul.hpp` | [ascend950_fp8_mx_allgather_matmul](../examples/ascend950_fp8_mx_allgather_matmul/README.md) |
| MX-FP4 AllGather + MatMul | `Ascend950Fp4MxAllGatherMatmul` | AllGather | MX-FP4 | Atlas 350 | `dgemm/kernel/mx_allgather_matmul.hpp` | [ascend950_fp4_mx_allgather_matmul](../examples/ascend950_fp4_mx_allgather_matmul/README.md) |
| MX-FP8 GroupedMatMul + AllToAllV | `Ascend950Fp8MxGroupedMatmulAllToAllV` | AllToAllV | MX-FP8 | Atlas 350 | `dgemm/kernel/grouped_matmul_alltoallv_mx.hpp` | [ascend950_fp8_mx_grouped_matmul_alltoallv](../examples/ascend950_fp8_mx_grouped_matmul_alltoallv/README.md) |
| MX-FP4 GroupedMatMul + AllToAllV | `Ascend950Fp4MxGroupedMatmulAllToAllV` | AllToAllV | MX-FP4 | Atlas 350 | `dgemm/kernel/grouped_matmul_alltoallv_mx.hpp` | [ascend950_fp4_mx_grouped_matmul_alltoallv](../examples/ascend950_fp4_mx_grouped_matmul_alltoallv/README.md) |
| MX-FP8 MatMul + AllToAll | `Ascend950MxFp8MatmulAllToAll` | AllToAll | MX-FP8 | Atlas 350 | `dgemm/kernel/ascend950_matmul_alltoall_mx_slice_n.hpp` | [ascend950_mxfp8_matmul_alltoall](../examples/ascend950_mxfp8_matmul_alltoall/main.cpp) |
| MX-FP8 AllToAll + MatMul（Split K） | `Ascend950MxFp8AllToAllMatmulSplitKUrma` | AllToAll | MX-FP8 | Atlas 350 | `dgemm/kernel/ascend950_alltoall_matmul_split_k_urma.hpp` | [ascend950_mxfp8_alltoall_matmul_split_k_urma](../examples/ascend950_mxfp8_alltoall_matmul_split_k_urma/main.cpp) |
| MX-FP4 MatMul + ReduceScatter | `Ascend950Fp4MxMatmulReduceScatter` | ReduceScatter | MX-FP4 | Atlas 350 | `dgemm/kernel/matmul_reduce_scatter_mx_tla.hpp` | [ascend950_fp4_mx_matmul_reduce_scatter](../examples/ascend950_fp4_mx_matmul_reduce_scatter/main.cpp) |
| MX-FP8 AllToAllV + GroupedMatMul | `Ascend950Fp8MxAllToAllVGroupedMatmul` | AllToAllV | MX-FP8 | Atlas 350 | `dgemm/kernel/mx_alltoallv_group_matmul.hpp` | [ascend950_fp8_mx_alltoallv_grouped_matmul](../examples/ascend950_fp8_mx_alltoallv_grouped_matmul/main.cpp) |
| MX-FP4 AllToAllV + GroupedMatMul | `Ascend950Fp4MxAllToAllVGroupedMatmul` | AllToAllV | MX-FP4 | Atlas 350 | `dgemm/kernel/mx_alltoallv_group_matmul.hpp` | [ascend950_fp4_mx_alltoallv_grouped_matmul](../examples/ascend950_fp4_mx_alltoallv_grouped_matmul/main.cpp) |

---

## 6. 纯通信 / 量化通信

不含 MMAD 的纯通信或量化通信 Kernel，位于 `include/catccos/comm/kernel/`。

| 算子名称 | Registry 名 | 通信模式 | 数据类型 | 硬件平台 | Kernel 头文件 | 示例 |
|---------|-------------|---------|---------|---------|--------------|------|
| Quant AllGather | `QuantAllGather` | AllGather | BF16 → HiF8 | Atlas 350 | `comm/kernel/quant_allgather.hpp` | [ascend950_quant_allgather](../examples/ascend950_quant_allgather/README.md) |
| Quant AllToAll | `QuantAllToAll` | AllToAll | BF16 → HiF8 | Atlas 350 | `comm/kernel/quant_alltoall.hpp` | [ascend950_quant_alltoall](../examples/ascend950_quant_alltoall/README.md) |
| MX Quant + AllGather | `MxQuantAllGather` | AllGather | BF16 → MX-FP8/FP4 | Atlas 350 | `comm/kernel/mx_quant_allgather.hpp` | [ascend950_mx_quant_allgather](../examples/ascend950_mx_quant_allgather/README.md) |

---

## 统计

PyTorch binding 复用 `AllGatherMatmul` 算子和 `dgemm/kernel/allgather_matmul.hpp`，使用 `CATCCOS_TORCH_EXTENSION` 构建，入口见 [torch_binding](../examples/torch_binding/CMakeLists.txt)。

| 类别 | 算子数量 |
|------|---------|
| MatMul + 集合通信 | 11 |
| 量化 AllGather-MatMul / MatMul-RS | 4 |
| MoE / GroupedMatMul + AllToAllV | 10 |
| Atlas 350 通用融合 | 11 |
| Atlas 350 MX 量化 | 9 |
| 纯通信 / 量化通信 | 3 |
| **C++ 样例合计** | **48**（含条件构建的 RDMA 和独立构建的 HCCL 变体） |
| PyTorch binding | 1（条件构建） |
| **样例入口总计** | **49** |

实现模式说明见 [operators/implementation_patterns.md](operators/implementation_patterns.md)。
