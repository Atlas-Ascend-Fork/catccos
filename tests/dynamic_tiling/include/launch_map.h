
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LAUNCH_MAP_H
#define LAUNCH_MAP_H

#include <unordered_map>

using KernelFuncPtr =
    void (*)(void*, uint32_t, uint64_t, KernelParams&, uint8_t*, uint8_t*, CocTilingParams&, uint32_t, uint32_t);

class KernelDispatcher {
private:
    static std::unordered_map<int, KernelFuncPtr>& GetKernelMap()
    {
        static std::unordered_map<int, KernelFuncPtr> kernelMap;
        return kernelMap;
    }

    static int Hash(CocCommType commType, CocDataType dataType) { return (commType << 8) | dataType; }

public:
    static KernelFuncPtr GetKernelFunc(CocCommType commType, CocDataType dataType)
    {
        auto& kernelMap = GetKernelMap();
        if (auto it = kernelMap.find(Hash(commType, dataType)); it != kernelMap.end()) {
            return it->second;
        }
        return nullptr;
    }

    static void RegisterKernelFunc(CocCommType commType, CocDataType dataType, KernelFuncPtr func)
    {
        auto& kernelMap = GetKernelMap();
        kernelMap.insert({Hash(commType, dataType), func});
    }
};

#define REGISTER_KERNEL_FUNC(kernelName, commType, dataType)                                                 \
    void Launch##kernelName##dataType(                                                                       \
        void*, uint32_t, uint64_t, KernelParams&, uint8_t*, uint8_t*, CocTilingParams&, uint32_t, uint32_t); \
    namespace {                                                                                              \
    struct AutoRegister##kernelName##dataType {                                                              \
        AutoRegister##kernelName##dataType()                                                                 \
        {                                                                                                    \
            KernelDispatcher::RegisterKernelFunc(commType, dataType, &Launch##kernelName##dataType);         \
        }                                                                                                    \
    } s_autoRegister##kernelName##dataType;                                                                  \
    }

REGISTER_KERNEL_FUNC(MatmulAllReduce, MATMUL_ALLREDUCE, FP16);
REGISTER_KERNEL_FUNC(AllGatherMatmul, ALLGATHER_MATMUL, FP16);
REGISTER_KERNEL_FUNC(MatmulReduceScatter, MATMUL_REDUCE_SCATTER, FP16);
REGISTER_KERNEL_FUNC(AllGatherMatmulWithGatherResult, ALLGATHER_MATMUL_WITH_GATHER_RESULT, FP16);
REGISTER_KERNEL_FUNC(AllGatherMatmulRemoteRead, ALLGATHER_MATMUL_REMOTE_READ, FP16);
REGISTER_KERNEL_FUNC(GroupedMatmulAllToAllV, GROUPED_MATMUL_ALLTOALLV, FP16);
REGISTER_KERNEL_FUNC(AllToAllVGroupedMatmul, ALLTOALLV_GROUPED_MATMUL, FP16);
REGISTER_KERNEL_FUNC(AllToAllVGMMV2, ALLTOALLV_GMM_V2, FP16);

REGISTER_KERNEL_FUNC(MatmulAllReduce, MATMUL_ALLREDUCE, BF16);
REGISTER_KERNEL_FUNC(AllGatherMatmul, ALLGATHER_MATMUL, BF16);
REGISTER_KERNEL_FUNC(MatmulReduceScatter, MATMUL_REDUCE_SCATTER, BF16);
REGISTER_KERNEL_FUNC(AllGatherMatmulWithGatherResult, ALLGATHER_MATMUL_WITH_GATHER_RESULT, BF16);
REGISTER_KERNEL_FUNC(AllGatherMatmulRemoteRead, ALLGATHER_MATMUL_REMOTE_READ, BF16);
REGISTER_KERNEL_FUNC(GroupedMatmulAllToAllV, GROUPED_MATMUL_ALLTOALLV, BF16);
REGISTER_KERNEL_FUNC(AllToAllVGroupedMatmul, ALLTOALLV_GROUPED_MATMUL, BF16);
REGISTER_KERNEL_FUNC(AllToAllVGMMV2, ALLTOALLV_GMM_V2, BF16);

#ifdef RDMA_TRANSPORT
REGISTER_KERNEL_FUNC(AllGatherMatmulRdma, ALLGATHER_MATMUL_RDMA, FP16);
REGISTER_KERNEL_FUNC(AllGatherMatmulRdma, ALLGATHER_MATMUL_RDMA, BF16);
#endif

REGISTER_KERNEL_FUNC(AllGatherMatmulDequantBias, ALLGATHER_MATMUL_DEQUANT_BIAS, INT8);
REGISTER_KERNEL_FUNC(AllGatherMatmulDequant, ALLGATHER_MATMUL_DEQUANT, INT8);
REGISTER_KERNEL_FUNC(AllGatherMatmulDequantPadding, ALLGATHER_MATMUL_DEQUANT_PADDING, INT8);
REGISTER_KERNEL_FUNC(MatmulDequantReduceScatterWrite, MATMUL_DEQUANT_REDUCE_SCATTER_WRITE, INT8);

#ifdef CATCCOS_ENABLE_A5_BUILD
REGISTER_KERNEL_FUNC(Ascend950AllGatherMatmul, ASCEND950_ALLGATHER_MATMUL, FP16);
REGISTER_KERNEL_FUNC(Ascend950MatmulReduceScatter, ASCEND950_MATMUL_REDUCE_SCATTER, FP16);
REGISTER_KERNEL_FUNC(Ascend950AllGatherMatmul, ASCEND950_ALLGATHER_MATMUL, BF16);
REGISTER_KERNEL_FUNC(Ascend950MatmulReduceScatter, ASCEND950_MATMUL_REDUCE_SCATTER, BF16);
REGISTER_KERNEL_FUNC(Ascend950Fp8MxAllGatherMatmul, ASCEND950_FP8_MX_ALLGATHER_MATMUL, FP16);
REGISTER_KERNEL_FUNC(Ascend950Fp4MxAllGatherMatmul, ASCEND950_FP4_MX_ALLGATHER_MATMUL, FP16);
REGISTER_KERNEL_FUNC(Ascend950BF16QuantAllGather, MX_QUANT_ALLGATHER, FP8E4M3FN);
REGISTER_KERNEL_FUNC(Ascend950GroupedMatmulAllToAllV, ASCEND950_GROUPED_MATMUL_ALLTOALLV, FP16);
REGISTER_KERNEL_FUNC(Ascend950Fp8MxGroupedMatmulAllToAllV, ASCEND950_FP8_MX_GROUPED_MATMUL_ALLTOALLV, FP16);
REGISTER_KERNEL_FUNC(Ascend950Fp4MxGroupedMatmulAllToAllV, ASCEND950_FP4_MX_GROUPED_MATMUL_ALLTOALLV, FP16);
REGISTER_KERNEL_FUNC(Ascend950Fp8MxAllToAllVGroupedMatmul, ASCEND950_FP8_MX_ALLTOALLV_GROUPED_MATMUL, FP16);
REGISTER_KERNEL_FUNC(Ascend950Fp4MxAllToAllVGroupedMatmul, ASCEND950_FP4_MX_ALLTOALLV_GROUPED_MATMUL, FP16);
REGISTER_KERNEL_FUNC(Ascend950AllToAllVGroupedMatmul, ASCEND950_ALLTOALLV_GROUPED_MATMUL, FP16);
REGISTER_KERNEL_FUNC(Ascend950Fp4MxMatmulReduceScatter, ASCEND950_FP4_MX_MATMUL_REDUCE_SCATTER, FP16);
REGISTER_KERNEL_FUNC(Ascend950MxFp8MatmulAllToAll, ASCEND950_MXFP8_MATMUL_ALLTOALL, FP16);
REGISTER_KERNEL_FUNC(Ascend950MxFp8AllToAllMatmulSplitKUrma, ASCEND950_MXFP8_ALLTOALL_MATMUL_SPLIT_K_URMA, FP16);
REGISTER_KERNEL_FUNC(Ascend950AllGatherMatmulUdma, ASCEND950_ALLGATHER_MATMUL_UDMA, FP16);
REGISTER_KERNEL_FUNC(Ascend950MatmulReduceScatterUdma, ASCEND950_MATMUL_REDUCE_SCATTER_UDMA, FP16);
#endif

#undef REGISTER_KERNEL_FUNC

#endif // LAUNCH_MAP_H
