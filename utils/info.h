
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INFO_H
#define INFO_H

#include <acl/acl.h>
#include <limits.h>

#include <cmath>
#include <map>

#include "utils.h"

static uint64_t SHMEM_MALLOC_MAX_SIZE = 1024UL * 1024UL * 1024;
constexpr uint32_t M0 = 128;
constexpr uint32_t N0 = 256;
constexpr uint32_t K0 = 256;
constexpr uint32_t TILE_SHAPE_64 = 64;
constexpr uint32_t TILE_SHAPE_128 = 128;
constexpr uint32_t TILE_SHAPE_256 = 256;
constexpr uint32_t TILE_SHAPE_512 = 512;
constexpr uint32_t WORKSPACE_STAGES = 2;
constexpr uint32_t UB_STAGES = 2;
constexpr uint32_t BLOCK_NUM = 20;
constexpr uint32_t WARM_UP_TIMES = 10;
constexpr uint32_t PERF_TEST_CYCLE_TIMES = 3;
constexpr uint32_t MAX_BLOCK_COUNT = 2;
constexpr int32_t SHMEM_BUFF_BYTES = 1004UL * 1024 * 1024;
constexpr uint32_t IPC_BUFF_MAX_SIZE = 1000UL * 1024 * 1024;
constexpr uint32_t IPC_DATA_OFFSET = 2 * 1024 * 1024;
constexpr uint32_t PING_PONG_SIZE = 2;
constexpr uint32_t SYNC_UNIT_SIZE = 4 * sizeof(int64_t);
constexpr int32_t FLAG_BUFF_BYTES = 5 * 512 * 1024;
constexpr int32_t INPUT_DTYPE = 2;
constexpr int32_t MAX_RANK_SIZE = 128;

enum CocCommType {
    MATMUL_ALLREDUCE = 0,
    ALLGATHER_MATMUL,
    ASCEND950_ALLGATHER_MATMUL,
    MATMUL_REDUCE_SCATTER,
    ASCEND950_MATMUL_REDUCE_SCATTER,
    ALLGATHER_MATMUL_WITH_GATHER_RESULT,
    GROUPED_MATMUL_ALLTOALLV,
    GROUPED_MATMUL_ALLTOALLV_TLA,
    ASCEND950_GROUPED_MATMUL_ALLTOALLV,
    ALLTOALLV_GROUPED_MATMUL,
    ALLGATHER_MATMUL_RDMA,
    ALLTOALLV_GMM_V2,
    ALLGATHER_MATMUL_DEQUANT_BIAS,
    ALLGATHER_MATMUL_DEQUANT,
    ALLGATHER_MATMUL_DEQUANT_PADDING,
    QUANT_ALLTOALL,
    QUANT_ALLGATHER,
    MATMUL_DEQUANT_REDUCE_SCATTER_WRITE,
    ASCEND950_MXFP8_MATMUL_REDUCE_SCATTER,
    ASCEND950_MXFP8_MATMUL_ALLTOALL,
    ALLGATHER_MATMUL_REMOTE_READ,
    DISPATCH_GMM_DEQUANT_SWIGLU,
    ASCEND950_FP8_MX_ALLGATHER_MATMUL,
    ASCEND950_FP4_MX_ALLGATHER_MATMUL,
    MX_QUANT_ALLGATHER,
    ASCEND950_FP8_MX_GROUPED_MATMUL_ALLTOALLV,
    ASCEND950_FP4_MX_GROUPED_MATMUL_ALLTOALLV,
    ASCEND950_FP8_MX_ALLTOALLV_GROUPED_MATMUL,
    ASCEND950_FP4_MX_ALLTOALLV_GROUPED_MATMUL,
    ASCEND950_ALLTOALLV_GROUPED_MATMUL,
    ASCEND950_FP4_MX_MATMUL_REDUCE_SCATTER,
    DISPATCH_FFN_COMBINE,
    ASCEND950_DISPATCH_FFN_COMBINE,
    ASCEND950_ALLGATHER_MATMUL_UDMA,
    ASCEND950_MXFP8_ALLTOALL_MATMUL_SPLIT_K_URMA,
    ALLGATHER_MATMUL_SHALLOW_FUSION,
    MATMUL_ALLREDUCE_SHALLOW_FUSION,
    MATMUL_REDUCE_SCATTER_SHALLOW_FUSION,
    GROUPED_MATMUL_ALLTOALLV_SMALL_M,
    ASCEND950_DISPATCH,
    TYPE_NUM,
    UNKNOWN
};

enum CocDataType { FP16 = 1, INT8 = 2, BF16 = 27, FP8E4M3FN = 36 };

struct CocTilingParams {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t transA = 0;
    uint32_t transB = 0;
    uint32_t m0 = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    uint32_t commTileM = 0;
    uint32_t commInterval = 0;
    uint32_t commNpuSplit = 0;
    uint32_t commDataSplit = 0;
    uint32_t commBlockM = 0;
    uint32_t rankSize = 0;
    uint32_t epSize = 0;
    uint32_t expertNum = 0;
    uint32_t topK = 1;
};

inline bool CheckCommonCocTilingParams(uint32_t rankSize, const CocTilingParams& params)
{
    return rankSize > 0 && params.rankSize > 0 && params.rankSize == rankSize && params.m > 0 && params.k > 0 &&
           params.n > 0 && params.m0 > 0 && params.k0 > 0 && params.n0 > 0;
}

inline bool CheckMoeCocTilingParams(const CocTilingParams& params)
{
    return params.epSize > 0 && params.expertNum > 0 && params.expertNum % params.epSize == 0 && params.topK > 0;
}

struct COCMatMulInfo {
    int64_t m;
    int64_t k;
    int64_t n;
};

struct KernelParams {
    uint8_t* ptrA;
    uint8_t* ptrB;
    uint8_t* ptrC;
    uint8_t** customPtrs;
    uint32_t customCount;

    KernelParams() : ptrA(nullptr), ptrB(nullptr), ptrC(nullptr), customPtrs(nullptr), customCount(0) {}

    // The custom pointer array is owned by this object; copies would share its lifetime.
    KernelParams(const KernelParams&) = delete;
    KernelParams& operator=(const KernelParams&) = delete;

    KernelParams(KernelParams&& other) noexcept
        : ptrA(other.ptrA),
          ptrB(other.ptrB),
          ptrC(other.ptrC),
          customPtrs(other.customPtrs),
          customCount(other.customCount)
    {
        other.ptrA = nullptr;
        other.ptrB = nullptr;
        other.ptrC = nullptr;
        other.customPtrs = nullptr;
        other.customCount = 0;
    }

    KernelParams& operator=(KernelParams&&) = delete;

    template <typename... Args>
    void SetKernelParams(uint8_t* a, uint8_t* b, uint8_t* c, Args... args)
    {
        ptrA = a;
        ptrB = b;
        ptrC = c;

        const int argsCount = sizeof...(args);
        if (customPtrs != nullptr) {
            ACL_CHECK(aclrtFreeHost(customPtrs));
            customPtrs = nullptr;
        }
        customCount = argsCount;

        if (argsCount > 0) {
            ACL_CHECK(aclrtMallocHost((void**)(&customPtrs), argsCount * sizeof(uint8_t*)));
            uint8_t* pointers[] = {args...};

            for (size_t i = 0; i < argsCount; ++i) {
                customPtrs[i] = pointers[i];
            }
        }
    }

    ~KernelParams()
    {
        if (customPtrs != nullptr) {
            ACL_CHECK(aclrtFreeHost(customPtrs));
        }
    }
};

inline void FreeDeviceSpace(KernelParams& params)
{
    if (params.ptrA != nullptr) {
        ACL_CHECK(aclrtFree(params.ptrA));
    }
    if (params.ptrB != nullptr) {
        ACL_CHECK(aclrtFree(params.ptrB));
    }
    if (params.ptrC != nullptr) {
        ACL_CHECK(aclrtFree(params.ptrC));
    }
    params.ptrA = nullptr;
    params.ptrB = nullptr;
    params.ptrC = nullptr;
    for (uint32_t i = 0; i < params.customCount; i++) {
        if (params.customPtrs[i] != nullptr) {
            ACL_CHECK(aclrtFree(params.customPtrs[i]));
        }
    }
}

// kernel缩写
const std::map<std::string, CocCommType> CommTypeMap = {
    {"mmar", CocCommType::MATMUL_ALLREDUCE},
    {"agmm", CocCommType::ALLGATHER_MATMUL},
    {"a5agmm", CocCommType::ASCEND950_ALLGATHER_MATMUL},
    {"mmrs", CocCommType::MATMUL_REDUCE_SCATTER},
    {"a5mmrs", CocCommType::ASCEND950_MATMUL_REDUCE_SCATTER},
    {"agmmwg", CocCommType::ALLGATHER_MATMUL_WITH_GATHER_RESULT},
    {"gmmata", CocCommType::GROUPED_MATMUL_ALLTOALLV},
    {"gmmatatla", CocCommType::GROUPED_MATMUL_ALLTOALLV_TLA},
    {"a5gmmata", CocCommType::ASCEND950_GROUPED_MATMUL_ALLTOALLV},
    {"atavgmm", CocCommType::ALLTOALLV_GROUPED_MATMUL},
    {"agmmrdma", CocCommType::ALLGATHER_MATMUL_RDMA},
    {"atavgmmv2", CocCommType::ALLTOALLV_GMM_V2},
    {"agmmdq", CocCommType::ALLGATHER_MATMUL_DEQUANT},
    {"agmmdqbs", CocCommType::ALLGATHER_MATMUL_DEQUANT_BIAS},
    {"mmdqrs", CocCommType::MATMUL_DEQUANT_REDUCE_SCATTER_WRITE},
    {"a5mxfp8mmrs", CocCommType::ASCEND950_MXFP8_MATMUL_REDUCE_SCATTER},
    {"a5mxfp8mmata", CocCommType::ASCEND950_MXFP8_MATMUL_ALLTOALL},
    {"agmmrr", CocCommType::ALLGATHER_MATMUL_REMOTE_READ},
    {"dgds", CocCommType::DISPATCH_GMM_DEQUANT_SWIGLU},
    {"a5fp8mxagmm", CocCommType::ASCEND950_FP8_MX_ALLGATHER_MATMUL},
    {"a5fp4mxagmm", CocCommType::ASCEND950_FP4_MX_ALLGATHER_MATMUL},
    {"mxqtag", CocCommType::MX_QUANT_ALLGATHER},
    {"a5fp8gmmata", CocCommType::ASCEND950_FP8_MX_GROUPED_MATMUL_ALLTOALLV},
    {"a5fp4gmmata", CocCommType::ASCEND950_FP4_MX_GROUPED_MATMUL_ALLTOALLV},
    {"a5fp8mxatavgmm", CocCommType::ASCEND950_FP8_MX_ALLTOALLV_GROUPED_MATMUL},
    {"a5fp4mxatavgmm", CocCommType::ASCEND950_FP4_MX_ALLTOALLV_GROUPED_MATMUL},
    {"a5atavgmm", CocCommType::ASCEND950_ALLTOALLV_GROUPED_MATMUL},
    {"a5fp4mmrs", CocCommType::ASCEND950_FP4_MX_MATMUL_REDUCE_SCATTER},
    {"moe", CocCommType::DISPATCH_FFN_COMBINE},
    {"a5moe", CocCommType::ASCEND950_DISPATCH_FFN_COMBINE},
    {"a5dispatch", CocCommType::ASCEND950_DISPATCH},
    {"a5agmmudma", CocCommType::ASCEND950_ALLGATHER_MATMUL_UDMA},
    {"a5mxfp8atammsplitkurma", CocCommType::ASCEND950_MXFP8_ALLTOALL_MATMUL_SPLIT_K_URMA},
    {"agmmsf", ALLGATHER_MATMUL_SHALLOW_FUSION},
    {"mmarsf", MATMUL_ALLREDUCE_SHALLOW_FUSION},
    {"mmrssf", MATMUL_REDUCE_SCATTER_SHALLOW_FUSION},
    {"gmmatasm", GROUPED_MATMUL_ALLTOALLV_SMALL_M},
    // 新增算子继续添加...
};

inline CocCommType GetCommType(const std::string& kernelName)
{
    auto it = CommTypeMap.find(kernelName);
    if (it != CommTypeMap.end()) {
        return it->second;
    }
    return CocCommType::UNKNOWN;
}

const std::map<CocCommType, std::string> CommTypeOpNameMap = {
    {ALLGATHER_MATMUL_SHALLOW_FUSION, "AllGatherMatmulShallowFusion"},
    {MATMUL_ALLREDUCE_SHALLOW_FUSION, "MatmulAllReduceShallowFusion"},
    {MATMUL_REDUCE_SCATTER_SHALLOW_FUSION, "MatmulReduceScatterShallowFusion"},
    {GROUPED_MATMUL_ALLTOALLV_SMALL_M, "GroupedMatmulAlltoAllvSmallM"},
    {MATMUL_ALLREDUCE, "MatmulAllReduce"},
    {ALLGATHER_MATMUL, "AllGatherMatmul"},
    {ASCEND950_ALLGATHER_MATMUL, "Ascend950AllGatherMatmul"},
    {MATMUL_REDUCE_SCATTER, "MatmulReduceScatter"},
    {ASCEND950_MATMUL_REDUCE_SCATTER, "Ascend950MatmulReduceScatter"},
    {ALLGATHER_MATMUL_WITH_GATHER_RESULT, "AllGatherMatmulWithGatherResult"},
    {GROUPED_MATMUL_ALLTOALLV, "GroupedMatmulAllToAllV"},
    {GROUPED_MATMUL_ALLTOALLV_TLA, "GroupedMatmulAllToAllVTla"},
    {ASCEND950_GROUPED_MATMUL_ALLTOALLV, "Ascend950GroupedMatmulAllToAllV"},
    {ALLTOALLV_GROUPED_MATMUL, "AllToAllVGroupedMatmul"},
    {ALLGATHER_MATMUL_RDMA, "AllGatherMatmulRdma"},
    {ALLTOALLV_GMM_V2, "AllToAllVGMMV2"},
    {ALLGATHER_MATMUL_DEQUANT, "AllGatherMatmulDequant"},
    {ALLGATHER_MATMUL_DEQUANT_BIAS, "AllGatherMatmulDequantBias"},
    {ASCEND950_MXFP8_MATMUL_REDUCE_SCATTER, "Ascend950MxFp8MatmulReduceScatter"},
    {ASCEND950_MXFP8_MATMUL_ALLTOALL, "Ascend950MxFp8MatmulAllToAll"},
    {ALLGATHER_MATMUL_REMOTE_READ, "AllGatherMatmulRemoteRead"},
    {DISPATCH_GMM_DEQUANT_SWIGLU, "DispatchGmmDequantSwiglu"},
    {ASCEND950_FP8_MX_ALLGATHER_MATMUL, "Ascend950Fp8MxAllGatherMatmul"},
    {ASCEND950_FP4_MX_ALLGATHER_MATMUL, "Ascend950Fp4MxAllGatherMatmul"},
    {MX_QUANT_ALLGATHER, "MxQuantAllGather"},
    {MATMUL_DEQUANT_REDUCE_SCATTER_WRITE, "MatmulDequantReduceScatterWrite"},
    {ASCEND950_FP8_MX_GROUPED_MATMUL_ALLTOALLV, "Ascend950Fp8MxGroupedMatmulAllToAllV"},
    {ASCEND950_FP4_MX_GROUPED_MATMUL_ALLTOALLV, "Ascend950Fp4MxGroupedMatmulAllToAllV"},
    {ASCEND950_FP8_MX_ALLTOALLV_GROUPED_MATMUL, "Ascend950Fp8MxAllToAllVGroupedMatmul"},
    {ASCEND950_FP4_MX_ALLTOALLV_GROUPED_MATMUL, "Ascend950Fp4MxAllToAllVGroupedMatmul"},
    {ASCEND950_ALLTOALLV_GROUPED_MATMUL, "Ascend950AllToAllVGroupedMatmul"},
    {ASCEND950_FP4_MX_MATMUL_REDUCE_SCATTER, "Ascend950Fp4MxMatmulReduceScatter"},
    {DISPATCH_FFN_COMBINE, "DispatchFFNCombine"},
    {ASCEND950_DISPATCH_FFN_COMBINE, "Ascend950DispatchFFNCombine"},
    {ASCEND950_ALLGATHER_MATMUL_UDMA, "Ascend950AllGatherMatmulUdma"},
    {ASCEND950_MXFP8_ALLTOALL_MATMUL_SPLIT_K_URMA, "Ascend950MxFp8AllToAllMatmulSplitKUrma"},
    {ASCEND950_DISPATCH, "Ascend950Dispatch"},
};

inline int32_t CeilDev(int32_t num, int32_t div)
{
    if (div == 0) {
        return 0;
    }
    return (num + div - 1) / div;
}

#endif // INFO_H
