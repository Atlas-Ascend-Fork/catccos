
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_MXFP8_ALLTOALL_MATMUL_SPLIT_K_URMA_HOST_H
#define ASCEND950_MXFP8_ALLTOALL_MATMUL_SPLIT_K_URMA_HOST_H

#include "catlass/detail/alignment.hpp"
#include "operator_registry.h"

namespace SplitKUrmaExample {
#if defined(CATCCOS_LOCAL_ATOMIC)
inline constexpr bool IS_LOCAL_ATOMIC = true;
inline constexpr auto OPERATOR_NAME = "Ascend950MxFp8AllToAllMatmulSplitKUrmaLocalAtomic";
inline constexpr auto EXECUTABLE_NAME = "ascend950_mxfp8_alltoall_matmul_split_k_urma_local_atomic";
#else
inline constexpr bool IS_LOCAL_ATOMIC = false;
inline constexpr auto OPERATOR_NAME = "Ascend950MxFp8AllToAllMatmulSplitKUrma";
inline constexpr auto EXECUTABLE_NAME = "ascend950_mxfp8_alltoall_matmul_split_k_urma";
#endif
} // namespace SplitKUrmaExample

class Ascend950MxFp8AllToAllMatmulSplitKUrmaOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(
        KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        constexpr uint32_t MX_SCALE_GROUP_NUM = 32;
        uint32_t chunkM = cocTiling.m / cocTiling.rankSize;
        uint32_t fullK = cocTiling.k * cocTiling.rankSize;
        size_t localScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(cocTiling.k);
        size_t alignedLocalScaleK = RoundUp<2>(localScaleK);
        size_t fullScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(fullK);
        size_t alignedFullScaleK = RoundUp<2>(fullScaleK);

        size_t aSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(int8_t);
        size_t bSize = static_cast<size_t>(fullK) * cocTiling.n * sizeof(int8_t);
        size_t cSize = static_cast<size_t>(chunkM) * cocTiling.n * sizeof(fp16_t);
        size_t aMxScaleSize = static_cast<size_t>(cocTiling.m) * alignedLocalScaleK * sizeof(int8_t);
        size_t bMxScaleSize = alignedFullScaleK * cocTiling.n * sizeof(int8_t);

        uint8_t* aDevice;
        ACL_CHECK(aclrtMalloc((void**)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t* aHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void**)(&aHost), aSize));
            ReadFileOrThrow(dataFile + "/input_a_" + std::to_string(rankId) + ".bin", aHost, aSize);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixA(static_cast<size_t>(cocTiling.m) * cocTiling.k, 1);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, matrixA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t* bDevice;
        ACL_CHECK(aclrtMalloc((void**)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t* bHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void**)(&bHost), bSize));
            ReadFileOrThrow(dataFile + "/input_b_" + std::to_string(rankId) + ".bin", bHost, bSize);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixB(static_cast<size_t>(fullK) * cocTiling.n, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t* aMxScaleDevice;
        ACL_CHECK(aclrtMalloc((void**)(&aMxScaleDevice), aMxScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t* aMxScaleHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void**)(&aMxScaleHost), aMxScaleSize));
            ReadFileOrThrow(dataFile + "/input_a_scale_" + std::to_string(rankId) + ".bin", aMxScaleHost, aMxScaleSize);
            ACL_CHECK(aclrtMemcpy(aMxScaleDevice, aMxScaleSize, aMxScaleHost, aMxScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixAMxScale(static_cast<size_t>(cocTiling.m) * alignedLocalScaleK, 1);
            ACL_CHECK(aclrtMemcpy(
                aMxScaleDevice, aMxScaleSize, matrixAMxScale.data(), aMxScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t* bMxScaleDevice;
        ACL_CHECK(aclrtMalloc((void**)(&bMxScaleDevice), bMxScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t* bMxScaleHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void**)(&bMxScaleHost), bMxScaleSize));
            ReadFileOrThrow(dataFile + "/input_b_scale_" + std::to_string(rankId) + ".bin", bMxScaleHost, bMxScaleSize);
            ACL_CHECK(aclrtMemcpy(bMxScaleDevice, bMxScaleSize, bMxScaleHost, bMxScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixBMxScale(alignedFullScaleK * cocTiling.n, 1);
            ACL_CHECK(aclrtMemcpy(
                bMxScaleDevice, bMxScaleSize, matrixBMxScale.data(), bMxScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t* cDevice;
        ACL_CHECK(aclrtMalloc((void**)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));

        params.SetKernelParams(aDevice, bDevice, cDevice, aMxScaleDevice, bMxScaleDevice);

        if (dataFile != "") {
            ACL_CHECK(aclrtFreeHost(aHost));
            ACL_CHECK(aclrtFreeHost(bHost));
            ACL_CHECK(aclrtFreeHost(aMxScaleHost));
            ACL_CHECK(aclrtFreeHost(bMxScaleHost));
        }
    }

    void WriteResultFile(
        const KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        uint32_t chunkM = cocTiling.m / cocTiling.rankSize;
        size_t cSize = static_cast<size_t>(chunkM) * cocTiling.n * sizeof(fp16_t);

        uint8_t* cDevice = params.ptrC;
        uint8_t* cHost;
        ACL_CHECK(aclrtMallocHost((void**)(&cHost), cSize));
        ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
        std::string outDir = dataFile == "" ? "." : dataFile;
        WriteFile(outDir + "/output_" + std::to_string(rankId) + ".bin", cHost, cSize);

        ACL_CHECK(aclrtFreeHost(cHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams& cocTiling) override { return 0; }

    CocCommType GetActualKernelType(const CocTilingParams& cocTiling) override
    {
        return CocCommType::ASCEND950_MXFP8_ALLTOALL_MATMUL_SPLIT_K_URMA;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override
    {
        constexpr uint32_t MX_SCALE_GROUP_NUM = 32;
        constexpr uint32_t FP8_ELE_NUM_PER_FRACTAL = 512;

        if (rankSize == 0 || cocTiling.m % rankSize != 0 || cocTiling.k0 == 0 || cocTiling.k % cocTiling.k0 != 0 ||
            cocTiling.k % FP8_ELE_NUM_PER_FRACTAL != 0) {
            return false;
        }

        uint64_t commSizeM = static_cast<uint64_t>(cocTiling.commInterval) * cocTiling.m0;
        uint64_t alignedLocalK = RoundUp<FP8_ELE_NUM_PER_FRACTAL>(static_cast<uint64_t>(cocTiling.k));
        uint64_t localScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(static_cast<uint64_t>(cocTiling.k));
        uint64_t alignedLocalScaleK = RoundUp<2>(localScaleK);
        uint64_t aWorkspaceBytes =
            static_cast<uint64_t>(WORKSPACE_STAGES) * rankSize * commSizeM * alignedLocalK * sizeof(int8_t);
        uint64_t scaleWorkspaceBytes =
            static_cast<uint64_t>(WORKSPACE_STAGES) * rankSize * commSizeM * alignedLocalScaleK * sizeof(int8_t);
        if (aWorkspaceBytes + scaleWorkspaceBytes > static_cast<uint64_t>(SHMEM_BUFF_BYTES)) {
            return false;
        }
        return true;
    }
};

REGISTER_OPERATOR(SplitKUrmaExample::OPERATOR_NAME, Ascend950MxFp8AllToAllMatmulSplitKUrmaOperator);

#endif // ASCEND950_MXFP8_ALLTOALL_MATMUL_SPLIT_K_URMA_HOST_H
