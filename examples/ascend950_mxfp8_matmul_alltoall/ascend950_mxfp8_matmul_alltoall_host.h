
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_MXFP8_MATMUL_ALLTOALL_HOST_H
#define ASCEND950_MXFP8_MATMUL_ALLTOALL_HOST_H

#include "operator_registry.h"
#include "catlass/detail/alignment.hpp"

class Ascend950MxFp8MatmulAllToAllOperator : public CatccosOperator {
public:
    static bool CheckProblemShape(uint32_t rankSize, const CocTilingParams& cocTiling, uint32_t tileN)
    {
        if (rankSize == 0 || rankSize > MAX_RANK_SIZE || cocTiling.m == 0 || cocTiling.n == 0 || cocTiling.k == 0 ||
            tileN == 0) {
            return false;
        }
        // Each GEMM tile must fit entirely within one rank's N-slice.
        return cocTiling.n % rankSize == 0 && (cocTiling.n / rankSize) % tileN == 0;
    }

    void AllocateDeviceSpace(
        KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        // A: M x K (full input), B: K x N (shared weight)
        size_t lenA = static_cast<size_t>(cocTiling.m) * cocTiling.k;
        size_t lenB = static_cast<size_t>(cocTiling.k) * cocTiling.n;
        // D: M x N (output after MatMul + AllToAll, assembled from R chunks)
        size_t lenC = static_cast<size_t>(cocTiling.m) * cocTiling.n;
        constexpr uint32_t MX_SCALE_GROUP_NUM = 32;
        size_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(cocTiling.k);
        size_t lenAScale = static_cast<size_t>(cocTiling.m) * mxScaleK;
        size_t lenBScale = static_cast<size_t>(cocTiling.n) * mxScaleK;

        size_t aSize = lenA * sizeof(int8_t);
        size_t bSize = lenB * sizeof(int8_t);
        size_t cSize = lenC * sizeof(fp16_t);
        size_t aScaleSize = lenAScale * sizeof(int8_t);
        size_t bScaleSize = lenBScale * sizeof(int8_t);

        // Allocate and fill matrix A
        uint8_t* aDevice;
        ACL_CHECK(aclrtMalloc((void**)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            uint8_t* aHost;
            ACL_CHECK(aclrtMallocHost((void**)(&aHost), aSize));
            ReadFileOrThrow(dataFile + "/rank_" + std::to_string(rankId) + "_a.bin", aHost, aSize);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtFreeHost(aHost));
        } else {
            std::vector<int8_t> matrixA(lenA, 1);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, matrixA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        // Allocate and fill matrix B
        uint8_t* bDevice;
        ACL_CHECK(aclrtMalloc((void**)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            uint8_t* bHost;
            ACL_CHECK(aclrtMallocHost((void**)(&bHost), bSize));
            ReadFileOrThrow(dataFile + "/rank_" + std::to_string(rankId) + "_b.bin", bHost, bSize);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtFreeHost(bHost));
        } else {
            std::vector<int8_t> matrixB(lenB, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        // Allocate output D
        uint8_t* cDevice;
        ACL_CHECK(aclrtMalloc((void**)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(cDevice, cSize, 0, cSize));

        // Allocate and fill A scale
        uint8_t* aScaleDevice;
        ACL_CHECK(aclrtMalloc((void**)(&aScaleDevice), aScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            uint8_t* aScaleHost;
            ACL_CHECK(aclrtMallocHost((void**)(&aScaleHost), aScaleSize));
            ReadFileOrThrow(dataFile + "/rank_" + std::to_string(rankId) + "_a_scale.bin", aScaleHost, aScaleSize);
            ACL_CHECK(aclrtMemcpy(aScaleDevice, aScaleSize, aScaleHost, aScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtFreeHost(aScaleHost));
        } else {
            std::vector<int8_t> scaleA(lenAScale, 1);
            ACL_CHECK(aclrtMemcpy(aScaleDevice, aScaleSize, scaleA.data(), aScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        // Allocate and fill B scale
        uint8_t* bScaleDevice;
        ACL_CHECK(aclrtMalloc((void**)(&bScaleDevice), bScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (dataFile != "") {
            uint8_t* bScaleHost;
            ACL_CHECK(aclrtMallocHost((void**)(&bScaleHost), bScaleSize));
            ReadFileOrThrow(dataFile + "/rank_" + std::to_string(rankId) + "_b_scale.bin", bScaleHost, bScaleSize);
            ACL_CHECK(aclrtMemcpy(bScaleDevice, bScaleSize, bScaleHost, bScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtFreeHost(bScaleHost));
        } else {
            std::vector<int8_t> scaleB(lenBScale, 1);
            ACL_CHECK(aclrtMemcpy(bScaleDevice, bScaleSize, scaleB.data(), bScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        params.SetKernelParams(aDevice, bDevice, cDevice, aScaleDevice, bScaleDevice);
        return;
    }

    void WriteResultFile(
        const KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        // AllToAll: each rank holds a full M x N output
        size_t lenC = static_cast<size_t>(cocTiling.m) * cocTiling.n;
        size_t cSize = lenC * sizeof(fp16_t);

        uint8_t* cDevice = params.ptrC;
        uint8_t* cHost;
        ACL_CHECK(aclrtMallocHost((void**)(&cHost), cSize));
        ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output_" + std::to_string(rankId) + ".bin", cHost, cSize);

        ACL_CHECK(aclrtFreeHost(cHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams& cocTiling) override { return 0; }

    CocCommType GetActualKernelType(const CocTilingParams& cocTiling) override
    {
        return CocCommType::ASCEND950_MXFP8_MATMUL_ALLTOALL;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override
    {
        // The dynamic launcher falls back to N=128 tiles when an N=256 tile would cross a rank boundary.
        if (!CheckProblemShape(rankSize, cocTiling, TILE_SHAPE_128)) {
            return false;
        }
        auto blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        int64_t product = static_cast<int64_t>(blockNum) * cocTiling.commInterval;

        if (product % rankSize != 0) {
            return false;
        }
        return true;
    }
};

REGISTER_OPERATOR("Ascend950MxFp8MatmulAllToAll", Ascend950MxFp8MatmulAllToAllOperator);

#endif // ASCEND950_MXFP8_MATMUL_ALLTOALL_HOST_H
