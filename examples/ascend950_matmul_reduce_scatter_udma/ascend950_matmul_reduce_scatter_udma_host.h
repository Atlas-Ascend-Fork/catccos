

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_MATMUL_REDUCE_SCATTER_UDMA_HOST_H
#define ASCEND950_MATMUL_REDUCE_SCATTER_UDMA_HOST_H

#include "operator_registry.h"

class Ascend950MatmulReduceScatterUdmaOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(
        KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        size_t aSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(fp16_t);
        size_t bSize = static_cast<size_t>(cocTiling.k) * cocTiling.n * sizeof(fp16_t);
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.n * sizeof(fp16_t);
        size_t cSizeScatter = cSize / cocTiling.rankSize;

        auto aicorenum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        size_t blockBytes = cocTiling.m0 * cocTiling.n0 * sizeof(fp16_t);
        uint32_t blockPerCommInRank = aicorenum * cocTiling.commInterval / cocTiling.rankSize;
        size_t sizePercommPerRank = blockPerCommInRank * blockBytes;
        size_t scratchSize = cocTiling.rankSize * sizePercommPerRank * WORKSPACE_STAGES;

        uint8_t* aDevice;
        ACL_CHECK(aclrtMalloc((void**)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t* aHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void**)(&aHost), aSize));
            ReadFileOrThrow(dataFile + "/rank_" + std::to_string(rankId) + "_a.bin", aHost, aSize);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<fp16_t> matrixA(static_cast<size_t>(cocTiling.m) * cocTiling.k, 1);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, matrixA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t* bDevice;
        ACL_CHECK(aclrtMalloc((void**)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t* bHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void**)(&bHost), bSize));
            ReadFileOrThrow(dataFile + "/rank_" + std::to_string(rankId) + "_b.bin", bHost, bSize);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<fp16_t> matrixB(static_cast<size_t>(cocTiling.k) * cocTiling.n, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t* cDevice;
        ACL_CHECK(aclrtMalloc((void**)(&cDevice), cSizeScatter, ACL_MEM_MALLOC_HUGE_FIRST));

        uint8_t* cScratchDevice;
        ACL_CHECK(aclrtMalloc((void**)(&cScratchDevice), scratchSize, ACL_MEM_MALLOC_HUGE_FIRST));

        params.SetKernelParams(aDevice, bDevice, cDevice, cScratchDevice);

        if (dataFile != "") {
            ACL_CHECK(aclrtFreeHost(aHost));
            ACL_CHECK(aclrtFreeHost(bHost));
        }

        return;
    }

    void WriteResultFile(
        const KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.n * sizeof(fp16_t);
        size_t cSizeScatter = cSize / cocTiling.rankSize;

        uint8_t* cDevice = params.ptrC;
        uint8_t* cHost;
        ACL_CHECK(aclrtMallocHost((void**)(&cHost), cSizeScatter));
        ACL_CHECK(aclrtMemcpy(cHost, cSizeScatter, cDevice, cSizeScatter, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output.bin", cHost, cSizeScatter, rankId * cSizeScatter);

        ACL_CHECK(aclrtFreeHost(cHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams& cocTiling) override { return 0; }

    CocCommType GetActualKernelType(const CocTilingParams& cocTiling) override
    {
        return CocCommType::ASCEND950_MATMUL_REDUCE_SCATTER_UDMA;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override
    {
        if (rankSize == 0 || cocTiling.rankSize != rankSize || cocTiling.m % rankSize != 0) {
            return false;
        }

        auto blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        int64_t product = static_cast<int64_t>(blockNum) * cocTiling.commInterval;

        size_t blockBytes = cocTiling.m0 * cocTiling.n0 * sizeof(fp16_t);
        size_t blockPerCommInRank = product / cocTiling.rankSize;
        size_t sizePercommPerRank = blockPerCommInRank * blockBytes;
        size_t footprint = cocTiling.rankSize * sizePercommPerRank * WORKSPACE_STAGES;

        if (product % rankSize != 0 || footprint > static_cast<size_t>(SHMEM_BUFF_BYTES)) {
            return false;
        }
        return true;
    }
};

// 注册这个算子
REGISTER_OPERATOR("Ascend950MatmulReduceScatterUdma", Ascend950MatmulReduceScatterUdmaOperator);

#endif // ASCEND950_MATMUL_REDUCE_SCATTER_UDMA_HOST_H
