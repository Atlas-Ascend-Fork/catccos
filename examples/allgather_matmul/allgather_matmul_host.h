

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLGATHER_MATMUL_HOST_H
#define ALLGATHER_MATMUL_HOST_H

#include <memory>
#include <stdexcept>

#include "operator_registry.h"
#include "padding.h"

inline bool IsAgmmNeedPaddingB(const CocTilingParams& cocTiling)
{
    constexpr uint32_t alignByByte = 512;
    constexpr uint32_t alignByElement = alignByByte / sizeof(int16_t);
    return IsNeedPadding(cocTiling.k, cocTiling.n, cocTiling.transB, alignByElement);
}

inline size_t GetAgmmWorkspaceSize(const CocTilingParams& cocTiling)
{
    if (!IsAgmmNeedPaddingB(cocTiling)) {
        return 0;
    }
    if (cocTiling.m0 == 128) {
        return GetWorkspaceLen(cocTiling.k, cocTiling.n, TILE_SHAPE_256, TILE_SHAPE_256) * sizeof(int16_t);
    }
    return GetWorkspaceLen(cocTiling.k, cocTiling.n, TILE_SHAPE_256, TILE_SHAPE_128) * sizeof(int16_t);
}

class AllGatherMatmulOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(
        KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        size_t aSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(fp16_t);
        size_t bSize = static_cast<size_t>(cocTiling.k) * cocTiling.n * sizeof(fp16_t);
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.rankSize * cocTiling.n * sizeof(fp16_t);

        auto deviceDeleter = [](uint8_t* ptr) {
            if (ptr != nullptr)
                ACL_CHECK(aclrtFree(ptr));
        };
        auto hostDeleter = [](uint8_t* ptr) {
            if (ptr != nullptr)
                ACL_CHECK(aclrtFreeHost(ptr));
        };
        using DevicePtr = std::unique_ptr<uint8_t, decltype(deviceDeleter)>;
        using HostPtr = std::unique_ptr<uint8_t, decltype(hostDeleter)>;
        auto allocateDevice = [&](size_t bytes) {
            uint8_t* ptr = nullptr;
            aclError status = aclrtMalloc(reinterpret_cast<void**>(&ptr), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            if (status != ACL_SUCCESS || ptr == nullptr)
                throw std::runtime_error("AllGatherMatmul device allocation failed");
            return DevicePtr(ptr, deviceDeleter);
        };
        auto allocateHost = [&](size_t bytes) {
            uint8_t* ptr = nullptr;
            aclError status = aclrtMallocHost(reinterpret_cast<void**>(&ptr), bytes);
            if (status != ACL_SUCCESS || ptr == nullptr)
                throw std::runtime_error("AllGatherMatmul pinned host allocation failed");
            return HostPtr(ptr, hostDeleter);
        };
        auto copyToDevice = [](uint8_t* dst, size_t size, const void* src) {
            if (aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
                throw std::runtime_error("AllGatherMatmul host-to-device copy failed");
        };

        auto aDevice = allocateDevice(aSize);
        if (dataFile != "") {
            auto aHost = allocateHost(aSize);
            ReadFileOrThrow(dataFile + "/rank_" + std::to_string(rankId) + "_a.bin", aHost.get(), aSize);
            copyToDevice(aDevice.get(), aSize, aHost.get());
        } else {
            std::vector<fp16_t> matrixA(static_cast<size_t>(cocTiling.m) * cocTiling.k, 1);
            copyToDevice(aDevice.get(), aSize, matrixA.data());
        }

        auto bDevice = allocateDevice(bSize);
        if (dataFile != "") {
            auto bHost = allocateHost(bSize);
            ReadFileOrThrow(dataFile + "/rank_" + std::to_string(rankId) + "_b.bin", bHost.get(), bSize);
            copyToDevice(bDevice.get(), bSize, bHost.get());
        } else {
            std::vector<fp16_t> matrixB(static_cast<size_t>(cocTiling.k) * cocTiling.n, 1);
            copyToDevice(bDevice.get(), bSize, matrixB.data());
        }

        auto cDevice = allocateDevice(cSize);

        params.SetKernelParams(aDevice.get(), bDevice.get(), cDevice.get());
        aDevice.release();
        bDevice.release();
        cDevice.release();

        return;
    }

    void WriteResultFile(
        const KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.rankSize * cocTiling.n * sizeof(fp16_t);

        uint8_t* cDevice = params.ptrC;
        uint8_t* cHost;
        ACL_CHECK(aclrtMallocHost((void**)(&cHost), cSize));
        ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output.bin", cHost, cSize);

        ACL_CHECK(aclrtFreeHost(cHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams& cocTiling) override { return GetAgmmWorkspaceSize(cocTiling); }

    CocCommType GetActualKernelType(const CocTilingParams& cocTiling) override { return CocCommType::ALLGATHER_MATMUL; }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override
    {
        auto blockCount = MAX_BLOCK_COUNT;
        uint32_t kLoops = CeilDev(cocTiling.k, cocTiling.k0);
        int32_t maxPeerMemPerRank = SHMEM_BUFF_BYTES / INPUT_DTYPE / rankSize / blockCount;
        if (cocTiling.commInterval * cocTiling.m0 * cocTiling.k0 * kLoops >= maxPeerMemPerRank) {
            return false;
        }
        return true;
    }
};

// 注册这个算子
REGISTER_OPERATOR("AllGatherMatmul", AllGatherMatmulOperator);

#endif // ALLGATHER_MATMUL_HOST_H
