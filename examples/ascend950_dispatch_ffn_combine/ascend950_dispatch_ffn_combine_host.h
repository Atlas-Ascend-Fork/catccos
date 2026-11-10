

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_DISPATCH_FFN_COMBINE_HOST_H
#define ASCEND950_DISPATCH_FFN_COMBINE_HOST_H

#include "operator_registry.h"
#include "catccos/detail/ascend950_moe_ready.hpp"

class Ascend950DispatchFFNCombineOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(
        KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        uint32_t expertPerRank = cocTiling.expertNum / cocTiling.epSize;
        uint32_t EP = cocTiling.rankSize;
        uint32_t k2 = cocTiling.n / 2;
        uint32_t n2 = cocTiling.k;

        uint32_t maxOutputSize = cocTiling.m * cocTiling.topK * cocTiling.rankSize;
        size_t aSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(fp16_t);
        size_t bSize = static_cast<size_t>(cocTiling.k) * cocTiling.n * expertPerRank;
        size_t b2Size = static_cast<size_t>(k2) * n2 * expertPerRank;
        size_t scaleK = RoundUp<size_t>(CeilDev(cocTiling.k, 32), 2);
        size_t scaleK2 = RoundUp<size_t>(CeilDev(k2, 32), 2);
        size_t bScaleSize = scaleK * cocTiling.n * expertPerRank;
        size_t b2ScaleSize = scaleK2 * n2 * expertPerRank;
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(fp16_t);

        size_t expertIdxSize = cocTiling.m * cocTiling.topK * sizeof(int32_t);

        uint8_t* aDevice;
        ACL_CHECK(aclrtMalloc((void**)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t* aHost;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void**)(&aHost), aSize));
            ReadFileOrThrow(dataFile + "/in_routing_matrix_a_" + std::to_string(rankId) + ".bin", aHost, aSize);
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
            ReadFileOrThrow(dataFile + "/in_gmm_matrix_b_" + std::to_string(rankId) + ".bin", bHost, bSize);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixB(static_cast<size_t>(cocTiling.k) * cocTiling.n * expertPerRank, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t* b2Device;
        ACL_CHECK(aclrtMalloc((void**)(&b2Device), b2Size, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t* b2Host;
        if (dataFile != "") {
            ACL_CHECK(aclrtMallocHost((void**)(&b2Host), b2Size));
            ReadFileOrThrow(dataFile + "/in_gmm_matrix_b2_" + std::to_string(rankId) + ".bin", b2Host, b2Size);
            ACL_CHECK(aclrtMemcpy(b2Device, b2Size, b2Host, b2Size, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<int8_t> matrixB2(static_cast<size_t>(k2) * n2 * expertPerRank, 1);
            ACL_CHECK(aclrtMemcpy(b2Device, b2Size, matrixB2.data(), b2Size, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t* bScaleDevice;
        uint8_t* bScaleHost;
        ACL_CHECK(aclrtMalloc((void**)(&bScaleDevice), bScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMallocHost((void**)(&bScaleHost), bScaleSize));
        ReadFileOrThrow(dataFile + "/in_gmm_matrix_b_scale_" + std::to_string(rankId) + ".bin", bScaleHost, bScaleSize);
        ACL_CHECK(aclrtMemcpy(bScaleDevice, bScaleSize, bScaleHost, bScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));

        uint8_t* b2ScaleDevice;
        uint8_t* b2ScaleHost;
        ACL_CHECK(aclrtMalloc((void**)(&b2ScaleDevice), b2ScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMallocHost((void**)(&b2ScaleHost), b2ScaleSize));
        ReadFileOrThrow(
            dataFile + "/in_gmm_matrix_b2_scale_" + std::to_string(rankId) + ".bin", b2ScaleHost, b2ScaleSize);
        ACL_CHECK(aclrtMemcpy(b2ScaleDevice, b2ScaleSize, b2ScaleHost, b2ScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));

        uint8_t* expertIdxDevice;
        uint8_t* expertIdxHost;
        ACL_CHECK(aclrtMalloc((void**)(&expertIdxDevice), expertIdxSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMallocHost((void**)(&expertIdxHost), expertIdxSize));
        ReadFileOrThrow(
            dataFile + "/in_routing_expert_idx_" + std::to_string(rankId) + ".bin", expertIdxHost, expertIdxSize);
        ACL_CHECK(aclrtMemcpy(expertIdxDevice, expertIdxSize, expertIdxHost, expertIdxSize, ACL_MEMCPY_HOST_TO_DEVICE));

        uint8_t* probsDevice;
        uint8_t* probsHost;
        size_t probsSize = cocTiling.m * cocTiling.topK * sizeof(float32_t);
        ACL_CHECK(aclrtMalloc((void**)(&probsDevice), probsSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMallocHost((void**)(&probsHost), probsSize));
        ReadFileOrThrow(dataFile + "/in_gather_gate_weight_" + std::to_string(rankId) + ".bin", probsHost, probsSize);
        ACL_CHECK(aclrtMemcpy(probsDevice, probsSize, probsHost, probsSize, ACL_MEMCPY_HOST_TO_DEVICE));

        uint8_t* cDevice;
        ACL_CHECK(aclrtMalloc((void**)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(cDevice, cSize, 0, cSize));

        // uint8_t *outputDevice;
        // size_t outputSize = cocTiling.m * cocTiling.k * sizeof(fp16_t);
        // ACL_CHECK(aclrtMalloc((void **)(&outputDevice), outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
        // ACL_CHECK(aclrtMemset(outputDevice, outputSize, 0, outputSize));

        params.SetKernelParams(
            aDevice, bDevice, cDevice, b2Device, expertIdxDevice, probsDevice, bScaleDevice, b2ScaleDevice);

        if (dataFile != "") {
            ACL_CHECK(aclrtFreeHost(aHost));
            ACL_CHECK(aclrtFreeHost(bHost));
            ACL_CHECK(aclrtFreeHost(b2Host));
        }
        ACL_CHECK(aclrtFreeHost(expertIdxHost));
        ACL_CHECK(aclrtFreeHost(probsHost));
        ACL_CHECK(aclrtFreeHost(bScaleHost));
        ACL_CHECK(aclrtFreeHost(b2ScaleHost));

        return;
    }

    void WriteResultFile(
        const KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        size_t cSize = static_cast<size_t>(cocTiling.m) * cocTiling.k * sizeof(fp16_t);
        uint8_t* cDevice = params.ptrC;
        uint8_t* cHost;
        ACL_CHECK(aclrtMallocHost((void**)(&cHost), cSize));
        ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/output_" + std::to_string(rankId) + ".bin", cHost, cSize);

        ACL_CHECK(aclrtFreeHost(cHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams& cocTiling) override
    {
        uint32_t expertPerRank = cocTiling.expertNum / cocTiling.epSize;
        uint32_t EP = cocTiling.rankSize;
        size_t maxOutputSize = static_cast<size_t>(cocTiling.m) * cocTiling.rankSize * cocTiling.topK;
        size_t metaInfoSize = static_cast<size_t>(EP) * EP * expertPerRank * sizeof(int32_t);
        size_t workspaceSize =
            RoundUp<size_t>(RoundUp<size_t>(cocTiling.m, 256) * cocTiling.topK * sizeof(int32_t), 512) +
            RoundUp<size_t>(maxOutputSize * cocTiling.k * sizeof(fp16_t), 512) + RoundUp<size_t>(metaInfoSize, 512) +
            RoundUp<size_t>(maxOutputSize * cocTiling.k, 512) +
            RoundUp<size_t>(maxOutputSize * RoundUp<size_t>(CeilDev(cocTiling.k, 32), 2), 512) +
            Catccos::Detail::Ascend950MoeReady::SwigluRegionBytes(
                maxOutputSize * (cocTiling.n / 2) * sizeof(fp16_t), expertPerRank) +
            RoundUp<size_t>(maxOutputSize * (cocTiling.n / 2), 512) +
            RoundUp<size_t>(maxOutputSize * RoundUp<size_t>(CeilDev(cocTiling.n / 2, 32), 2), 512) +
            Catccos::Detail::Ascend950MoeReady::DispatchRegionBytes(
                maxOutputSize * cocTiling.k * sizeof(fp16_t), EP, expertPerRank) +
            RoundUp<size_t>(metaInfoSize, 512);
        return workspaceSize;
    }

    CocCommType GetActualKernelType(const CocTilingParams& cocTiling) override
    {
        return CocCommType::ASCEND950_DISPATCH_FFN_COMBINE;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override
    {
        if (!CheckMoeCocTilingParams(cocTiling)) {
            return false;
        }
        if (cocTiling.topK > cocTiling.expertNum) {
            std::cerr << "TopK must not exceed the global expert count" << std::endl;
            return false;
        }
        if (rankSize != cocTiling.epSize) {
            std::cerr << "Ascend950DispatchFFNCombine requires rankSize == epSize" << std::endl;
            return false;
        }
        auto blockCount = MAX_BLOCK_COUNT;
        uint32_t kLoop = CeilDev(cocTiling.k, cocTiling.k0);
        int32_t maxPeerMemPerRank = IPC_BUFF_MAX_SIZE / INPUT_DTYPE / rankSize / blockCount;
        if (cocTiling.commInterval * cocTiling.m0 * cocTiling.k0 * kLoop >= maxPeerMemPerRank) {
            return false;
        }
        if (cocTiling.k % 256 != 0 || cocTiling.n % 512 != 0) {
            std::cerr << "MXFP8 FFN requires K % 256 == 0 and N % 512 == 0" << std::endl;
            return false;
        }
        size_t expandedM = static_cast<size_t>(cocTiling.m) * cocTiling.topK;
        size_t scaleK = RoundUp<size_t>(CeilDev(cocTiling.k, 32), 2);
        size_t symmetricBytes =
            RoundUp<size_t>(expandedM * cocTiling.k * sizeof(fp16_t), 512) +
            RoundUp<size_t>(expandedM * RoundUp<size_t>(cocTiling.k + scaleK, 32), 512) +
            RoundUp<size_t>(
                static_cast<size_t>(rankSize) * rankSize * (cocTiling.expertNum / rankSize) * sizeof(int32_t), 512);
        if (symmetricBytes > SHMEM_BUFF_BYTES) {
            std::cerr << "Packed input and BF16 return regions exceed symmetric buffer capacity" << std::endl;
            return false;
        }
        return true;
    }
};

REGISTER_OPERATOR("Ascend950DispatchFFNCombine", Ascend950DispatchFFNCombineOperator);

#endif // ASCEND950_DISPATCH_FFN_COMBINE_HOST_H
