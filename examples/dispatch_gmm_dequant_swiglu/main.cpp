

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dispatch_gmm_dequant_swiglu_device.h"
#include "dispatch_gmm_dequant_swiglu_host.h"

using namespace AscendC;
using namespace Catccos;

using ElementA0 = bfloat16_t;
using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = half;
using ElementD = bfloat16_t;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;

static uint32_t gNpuNum = 8;

int main(int argc, char** argv)
{
    if (argc < 12) {
        std::cerr << "Usage: " << argv[0]
                  << " rank_size rank_id ip_port device_id m k n expert_per_rank data_type weight_nz trans_b"
                  << std::endl;
        return -1;
    }

    int status = ACLSHMEM_SUCCESS;
    int rankSize = atoi(argv[1]);
    uint32_t rankId = atoi(argv[2]);
    std::string ipPort = argv[3];
    int32_t deviceId = atoi(argv[4]) + rankId % gNpuNum;

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipPort
              << std::endl;

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t default_flag_uid;
    set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, ipPort.c_str(), &attributes, &default_flag_uid);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    // Prepare FFTS address
    uint64_t fftsAddr = shmemx_get_ffts_config();

    uint32_t m = atoi(argv[5]);
    uint32_t k = atoi(argv[6]);
    uint32_t n = atoi(argv[7]);
    uint32_t EP = rankSize;
    uint32_t expertPerRank = atoi(argv[8]);
    uint32_t dataType = atoi(argv[9]);
    uint32_t weightNz = atoi(argv[10]);
    uint32_t transB = atoi(argv[11]);

    uint32_t m0 = 10;
    uint32_t k0 = 10;
    uint32_t n0 = 10;
    uint32_t swizzleDirect = 1;
    uint32_t swizzleOffset = 2;
    uint32_t ubMoveNum = 8192;
    uint32_t pValue = 1;
    uint32_t commNpuSplit = rankSize;
    uint32_t commDataSplit = 1;
    uint32_t lenPerLoop = m0 * n0 / 2;
    uint32_t topK = 4;
    int64_t tilingKey = 0;
    uint32_t maxOutputSize = m * topK * rankSize;

    CocTilingParams cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    COCMatMulInfo info{int64_t(m), int64_t(k), int64_t(n)};
    cocTiling.m0 = M0;
    cocTiling.n0 = N0;
    cocTiling.k0 = K0;
    cocTiling.commTileM = 64;
    cocTiling.commInterval = 10;
    cocTiling.commNpuSplit = rankSize;
    cocTiling.commDataSplit = 1;
    cocTiling.commBlockM = 64;
    cocTiling.rankSize = rankSize;
    cocTiling.epSize = EP;
    cocTiling.expertNum = EP * expertPerRank;
    cocTiling.topK = topK;

    optiling::MoeInitRoutingQuantV2TilingBase moeInitRoutingQuantV2TilingBase;
    int64_t inuptXDtypeSize = sizeof(ElementA0);
    int64_t scaleDim0 = 0;
    int64_t ubSize = 196352;
    int64_t expertCapacity = 0;
    int64_t activeNum = 0;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t quantMode = 1;
    uint32_t aivNum = 2 * BLOCK_NUM;
    moeInitRoutingQuantV2TilingBase.DoTiling(
        m, k, topK, expertCapacity, cocTiling.expertNum, activeNum, dropPadMode, expertTokensCountOrCumsumFlag,
        expertTokensBeforeCapacityFlag, inuptXDtypeSize, quantMode, scaleDim0, aivNum, ubSize);

    size_t initRoutingWorkspace = moeInitRoutingQuantV2TilingBase.workspaceSize_;

    MoeInitRoutingQuantV2Tiling moeTiling{
        moeInitRoutingQuantV2TilingBase.quantTilingData, moeInitRoutingQuantV2TilingBase.tilingKey_};

    printf("tiling key: %d\n", moeInitRoutingQuantV2TilingBase.tilingKey_);

    auto op = OperatorRegistry::Instance().CreateOperator("DispatchGmmDequantSwiglu");
    if (!op) {
        std::cout << "Operator DispatchGmmDequantSwiglu not found!" << std::endl;
        return -1;
    }
    if (!op->ValidateCocTilingParams(rankSize, cocTiling)) {
        std::cerr << "Invalid CocTilingParams" << std::endl;
        return -1;
    }

    KernelParams kernelParams;
    op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, "./output");

    void* symmPtr = shmem_malloc(SHMEM_BUFF_BYTES);
    uint8_t* symmetricPtr = (uint8_t*)symmPtr;

    size_t allToAllVGmmWorkspace = op->GetWorkspaceSize(cocTiling);
    size_t expandedRowIdxSize = (cocTiling.m + 255) / 256 * 256 * cocTiling.topK * sizeof(int32_t);
    auto workSpaceSize = expandedRowIdxSize + std::max(allToAllVGmmWorkspace, initRoutingWorkspace);

    uint8_t* workspaceDevice{nullptr};
    if (workSpaceSize > 0) {
        ACL_CHECK(aclrtMalloc((void**)(&workspaceDevice), workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    uint8_t* aPtr = kernelParams.ptrA;
    uint8_t* bPtr = kernelParams.ptrB;
    uint8_t* cPtr = kernelParams.ptrC;
    uint8_t* scalePtr = kernelParams.customPtrs[0];
    uint8_t* expertIdxPtr = kernelParams.customPtrs[1];

    ACL_CHECK(aclrtSynchronizeStream(stream));

    for (int i = 0; i < 1; i++) {
        DispatchGmmDequantSwiglu<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementD, LayoutD>
            <<<BLOCK_NUM, nullptr, stream>>>(
                fftsAddr, aPtr, bPtr, cPtr, scalePtr, expertIdxPtr, workspaceDevice, symmetricPtr, cocTiling,
                moeTiling);
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));

    op->WriteResultFile(kernelParams, cocTiling, rankId, "./output");

    {
        uint8_t* tokensPerExpertDevice = symmetricPtr + SHMEM_BUFF_BYTES - 2 * 1024 * 1024UL;
        uint8_t* tokensPerExpertHost;
        int32_t size = cocTiling.epSize * cocTiling.epSize * expertPerRank;
        ACL_CHECK(aclrtMallocHost((void**)(&tokensPerExpertHost), size * sizeof(int32_t)));
        ACL_CHECK(aclrtMemcpy(
            tokensPerExpertHost, size * sizeof(int32_t), tokensPerExpertDevice, size * sizeof(int32_t),
            ACL_MEMCPY_DEVICE_TO_HOST));
        printf("tokensPerExpertHost: ========================\n");
        for (int i = 0; i < size; i++) {
            printf("%d ", *(reinterpret_cast<int32_t*>(tokensPerExpertHost) + i));
        }
        printf("=================================\n");
        ACL_CHECK(aclrtFreeHost(tokensPerExpertHost));
    }

    shmem_free(symmPtr);

    FreeDeviceSpace(kernelParams);
    if (workSpaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceDevice));
    }

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}
