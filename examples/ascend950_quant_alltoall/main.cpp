

/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>

#include "quant_alltoall_device.h"
#include "quant_alltoall_host.h"

using namespace AscendC;
using namespace Catccos;

using LayoutInput = Catlass::layout::RowMajor;
using LayoutOutput = Catlass::layout::RowMajor;

using ElementInput = bfloat16_t;
using ElementOutput = hifloat8_t;
using ElementScale = float;

struct Options {
    static constexpr auto HELPER = "Usage: quant_alltoall rank_size rank_id ip_port m n k [device_id_list]\n";

    int rankSize;
    int rankId;
    std::string ipPort;
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::string dataPath;
    std::vector<int> deviceIdList{};
    bool perfMode{false};

    int Parse(int argc, char** argv)
    {
        enum class ArgsIndex {
            RANK_SIZE_INDEX = 1,
            RANK_ID_INDEX,
            IP_PORT_INDEX,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            DATA_PATH_INDEX,
            DEVICE_LIST_INDEX,
            INDEX_MAX
        };

        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "-p") {
                perfMode = true;
            }
        }

        int positionalArgc = 0;
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) != "-p") {
                positionalArgc++;
            }
        }
        if (positionalArgc < static_cast<int>(ArgsIndex::DATA_PATH_INDEX) ||
            positionalArgc >= static_cast<int>(ArgsIndex::INDEX_MAX)) {
            printf(HELPER);
            return -1;
        }

        rankSize = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_SIZE_INDEX)]);
        if (rankSize <= 0 || rankSize > MAX_RANK_SIZE) {
            std::cerr << "rankSize must be in [1, " << MAX_RANK_SIZE << "]" << std::endl;
            return -1;
        }
        rankId = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_ID_INDEX)]);
        if (rankId < 0 || rankId >= rankSize) {
            std::cerr << "rankId must be in [0, rankSize)" << std::endl;
            return -1;
        }
        ipPort = argv[static_cast<int>(ArgsIndex::IP_PORT_INDEX)];
        m = std::atoi(argv[static_cast<int>(ArgsIndex::M_INDEX)]);
        n = std::atoi(argv[static_cast<int>(ArgsIndex::N_INDEX)]);
        k = std::atoi(argv[static_cast<int>(ArgsIndex::K_INDEX)]);
        dataPath = argv[static_cast<int>(ArgsIndex::DATA_PATH_INDEX)];
        if (argc > static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)) {
            char* idListStr = argv[static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)];
            for (char* idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) {
                deviceIdList.push_back(std::atoi(idToken));
            }
        } else {
            for (size_t i = 0; i < rankSize; ++i) {
                deviceIdList.push_back(i);
            }
        }
        if (static_cast<size_t>(rankId) >= deviceIdList.size()) {
            std::cerr << "device_id_list does not contain rankId " << rankId << std::endl;
            return -1;
        }
        return 0;
    }

    std::string GetDataPath() const { return dataPath; }
};

CocTilingParams SetupTilingParams(uint32_t m, uint32_t n, uint32_t k)
{
    CocTilingParams cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    cocTiling.m0 = 32;
    cocTiling.n0 = 256;
    cocTiling.k0 = 256;
    cocTiling.commTileM = 64;
    cocTiling.commInterval = 3;
    cocTiling.commNpuSplit = 1;
    cocTiling.commDataSplit = 20;
    cocTiling.commBlockM = 32;
    return cocTiling;
}

int main(int argc, char** argv)
{
    int status = ACLSHMEM_SUCCESS;
    Options options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }
    int rankSize = options.rankSize;
    int rankId = options.rankId;
    std::string ipPort = options.ipPort;
    int32_t deviceId = options.deviceIdList[rankId];

    CocTilingParams cocTiling = SetupTilingParams(options.m, options.n, options.k);
    if (!QuantAllToAllOperator::CheckProblemShape(rankSize, cocTiling)) {
        std::cerr << "[ERROR] M, N and K must be positive, and M * N * K must be divisible by rankSize\n";
        return 1;
    }

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t default_flag_uid;
    set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, ipPort.c_str(), &attributes, &default_flag_uid);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (status != ACLSHMEM_SUCCESS) {
        ERROR_LOG("aclshmemx_init_attr failed, rankId=%d, status=%d", rankId, status);
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());
        return -1;
    }

    cocTiling.rankSize = rankSize;

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << std::endl;

    auto op = OperatorRegistry::Instance().CreateOperator("QuantAllToAll");
    if (!op) {
        std::cout << "Operator QuantAllToAll not found!" << std::endl;
        return -1;
    }

    KernelParams kernelParams;
    op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, options.GetDataPath());

    void* symmPtr = aclshmem_calloc(1, SHMEM_BUFF_BYTES);
    if (symmPtr == nullptr) {
        ERROR_LOG("aclshmem_calloc failed, rankId=%d", rankId);
        FreeDeviceSpace(kernelParams);
        status = shmem_finalize();
        if (status != ACLSHMEM_SUCCESS) {
            ERROR_LOG("shmem_finalize failed after aclshmem_calloc failure, rankId=%d, status=%d", rankId, status);
        }
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());
        return -1;
    }
    uint8_t* symmetricPtr = (uint8_t*)symmPtr;
    ACL_CHECK(aclrtMemset(symmetricPtr, SHMEM_BUFF_BYTES, 0, SHMEM_BUFF_BYTES));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    aclshmem_barrier_all();
    uint8_t* inputPtr = kernelParams.ptrA;
    uint8_t* scalePtr = kernelParams.ptrB;
    uint8_t* outputPtr = kernelParams.ptrC;
    uint32_t curBlockNum = rankSize;
    constexpr uint32_t KERNEL_ITERATIONS = 5;
    constexpr uint32_t WARM_UP_ITERS = 5;
    constexpr uint32_t PERF_ITERS = 50;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    uint64_t fftsAddr = shmemx_get_ffts_config();
    if (options.perfMode) {
        // Warm up
        std::cout << "[PERF] Warming up (" << WARM_UP_ITERS << " iters)..." << std::endl;
        for (uint32_t i = 0; i < WARM_UP_ITERS; i++) {
            QuantAllToAll<ElementInput, LayoutInput, ElementOutput, LayoutOutput, ElementScale>
                <<<curBlockNum, nullptr, stream>>>(fftsAddr, inputPtr, scalePtr, outputPtr, symmetricPtr, cocTiling, i);
            ACL_CHECK(aclrtSynchronizeStream(stream));
            aclshmem_barrier_all();
        }

        // Timed perf test
        std::cout << "[PERF] Running performance test (" << PERF_ITERS << " iters)..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < PERF_ITERS; i++) {
            QuantAllToAll<ElementInput, LayoutInput, ElementOutput, LayoutOutput, ElementScale>
                <<<curBlockNum, nullptr, stream>>>(fftsAddr, inputPtr, scalePtr, outputPtr, symmetricPtr, cocTiling, i);
            ACL_CHECK(aclrtSynchronizeStream(stream));
            aclshmem_barrier_all();
        }
        auto endTime = std::chrono::high_resolution_clock::now();

        double totalUs = std::chrono::duration<double, std::micro>(endTime - startTime).count();
        double avgUs = totalUs / PERF_ITERS;
        // Data volume: each rankId sends M elements (bf16=2B) and receives M elements (hif8=1B)
        // Total bytes moved per rankId = M*2 (input read) + M*1 (output written) = M*3
        double dataBytes = static_cast<double>(options.m) * 3.0;
        double bandwidthGBs = dataBytes / (avgUs * 1e-6) / 1e9;

        if (rankId == 0) {
            std::printf("[PERF] QuantAllToAll M=%u rankSize=%d\n", options.m, rankSize);
            std::printf("[PERF]   Total time: %.2f us / %d iters\n", totalUs, PERF_ITERS);
            std::printf("[PERF]   Avg latency: %.2f us/iter\n", avgUs);
            std::printf("[PERF]   Bandwidth: %.2f GB/s\n", bandwidthGBs);
        }
    } else {
        std::cout << "Before calling QuantAllToAll kernel " << std::endl;
        for (uint32_t i = 0; i < KERNEL_ITERATIONS; i++) {
            QuantAllToAll<ElementInput, LayoutInput, ElementOutput, LayoutOutput, ElementScale>
                <<<curBlockNum, nullptr, stream>>>(fftsAddr, inputPtr, scalePtr, outputPtr, symmetricPtr, cocTiling, i);
            ACL_CHECK(aclrtSynchronizeStream(stream));
            aclshmem_barrier_all();
        }
        std::cout << "After calling QuantAllToAll kernel " << std::endl;

        op->WriteResultFile(kernelParams, cocTiling, rankId, options.GetDataPath());
        std::printf("rankId %d test finished\n", rankId);
    }
    aclshmem_barrier_all();
    shmem_free(symmPtr);
    FreeDeviceSpace(kernelParams);
    status = shmem_finalize();
    if (status != ACLSHMEM_SUCCESS) {
        ERROR_LOG("shmem_finalize failed after aclshmem_calloc failure, rankId=%d, status=%d", rankId, status);
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    return status == ACLSHMEM_SUCCESS ? 0 : -1;
    return 0;
}
