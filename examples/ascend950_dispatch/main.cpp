/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <tiling/platform/platform_ascendc.h>

#include "ascend950_dispatch_device.h"
#include "ascend950_dispatch_host.h"
#include "info.h"
#include "shmem.h"
#include "shmem_init.h"

using Config = Ascend950DispatchConfig<ElementToken, ElementScale, ElementTopkIdx, ElementTopkW, ElementIdx>;
using DirectConfig =
    Ascend950DispatchDirectConfig<ElementToken, ElementScale, ElementTopkIdx, ElementTopkW, ElementIdx>;
using DeviceOp = Config::Device;
using DirectDeviceOp = DirectConfig::Device;

struct Options {
    static constexpr const char* HELPER = "Usage: ascend950_dispatch rank_size rank_id ip_port num_tokens hidden "
                                          "group_size top_k experts_per_rank comm_interval "
                                          "comm_block_m data_path [device_id_list] "
                                          "[warmup_iterations performance_iterations]\n";
    uint32_t rankSize{0};
    uint32_t rankIdx{0};
    std::string ipPort;
    uint32_t numTokens{0};
    uint32_t hidden{0};
    uint32_t groupSize{0};
    uint32_t topK{0};
    uint32_t expertsPerRank{0};
    uint32_t commInterval{0};
    uint32_t commBlockM{0};
    std::string dataPath;
    std::vector<int32_t> deviceIds;
    uint32_t warmupIterations{0};
    uint32_t performanceIterations{0};

    bool IsPerformanceMode() const { return performanceIterations > 0; }

    bool Parse(int argc, char** argv)
    {
        if (argc != 12 && argc != 13 && argc != 15) {
            std::cerr << HELPER;
            return false;
        }
        rankSize = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));
        rankIdx = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
        ipPort = argv[3];
        numTokens = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10));
        hidden = static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 10));
        groupSize = static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10));
        topK = static_cast<uint32_t>(std::strtoul(argv[7], nullptr, 10));
        expertsPerRank = static_cast<uint32_t>(std::strtoul(argv[8], nullptr, 10));
        commInterval = static_cast<uint32_t>(std::strtoul(argv[9], nullptr, 10));
        commBlockM = static_cast<uint32_t>(std::strtoul(argv[10], nullptr, 10));
        dataPath = argv[11];
        if (argc >= 13) {
            std::stringstream stream(argv[12]);
            std::string item;
            while (std::getline(stream, item, ',')) {
                deviceIds.push_back(std::stoi(item));
            }
        } else {
            for (uint32_t rank = 0; rank < rankSize; ++rank) {
                deviceIds.push_back(rank);
            }
        }
        if (argc == 15) {
            warmupIterations = static_cast<uint32_t>(std::strtoul(argv[13], nullptr, 10));
            performanceIterations = static_cast<uint32_t>(std::strtoul(argv[14], nullptr, 10));
        }
        const bool valid =
            !dataPath.empty() && deviceIds.size() == rankSize && (argc != 15 || performanceIterations > 0);
        if (!valid) {
            std::cerr << "Invalid Dispatch shape or topology.\n" << HELPER;
        }
        return valid;
    }
};

bool CheckAcl(aclError status, const char* operation)
{
    if (status == ACL_ERROR_NONE) {
        return true;
    }
    std::cerr << "[FAILED] " << operation << ", aclError=" << status << std::endl;
    return false;
}

int main(int argc, char** argv)
{
    Options options;
    if (!options.Parse(argc, argv)) {
        return 1;
    }
    const bool useDirect = options.numTokens < 512;

    DispatchCocTilingParams cocTiling{
        options.rankSize, options.rankIdx,        options.numTokens,    options.hidden,     options.groupSize,
        options.topK,     options.expertsPerRank, options.commInterval, options.commBlockM, 0};
    if (!CheckCocTilingParams(cocTiling)) {
        std::cerr << "Invalid Dispatch CocTilingParams.\n" << Options::HELPER;
        return 1;
    }

    DispatchHostData host;
    if (!LoadDispatchHostData(
            host, options.dataPath, options.rankIdx, options.rankSize, options.numTokens, options.hidden, options.topK,
            options.expertsPerRank)) {
        return 1;
    }
    cocTiling.numOutputTokens = host.numOutputTokens;
    if (!CheckCocTilingParams(cocTiling)) {
        std::cerr << "[FAILED] Dispatch output exceeds symmetric receive capacity" << std::endl;
        return 1;
    }
    const int32_t deviceId = options.deviceIds[options.rankIdx];
    aclrtStream stream = nullptr;
    DispatchDeviceData device;
    if (!CheckAcl(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    if (!CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice")) {
        aclFinalize();
        return 1;
    }

    const uint32_t blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t uniqueId;
    const int32_t attrStatus = set_attr(
        options.rankIdx, options.rankSize, SHMEM_MALLOC_MAX_SIZE, options.ipPort.c_str(), &attributes, &uniqueId);
    if (attrStatus != ACLSHMEM_SUCCESS) {
        std::cerr << "[FAILED] set_attr, status=" << attrStatus << std::endl;
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }
    if (aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes) != ACLSHMEM_SUCCESS) {
        std::cerr << "[FAILED] aclshmemx_init_attr" << std::endl;
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }
    if (!AllocateDeviceSpace(device, host, cocTiling)) {
        std::cerr << "[FAILED] allocate Dispatch inputs" << std::endl;
        FreeDispatchDeviceData(device);
        shmem_finalize();
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    // Shape-only upper bounds keep the allocation and offsets identical on all ranks.
    // Round capacities to 32 rows so every field starts at a 32B-aligned address;
    // the valid rows and their packed GM strides are unchanged.
    const uint64_t recvCapacity = DispatchRoundUp(static_cast<uint64_t>(options.rankSize) * options.numTokens, 32);
    const uint64_t stageCapacity =
        useDirect ? 0 : DispatchRoundUp(static_cast<uint64_t>(options.numTokens) * options.topK, 32);
    const uint64_t quantCapacity = DispatchRoundUp(options.numTokens, 32);
    const uint64_t xRowBytes = static_cast<uint64_t>(options.hidden) * sizeof(ElementToken);
    const uint64_t topkIdxRowBytes = static_cast<uint64_t>(options.topK) * sizeof(ElementTopkIdx);
    const uint64_t topkWeightRowBytes = static_cast<uint64_t>(options.topK) * sizeof(ElementTopkW);
    const uint64_t rowBytes =
        xRowBytes + topkIdxRowBytes + topkWeightRowBytes + sizeof(ElementIdx) + sizeof(ElementScale);
    const uint64_t symmetricBytes =
        (recvCapacity + stageCapacity) * rowBytes + quantCapacity * (xRowBytes + sizeof(ElementScale));

    const uint64_t recvXOffset = 0;
    const uint64_t recvTopkIdxOffset = recvCapacity * xRowBytes;
    const uint64_t recvTopkWeightsOffset = recvTopkIdxOffset + recvCapacity * topkIdxRowBytes;
    const uint64_t recvSrcIdxOffset = recvTopkWeightsOffset + recvCapacity * topkWeightRowBytes;
    const uint64_t recvScalesOffset = recvSrcIdxOffset + recvCapacity * sizeof(ElementIdx);
    const uint64_t stageXOffset = recvCapacity * rowBytes;
    const uint64_t stageTopkIdxOffset = stageXOffset + stageCapacity * xRowBytes;
    const uint64_t stageTopkWeightsOffset = stageTopkIdxOffset + stageCapacity * topkIdxRowBytes;
    const uint64_t stageSrcIdxOffset = stageTopkWeightsOffset + stageCapacity * topkWeightRowBytes;
    const uint64_t stageScalesOffset = stageSrcIdxOffset + stageCapacity * sizeof(ElementIdx);
    const uint64_t quantXOffset = (recvCapacity + stageCapacity) * rowBytes;
    const uint64_t quantScalesOffset = quantXOffset + quantCapacity * xRowBytes;
    auto symmetric = reinterpret_cast<uint8_t*>(shmem_malloc(symmetricBytes));
    if (symmetric == nullptr) {
        std::cerr << "[FAILED] shmem_malloc" << std::endl;
        FreeDispatchDeviceData(device);
        shmem_finalize();
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }
    if (!CheckAcl(aclrtMemset(symmetric, symmetricBytes, 0, symmetricBytes), "aclrtMemset") ||
        !CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
        shmem_free(symmetric);
        FreeDispatchDeviceData(device);
        shmem_finalize();
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }
    aclshmem_barrier_all();

    auto recvX = symmetric + recvXOffset;
    auto recvTopkIdx = symmetric + recvTopkIdxOffset;
    auto recvTopkWeights = symmetric + recvTopkWeightsOffset;
    auto recvSrcIdx = symmetric + recvSrcIdxOffset;
    auto recvScales = symmetric + recvScalesOffset;
    auto makeArgs = [&](auto* argsType) {
        using Arguments = std::remove_pointer_t<decltype(argsType)>;
        return Arguments{
            Catlass::GemmCoord{options.numTokens, 1U, options.hidden},
            Catlass::MatrixCoord{options.commBlockM, options.hidden},
            options.rankIdx,
            options.rankSize,
            options.groupSize,
            options.commInterval,
            options.topK,
            host.numExperts,
            host.numOutputTokens,
            true,
            device.x,
            device.topkIdx,
            device.topkWeights,
            device.numTokensPerRank,
            device.isTokenInRank,
            device.rankPrefixMatrix,
            symmetric,
            recvXOffset,
            recvTopkIdxOffset,
            recvTopkWeightsOffset,
            recvSrcIdxOffset,
            stageXOffset,
            stageTopkIdxOffset,
            stageTopkWeightsOffset,
            stageSrcIdxOffset,
            recvX,
            recvTopkIdx,
            recvTopkWeights,
            recvSrcIdx,
            symmetric + quantXOffset,
            symmetric + quantScalesOffset,
            stageScalesOffset,
            recvScalesOffset,
            recvScales};
    };
    auto args = makeArgs(static_cast<DeviceOp::Arguments*>(nullptr));
    auto directArgs = makeArgs(static_cast<DirectDeviceOp::Arguments*>(nullptr));

    const size_t workspaceSize =
        useDirect ? DirectDeviceOp::GetWorkspaceSize(directArgs) : DeviceOp::GetWorkspaceSize(args);
    if (GetWorkspaceSize(cocTiling) != workspaceSize) {
        std::cerr << "[FAILED] Dispatch workspace size mismatch" << std::endl;
        shmem_free(symmetric);
        FreeDispatchDeviceData(device);
        shmem_finalize();
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    const uint64_t fftsAddr = shmemx_get_ffts_config();
    const uint64_t iterations = options.IsPerformanceMode() ?
                                    static_cast<uint64_t>(options.warmupIterations) + options.performanceIterations :
                                    1;
    if (useDirect) {
        DirectDeviceOp deviceOp;
        deviceOp.Initialize(directArgs);
        for (uint64_t i = 0; i < iterations; ++i) {
            deviceOp.Run(stream, blockNum, fftsAddr);
        }
    } else {
        DeviceOp deviceOp;
        deviceOp.Initialize(args);
        for (uint64_t i = 0; i < iterations; ++i) {
            deviceOp.Run(stream, blockNum, fftsAddr);
        }
    }
    int status = 0;
    if (!CheckAcl(aclrtSynchronizeStream(stream), "Dispatch synchronize")) {
        status = 1;
    } else {
        aclshmem_barrier_all();
        if (!options.IsPerformanceMode() && !WriteResultFile(
                                                options.dataPath, options.rankIdx, cocTiling, recvX, recvTopkIdx,
                                                recvTopkWeights, recvSrcIdx, recvScales)) {
            status = 1;
        }
    }
    shmem_free(symmetric);
    FreeDispatchDeviceData(device);
    shmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return status;
}
