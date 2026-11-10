

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ascend950_dispatch_ffn_combine_device.h"
#include "ascend950_dispatch_ffn_combine_host.h"
#include "catccos/detail/ascend950_moe_ub.h"

using namespace AscendC;
using namespace Catccos;

using ElementA = bfloat16_t;
using ElementB = float8_e4m3_t;
using ElementC = bfloat16_t;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::ColumnMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutA0 = Catlass::layout::RowMajor;
using LayoutB0 = Catlass::layout::ColumnMajor;

static uint32_t gNpuNum = 8;

// The shared ACL_CHECK only logs. A regression must stop on the first runtime
// failure instead of submitting further kernels to a failed stream.
void CheckNativeAcl(aclError status)
{
    if (status != ACL_SUCCESS) {
        std::cerr << "[ERROR] Native ACL call failed: " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

std::vector<int32_t> ParseDeviceIdList(char* deviceIdList)
{
    std::vector<int32_t> result;
    if (deviceIdList == nullptr) {
        return result;
    }

    for (char* idToken = std::strtok(deviceIdList, ","); idToken; idToken = std::strtok(nullptr, ",")) {
        result.push_back(std::atoi(idToken));
    }
    return result;
}

uint32_t ParsePositiveCount(const std::string& value, const char* name)
{
    uint32_t count = 0;
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || count == 0) {
        throw std::runtime_error(std::string(name) + " must be a positive uint32 integer");
    }
    return count;
}

int RunExample(int argc, char** argv)
{
    if (argc < 12) {
        std::cerr << "Usage: rankSize rankId ipPort firstDevice M K N expertsPerRank dtype nz transB "
                     "[devices] [topK] [gmm1MTile]"
                  << std::endl;
        return 1;
    }
    int status = ACLSHMEM_SUCCESS;
    int rankSize = atoi(argv[1]);
    uint32_t rankId = atoi(argv[2]);
    std::string ipPort = argv[3];
    auto deviceIdList = argc > 12 ? ParseDeviceIdList(argv[12]) : std::vector<int32_t>{};
    if (rankSize <= 0 || rankId >= static_cast<uint32_t>(rankSize) ||
        (!deviceIdList.empty() && deviceIdList.size() != static_cast<size_t>(rankSize))) {
        std::cerr << "Invalid rank or device list" << std::endl;
        return 1;
    }
    int32_t deviceId = deviceIdList.empty() ? atoi(argv[4]) + rankId % gNpuNum : deviceIdList[rankId];

    uint32_t m = ParsePositiveCount(argv[5], "M");
    uint32_t k = ParsePositiveCount(argv[6], "K");
    uint32_t n = ParsePositiveCount(argv[7], "N");
    uint32_t expertPerRank = ParsePositiveCount(argv[8], "Experts per rank");
    uint32_t topK = argc > 13 ? ParsePositiveCount(argv[13], "TopK") : 4;
    uint32_t gmm1MTile = argc > 14 ? ParsePositiveCount(argv[14], "GMM1 M tile") : M0;
    if ((gmm1MTile != 128 && gmm1MTile != 256) ||
        static_cast<uint64_t>(expertPerRank) * rankSize > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("Invalid expert count or GMM1 M tile");
    }

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " device_id:" << deviceId
              << " input_ip: " << ipPort << std::endl;

    CocTilingParams cocTiling{};
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    cocTiling.m0 = gmm1MTile;
    cocTiling.n0 = N0;
    cocTiling.k0 = K0;
    cocTiling.commTileM = 64;
    cocTiling.commInterval = 10;
    cocTiling.commNpuSplit = rankSize;
    cocTiling.commDataSplit = 1;
    cocTiling.commBlockM = 64;
    cocTiling.rankSize = rankSize;
    cocTiling.epSize = rankSize;
    cocTiling.expertNum = rankSize * expertPerRank;
    cocTiling.topK = topK;

    auto op = OperatorRegistry::Instance().CreateOperator("Ascend950DispatchFFNCombine");
    if (!op) {
        std::cout << "Operator Ascend950DispatchFFNCombine not found!" << std::endl;
        return -1;
    }
    // A case owns its inputs and all outputs until the queued batch finishes.
    struct TestCase {
        std::string directory;
        CocTilingParams tiling;
        uint32_t repeats;
        MoeInitRoutingQuantV2Tiling routing;
        KernelParams params;
        std::vector<uint8_t*> outputs;
    };
    std::vector<TestCase> cases;
    auto addCase = [&](const std::string& directory, uint32_t rows, uint32_t repeats) {
        auto tiling = cocTiling;
        tiling.m = rows;
        if (static_cast<uint64_t>(rows) * topK > std::numeric_limits<int32_t>::max() / rankSize ||
            !op->ValidateCocTilingParams(rankSize, tiling)) {
            throw std::runtime_error("Invalid case shape: " + directory);
        }
        cases.push_back({directory, tiling, repeats, {}, {}, {}});
    };
    // Optional manifest entries are "directory M repeats".
    const char* manifest = std::getenv("CATCCOS_CASE_LIST");
    if (manifest) {
        std::ifstream input(manifest);
        if (!input) {
            throw std::runtime_error("Cannot read CATCCOS_CASE_LIST");
        }
        std::string directory;
        while (input >> directory) {
            std::string rows, repeats;
            if (!(input >> rows >> repeats)) {
                throw std::runtime_error("Manifest requires directory, M and repeats on every row");
            }
            addCase(directory, ParsePositiveCount(rows, "Manifest M"), ParsePositiveCount(repeats, "Manifest repeats"));
        }
        if (!input.eof() || cases.empty()) {
            throw std::runtime_error("Malformed or empty regression manifest");
        }
    } else {
        addCase("./output", m, 1);
    }

    aclrtStream stream = nullptr;
    CheckNativeAcl(aclInit(nullptr));
    CheckNativeAcl(aclrtSetDevice(deviceId));
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aicCoreNum = platform == nullptr ? 0 : static_cast<uint32_t>(platform->GetCoreNumAic());
    uint32_t aivCoreNum = platform == nullptr ? 0 : static_cast<uint32_t>(platform->GetCoreNumAiv());
    uint32_t hardwareBlockNum = std::min(aicCoreNum, aivCoreNum / 2);
    uint32_t aclshmemBlockNum = ACLSHMEM_MAX_AIV_PER_NPU / 2;
    uint32_t blockNum = std::min(hardwareBlockNum, aclshmemBlockNum);
    if (blockNum == 0) {
        std::cerr << "[ERROR] Invalid platform core counts: AIC=" << aicCoreNum << " AIV=" << aivCoreNum << std::endl;
        CheckNativeAcl(aclrtResetDevice(deviceId));
        CheckNativeAcl(aclFinalize());
        return 1;
    }
    uint32_t launchAivNum = 2 * blockNum;
    std::cout << "[CONFIG] platform_aic=" << aicCoreNum << " platform_aiv=" << aivCoreNum
              << " hardware_mixed_1_2_block_num=" << hardwareBlockNum << " launch_block_num=" << blockNum
              << " launch_aiv_num=" << launchAivNum << " aclshmem_aiv_limit=" << ACLSHMEM_MAX_AIV_PER_NPU << std::endl;
    for (const auto& item : cases) {
        if (!Catccos::Detail::Ascend950MoeUb::Fits(item.tiling.m, k, topK, launchAivNum)) {
            std::cerr << "[ERROR] Shape-dependent buffers exceed Ascend950 UB: " << item.directory
                      << " M=" << item.tiling.m << std::endl;
            CheckNativeAcl(aclrtResetDevice(deviceId));
            CheckNativeAcl(aclFinalize());
            return 1;
        }
    }
    CheckNativeAcl(aclrtCreateStream(&stream));
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t default_flag_uid;
    set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, ipPort.c_str(), &attributes, &default_flag_uid);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    std::cout << "[TEST] after shmem init rankId: " << rankId << " status: " << status << std::endl;
    if (status != ACLSHMEM_SUCCESS) {
        std::cerr << "[ERROR] aclshmemx_init_attr failed rankId: " << rankId << " status: " << status << std::endl;
        CheckNativeAcl(aclrtDestroyStream(stream));
        CheckNativeAcl(aclrtResetDevice(deviceId));
        CheckNativeAcl(aclFinalize());
        return status;
    }

    uint64_t fftsAddr = shmemx_get_ffts_config();

    size_t workspaceSize = 0;
    const bool poison = manifest && std::getenv("CATCCOS_POISON_WORKSPACE");
    for (auto& item : cases) {
        optiling::MoeInitRoutingV2TilingBase routing;
        routing.DoTiling(
            item.tiling.m, k, topK, 0, item.tiling.expertNum, item.tiling.m * topK, 0, 2, false, sizeof(ElementA), -1,
            0, launchAivNum, 196352);
        auto key = routing.tilingKey_;
        if (key == optiling::TILING_KEY_HIGH_PERFORMANCE) {
            key = optiling::TILING_KEY_DROPLESS_SORT_ONE_CORE;
        }
        item.routing = {routing.moeInitRoutingTilingData, key};
        workspaceSize = std::max(workspaceSize, op->GetWorkspaceSize(item.tiling));
        op->AllocateDeviceSpace(item.params, item.tiling, rankId, item.directory);
        item.outputs.push_back(item.params.ptrC);
        size_t bytes = static_cast<size_t>(item.tiling.m) * k * sizeof(ElementC);
        for (uint32_t repeat = 1; repeat < item.repeats; ++repeat) {
            uint8_t* output = nullptr;
            CheckNativeAcl(aclrtMalloc(reinterpret_cast<void**>(&output), bytes, ACL_MEM_MALLOC_HUGE_FIRST));
            item.outputs.push_back(output);
        }
        if (poison) {
            for (auto* output : item.outputs) {
                CheckNativeAcl(aclrtMemset(output, bytes, 0xA5, bytes));
            }
        }
    }
    uint8_t* workspaceDevice = nullptr;
    CheckNativeAcl(aclrtMalloc(reinterpret_cast<void**>(&workspaceDevice), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto symmetricPtr = static_cast<uint8_t*>(shmem_malloc(SHMEM_BUFF_BYTES));
    if (!symmetricPtr) {
        throw std::runtime_error("Symmetric allocation failed");
    }
    if (poison) {
        CheckNativeAcl(aclrtMemset(workspaceDevice, workspaceSize, 0xA5, workspaceSize));
        CheckNativeAcl(aclrtMemset(symmetricPtr, SHMEM_BUFF_BYTES, 0xA5, SHMEM_BUFF_BYTES));
    }

    CheckNativeAcl(aclrtSynchronizeStream(stream));
    aclshmem_barrier_all();
    aclshmemx_barrier_all_on_stream(stream);
    for (const auto& item : cases) {
        const auto& params = item.params;
        for (auto* output : item.outputs) {
            DispatchFFNCombine<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
                <<<blockNum, nullptr, stream>>>(
                    fftsAddr, params.ptrA, params.ptrB, params.customPtrs[3], params.customPtrs[0],
                    params.customPtrs[4], output, params.customPtrs[1], params.customPtrs[2], workspaceDevice,
                    symmetricPtr, item.tiling, item.routing);
        }
    }
    CheckNativeAcl(aclrtSynchronizeStream(stream));
    for (size_t i = 0; i < cases.size(); ++i) {
        auto& item = cases[i];
        for (size_t repeat = 0; repeat < item.outputs.size(); ++repeat) {
            item.params.ptrC = item.outputs[repeat];
            op->WriteResultFile(item.params, item.tiling, rankId, item.directory);
            item.params.ptrC = item.outputs.front();
            if (manifest) {
                auto output = item.directory + "/output_" + std::to_string(rankId);
                auto saved = output + "_case" + std::to_string(i) + "_repeat" + std::to_string(repeat) + ".bin";
                if (std::rename((output + ".bin").c_str(), saved.c_str()) != 0) {
                    throw std::runtime_error("Cannot save launch output: " + saved);
                }
            }
            if (repeat != 0) {
                CheckNativeAcl(aclrtFree(item.outputs[repeat]));
            }
        }
        FreeDeviceSpace(item.params);
    }
    cases.clear();
    shmem_free(symmetricPtr);
    CheckNativeAcl(aclrtFree(workspaceDevice));

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = shmem_finalize();
    CheckNativeAcl(aclrtDestroyStream(stream));
    CheckNativeAcl(aclrtResetDevice(deviceId));
    CheckNativeAcl(aclFinalize());

    return 0;
}

int main(int argc, char** argv)
{
    try {
        return RunExample(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << std::endl;
        return 1;
    }
}
