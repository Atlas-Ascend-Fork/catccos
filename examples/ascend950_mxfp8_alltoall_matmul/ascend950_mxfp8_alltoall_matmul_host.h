/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_MXFP8_ALLTOALL_MATMUL_HOST_H
#define ASCEND950_MXFP8_ALLTOALL_MATMUL_HOST_H

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "catccos/dgemm/kernel/ascend950_mxfp8_alltoall_matmul.hpp"
#include "operator_registry.h"
#include "shmem.h"
#include "host/utils/shmem_log.h"

namespace MxFp8AllToAllMatmulExample {
inline constexpr auto OPERATOR_NAME = "Ascend950MxFp8AllToAllMatmul";
inline constexpr auto EXECUTABLE_NAME = "a2a_matmul";
namespace AllToAllLayout = Catccos::DGemm::Kernel::MxFp8AllToAllMatmul;

inline void Check(int status, const char* operation)
{
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
    }
}

inline uint32_t shmemLogRank = 0;
inline int shmemLogLevel = 3;

inline void ShmemLogger(int level, const char* message)
{
    if (message == nullptr || level < shmemLogLevel) {
        return;
    }
    static constexpr const char* levels[] = {"DEBUG", "INFO", "WARN", "ERROR", "FATAL"};
    const char* name = level >= 0 && level < 5 ? levels[level] : "UNKNOWN";
    std::fprintf(stderr, "[shmem] rank=%u %s %s\n", shmemLogRank, name, message);
    std::fflush(stderr);
}

inline void ConfigureShmemLogger(uint32_t rank)
{
    shmemLogRank = rank;
    shmemLogLevel = 3;
    const char* requested = std::getenv("SHMEM_LOG_LEVEL");
    static constexpr const char* levels[] = {"DEBUG", "INFO", "WARN"};
    for (int level = 0; requested != nullptr && level < 3; ++level) {
        if (std::strcmp(requested, levels[level]) == 0) {
            shmemLogLevel = level;
        }
    }
    // The external callback bypasses SHMEM's level filter and file/stdout sink.
    // Filter here, always retain errors, and flush before an uncaught exception.
    Check(aclshmemx_set_extern_logger(ShmemLogger), "set SHMEM logger");
}

inline AllToAllLayout::Layout GetLayout(const CocTilingParams& cocTiling)
{
    AllToAllLayout::Layout layout;
    if (!AllToAllLayout::MakeLayout(
            cocTiling.m, cocTiling.n, cocTiling.k, cocTiling.rankSize, cocTiling.commTileM, layout)) {
        throw std::invalid_argument("invalid symmetric layout shape");
    }
    return layout;
}

// Bounded host staging avoids allocating one full large matrix per process in host RAM.
inline void ReadDeviceFile(const std::string& path, uint8_t* device, size_t bytes)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || static_cast<uint64_t>(file.tellg()) != bytes) {
        throw std::runtime_error("missing or wrong-size input: " + path);
    }
    file.seekg(0);
    std::vector<uint8_t> buffer(std::min(bytes, size_t{8 * 1024 * 1024}));
    for (size_t offset = 0; offset < bytes; offset += buffer.size()) {
        const size_t count = std::min(buffer.size(), bytes - offset);
        if (!file.read(reinterpret_cast<char*>(buffer.data()), count)) {
            throw std::runtime_error("failed to read: " + path);
        }
        Check(aclrtMemcpy(device + offset, count, buffer.data(), count, ACL_MEMCPY_HOST_TO_DEVICE), "input H2D");
    }
}

inline void WriteDeviceFile(std::ofstream& file, const uint8_t* device, size_t bytes)
{
    std::vector<uint8_t> buffer(std::min(bytes, size_t{8 * 1024 * 1024}));
    for (size_t offset = 0; offset < bytes; offset += buffer.size()) {
        const size_t count = std::min(buffer.size(), bytes - offset);
        Check(aclrtMemcpy(buffer.data(), count, device + offset, count, ACL_MEMCPY_DEVICE_TO_HOST), "output D2H");
        if (!file.write(reinterpret_cast<const char*>(buffer.data()), count)) {
            throw std::runtime_error("failed to write output");
        }
    }
}

// Ordinary GM inputs stay owned by KernelParams and are staged on every timed call.
inline void StageInputs(
    const KernelParams& params, const CocTilingParams& cocTiling, uint8_t* symmetric, aclrtStream stream)
{
    const auto layout = GetLayout(cocTiling);
    const size_t aBytes = static_cast<size_t>(layout.m) * layout.localK;
    const size_t scaleBytes = static_cast<size_t>(layout.m) * layout.scaleK;
    Check(
        aclrtMemcpyAsync(symmetric + layout.sourceA, aBytes, params.ptrA, aBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
        "stage A");
    Check(
        aclrtMemcpyAsync(
            symmetric + layout.sourceScaleA, scaleBytes, params.customPtrs[0], scaleBytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
            stream),
        "stage scaleA");
}

// Initialize receive padding and the shared readiness workspace.
inline void InitializeSymmetricBuffer(const CocTilingParams& cocTiling, uint8_t* symmetric)
{
    const auto layout = GetLayout(cocTiling);
    const size_t scaleBytes = static_cast<size_t>(cocTiling.rankSize) * layout.recvSlotRows * layout.scaleK;
    // Initialize padded rows before the first window arrives.
    const size_t recvBytes = static_cast<size_t>(cocTiling.rankSize) * layout.recvSlotRows * layout.localK;
    Check(aclrtMemset(symmetric + layout.recvA, recvBytes, 0, recvBytes), "clear receive A");
    Check(aclrtMemset(symmetric + layout.recvScaleA, scaleBytes, 127, scaleBytes), "clear receive scaleA");
    Check(aclrtMemset(symmetric + layout.ready, layout.readyBytes, 0, layout.readyBytes), "initialize ready");
    std::vector<uint32_t> readyValues(layout.windowCount);
    for (uint32_t window = 0; window < layout.windowCount; ++window) {
        readyValues[window] = window + 1;
    }
    const size_t readyValueBytes = readyValues.size() * sizeof(uint32_t);
    Check(
        aclrtMemcpy(
            symmetric + layout.readyValues, readyValueBytes, readyValues.data(), readyValueBytes,
            ACL_MEMCPY_HOST_TO_DEVICE),
        "initialize immutable ready values");
}

inline void ResetSyncBuffer(const CocTilingParams& cocTiling, uint8_t* symmetric, aclrtStream stream)
{
    const auto layout = GetLayout(cocTiling);
    // The previous call completed on every rank before this reset; never reset between windows.
    const size_t bytes = layout.readyBytes;
    Check(aclrtMemsetAsync(symmetric + layout.ready, bytes, 0, bytes, stream), "reset ready/frontier");
}
} // namespace MxFp8AllToAllMatmulExample

class Ascend950MxFp8AllToAllMatmulOperator : public CatccosOperator {
public:
    void AllocateDeviceSpace(
        KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        using namespace MxFp8AllToAllMatmulExample;
        const auto layout = GetLayout(cocTiling);
        params.SetKernelParams(
            nullptr, nullptr, nullptr, static_cast<uint8_t*>(nullptr), static_cast<uint8_t*>(nullptr));
        const std::string suffix = "_" + std::to_string(rankId) + ".bin";
        auto allocate = [&](uint8_t*& ptr, size_t bytes, const std::string& name, int defaultValue) {
            Check(aclrtMalloc(reinterpret_cast<void**>(&ptr), bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
            if (!dataFile.empty()) {
                ReadDeviceFile(dataFile + "/" + name, ptr, bytes);
            } else {
                Check(aclrtMemset(ptr, bytes, defaultValue, bytes), "initialize input");
            }
        };
        allocate(params.ptrA, static_cast<size_t>(layout.m) * layout.localK, "input_a" + suffix, 0x38);
        allocate(params.ptrB, layout.BBytes(), "input_b" + suffix, 0x38);
        allocate(params.customPtrs[0], static_cast<size_t>(layout.m) * layout.scaleK, "input_a_scale" + suffix, 127);
        allocate(params.customPtrs[1], layout.ScaleBBytes(), "input_b_scale" + suffix, 127);
        Check(
            aclrtMalloc(reinterpret_cast<void**>(&params.ptrC), layout.CBytes(), ACL_MEM_MALLOC_HUGE_FIRST),
            "malloc C");
    }

    void WriteResultFile(
        const KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        using namespace MxFp8AllToAllMatmulExample;
        const auto layout = GetLayout(cocTiling);
        const std::string directory = dataFile.empty() ? "." : dataFile;
        const std::string suffix = "_" + std::to_string(rankId) + ".bin";
        std::ifstream rowsFile(directory + "/sample_rows.bin", std::ios::binary | std::ios::ate);
        if (!rowsFile) {
            std::ofstream output(directory + "/output" + suffix, std::ios::binary);
            WriteDeviceFile(output, params.ptrC, layout.CBytes());
            return;
        }
        const auto bytes = static_cast<size_t>(rowsFile.tellg());
        if (bytes == 0 || bytes % sizeof(uint32_t) != 0 || bytes / sizeof(uint32_t) > layout.chunkM) {
            throw std::runtime_error("invalid sample_rows.bin size");
        }
        std::vector<uint32_t> rows(bytes / sizeof(uint32_t));
        rowsFile.seekg(0);
        if (!rowsFile.read(reinterpret_cast<char*>(rows.data()), bytes)) {
            throw std::runtime_error("failed to read sample_rows.bin");
        }
        std::ofstream output(directory + "/output_samples" + suffix, std::ios::binary);
        const size_t rowBytes = layout.n * sizeof(bfloat16_t);
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] >= layout.chunkM || (i > 0 && rows[i] <= rows[i - 1])) {
                throw std::runtime_error("sample rows must be sorted, distinct and in range");
            }
            WriteDeviceFile(output, params.ptrC + static_cast<size_t>(rows[i]) * rowBytes, rowBytes);
        }
    }

    size_t GetSymmetricSize(const CocTilingParams& cocTiling) override
    {
        return MxFp8AllToAllMatmulExample::GetLayout(cocTiling).symmetricBytes;
    }
    size_t GetWorkspaceSize(const CocTilingParams&) override { return 0; }
    CocCommType GetActualKernelType(const CocTilingParams&) override
    {
        return CocCommType::ASCEND950_MXFP8_ALLTOALL_MATMUL;
    }
    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override
    {
        using namespace MxFp8AllToAllMatmulExample;
        AllToAllLayout::Layout layout;
        return (rankSize == 1 || rankSize == 2 || rankSize == 4 || rankSize == 8 || rankSize == 16 ||
                rankSize == AllToAllLayout::MAX_RANKS) &&
               cocTiling.rankSize == rankSize && cocTiling.transA == 0 && cocTiling.transB == 1 &&
               cocTiling.m0 == 256 && cocTiling.n0 == 256 && cocTiling.k0 == AllToAllLayout::SelectTileK(cocTiling.k) &&
               (cocTiling.commTileM == 256 || cocTiling.commTileM == 512 || cocTiling.commTileM == 1024) &&
               AllToAllLayout::MakeLayout(cocTiling.m, cocTiling.n, cocTiling.k, rankSize, cocTiling.commTileM, layout);
    }
};

REGISTER_OPERATOR(MxFp8AllToAllMatmulExample::OPERATOR_NAME, Ascend950MxFp8AllToAllMatmulOperator);

#endif // ASCEND950_MXFP8_ALLTOALL_MATMUL_HOST_H
