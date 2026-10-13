/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef ASCEND950_DISPATCH_HOST_H
#define ASCEND950_DISPATCH_HOST_H

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "info.h"
#include "catccos/comm/block/block_quantize.hpp"

using DispatchInputElement = bfloat16_t;
using ElementToken = int8_t;
using DispatchHostElement = uint16_t;
using ElementScale = float;
using ElementTopkIdx = int64_t;
using ElementTopkW = float;
using ElementIdx = int32_t;
static_assert(
    sizeof(DispatchInputElement) == sizeof(DispatchHostElement),
    "Dispatch device and host BF16 storage must have the same width");

struct DispatchHostData {
    std::vector<DispatchHostElement> x;
    std::vector<ElementTopkIdx> topkIdx;
    std::vector<ElementTopkW> topkWeights;
    std::vector<int32_t> numTokensPerRank;
    std::vector<uint8_t> isTokenInRank;
    std::vector<int32_t> rankPrefixMatrix;
    uint32_t numRecvTokens{0};
    uint32_t numOutputTokens{0};
    uint32_t numExperts{0};
};

struct DispatchDeviceData {
    uint8_t* x{nullptr};
    uint8_t* topkIdx{nullptr};
    uint8_t* topkWeights{nullptr};
    uint8_t* numTokensPerRank{nullptr};
    uint8_t* isTokenInRank{nullptr};
    uint8_t* rankPrefixMatrix{nullptr};
};

struct DispatchCocTilingParams {
    uint32_t rankSize{0};
    uint32_t rankIdx{0};
    uint32_t numTokens{0};
    uint32_t hidden{0};
    uint32_t groupSize{0};
    uint32_t numTopk{0};
    uint32_t expertsPerRank{0};
    uint32_t commInterval{0};
    uint32_t commBlockM{0};
    uint32_t numOutputTokens{0};
};

inline bool CheckCocTilingParams(DispatchCocTilingParams const& cocTiling)
{
    const bool validShape =
        cocTiling.rankSize > 0 && cocTiling.rankSize <= MAX_RANK_SIZE && cocTiling.rankIdx < cocTiling.rankSize &&
        cocTiling.numTokens > 0 && cocTiling.hidden > 0 && cocTiling.groupSize > 0 &&
        cocTiling.groupSize <= cocTiling.rankSize && cocTiling.numTopk > 0 && cocTiling.numTopk <= cocTiling.rankSize &&
        cocTiling.numTopk <= 128 && cocTiling.expertsPerRank > 0 && cocTiling.commInterval > 0 &&
        cocTiling.commBlockM > 0 && Catccos::Comm::Block::BlockQuantizeBf16Int8::CanImplement(cocTiling.hidden);
    if (!validShape) {
        return false;
    }

    const uint64_t capacity = static_cast<uint64_t>(cocTiling.rankSize) * cocTiling.numTokens;
    return capacity <= std::numeric_limits<uint32_t>::max() && cocTiling.numOutputTokens <= capacity;
}

inline size_t GetWorkspaceSize(DispatchCocTilingParams const&) { return 0; }

inline CocCommType GetActualKernelType(DispatchCocTilingParams const&) { return CocCommType::ASCEND950_DISPATCH; }

inline uint64_t DispatchRoundUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

inline std::string DispatchDataFile(std::string const& dataPath, std::string const& name, uint32_t rankIdx)
{
    return dataPath + "/input_" + name + "_" + std::to_string(rankIdx) + ".bin";
}

template <class T>
inline bool DispatchReadVector(std::string const& filePath, std::vector<T>& data, size_t elementCount)
{
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[FAILED] open input file " << filePath << std::endl;
        return false;
    }
    const auto fileBytes = file.tellg();
    const size_t expectedBytes = elementCount * sizeof(T);
    if (fileBytes < 0 || static_cast<size_t>(fileBytes) != expectedBytes) {
        std::cerr << "[FAILED] input file size " << filePath << ": actual=" << fileBytes
                  << " expected=" << expectedBytes << std::endl;
        return false;
    }
    data.resize(elementCount);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(expectedBytes));
    if (!file) {
        std::cerr << "[FAILED] read input file " << filePath << std::endl;
        return false;
    }
    return true;
}

inline bool LoadDispatchHostData(
    DispatchHostData& data, std::string const& dataPath, uint32_t rankIdx, uint32_t rankSize, uint32_t numTokens,
    uint32_t hidden, uint32_t numTopk, uint32_t expertsPerRank)
{
    data.numExperts = rankSize * expertsPerRank;
    const bool loaded =
        DispatchReadVector(DispatchDataFile(dataPath, "x", rankIdx), data.x, static_cast<size_t>(numTokens) * hidden) &&
        DispatchReadVector(
            DispatchDataFile(dataPath, "topk_idx", rankIdx), data.topkIdx, static_cast<size_t>(numTokens) * numTopk) &&
        DispatchReadVector(
            DispatchDataFile(dataPath, "topk_weights", rankIdx), data.topkWeights,
            static_cast<size_t>(numTokens) * numTopk) &&
        DispatchReadVector(
            DispatchDataFile(dataPath, "num_tokens_per_rank", rankIdx), data.numTokensPerRank, rankSize) &&
        DispatchReadVector(
            DispatchDataFile(dataPath, "is_token_in_rank", rankIdx), data.isTokenInRank,
            static_cast<size_t>(numTokens) * rankSize) &&
        DispatchReadVector(
            DispatchDataFile(dataPath, "rank_prefix_matrix", rankIdx), data.rankPrefixMatrix,
            static_cast<size_t>(rankSize) * rankSize);
    if (!loaded) {
        return false;
    }
    // The per-token quantizer requires finite input and a finite FP32
    // reciprocal scale. Reject unsupported values before collective launch.
    for (uint32_t row = 0; row < numTokens; ++row) {
        uint16_t maxBits = 0;
        for (uint32_t col = 0; col < hidden; ++col) {
            const uint16_t bits = data.x[static_cast<size_t>(row) * hidden + col] & 0x7fff;
            if ((bits & 0x7f80) == 0x7f80) {
                std::cerr << "[FAILED] INT8 quantization requires finite BF16 inputs" << std::endl;
                return false;
            }
            maxBits = bits > maxBits ? bits : maxBits;
        }
        const uint32_t fp32Bits = static_cast<uint32_t>(maxBits) << 16;
        float amax = 0;
        std::memcpy(&amax, &fp32Bits, sizeof(amax));
        if (amax != 0 && amax < 127.0F / std::numeric_limits<float>::max()) {
            std::cerr << "[FAILED] INT8 reciprocal scale overflows FP32" << std::endl;
            return false;
        }
    }

    const int32_t numRecvTokens = data.rankPrefixMatrix[static_cast<size_t>(rankSize - 1) * rankSize + rankIdx];
    if (numRecvTokens < 0 || static_cast<uint64_t>(numRecvTokens) > static_cast<uint64_t>(rankSize) * numTokens) {
        std::cerr << "[FAILED] Invalid Dispatch receive count" << std::endl;
        return false;
    }
    data.numRecvTokens = static_cast<uint32_t>(numRecvTokens);
    data.numOutputTokens = data.numRecvTokens;
    return true;
}

inline bool DispatchAllocateAndCopy(uint8_t** device, const void* host, size_t bytes)
{
    if (aclrtMalloc(reinterpret_cast<void**>(device), bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_ERROR_NONE) {
        return false;
    }
    if (aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
        aclrtFree(*device);
        *device = nullptr;
        return false;
    }
    return true;
}

template <class T>
inline bool DispatchAllocateVector(uint8_t** device, std::vector<T> const& host)
{
    if (host.empty()) {
        return true;
    }
    return DispatchAllocateAndCopy(device, host.data(), host.size() * sizeof(T));
}

inline bool AllocateDeviceSpace(
    DispatchDeviceData& device, DispatchHostData const& host, DispatchCocTilingParams const&)
{
    return DispatchAllocateVector(&device.x, host.x) && DispatchAllocateVector(&device.topkIdx, host.topkIdx) &&
           DispatchAllocateVector(&device.topkWeights, host.topkWeights) &&
           DispatchAllocateVector(&device.numTokensPerRank, host.numTokensPerRank) &&
           DispatchAllocateVector(&device.isTokenInRank, host.isTokenInRank) &&
           DispatchAllocateVector(&device.rankPrefixMatrix, host.rankPrefixMatrix);
}

inline void FreeDispatchDeviceData(DispatchDeviceData& device)
{
    uint8_t* ptrs[] = {device.x,
                       device.topkIdx,
                       device.topkWeights,
                       device.numTokensPerRank,
                       device.isTokenInRank,
                       device.rankPrefixMatrix};
    for (auto ptr : ptrs) {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
    device = {};
}

template <class T>
inline bool DispatchCopyAndWrite(std::string const& filePath, const void* device, size_t elementCount)
{
    std::vector<T> host(elementCount);
    const size_t bytes = elementCount * sizeof(T);
    if (bytes > 0 && aclrtMemcpy(host.data(), bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
        std::cerr << "[FAILED] copy output " << filePath << std::endl;
        return false;
    }
    std::ofstream file(filePath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "[FAILED] open output file " << filePath << std::endl;
        return false;
    }
    if (bytes > 0) {
        file.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(bytes));
    }
    if (!file) {
        std::cerr << "[FAILED] write output file " << filePath << std::endl;
        return false;
    }
    return true;
}

inline bool WriteResultFile(
    std::string const& dataPath, uint32_t rankIdx, DispatchCocTilingParams const& cocTiling, const void* recvX,
    const void* recvTopkIdx, const void* recvTopkWeights, const void* recvSrcIdx, const void* recvScales)
{
    const auto output = [&](std::string const& name) {
        return dataPath + "/output_" + name + "_" + std::to_string(rankIdx) + ".bin";
    };
    return DispatchCopyAndWrite<uint8_t>(
               output("recv_x"), recvX,
               static_cast<size_t>(cocTiling.numOutputTokens) * cocTiling.hidden * sizeof(ElementToken)) &&
           DispatchCopyAndWrite<ElementScale>(output("recv_x_scales"), recvScales, cocTiling.numOutputTokens) &&
           DispatchCopyAndWrite<ElementTopkIdx>(
               output("recv_topk_idx"), recvTopkIdx,
               static_cast<size_t>(cocTiling.numOutputTokens) * cocTiling.numTopk) &&
           DispatchCopyAndWrite<ElementTopkW>(
               output("recv_topk_weights"), recvTopkWeights,
               static_cast<size_t>(cocTiling.numOutputTokens) * cocTiling.numTopk) &&
           DispatchCopyAndWrite<ElementIdx>(output("recv_src_idx"), recvSrcIdx, cocTiling.numOutputTokens);
}

#endif // ASCEND950_DISPATCH_HOST_H
