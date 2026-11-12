/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_HIERARCHICAL_ALLTOALL_HOST_H
#define ASCEND950_HIERARCHICAL_ALLTOALL_HOST_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "catccos_operator.h"
#include "catccos/comm/tile/tile_hierarchical_ready.hpp"
#include "acl/acl.h"
#include "shmem.h"
#include "host/utils/shmem_log.h"
#include "shmem_init.h"
#include "tiling/platform/platform_ascendc.h"

namespace HierarchicalAllToAllHost {

constexpr uint32_t CORE_COUNT = 32;
constexpr uint64_t HEAP_BYTES = 512ULL * 1024 * 1024;
constexpr uint64_t HEAP_RESERVE_BYTES = 16ULL * 1024 * 1024;
constexpr uint8_t PADDING_VALUE = 0xa5;
constexpr uint32_t MX_SCALE_GROUP_SIZE = 32;

struct TensorLayout {
    uint64_t payloadBytes = 0;
    uint64_t peerStrideBytes = 0;
    uint64_t rankBytes = 0;
};

struct Layout {
    TensorLayout data;
    TensorLayout scale;
    uint32_t scaleN = 0;
    uint64_t allocationBytes = 0;
};

// Validate before multiplying byte counts.
template <class Element>
bool MakeTensorLayout(uint32_t rankSize, uint32_t m, uint32_t n, TensorLayout& layout)
{
    static_assert(sizeof(Element) <= 32 && 32 % sizeof(Element) == 0, "Elements must divide a 32-byte slot");
    layout = {};
    if (rankSize == 0 || rankSize > CORE_COUNT) {
        return false;
    }
    const uint64_t elements = static_cast<uint64_t>(m) * n;
    const uint64_t maxPayload = (HEAP_BYTES - HEAP_RESERVE_BYTES) / (2 * rankSize);
    if (elements > maxPayload / sizeof(Element)) {
        return false;
    }
    layout.payloadBytes = elements * sizeof(Element);
    layout.peerStrideBytes = (layout.payloadBytes + 31) / 32 * 32;
    layout.rankBytes = layout.peerStrideBytes * rankSize;
    return true;
}

template <class Element, class ElementScale>
bool MakeLayout(uint32_t rankSize, uint32_t m, uint32_t n, Layout& layout)
{
    layout = {};
    if (m == 0 || n == 0 || n % MX_SCALE_GROUP_SIZE != 0) {
        return false;
    }
    layout.scaleN = n / MX_SCALE_GROUP_SIZE;
    if (!MakeTensorLayout<Element>(rankSize, m, n, layout.data) ||
        !MakeTensorLayout<ElementScale>(rankSize, m, layout.scaleN, layout.scale)) {
        return false;
    }
    layout.allocationBytes = 2 * (layout.data.rankBytes + layout.scale.rankBytes);
    return layout.allocationBytes <= HEAP_BYTES - HEAP_RESERVE_BYTES;
}

inline void CheckStatus(int status, const char* operation)
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
    CheckStatus(aclshmemx_set_extern_logger(ShmemLogger), "set SHMEM logger");
}

} // namespace HierarchicalAllToAllHost

// Host allocation and binary input/output for the communication-only example.
template <class Element, class ElementScale, class BarrierDevice>
class Ascend950HierarchicalAllToAllOperator : public CatccosOperator {
public:
    using Layout = HierarchicalAllToAllHost::Layout;
    using Ready = Catccos::Comm::Tile::TileHierarchicalReady;

    Ascend950HierarchicalAllToAllOperator(uint32_t rankSize, uint32_t rankId, uint32_t groupSize, uint32_t qpNum = 1)
        : rankSize_(rankSize), rankId_(rankId), groupSize_(groupSize), qpNum_(qpNum)
    {
        if (qpNum_ != 1 && qpNum_ != 2 && qpNum_ != 4) {
            throw std::invalid_argument("QP_NUM must be 1, 2 or 4");
        }
    }

    Ascend950HierarchicalAllToAllOperator(const Ascend950HierarchicalAllToAllOperator&) = delete;
    Ascend950HierarchicalAllToAllOperator& operator=(const Ascend950HierarchicalAllToAllOperator&) = delete;

    ~Ascend950HierarchicalAllToAllOperator() override
    {
        if (stream_ != nullptr) {
            (void)aclrtSynchronizeStream(stream_);
        }
        if (readyWorkspace_ != nullptr) {
            aclshmem_free(readyWorkspace_);
        }
        if (barrierWorkspace_ != nullptr) {
            aclshmem_free(barrierWorkspace_);
        }
        if (symmetricPtr_ != nullptr) {
            aclshmem_free(symmetricPtr_);
        }
        if (shmemInitialized_) {
            (void)aclshmem_finalize();
        }
        if (stream_ != nullptr) {
            (void)aclrtDestroyStream(stream_);
        }
        if (deviceSet_) {
            (void)aclrtResetDevice(deviceId_);
        }
        if (aclInitialized_) {
            (void)aclFinalize();
        }
    }

    void Initialize(int32_t deviceId, const std::string& ipPort)
    {
        using HierarchicalAllToAllHost::CheckStatus;
        HierarchicalAllToAllHost::ConfigureShmemLogger(rankId_);
        CheckStatus(aclInit(nullptr), "aclInit");
        aclInitialized_ = true;
        deviceId_ = deviceId;
        CheckStatus(aclrtSetDevice(deviceId_), "aclrtSetDevice");
        deviceSet_ = true;
        auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (platform == nullptr || platform->GetCoreNumAic() < static_cast<int>((rankSize_ + 1U) / 2U) ||
            platform->GetCoreNumAiv() < static_cast<int>(rankSize_)) {
            throw std::runtime_error("This example requires ceil(rank_size/2) AIC and rank_size AIV cores");
        }
        const char* soc = aclrtGetSocName();
        if (soc == nullptr || std::string(soc).find("Ascend950") != 0) {
            throw std::runtime_error("This example requires Ascend950");
        }
        CheckStatus(aclrtCreateStream(&stream_), "aclrtCreateStream");

        const bool crossNode = rankSize_ > groupSize_;
        CheckStatus(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qpNum_), "set UDMA QP count");
        if (crossNode) {
            CheckStatus(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE, qpNum_), "set RDMA QP count");
        }
        aclshmemx_uniqueid_t defaultFlagUid{};
        aclshmemx_init_attr_t attributes{};
        CheckStatus(
            set_attr(
                rankId_, rankSize_, HierarchicalAllToAllHost::HEAP_BYTES, ipPort.c_str(), &attributes, &defaultFlagUid),
            "set SHMEM attributes");
        attributes.option_attr.data_op_engine_type =
            static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_UDMA | (crossNode ? ACLSHMEM_DATA_OP_ROCE : 0));
        // Different peers are submitted concurrently by different AIVs. Shared
        // jetties would require serialization across all peers sharing one SQ/CQ.
        attributes.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_DISABLED;
        const int initStatus = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
        if (initStatus != ACLSHMEM_SUCCESS) {
            std::fprintf(
                stderr, "[shmem-init] rank=%u world=%u group=%u endpoint=%s status=%d\n", rankId_, rankSize_,
                groupSize_, ipPort.c_str(), initStatus);
            std::fflush(stderr);
        }
        CheckStatus(initStatus, "aclshmemx_init_attr");
        shmemInitialized_ = true;
        ValidateUdmaRoutes();
    }

    // m/n describe each peer's matrix; MTE block rows are fixed by the scheduler template.
    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override
    {
        Layout layout;
        return rankSize == rankSize_ && cocTiling.rankSize == rankSize && rankId_ < rankSize && groupSize_ > 0 &&
               groupSize_ <= rankSize &&
               HierarchicalAllToAllHost::MakeLayout<Element, ElementScale>(rankSize, cocTiling.m, cocTiling.n, layout);
    }

    size_t GetWorkspaceSize(const CocTilingParams&) override { return Ready::GetWorkspaceSize(rankSize_); }

    CocCommType GetActualKernelType(const CocTilingParams&) override
    {
        return CocCommType::ASCEND950_HIERARCHICAL_ALLTOALL;
    }

    void AllocateDeviceSpace(
        KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile = "") override
    {
        if (rankId != rankId_ || !CheckCocTilingParams(rankSize_, cocTiling)) {
            throw std::invalid_argument("Invalid hierarchical AllToAll tiling or rank");
        }
        if (dataFile.empty()) {
            throw std::invalid_argument("Input data directory is required");
        }
        HierarchicalAllToAllHost::MakeLayout<Element, ElementScale>(rankSize_, cocTiling.m, cocTiling.n, layout_);
        symmetricPtr_ = static_cast<uint8_t*>(aclshmem_malloc(layout_.allocationBytes));
        if (symmetricPtr_ == nullptr) {
            throw std::runtime_error("aclshmem_malloc failed");
        }
        const typename BarrierDevice::Arguments barrierArgs{rankId_, rankSize_, groupSize_, 0};
        // calloc zeroes the signals and performs a control-plane barrier before
        // returning, so an early peer cannot signal a buffer still being cleared.
        barrierWorkspace_ = static_cast<uint8_t*>(aclshmem_calloc(1, BarrierDevice::GetWorkspaceSize(barrierArgs)));
        if (barrierWorkspace_ == nullptr) {
            throw std::runtime_error("allocate barrier workspace failed");
        }
        readyWorkspace_ = static_cast<uint8_t*>(aclshmem_calloc(1, GetWorkspaceSize(cocTiling)));
        if (readyWorkspace_ == nullptr) {
            throw std::runtime_error("allocate ready workspace failed");
        }
        const std::string prefix = dataFile + "/rank_" + std::to_string(rankId);
        LoadTensor(prefix + "_a.bin", layout_.data, GetInput(), GetOutput());
        LoadTensor(prefix + "_a_scale.bin", layout_.scale, GetInputScale(), GetOutputScale());
        // These are views into one SHMEM allocation owned by this operator.
        // Do not pass them to FreeDeviceSpace, which uses aclrtFree.
        params.SetKernelParams(GetInput(), nullptr, GetOutput(), GetInputScale(), GetOutputScale());
    }

    void WriteResultFile(
        const KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId,
        std::string dataFile = "") override
    {
        if (dataFile.empty() || rankId != rankId_ || !CheckCocTilingParams(rankSize_, cocTiling) ||
            params.ptrC != GetOutput()) {
            throw std::invalid_argument("Invalid hierarchical AllToAll output parameters");
        }
        SaveTensor(dataFile + "/output_" + std::to_string(rankId) + ".bin", layout_.data, GetOutput());
        SaveTensor(dataFile + "/output_scale_" + std::to_string(rankId) + ".bin", layout_.scale, GetOutputScale());
    }

    void Barrier(const char* name)
    {
        using HierarchicalAllToAllHost::CheckStatus;
        // Data kernels quiet their QPs before returning. Complete the stream
        // before the barrier reuses those QPs to publish arrival signals.
        CheckStatus(aclrtSynchronizeStream(stream_), name);
        if (barrierEpoch_ == UINT32_MAX) {
            throw std::runtime_error("barrier epoch exhausted");
        }
        const typename BarrierDevice::Arguments args{rankId_, rankSize_, groupSize_, ++barrierEpoch_};
        BarrierDevice barrierDevice;
        CheckStatus(static_cast<int>(barrierDevice.Initialize(args, barrierWorkspace_)), name);
        CheckStatus(static_cast<int>(barrierDevice.Run(stream_, 1, shmemx_get_ffts_config())), name);
    }

    aclrtStream GetStream() const { return stream_; }
    uint8_t* GetWorkspace() const { return readyWorkspace_; }

private:
    uint8_t* GetInput() const { return symmetricPtr_; }
    uint8_t* GetOutput() const { return symmetricPtr_ + layout_.data.rankBytes; }
    uint8_t* GetInputScale() const { return symmetricPtr_ + 2 * layout_.data.rankBytes; }
    uint8_t* GetOutputScale() const { return GetInputScale() + layout_.scale.rankBytes; }

    // Stage raw bytes: MXFP8 proxy types need not be default-constructible on the host.
    void LoadTensor(
        const std::string& file, const HierarchicalAllToAllHost::TensorLayout& layout, uint8_t* input, uint8_t* output)
    {
        using namespace HierarchicalAllToAllHost;
        std::vector<uint8_t> hostInput(layout.rankBytes);
        ReadFileOrThrow(file, hostInput.data(), layout.rankBytes);
        CheckStatus(
            aclrtMemcpy(input, layout.rankBytes, hostInput.data(), layout.rankBytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "upload input");
        CheckStatus(aclrtMemset(output, layout.rankBytes, PADDING_VALUE, layout.rankBytes), "initialize output");
    }

    void SaveTensor(const std::string& file, const HierarchicalAllToAllHost::TensorLayout& layout, uint8_t* output)
    {
        std::vector<uint8_t> hostOutput(layout.rankBytes);
        HierarchicalAllToAllHost::CheckStatus(
            aclrtMemcpy(hostOutput.data(), layout.rankBytes, output, layout.rankBytes, ACL_MEMCPY_DEVICE_TO_HOST),
            "download output");
        if (!WriteFile(file, hostOutput.data(), layout.rankBytes)) {
            throw std::runtime_error("Failed to write file: " + file);
        }
    }

    void ValidateUdmaRoutes() const
    {
        for (uint32_t peer = 0; peer < rankSize_; ++peer) {
            if (peer == rankId_) {
                continue;
            }
            if (peer / groupSize_ != rankId_ / groupSize_) {
                continue;
            }
            // The public route query supports UDMA only. Cross-node RDMA setup
            // is checked by init, then exercised by the first barrier/transfer.
            for (uint32_t qp = 0; qp < qpNum_; ++qp) {
                aclshmemx_net_layer_t netLayer;
                if (aclshmemx_get_qp_net_layer(ACLSHMEM_DATA_OP_UDMA, peer, qp, &netLayer) != ACLSHMEM_SUCCESS) {
                    throw std::runtime_error(
                        "peer " + std::to_string(peer) + " QP " + std::to_string(qp) +
                        " has no UDMA route; rebuild SHMEM with -multi_link and check this server's driver "
                        "topology/EIDs");
                }
            }
        }
    }

    uint32_t rankSize_;
    uint32_t rankId_;
    uint32_t groupSize_;
    uint32_t qpNum_;
    int32_t deviceId_ = 0;
    bool aclInitialized_ = false;
    bool deviceSet_ = false;
    bool shmemInitialized_ = false;
    aclrtStream stream_ = nullptr;
    uint8_t* symmetricPtr_ = nullptr;
    uint8_t* barrierWorkspace_ = nullptr;
    uint8_t* readyWorkspace_ = nullptr;
    uint32_t barrierEpoch_ = 0;
    Layout layout_;
};

#endif // ASCEND950_HIERARCHICAL_ALLTOALL_HOST_H
