
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ascend950_quant_matmul_alltoall_device.h"
#include "ascend950_quant_matmul_alltoall_host.h"
#include "catccos/comm/kernel/ascend950_hierarchical_barrier.hpp"
#include "catccos/dgemm/kernel/ascend950_quant_matmul_alltoall.hpp"
#include "shmem_init.h"
#include "host/data_plane/shmem_host_cc.h"

using namespace AscendC;
using namespace Catccos;
using namespace Catlass;
using namespace tla;
using namespace Catccos::DGemm::Kernel;
using SymLayout = Catccos::DGemm::Kernel::detail::QuantA2ASymLayout;

namespace QmmaHost {

constexpr uint64_t MIB = 1024ULL * 1024;
constexpr uint64_t HEAP_ALIGNMENT = 2 * MIB;
const auto TRACE_START = std::chrono::steady_clock::now();
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

inline int ConfigureShmemLogger(uint32_t rank)
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
    return aclshmemx_set_extern_logger(ShmemLogger);
}

inline void TracePhase(uint32_t rankId, const char* phase, const std::string& detail = {})
{
    const auto elapsedMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - TRACE_START).count();
    std::cout << "[TRACE] t=+" << elapsedMs << "ms rank=" << rankId << " phase=" << phase;
    if (!detail.empty()) {
        std::cout << " " << detail;
    }
    std::cout << std::endl;
}

inline bool SyncStreamChecked(const char* phase, uint32_t rankId, aclrtStream stream)
{
    const aclError status = aclrtSynchronizeStream(stream);
    if (status != ACL_SUCCESS) {
        std::cerr << "[ERROR] rank=" << rankId << " phase=" << phase << " aclrtSynchronizeStream failed: " << status
                  << std::endl;
        return false;
    }
    return true;
}

// The generic SHMEM barrier can trap on a two-host UDMA topology. Use the same
// hierarchical barrier proven by the reference AllToAll operator.
inline bool BarrierAllChecked(
    const char* phase, uint32_t rankId, uint32_t rankSize, uint32_t groupSize, uint32_t& barrierEpoch,
    uint8_t* barrierWorkspace, aclrtStream stream)
{
    const std::string pendingPhase = std::string(phase) + ".before_barrier";
    if (!SyncStreamChecked(pendingPhase.c_str(), rankId, stream)) {
        return false;
    }
    using BarrierKernel = Catccos::Comm::Kernel::HierarchicalBarrier;
    if (barrierWorkspace == nullptr || barrierEpoch == UINT32_MAX) {
        std::cerr << "[ERROR] barrier phase=" << phase << " rank=" << rankId << " invalid workspace/epoch" << std::endl;
        return false;
    }

    const BarrierKernel::Arguments args{rankId, rankSize, groupSize, ++barrierEpoch};
    const BarrierKernel::Params params = BarrierKernel::ToUnderlyingArguments(args, barrierWorkspace);
    const uint64_t fftsAddr = shmemx_get_ffts_config();
    TracePhase(rankId, (std::string(phase) + ".barrier.enter").c_str());
    if (fftsAddr == 0) {
        Catccos::DGemm::Device::KernelAdapter<BarrierKernel><<<1, nullptr, stream>>>(params, nullptr);
    } else {
        Catccos::DGemm::Device::KernelAdapter<BarrierKernel><<<1, nullptr, stream>>>(params, fftsAddr, nullptr);
    }
    if (!SyncStreamChecked(phase, rankId, stream)) {
        return false;
    }
    TracePhase(rankId, (std::string(phase) + ".barrier.done").c_str());
    return true;
}

inline uint64_t AlignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

inline bool ReadMiB(const char* text, uint64_t& result)
{
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
        return false;
    }
    result = value;
    return true;
}

/// 自动堆容量:对称区 + 16MiB SHMEM 控制预留,2MiB 对齐,下限 64MiB;
/// QMMA_SHMEM_HEAP_MB / SHMEM_HEAP_MB 可覆盖(须为偶数 MiB 且不低于下限)。
inline uint64_t SelectHeapBytes(uint64_t symmetricBytes, std::string& error)
{
    constexpr uint64_t reserve = 16 * MIB;
    constexpr uint64_t minimumHeap = 64 * MIB;
    const uint64_t minimum = std::max(minimumHeap, AlignUp(symmetricBytes + reserve, HEAP_ALIGNMENT));
    uint64_t requestedMiB = 0;
    for (const char* name : {"QMMA_SHMEM_HEAP_MB", "SHMEM_HEAP_MB"}) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            continue;
        }
        if (!ReadMiB(value, requestedMiB)) {
            error = std::string(name) + " must be an unsigned integer number of MiB";
            return 0;
        }
        break;
    }

    const uint64_t requested = requestedMiB * MIB;
    if (requested != 0 && (requested < minimum || requested % HEAP_ALIGNMENT != 0)) {
        error = "SHMEM heap must be even and at least " + std::to_string(minimum / MIB) +
                " MiB; leave it unset for automatic sizing";
        return 0;
    }
    return requested == 0 ? minimum : requested;
}

} // namespace QmmaHost

using ElementAScale = float8_e8m0_t;
using ElementBScale = float8_e8m0_t;
using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutY = Catlass::layout::RowMajor;

using ConfigFp8 = Ascend950QuantMatmulAllToAllConfigFp8<
    float8_e4m3_t, LayoutA, float8_e4m3_t, LayoutB, ElementAScale, LayoutA, ElementBScale, LayoutB, bfloat16_t,
    LayoutY>;
using ConfigFp4 = Ascend950QuantMatmulAllToAllConfigFp4<
    float4_e2m1x2_t, LayoutA, float4_e2m1x2_t, LayoutB, ElementAScale, LayoutA, ElementBScale, LayoutB, bfloat16_t,
    LayoutY>;
using ConfigBf16 =
    Ascend950QuantMatmulAllToAllConfigBf16<bfloat16_t, LayoutA, bfloat16_t, LayoutB, bfloat16_t, LayoutY>;

struct Options {
    static constexpr auto HELPER =
        "Usage: ascend950_quant_matmul_alltoall rank_size rank_id ip_port m N K [data_path] [device_list] "
        "[fmt] [slot_num] [lane_num] [ranks_per_node]\n";

    int rankSize{0};
    int rankId{-1};
    std::string ipPort;
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::string dataPath{"."};
    std::vector<int> deviceIdList;
    uint32_t fmt{0};
    uint32_t laneNum{1};
    uint32_t commInterval{1};
    bool commIntervalAuto{true};
    uint32_t ranksPerNode{CATCCOS_HIERARCHICAL_GROUP_SIZE};

    static bool ParseInt(const char* text, int& result)
    {
        char* end = nullptr;
        errno = 0;
        const long value = std::strtol(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0' || value < INT_MIN || value > INT_MAX) {
            return false;
        }
        result = static_cast<int>(value);
        return true;
    }

    static bool ParseUint32(const char* text, uint32_t& result)
    {
        if (text == nullptr || text[0] == '\0' || text[0] == '-') {
            return false;
        }
        char* end = nullptr;
        errno = 0;
        const unsigned long long value = std::strtoull(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
            return false;
        }
        result = static_cast<uint32_t>(value);
        return true;
    }

    int Parse(int argc, char** argv)
    {
        if (argc < 7 || argc > 13) {
            std::printf(HELPER);
            return -1;
        }
        if (!ParseInt(argv[1], rankSize) || !ParseInt(argv[2], rankId) || !ParseUint32(argv[4], m) ||
            !ParseUint32(argv[5], n) || !ParseUint32(argv[6], k)) {
            return -1;
        }
        ipPort = argv[3];
        if (argc > 7 && argv[7][0] != '\0') {
            dataPath = argv[7];
        }
        if (argc > 8 && argv[8][0] != '\0') {
            std::istringstream devices(argv[8]);
            std::string token;
            while (std::getline(devices, token, ',')) {
                int deviceId = -1;
                if (!ParseInt(token.c_str(), deviceId) || deviceId < 0) {
                    return -1;
                }
                deviceIdList.push_back(deviceId);
            }
        } else {
            for (int rank = 0; rank < rankSize && rank < 32; ++rank) {
                deviceIdList.push_back(rank);
            }
        }

        // argv[10] is the legacy slot argument and is intentionally ignored.
        if ((argc > 9 && !ParseUint32(argv[9], fmt)) || (argc > 11 && !ParseUint32(argv[11], laneNum)) ||
            (argc > 12 && !ParseUint32(argv[12], ranksPerNode))) {
            return -1;
        }
        const char* interval = std::getenv("QMMA_OPT_COMM_INTERVAL");
        if (interval != nullptr && interval[0] != '\0') {
            if (!ParseUint32(interval, commInterval)) {
                std::cerr << "QMMA_OPT_COMM_INTERVAL must be an unsigned integer\n";
                return -1;
            }
            commIntervalAuto = false;
        }
        return 0;
    }
};

static bool CheckLocalUdmaRoutes(uint32_t rankId, uint32_t rankSize, uint32_t groupSize, bool required, uint32_t qpNum)
{
    bool valid = true;
    for (uint32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rankId || peer / groupSize != rankId / groupSize) {
            continue;
        }
        for (uint32_t qp = 0; qp < qpNum; ++qp) {
            aclshmemx_net_layer_t layer{};
            const int rc = aclshmemx_get_qp_net_layer(ACLSHMEM_DATA_OP_UDMA, static_cast<int>(peer), qp, &layer);
            if (rc != ACLSHMEM_SUCCESS) {
                std::cerr << (required ? "[ERROR]" : "[WARN]") << " rank=" << rankId
                          << " has no UDMA route to local peer " << peer << " QP " << qp << " rc=" << rc << std::endl;
                valid = false;
            }
        }
    }
    return valid || !required;
}

/// 单次融合 Kernel 调用:epoch 固定为 1(当前实现按单次调用设计,
/// 无跨调用 ticket 推进)→ Initialize → Run(下发 Kernel 并同步 stream)。
template <class Config>
static bool RunQuantA2A(
    const Options& opt, uint32_t blockNum, uint8_t* symmetricPtr, const KernelParams& kernelParams, aclrtStream stream)
{
    using DeviceOp = typename Config::Device;
    typename DeviceOp::Arguments args{
        GemmCoord{opt.m, opt.n, opt.k},
        static_cast<uint32_t>(opt.rankId),
        static_cast<uint32_t>(opt.rankSize),
        opt.ranksPerNode,
        blockNum,
        opt.commInterval,
        opt.laneNum,
        1U,
        kernelParams.ptrA,
        kernelParams.ptrB,
        kernelParams.customPtrs[0],
        kernelParams.customPtrs[1],
        kernelParams.ptrC,
        symmetricPtr,
    };

    DeviceOp deviceOp;
    deviceOp.Initialize(args);
    if (deviceOp.Run(stream, blockNum, shmemx_get_ffts_config()) != Catlass::Status::kSuccess) {
        std::cerr << "[ERROR] compute kernel launch failed: rank=" << opt.rankId << std::endl;
        return false;
    }
    return true;
}

/// 按 fmt 选择模板档位:0=MXFP8,1=MXFP4,2=BF16;三档共用同一 Kernel。
template <class... Args>
static bool RunConfiguredKernel(const Options& opt, Args&&... args)
{
    switch (opt.fmt) {
        case 0:
            return RunQuantA2A<ConfigFp8>(opt, std::forward<Args>(args)...);
        case 1:
            return RunQuantA2A<ConfigFp4>(opt, std::forward<Args>(args)...);
        case 2:
            return RunQuantA2A<ConfigBf16>(opt, std::forward<Args>(args)...);
        default:
            return false;
    }
}

int main(int argc, char** argv)
{
    std::cout << std::unitbuf;
    Options options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }
    if (options.rankSize < 2 || options.rankSize > 32 || options.rankId < 0 || options.rankId >= options.rankSize) {
        std::cerr << "[ERROR] rank_size in [2, 32] and rank_id in [0, rank_size) required\n";
        return 1;
    }
    if (options.m == 0 || options.n == 0 || options.k == 0 || options.fmt > 2 || options.ipPort.empty()) {
        std::cerr << "[ERROR] positive m/N/K, fmt in [0, 2], and a nonempty ip_port required\n";
        return 1;
    }
    if (options.ranksPerNode == 0 || options.ranksPerNode > 16 ||
        options.ranksPerNode != CATCCOS_HIERARCHICAL_GROUP_SIZE || options.rankSize > 2 * options.ranksPerNode) {
        std::cerr << "[ERROR] ranks_per_node must match CATCCOS_HIERARCHICAL_GROUP_SIZE="
                  << CATCCOS_HIERARCHICAL_GROUP_SIZE << " and cover one or two nodes\n";
        return 1;
    }

    const uint32_t rankSize = static_cast<uint32_t>(options.rankSize);
    const uint32_t rankId = static_cast<uint32_t>(options.rankId);
    const uint32_t localRankId = rankId % options.ranksPerNode;
    const uint32_t nodeBase = rankId - localRankId;
    const uint32_t localRankCount = std::min(options.ranksPerNode, rankSize - nodeBase);
    if (options.deviceIdList.size() < localRankCount) {
        std::cerr << "[ERROR] device_list must contain all " << localRankCount << " local ranks\n";
        return 1;
    }
    for (uint32_t i = 0; i < localRankCount; ++i) {
        for (uint32_t j = 0; j < i; ++j) {
            if (options.deviceIdList[i] == options.deviceIdList[j]) {
                std::cerr << "[ERROR] device_list contains duplicate local devices\n";
                return 1;
            }
        }
    }
    const int32_t deviceId = options.deviceIdList[localRankId];

    // 未显式设置轮粒度时按 M 自适应:H = commInterval×128,上限 4096 行,
    // 用更少、更大的轮摊薄每个 peer 的控制开销。
    if (options.commIntervalAuto) {
        options.commInterval = std::min<uint32_t>(32, std::max<uint32_t>(1, (options.m + 255) / 256));
    }
    // ---- tiling 参数与算子校验(slotNum 已废弃:TX 槽按轮数自动决定) ----
    QuantMatmulAllToAllTilingParams cocTiling;
    cocTiling.m = options.m;
    cocTiling.n = options.n;
    cocTiling.k = options.k;
    cocTiling.fmt = options.fmt;
    cocTiling.m0 = 128;
    cocTiling.n0 = 256;
    cocTiling.k0 = options.fmt == 1 ? 512 : 256;
    cocTiling.commInterval = options.commInterval;
    cocTiling.laneNum = options.laneNum;
    cocTiling.ranksPerNode = options.ranksPerNode;
    cocTiling.rankSize = rankSize;
    cocTiling.slotNum = 0;

    auto op = OperatorRegistry::Instance().CreateOperator("Ascend950QuantMatmulAllToAll");
    auto* quantOp = dynamic_cast<Ascend950QuantMatmulAllToAllOperator*>(op.get());
    if (quantOp == nullptr) {
        std::cerr << "[ERROR] operator Ascend950QuantMatmulAllToAll not found\n";
        return 1;
    }
    if (!quantOp->ValidateQuantTilingParams(rankSize, cocTiling)) {
        std::cerr << "[ERROR] invalid tiling or problem shape: M=" << options.m << " N=" << options.n
                  << " K=" << options.k << " P=" << rankSize << " lane=" << options.laneNum
                  << " comm_interval=" << options.commInterval << std::endl;
        return 1;
    }

    const char* fmtName = options.fmt == 0 ? "mxfp8" : (options.fmt == 1 ? "mxfp4" : "bf16");
    std::cout << "[TEST] rank_size: " << rankSize << " rank_id: " << rankId << " ip: " << options.ipPort
              << " m: " << options.m << " N: " << options.n << " K: " << options.k << " fmt: " << fmtName
              << " lane: " << options.laneNum << " comm_interval: " << options.commInterval << std::endl;

    // ---- 初始化 ACL 并探测资源:AIV context 总数(blockNum×2)必须能覆盖
    // peer 数——PUT 模式下一 AIV 绑一 peer,worker 不足会把搬运串行化 ----
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    const uint32_t blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    if (blockNum == 0 || blockNum > 64 || blockNum * 2 < rankSize) {
        std::cerr << "[ERROR] insufficient AIC/AIV resources: aic_groups=" << blockNum << " peer_count=" << rankSize
                  << std::endl;
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());
        return 1;
    }

    // ---- 多卡:对称区 = [TX 各轮][Y 输出][Ready 工作区],Y 在对称区供对端 PUT 直写 ----
    QmmaHost::TracePhase(rankId, "layout.begin");
    const uint32_t h = options.commInterval * cocTiling.m0;
    const uint32_t q = options.n / rankSize;
    const uint32_t rounds = static_cast<uint32_t>((static_cast<uint64_t>(options.m) + h - 1) / h);
    const uint64_t yOffset = SymLayout::TxBytes(rounds, rankSize, h, q, sizeof(bfloat16_t));
    const uint64_t yBytes = SymLayout::YBytes(options.m, rankSize, q, sizeof(bfloat16_t));
    const uint64_t symmetricBytes =
        QmmaHost::AlignUp(SymLayout::TotalBytes(rounds, rankSize, options.m, h, q, sizeof(bfloat16_t), blockNum), 4096);
    const uint32_t groupSize = options.ranksPerNode;
    using BarrierKernel = Catccos::Comm::Kernel::HierarchicalBarrier;
    const BarrierKernel::Arguments barrierArgs{rankId, rankSize, groupSize, 0};
    const uint64_t barrierBytes = BarrierKernel::GetWorkspaceSize(barrierArgs);
    QmmaHost::TracePhase(
        rankId, "layout.done",
        "rounds=" + std::to_string(rounds) + " h=" + std::to_string(h) + " q=" + std::to_string(q) +
            " tx_bytes=" + std::to_string(yOffset) + " y_bytes=" + std::to_string(yBytes));
    std::string heapError;
    QmmaHost::TracePhase(rankId, "heap.size.begin", "symmetric_bytes=" + std::to_string(symmetricBytes));
    const uint64_t heapBytes = QmmaHost::SelectHeapBytes(symmetricBytes + barrierBytes, heapError);
    QmmaHost::TracePhase(rankId, "heap.size.done", "heap_bytes=" + std::to_string(heapBytes));
    if (heapBytes == 0) {
        std::cerr << "[ERROR] " << heapError << std::endl;
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());
        return 1;
    }
    std::cout << "[TEST] symmetric bytes: " << symmetricBytes << " (TX " << yOffset << " + Y " << yBytes
              << "); barrier bytes: " << barrierBytes << "; heap bytes: " << heapBytes << "; group size: " << groupSize
              << std::endl;

    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    KernelParams kernelParams;
    QmmaHost::TracePhase(rankId, "acl.stream.created");

    const int loggerStatus = QmmaHost::ConfigureShmemLogger(rankId);
    if (loggerStatus != ACLSHMEM_SUCCESS) {
        std::cerr << "[WARN] rank=" << rankId << " aclshmemx_set_extern_logger failed: " << loggerStatus << std::endl;
    } else {
        QmmaHost::TracePhase(rankId, "shmem.logger.ready");
    }

    const bool crossNode = rankSize > options.ranksPerNode;
    // 数据 QP 数 = laneNum(无独立控制 QP——尾部 Ready 通知先 quiet 全部
    // payload QP 后复用 QP0 推通知,因此这里不需要额外 +1)
    const uint32_t qpNum = options.laneNum;
    QmmaHost::TracePhase(rankId, "shmem.qp_num.begin", "qp_num=" + std::to_string(qpNum));
    int qpStatus = aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qpNum);
    if (crossNode) {
        const int roceStatus = aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE, qpNum);
        if (qpStatus == ACLSHMEM_SUCCESS) {
            qpStatus = roceStatus;
        }
    }
    if (qpStatus != ACLSHMEM_SUCCESS) {
        std::cerr << "[ERROR] aclshmemx_set_qp_num failed: " << qpStatus << std::endl;
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());
        return 1;
    }
    QmmaHost::TracePhase(rankId, "shmem.qp_num.done", "status=" + std::to_string(qpStatus));

    aclshmemx_init_attr_t attributes{};
    aclshmemx_uniqueid_t uniqueId{};
    QmmaHost::TracePhase(rankId, "shmem.attributes.begin");
    const int attrStatus = set_attr(
        static_cast<int32_t>(rankId), static_cast<int32_t>(rankSize), heapBytes, options.ipPort.c_str(), &attributes,
        &uniqueId);
    if (attrStatus != ACLSHMEM_SUCCESS) {
        std::cerr << "[ERROR] set_attr failed: " << attrStatus << std::endl;
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());
        return 1;
    }
    QmmaHost::TracePhase(
        rankId, "shmem.attributes.ready",
        "qp_num=" + std::to_string(qpNum) + " cross_node=" + (crossNode ? "true" : "false"));
    attributes.option_attr.data_op_engine_type =
        crossNode ? static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_UDMA | ACLSHMEM_DATA_OP_ROCE) :
                    ACLSHMEM_DATA_OP_UDMA;
    attributes.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_DISABLED;

    QmmaHost::TracePhase(
        rankId, "shmem.init.begin",
        "qp_num=" + std::to_string(qpNum) + " cross_node=" + (crossNode ? "true" : "false"));
    const int status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    QmmaHost::TracePhase(rankId, "shmem.init.end", "status=" + std::to_string(status));
    if (status != ACLSHMEM_SUCCESS) {
        std::cerr << "[ERROR] aclshmemx_init_attr failed: " << status << " bootstrap=" << options.ipPort << std::endl;
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());
        return 1;
    }
    QmmaHost::TracePhase(rankId, "shmem.route_check.begin");
    if (!CheckLocalUdmaRoutes(rankId, rankSize, options.ranksPerNode, crossNode, options.laneNum)) {
        std::cerr << "[ERROR] rank=" << rankId << " missing a local UDMA route; aborting\n";
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());
        return 1;
    }
    QmmaHost::TracePhase(rankId, "shmem.route_check.done");

    // Match the proven operator: initialize SHMEM before allocating and loading
    // the large A/B inputs so initialization is not displaced by tensor traffic.
    auto abortAfterShmemInit = [&]() {
        // Once a world-level setup or kernel exchange has started, shmem_finalize()
        // can wait for a peer that already failed. Match the proven operator and
        // release only process-local state before the launcher tears down the ranks.
        kernelParams.ptrC = nullptr;
        FreeDeviceSpace(kernelParams);
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());
    };

    QmmaHost::TracePhase(rankId, "host.alloc.begin");
    try {
        quantOp->AllocateDeviceSpace(kernelParams, cocTiling, rankId, options.dataPath);
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] rank=" << rankId << " host input allocation failed: " << error.what() << std::endl;
        abortAfterShmemInit();
        return 1;
    }
    QmmaHost::TracePhase(rankId, "host.alloc.done");

    // ---- 分配对称区并把 Y 重定向进去:对端 PUT 的目的地址就在这里 ----
    QmmaHost::TracePhase(rankId, "shmem.symm_alloc.begin");
    void* symmetricMemory = aclshmem_align(SymLayout::FLAG_BYTES, symmetricBytes);
    if (symmetricMemory == nullptr) {
        std::cerr << "[ERROR] symmetric allocation failed: " << symmetricBytes << " bytes\n";
        abortAfterShmemInit();
        return 1;
    }
    uint8_t* symmetricPtr = static_cast<uint8_t*>(symmetricMemory);
    QmmaHost::TracePhase(rankId, "shmem.symm_alloc.done", "bytes=" + std::to_string(symmetricBytes));
    QmmaHost::TracePhase(rankId, "shmem.barrier_alloc.begin");
    auto* barrierWorkspace = static_cast<uint8_t*>(aclshmem_calloc(1, barrierBytes));
    if (barrierWorkspace == nullptr) {
        std::cerr << "[ERROR] barrier workspace allocation failed: " << barrierBytes << " bytes\n";
        abortAfterShmemInit();
        return 1;
    }
    QmmaHost::TracePhase(rankId, "shmem.barrier_alloc.done", "bytes=" + std::to_string(barrierBytes));
    kernelParams.ptrC = symmetricPtr + yOffset;
    // TX/Y are fully overwritten by AIC/PUT. Only Ready mailboxes and producer
    // tickets require zero initialization. Avoid a multi-GiB host-side memset.
    const uint64_t controlOffset = yOffset + yBytes;
    const uint64_t controlBytes = symmetricBytes - controlOffset;
    QmmaHost::TracePhase(rankId, "shmem.control_clear.begin", "bytes=" + std::to_string(controlBytes));
    const aclError clearStatus = aclrtMemset(symmetricPtr + controlOffset, controlBytes, 0, controlBytes);
    if (clearStatus != ACL_SUCCESS) {
        std::cerr << "[ERROR] rank=" << rankId << " control-area memset failed: " << clearStatus << std::endl;
        abortAfterShmemInit();
        return 1;
    }
    QmmaHost::TracePhase(rankId, "shmem.control_clear.done", "bytes=" + std::to_string(controlBytes));

    uint32_t barrierEpoch = 0;
    if (!QmmaHost::BarrierAllChecked("init", rankId, rankSize, groupSize, barrierEpoch, barrierWorkspace, stream)) {
        std::cerr << "[ERROR] rank=" << rankId << " aborting without shmem_finalize after init barrier" << std::endl;
        abortAfterShmemInit();
        return 1;
    }

    // ---- 单次调用:kernel 内部已含逐轮生产票同步与尾部 Ready 收账
    // (AIV0 的 WaitAll 确认所有对端推完),host 只需同步 stream ----
    QmmaHost::TracePhase(rankId, "compute.kernel.submit");
    const bool launched = RunConfiguredKernel(options, blockNum, symmetricPtr, kernelParams, stream);
    if (!launched) {
        abortAfterShmemInit();
        return 1;
    }
    if (!QmmaHost::SyncStreamChecked("compute.kernel", rankId, stream)) {
        abortAfterShmemInit();
        return 1;
    }
    QmmaHost::TracePhase(rankId, "compute.kernel.sync_returned");

    if (!QmmaHost::BarrierAllChecked("final", rankId, rankSize, groupSize, barrierEpoch, barrierWorkspace, stream)) {
        std::cerr << "[ERROR] rank=" << rankId << " aborting without shmem_finalize after final barrier" << std::endl;
        abortAfterShmemInit();
        return 1;
    }

    QmmaHost::TracePhase(rankId, "result.write.begin");
    try {
        quantOp->WriteResultFile(kernelParams, cocTiling, rankId, options.dataPath);
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] rank=" << rankId << " result write failed: " << error.what() << std::endl;
        abortAfterShmemInit();
        return 1;
    }
    QmmaHost::TracePhase(rankId, "result.write.done");
    kernelParams.ptrC = nullptr;
    FreeDeviceSpace(kernelParams);

    // The hierarchical final barrier only synchronizes kernel completion. It is
    // not a replacement for the world-team release barrier: output writeback and
    // device cleanup can still finish at different times across ranks, and
    // freeing the SHMEM heap while a peer is still releasing resources can make
    // shmem_free wait until timeout. Keep this barrier immediately before the
    // collective SHMEM frees.
    QmmaHost::TracePhase(rankId, "shmem.release_barrier.enter");
    aclshmem_barrier_all();
    QmmaHost::TracePhase(rankId, "shmem.release_barrier.done");

    shmem_free(barrierWorkspace);
    shmem_free(symmetricMemory);
    QmmaHost::TracePhase(rankId, "shmem.finalize.begin");
    const int finalizeStatus = shmem_finalize();
    QmmaHost::TracePhase(rankId, "shmem.finalize.done", "status=" + std::to_string(finalizeStatus));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    if (rankId == 0) {
        std::printf("test finished\n");
    }
    return finalizeStatus == ACLSHMEM_SUCCESS ? 0 : 1;
}
