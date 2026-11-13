/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascend950_mxfp8_alltoall_matmul_device.h"
#include "ascend950_mxfp8_alltoall_matmul_host.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#ifdef ENABLE_TIMER
#include <filesystem>
#include "AscendTimer_host.hpp"
#endif

using namespace AscendC;
using namespace Catccos;
using namespace MxFp8AllToAllMatmulExample;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::ColumnMajor;
using LayoutMxScaleA = Catlass::layout::RowMajor;
using LayoutMxScaleB = Catlass::layout::ColumnMajor;
using LayoutC = Catlass::layout::RowMajor;
using ElementA = float8_e4m3_t;
using ElementB = float8_e4m3_t;
using ElementMxScaleA = float8_e8m0_t;
using ElementMxScaleB = float8_e8m0_t;
using ElementC = bfloat16_t;

namespace {
uint32_t ParseUint(const std::string& text)
{
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("invalid unsigned integer: " + text);
    }
    const uint64_t value = std::stoull(text);
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("integer exceeds uint32: " + text);
    }
    return static_cast<uint32_t>(value);
}

uint32_t ReadEnvUint(const char* name, uint32_t fallback)
{
    const char* value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? fallback : ParseUint(value);
}

uint64_t SelectHeapBytes(uint64_t symmetricBytes, uint64_t barrierBytes)
{
    constexpr uint64_t mib = 1024ULL * 1024;
    constexpr uint64_t reserve = 16 * mib;
    constexpr uint64_t alignment = 2 * mib;
    const uint64_t minimum =
        std::max(64 * mib, AllToAllLayout::AlignUp(symmetricBytes + barrierBytes + reserve, alignment));
    const uint64_t requested = static_cast<uint64_t>(ReadEnvUint("SHMEM_HEAP_MB", 0)) * mib;
    if (requested != 0 && (requested < minimum || requested % alignment != 0)) {
        throw std::invalid_argument(
            "SHMEM_HEAP_MB must be even and >= " + std::to_string(minimum / mib) + "; use 0 for automatic sizing");
    }
    return requested == 0 ? minimum : requested;
}

uint64_t NowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Cleanup is local to the executable. KernelParams owns ACL allocations, while
// symmetric is released with SHMEM; no interior SHMEM pointer is passed to aclrtFree.
struct RuntimeResources {
    bool aclInitialized = false;
    bool deviceSet = false;
    bool shmemInitialized = false;
    int device = 0;
    aclrtStream stream = nullptr;
    uint8_t* symmetric = nullptr;
    uint8_t* barrierWorkspace = nullptr;
    uint8_t* timerBuffer = nullptr;
    uint32_t barrierEpoch = 0;
    KernelParams kernelParams;

    ~RuntimeResources()
    {
        // A failed rank must report its error and exit. Collective finalization
        // would wait for peers still blocked in the failed communication round.
        // The process exit releases device resources on this exceptional path.
        if (std::uncaught_exceptions() != 0) {
            return;
        }
        if (stream) {
            (void)aclrtSynchronizeStream(stream);
        }
        FreeDeviceSpace(kernelParams);
        if (timerBuffer) {
            (void)aclrtFree(timerBuffer);
        }
        if (kernelParams.customPtrs != nullptr) {
            (void)aclrtFreeHost(kernelParams.customPtrs);
            kernelParams.customPtrs = nullptr;
            kernelParams.customCount = 0;
        }
        if (barrierWorkspace) {
            aclshmem_free(barrierWorkspace);
        }
        if (symmetric) {
            aclshmem_free(symmetric);
        }
        if (shmemInitialized) {
            (void)aclshmem_finalize();
        }
        if (stream) {
            (void)aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            (void)aclrtResetDevice(device);
        }
        if (aclInitialized) {
            (void)aclFinalize();
        }
    }
};

void WorldBarrier(
    RuntimeResources& resources, uint32_t rankId, uint32_t rankSize, uint32_t groupSize, const char* operation)
{
    using BarrierKernel = Comm::Kernel::HierarchicalBarrier;
    using BarrierDevice = DGemm::Device::DeviceDGemm<BarrierKernel>;
    if (resources.barrierEpoch == UINT32_MAX) {
        throw std::runtime_error("barrier epoch exhausted");
    }
    const BarrierKernel::Arguments args{rankId, rankSize, groupSize, ++resources.barrierEpoch};
    BarrierDevice device;
    Check(static_cast<int>(device.Initialize(args, resources.barrierWorkspace)), operation);
    DGemm::Device::KernelAdapter<BarrierKernel>
        <<<1, nullptr, resources.stream>>>(device.params(), shmemx_get_ffts_config(), nullptr);
    Check(aclrtSynchronizeStream(resources.stream), operation);
}

template <uint32_t TileK>
void RunKernel(
    const CocTilingParams& cocTiling, uint32_t rankId, uint32_t groupSize, const KernelParams& params,
    uint8_t* gmSymmetric, RuntimeResources& resources, uint32_t blockNum, uint32_t iterations)
{
    using Config = Ascend950MxFp8AllToAllMatmulConfig<
        ElementA, LayoutA, ElementB, LayoutB, ElementMxScaleA, LayoutMxScaleA, ElementMxScaleB, LayoutMxScaleB,
        ElementC, LayoutC, TileK>;
    using DeviceOp = typename Config::Device;
    aclrtStream stream = resources.stream;
    const auto layout = GetLayout(cocTiling);
    Catlass::GemmCoord problemShape{cocTiling.m, cocTiling.n, cocTiling.k};
    typename DeviceOp::Arguments args{
        problemShape,
        rankId,
        cocTiling.rankSize,
        cocTiling.commTileM,
        gmSymmetric + layout.sourceA,
        params.ptrB,
        gmSymmetric + layout.sourceScaleA,
        params.customPtrs[1],
        params.ptrC,
        gmSymmetric};
    DeviceOp deviceOp;
    deviceOp.Initialize(args);
    const auto fftsAddr = shmemx_get_ffts_config();
    std::cout << "READY rank=" << rankId << " M=" << layout.chunkM << " input_rows=" << layout.m
              << " iterations=" << iterations << std::endl;
    struct Timing {
        uint64_t begin;
        uint64_t done;
    };
    std::vector<Timing> timings;
    timings.reserve(iterations);
#ifdef ENABLE_TIMER
    AscendTimerHost::ConfigureCoreNum(static_cast<int>(blockNum));
    const size_t timerBytes = AscendTimerHost::GetTotalTimerBufferSize() * sizeof(int64_t);
    Check(
        aclrtMalloc(reinterpret_cast<void**>(&resources.timerBuffer), timerBytes, ACL_MEM_MALLOC_HUGE_FIRST),
        "allocate timer buffer");
    Check(aclrtMemset(resources.timerBuffer, timerBytes, 0, timerBytes), "clear timer buffer");
    const uint32_t traceIteration = ReadEnvUint("TRACE_ITERATION", iterations - 1);
    if (traceIteration >= iterations) {
        throw std::invalid_argument("TRACE_ITERATION must be less than ITERATIONS");
    }
    const char* traceEnv = std::getenv("TRACE_DIR");
    const std::string traceDir = traceEnv == nullptr ? "output_timer" : traceEnv;
    std::filesystem::create_directories(traceDir);
#endif
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        uint8_t* iterationTimer = nullptr;
#ifdef ENABLE_TIMER
        // Only the selected call records events; warmup calls pass a null timer.
        if (iteration == traceIteration) {
            iterationTimer = resources.timerBuffer;
        }
#endif
        Timing timing;
        WorldBarrier(resources, rankId, cocTiling.rankSize, groupSize, "iteration finish barrier");
        timing.begin = NowNs();
        ResetSyncBuffer(cocTiling, gmSymmetric, stream);
        StageInputs(params, cocTiling, gmSymmetric, stream);
        // This barrier follows input preparation on the same stream.
        WorldBarrier(resources, rankId, cocTiling.rankSize, groupSize, "iteration start barrier");
        // DeviceDGemm::Run includes stream synchronization. Use its adapter here
        // to keep the completion boundary explicit in the operator timing.
        DGemm::Device::KernelAdapter<typename DeviceOp::Kernel>
            <<<blockNum, nullptr, stream>>>(deviceOp.params(), fftsAddr, iterationTimer);
        const auto kernelStatus = aclrtSynchronizeStream(stream);
        if (kernelStatus != ACL_SUCCESS) {
            const size_t snapshotBytes = layout.readyBytes;
            std::vector<uint8_t> snapshot(snapshotBytes);
            const auto copyStatus = aclrtMemcpy(
                snapshot.data(), snapshotBytes, gmSymmetric + layout.ready, snapshotBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            std::cerr << "READY_SNAPSHOT rank=" << rankId << " copy_status=" << copyStatus;
            if (copyStatus == ACL_SUCCESS) {
                for (uint32_t source = 0; source < cocTiling.rankSize; ++source) {
                    uint32_t value = 0;
                    const size_t offset = layout.ReadyOffset(source) - layout.ready;
                    std::memcpy(&value, snapshot.data() + offset, sizeof(value));
                    std::cerr << " source" << source << "=" << value;
                }
                uint32_t frontier = 0;
                std::memcpy(&frontier, snapshot.data() + (layout.FrontierOffset() - layout.ready), sizeof(frontier));
                std::cerr << " frontier=" << frontier;
            }
            std::cerr << std::endl;
        }
        Check(kernelStatus, "fused kernel");
        timing.done = NowNs();
        timings.push_back(timing);
#ifdef ENABLE_TIMER
        std::cout << "TRACE_MODE rank=" << rankId << " iter=" << iteration << " enabled=" << (iterationTimer != nullptr)
                  << std::endl;
        if (iteration == traceIteration) {
            const auto result = AscendTimerHost::GetTimeResult(resources.timerBuffer);
            const std::string prefix =
                traceDir + "/rank" + std::to_string(rankId) + "_iter" + std::to_string(iteration);
            if (!AscendTimerHost::WriteAllTimerResults(prefix + "_raw.csv", prefix + "_duration.csv", result)) {
                throw std::runtime_error("timer CSV export failed");
            }
            std::cout << "TRACE rank=" << rankId << " iter=" << iteration << " raw=" << prefix << "_raw.csv"
                      << " cycles_per_us=" << AscendTimerHost::kCycleToTimeBase
                      << " clock_domain=rank_local max_events=" << AscendTimer::MAX_DYNAMIC_ITER << std::endl;
        }
#endif
    }
    WorldBarrier(resources, rankId, cocTiling.rankSize, groupSize, "benchmark finish barrier");
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "SHAPE rank=" << rankId << " world=" << cocTiling.rankSize << " M=" << layout.chunkM
              << " input_rows=" << layout.m << " K=" << layout.fullK << " N=" << cocTiling.n
              << " localK=" << layout.localK << " tileK=" << TileK << " window_rows=" << layout.windowRows
              << " windows=" << layout.windowCount << " symmetric_bytes=" << layout.symmetricBytes
              << " c_bytes=" << layout.CBytes() << "\n";
    for (uint32_t i = 0; i < timings.size(); ++i) {
        const auto& t = timings[i];
        std::cout << "TIMING rank=" << rankId << " iter=" << i << " M=" << layout.chunkM
                  << " world=" << cocTiling.rankSize << " group_size=" << groupSize << " K=" << layout.fullK
                  << " N=" << layout.n << " window_rows=" << layout.windowRows << " timing_scope=gm_input_to_output"
                  << " begin_ns=" << t.begin << " done_ns=" << t.done << " total_ms=" << (t.done - t.begin) / 1e6
                  << "\n";
    }
    std::cout.flush();
}
} // namespace

struct Options {
    static constexpr auto HELP_SUFFIX = " rank_size rank_id ip_port M N K [data_path] [device_id_list]\n"
                                        "  M: output rows per rank; N: output columns; K: full K after AllToAll.\n";
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t groupSize = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t iterations = 20;
    uint32_t windowRows = 512;
    std::string ipPort;
    std::string dataPath;
    std::vector<int> deviceIdList;

    int Parse(int argc, char** argv)
    {
        enum class ArgsIndex { RANK_SIZE = 1, RANK_ID, IP_PORT, M, N, K, DATA_PATH, DEVICE_LIST, INDEX_MAX };
        if (argc < static_cast<int>(ArgsIndex::DATA_PATH) || argc > static_cast<int>(ArgsIndex::INDEX_MAX)) {
            std::cerr << "Usage: " << EXECUTABLE_NAME << HELP_SUFFIX;
            return -1;
        }
        try {
            rankSize = ParseUint(argv[static_cast<int>(ArgsIndex::RANK_SIZE)]);
            rankId = ParseUint(argv[static_cast<int>(ArgsIndex::RANK_ID)]);
            ipPort = argv[static_cast<int>(ArgsIndex::IP_PORT)];
            m = ParseUint(argv[static_cast<int>(ArgsIndex::M)]);
            n = ParseUint(argv[static_cast<int>(ArgsIndex::N)]);
            k = ParseUint(argv[static_cast<int>(ArgsIndex::K)]);
            if (argc > static_cast<int>(ArgsIndex::DATA_PATH)) {
                dataPath = argv[static_cast<int>(ArgsIndex::DATA_PATH)];
            }
            if (const char* count = std::getenv("ITERATIONS")) {
                iterations = ParseUint(count);
            }
            if (const char* rows = std::getenv("WINDOW_ROWS")) {
                windowRows = ParseUint(rows);
            }
            if (windowRows != 256 && windowRows != 512 && windowRows != 1024) {
                throw std::invalid_argument("WINDOW_ROWS must be 256, 512, or 1024; MatMul tiles remain 256 rows");
            }
            if ((rankSize != 1 && rankSize != 2 && rankSize != 4 && rankSize != 8 && rankSize != 16 &&
                 rankSize != AllToAllLayout::MAX_RANKS) ||
                rankId >= rankSize || iterations == 0 || iterations > 1000 || ipPort.compare(0, 6, "tcp://") != 0) {
                throw std::invalid_argument("use 1/2/4/8/16/32 ranks, a tcp endpoint and ITERATIONS in [1,1000]");
            }
            if (m == 0 || m > 300000 || n == 0 || n % 16 != 0 || k == 0 || k % (rankSize * 64) != 0) {
                throw std::invalid_argument("require M in [1,300000], N positive and divisible by 16, "
                                            "K positive and divisible by rank_size*64");
            }
            const size_t separator = ipPort.rfind(':');
            if (separator == std::string::npos || separator <= 6) {
                throw std::invalid_argument("invalid ip_port");
            }
            const uint32_t port = ParseUint(ipPort.substr(separator + 1));
            if (port == 0 || port > 65535) {
                throw std::invalid_argument("invalid bootstrap port");
            }
            if (argc > static_cast<int>(ArgsIndex::DEVICE_LIST)) {
                const std::string devices = argv[static_cast<int>(ArgsIndex::DEVICE_LIST)];
                if (devices.empty() || devices.back() == ',') {
                    throw std::invalid_argument("invalid device list");
                }
                std::istringstream list(devices);
                std::string token;
                while (std::getline(list, token, ',')) {
                    const uint32_t device = ParseUint(token);
                    if (device >= AllToAllLayout::MAX_LOCAL_DEVICES ||
                        std::find(deviceIdList.begin(), deviceIdList.end(), static_cast<int>(device)) !=
                            deviceIdList.end()) {
                        throw std::invalid_argument("device IDs must be distinct and in [0,15]");
                    }
                    deviceIdList.push_back(static_cast<int>(device));
                }
            } else {
                const uint32_t localRanks = std::min(rankSize, AllToAllLayout::MAX_LOCAL_DEVICES);
                for (uint32_t device = 0; device < localRanks; ++device) {
                    deviceIdList.push_back(static_cast<int>(device));
                }
            }
            groupSize = static_cast<uint32_t>(deviceIdList.size());
            if (groupSize == 0 || groupSize > AllToAllLayout::MAX_LOCAL_DEVICES ||
                (rankSize != groupSize && rankSize != 2 * groupSize)) {
                throw std::invalid_argument("use one or two nodes with equal local device counts");
            }
            if (groupSize != CATCCOS_MXFP8_ALLTOALL_GROUP_SIZE) {
                throw std::invalid_argument(
                    "local device count must match compiled GROUP_SIZE=" +
                    std::to_string(CATCCOS_MXFP8_ALLTOALL_GROUP_SIZE) +
                    "; rebuild with GROUP_SIZE=" + std::to_string(groupSize));
            }
            if (rankSize > groupSize &&
                (ipPort.compare(0, 10, "tcp://127.") == 0 || ipPort.compare(0, 16, "tcp://localhost:") == 0)) {
                throw std::invalid_argument("two nodes require the rank-0 server's non-loopback TCP endpoint");
            }
        } catch (const std::exception& error) {
            std::cerr << error.what() << "\n";
            return -1;
        }
        return 0;
    }
};

int main(int argc, char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return 2;
    }
    CocTilingParams cocTiling;
    // Convert the public output M/full K once to the kernel's input rows/local K.
    cocTiling.m = options.m * options.rankSize;
    cocTiling.n = options.n;
    cocTiling.k = options.k / options.rankSize;
    cocTiling.transB = 1;
    cocTiling.m0 = cocTiling.n0 = 256;
    cocTiling.k0 = AllToAllLayout::SelectTileK(cocTiling.k);
    cocTiling.commTileM = options.windowRows;
    cocTiling.commInterval = 1;
    cocTiling.commNpuSplit = 1;
    cocTiling.commDataSplit = options.rankSize;
    cocTiling.commBlockM = options.windowRows;
    cocTiling.rankSize = options.rankSize;

    try {
        auto op = OperatorRegistry::Instance().CreateOperator(OPERATOR_NAME);
        if (!op || !op->ValidateCocTilingParams(options.rankSize, cocTiling)) {
            throw std::invalid_argument("unsupported M/N/K layout or tiling configuration");
        }
        const Comm::Kernel::HierarchicalBarrier::Arguments barrierArgs{
            options.rankId, options.rankSize, options.groupSize, 0};
        const uint64_t barrierBytes = Comm::Kernel::HierarchicalBarrier::GetWorkspaceSize(barrierArgs);
        const uint64_t symmetricBytes = op->GetSymmetricSize(cocTiling);
        const uint64_t heapBytes = SelectHeapBytes(symmetricBytes, barrierBytes);
        RuntimeResources resources;
        ConfigureShmemLogger(options.rankId);
        Check(aclInit(nullptr), "aclInit");
        resources.aclInitialized = true;
        resources.device = options.deviceIdList[options.rankId % options.deviceIdList.size()];
        Check(aclrtSetDevice(resources.device), "aclrtSetDevice");
        resources.deviceSet = true;
        const char* soc = aclrtGetSocName();
        if (soc == nullptr || std::string(soc).find("Ascend950") != 0) {
            throw std::runtime_error("Ascend950 required");
        }
        const uint32_t requiredAiv = options.rankSize + 1;
        auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (platform == nullptr || platform->GetCoreNumAic() == 0 ||
            platform->GetCoreNumAiv() < static_cast<int>(requiredAiv) ||
            platform->GetCoreNumAic() * 2 < static_cast<int>(requiredAiv)) {
            throw std::runtime_error("insufficient AIC/AIV for senders and coordinator");
        }
        Check(aclrtCreateStream(&resources.stream), "aclrtCreateStream");
        const bool crossNode = options.rankSize > options.groupSize;
        Check(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, 1), "set UDMA qp0");
        if (crossNode) {
            Check(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_ROCE, 1), "set RDMA qp0");
        }
        aclshmemx_uniqueid_t uid{};
        aclshmemx_init_attr_t attributes{};
        attributes.my_pe = static_cast<int>(options.rankId);
        attributes.n_pes = static_cast<int>(options.rankSize);
        attributes.local_mem_size = heapBytes;
        if (options.ipPort.size() >= sizeof(attributes.ip_port)) {
            throw std::invalid_argument("ip_port too long");
        }
        std::memcpy(attributes.ip_port, options.ipPort.c_str(), options.ipPort.size() + 1);
        attributes.comm_args = &uid;
        attributes.option_attr.data_op_engine_type =
            static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_UDMA | (crossNode ? ACLSHMEM_DATA_OP_ROCE : 0));
        const uint32_t shmemTimeout = ReadEnvUint("SHMEM_INIT_TIMEOUT", 120);
        attributes.option_attr.shm_init_timeout = shmemTimeout;
        attributes.option_attr.shm_create_timeout = shmemTimeout;
        attributes.option_attr.control_operation_timeout = shmemTimeout;
        // One sender AIV owns each peer QP; shared jetty would make them race on one SQ/CQ.
        attributes.runtime_config.udma_shared_jetty = ACLSHMEM_CONFIG_DISABLED;
        std::cout << "CONFIG rank=" << options.rankId << " world=" << options.rankSize << " device=" << resources.device
                  << " group_size=" << options.groupSize << " window_rows=" << options.windowRows
                  << " rdma_qps=1 rdma_quiet=final ready_notify=window"
                  << " active_aiv=" << requiredAiv << " symmetric_bytes=" << symmetricBytes
                  << " heap_bytes=" << heapBytes << std::endl;
        Check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes), "aclshmemx_init_attr");
        resources.shmemInitialized = true;
        uint32_t udmaPeers = 0;
        uint32_t rdmaPeers = 0;
        for (uint32_t peer = 0; peer < options.rankSize; ++peer) {
            if (peer == options.rankId) {
                continue;
            }
            if (peer / options.groupSize != options.rankId / options.groupSize) {
                ++rdmaPeers;
                continue;
            }
            aclshmemx_net_layer_t netLayer;
            Check(aclshmemx_get_qp_net_layer(ACLSHMEM_DATA_OP_UDMA, peer, 0, &netLayer), "check same-node UDMA route");
            ++udmaPeers;
        }
        std::cout << "TRANSPORT rank=" << options.rankId << " udma_peers=" << udmaPeers << " rdma_peers=" << rdmaPeers
                  << std::endl;
        op->AllocateDeviceSpace(resources.kernelParams, cocTiling, options.rankId, options.dataPath);
        resources.symmetric = static_cast<uint8_t*>(aclshmem_malloc(symmetricBytes));
        if (resources.symmetric == nullptr) {
            throw std::runtime_error("aclshmem_malloc failed");
        }
        InitializeSymmetricBuffer(cocTiling, resources.symmetric);
        resources.barrierWorkspace = static_cast<uint8_t*>(aclshmem_calloc(1, barrierBytes));
        if (resources.barrierWorkspace == nullptr) {
            throw std::runtime_error("allocate barrier workspace failed");
        }
        // Select once on the host; all supported shapes share this executable.
        const auto run =
            cocTiling.k0 == 256 ? &RunKernel<256> : (cocTiling.k0 == 128 ? &RunKernel<128> : &RunKernel<64>);
        run(cocTiling, options.rankId, options.groupSize, resources.kernelParams, resources.symmetric, resources,
            static_cast<uint32_t>(platform->GetCoreNumAic()), options.iterations);
        op->WriteResultFile(resources.kernelParams, cocTiling, options.rankId, options.dataPath);
        WorldBarrier(resources, options.rankId, options.rankSize, options.groupSize, "output finish barrier");
        std::cout << "Rank " << options.rankId << " test finished\n";
    } catch (const std::exception& error) {
        std::cerr << "rank " << options.rankId << ": " << error.what() << "\n";
        return 1;
    }
    return 0;
}
