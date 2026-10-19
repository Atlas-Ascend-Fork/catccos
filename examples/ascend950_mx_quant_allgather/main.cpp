

/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * MX Quant + AllGather fused operator example
 */

#include <chrono>

#include "mx_quant_allgather_device.h"
#include "mx_quant_allgather_host.h"

using namespace AscendC;
using namespace Catccos;

using LayoutInput = Catlass::layout::RowMajor;
using LayoutOutput = Catlass::layout::RowMajor;

using ElementInput = bfloat16_t;

#ifndef MX_QUANT_DTYPE
inline constexpr int MX_QUANT_DTYPE_DEFAULT = 0;
#define MX_QUANT_DTYPE MX_QUANT_DTYPE_DEFAULT
#endif

#if MX_QUANT_DTYPE == 0
using ElementOutput = float8_e4m3_t;
static constexpr const char* DTYPE_NAME = "fp8_e4m3";
#elif MX_QUANT_DTYPE == 1
using ElementOutput = float8_e5m2_t;
static constexpr const char* DTYPE_NAME = "fp8_e5m2";
#elif MX_QUANT_DTYPE == 2
using ElementOutput = float4_e2m1x2_t;
static constexpr const char* DTYPE_NAME = "fp4_e2m1";
#elif MX_QUANT_DTYPE == 3
using ElementOutput = float4_e1m2x2_t;
static constexpr const char* DTYPE_NAME = "fp4_e1m2";
#else
#error "Invalid MX_QUANT_DTYPE. Must be 0 (fp8_e4m3), 1 (fp8_e5m2), 2 (fp4_e2m1), or 3 (fp4_e1m2)."
#endif

struct Options {
    static constexpr auto HELPER =
        "Usage: mx_quant_allgather rank_size rank_id ip_port m n k data_path [device_id_list]\n";

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
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "-p")
                perfMode = true;
        }

        int positionalArgc = 0;
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) != "-p")
                positionalArgc++;
        }
        if (positionalArgc < 7) {
            printf(HELPER);
            return -1;
        }

        rankSize = std::atoi(argv[1]);
        if (rankSize <= 0 || rankSize > MAX_RANK_SIZE) {
            std::cerr << "rankSize must be in [1, " << MAX_RANK_SIZE << "]" << std::endl;
            return -1;
        }
        rankId = std::atoi(argv[2]);
        if (rankId < 0 || rankId >= rankSize) {
            std::cerr << "rankId must be in [0, rankSize)" << std::endl;
            return -1;
        }
        ipPort = argv[3];
        m = std::atoi(argv[4]);
        n = std::atoi(argv[5]);
        k = std::atoi(argv[6]);
        dataPath = argv[7];
        if (argc > 8) {
            char* idListStr = argv[8];
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
    if (!MxQuantAllGatherOperator::CheckProblemShape(rankSize, cocTiling)) {
        std::cerr << "[ERROR] M and N must be positive, and N must be divisible by "
                  << MxQuantAllGatherOperator::MX_BLOCK_SIZE << "\n";
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

    cocTiling.rankSize = rankSize;

    std::cout << "[TEST] MxQuantAllGather rank_size: " << rankSize << " rank_id: " << rankId << " M=" << options.m
              << " N=" << options.n << " dtype=" << DTYPE_NAME << std::endl;

    auto op = OperatorRegistry::Instance().CreateOperator("MxQuantAllGather");
    if (!op) {
        std::cout << "Operator MxQuantAllGather not found!" << std::endl;
        return -1;
    }

    KernelParams kernelParams;
    op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, options.GetDataPath());

    void* symmPtr = aclshmem_calloc(1, SHMEM_BUFF_BYTES);
    uint8_t* symmetricPtr = (uint8_t*)symmPtr;

    uint8_t* inputPtr = kernelParams.ptrA;
    uint8_t* mxScalePtr = kernelParams.ptrB; // mxscale
    uint8_t* outputPtr = kernelParams.ptrC;  // quantized data
    uint32_t curBlockNum = rankSize;
    constexpr uint32_t KERNEL_ITERATIONS = 10;
    constexpr uint32_t WARM_UP_ITERS = 5;
    constexpr uint32_t PERF_ITERS = 50;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    uint64_t fftsAddr = shmemx_get_ffts_config();

    if (options.perfMode) {
        std::cout << "[PERF] Warming up (" << WARM_UP_ITERS << " iters)..." << std::endl;
        for (uint32_t i = 0; i < WARM_UP_ITERS; i++) {
            MxQuantAllGather<ElementInput, LayoutInput, ElementOutput, LayoutOutput><<<curBlockNum, nullptr, stream>>>(
                fftsAddr, inputPtr, outputPtr, mxScalePtr, symmetricPtr, cocTiling, i);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));

        std::cout << "[PERF] Running (" << PERF_ITERS << " iters)..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < PERF_ITERS; i++) {
            MxQuantAllGather<ElementInput, LayoutInput, ElementOutput, LayoutOutput><<<curBlockNum, nullptr, stream>>>(
                fftsAddr, inputPtr, outputPtr, mxScalePtr, symmetricPtr, cocTiling, i);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        auto endTime = std::chrono::high_resolution_clock::now();

        double totalUs = std::chrono::duration<double, std::micro>(endTime - startTime).count();
        double avgUs = totalUs / PERF_ITERS;
        if (rankId == 0) {
            std::printf("[PERF] MxQuantAllGather M=%u N=%u rankSize=%d\n", options.m, options.n, rankSize);
            std::printf("[PERF]   Avg latency: %.2f us/iter\n", avgUs);
        }
    } else {
        std::cout << "Before calling MxQuantAllGather kernel" << std::endl;
        for (uint32_t i = 0; i < KERNEL_ITERATIONS; i++) {
            MxQuantAllGather<ElementInput, LayoutInput, ElementOutput, LayoutOutput><<<curBlockNum, nullptr, stream>>>(
                fftsAddr, inputPtr, outputPtr, mxScalePtr, symmetricPtr, cocTiling, i);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::cout << "After calling MxQuantAllGather kernel" << std::endl;

        op->WriteResultFile(kernelParams, cocTiling, rankId, options.GetDataPath());
        std::printf("rankId %d test finished\n", rankId);
    }

    shmem_free(symmPtr);
    FreeDeviceSpace(kernelParams);
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    std::cout << "[TEST] exit rankId: " << rankId << std::endl;
    return 0;
}
