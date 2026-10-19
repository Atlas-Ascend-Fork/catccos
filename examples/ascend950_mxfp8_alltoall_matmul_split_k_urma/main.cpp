

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascend950_mxfp8_alltoall_matmul_split_k_urma_device.h"
#include "ascend950_mxfp8_alltoall_matmul_split_k_urma_host.h"

using namespace AscendC;
using namespace Catccos;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::ColumnMajor;
using LayoutMxScaleA = Catlass::layout::RowMajor;
using LayoutMxScaleB = Catlass::layout::ColumnMajor;
using LayoutC = Catlass::layout::RowMajor;

using ElementA = float8_e4m3_t;
using ElementB = float8_e4m3_t;
using ElementMxScaleA = float8_e8m0_t;
using ElementMxScaleB = float8_e8m0_t;
using ElementC = half;

namespace {

constexpr uint32_t MAX_SWIZZLE_OFFSET = 7;

uint32_t SelectCommInterval(uint32_t m, uint32_t rankSize, uint32_t l1TileM)
{
    uint32_t chunkM = m / rankSize;
    uint32_t mTiles = (chunkM + l1TileM - 1) / l1TileM;

    if (mTiles <= 4) {
        return 2;
    }
    if (mTiles <= 8) {
        return (mTiles + 1) / 2;
    }
    return 5;
}

uint32_t SelectSwizzleOffset(const CocTilingParams& cocTiling)
{
    uint32_t chunkM = cocTiling.m / cocTiling.rankSize;
    uint32_t mTiles = (chunkM + cocTiling.m0 - 1) / cocTiling.m0;
    uint32_t tileMPerComm = cocTiling.commInterval < mTiles ? cocTiling.commInterval : mTiles;

    if (tileMPerComm == 0) {
        return 1;
    }
    return tileMPerComm > MAX_SWIZZLE_OFFSET ? MAX_SWIZZLE_OFFSET : tileMPerComm;
}

template <uint32_t SWIZZLE_OFFSET>
int RunKernel(
    const CocTilingParams& cocTiling, uint32_t rankId, uint32_t rankSize, uint8_t* aPtr, uint8_t* bPtr,
    uint8_t* aMxScalePtr, uint8_t* bMxScalePtr, uint8_t* cPtr, uint8_t* gmSymmetric, aclrtStream stream,
    uint32_t blockNum, uint64_t fftsAddr)
{
    using Config = Ascend950MxFp8AllToAllMatmulSplitKUrmaConfig<
        ElementA, LayoutA, ElementB, LayoutB, ElementMxScaleA, LayoutMxScaleA, ElementMxScaleB, LayoutMxScaleB,
        ElementC, LayoutC, SWIZZLE_OFFSET, SplitKUrmaExample::IS_LOCAL_ATOMIC>;
    using DeviceOp = typename Config::Device;

    Catlass::GemmCoord problemShape{cocTiling.m, cocTiling.n, cocTiling.k};

    typename DeviceOp::Arguments args{problemShape, rankId, rankSize,   cocTiling.commInterval, aPtr, bPtr, aMxScalePtr,
                                      bMxScalePtr,  cPtr,   gmSymmetric};

    DeviceOp deviceOp;
    deviceOp.Initialize(args);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "Before calling " << SplitKUrmaExample::OPERATOR_NAME << " kernel " << std::endl;
    for (int i = 0; i < 10; i++) {
        deviceOp.Run(stream, blockNum, fftsAddr);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "After calling " << SplitKUrmaExample::OPERATOR_NAME << " kernel " << std::endl;

    return 0;
}

int RunKernelBySwizzleOffset(
    uint32_t swizzleOffset, const CocTilingParams& cocTiling, uint32_t rankId, uint32_t rankSize, uint8_t* aPtr,
    uint8_t* bPtr, uint8_t* aMxScalePtr, uint8_t* bMxScalePtr, uint8_t* cPtr, uint8_t* gmSymmetric, aclrtStream stream,
    uint32_t blockNum, uint64_t fftsAddr)
{
    switch (swizzleOffset) {
        case 1:
            return RunKernel<1>(
                cocTiling, rankId, rankSize, aPtr, bPtr, aMxScalePtr, bMxScalePtr, cPtr, gmSymmetric, stream, blockNum,
                fftsAddr);
        case 2:
            return RunKernel<2>(
                cocTiling, rankId, rankSize, aPtr, bPtr, aMxScalePtr, bMxScalePtr, cPtr, gmSymmetric, stream, blockNum,
                fftsAddr);
        case 3:
            return RunKernel<3>(
                cocTiling, rankId, rankSize, aPtr, bPtr, aMxScalePtr, bMxScalePtr, cPtr, gmSymmetric, stream, blockNum,
                fftsAddr);
        case 4:
            return RunKernel<4>(
                cocTiling, rankId, rankSize, aPtr, bPtr, aMxScalePtr, bMxScalePtr, cPtr, gmSymmetric, stream, blockNum,
                fftsAddr);
        case 5:
            return RunKernel<5>(
                cocTiling, rankId, rankSize, aPtr, bPtr, aMxScalePtr, bMxScalePtr, cPtr, gmSymmetric, stream, blockNum,
                fftsAddr);
        case 6:
            return RunKernel<6>(
                cocTiling, rankId, rankSize, aPtr, bPtr, aMxScalePtr, bMxScalePtr, cPtr, gmSymmetric, stream, blockNum,
                fftsAddr);
        default:
            return RunKernel<7>(
                cocTiling, rankId, rankSize, aPtr, bPtr, aMxScalePtr, bMxScalePtr, cPtr, gmSymmetric, stream, blockNum,
                fftsAddr);
    }
}

} // namespace

struct Options {
    static constexpr auto HELP_SUFFIX = " rank_size rank_id ip_port m n k [data_path] [device_id_list]\n";

    int rankSize;
    int rankId;
    std::string ipPort;
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::string dataPath;
    std::vector<int> deviceIdList{};

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

        if (argc < static_cast<int>(ArgsIndex::DATA_PATH_INDEX) || argc > static_cast<int>(ArgsIndex::INDEX_MAX)) {
            std::printf("Usage: %s%s", SplitKUrmaExample::EXECUTABLE_NAME, HELP_SUFFIX);
            return -1;
        }

        rankSize = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_SIZE_INDEX)]);
        rankId = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_ID_INDEX)]);
        ipPort = argv[static_cast<int>(ArgsIndex::IP_PORT_INDEX)];
        m = std::atoi(argv[static_cast<int>(ArgsIndex::M_INDEX)]);
        n = std::atoi(argv[static_cast<int>(ArgsIndex::N_INDEX)]);
        k = std::atoi(argv[static_cast<int>(ArgsIndex::K_INDEX)]);
        if (rankSize <= 0 || rankId < 0 || rankId >= rankSize) {
            std::printf("Usage: %s%s", SplitKUrmaExample::EXECUTABLE_NAME, HELP_SUFFIX);
            return -1;
        }
        if (argc > static_cast<int>(ArgsIndex::DATA_PATH_INDEX)) {
            dataPath = argv[static_cast<int>(ArgsIndex::DATA_PATH_INDEX)];
        }
        if (argc > static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)) {
            char* idListStr = argv[static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)];
            for (char* idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) {
                deviceIdList.push_back(std::atoi(idToken));
            }
        } else {
            for (size_t i = 0; i < static_cast<size_t>(rankSize); ++i) {
                deviceIdList.push_back(i);
            }
        }
        if (static_cast<int>(deviceIdList.size()) <= rankId) {
            std::printf("Usage: %s%s", SplitKUrmaExample::EXECUTABLE_NAME, HELP_SUFFIX);
            return -1;
        }
        return 0;
    }

    std::string GetDataPath() const { return dataPath; }
};

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
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t k = options.k;

    if (rankSize <= 0 || m % rankSize != 0) {
        std::cerr << "[ERROR] M (" << m << ") must be divisible by rankSize (" << rankSize
                  << ") for split-K AllToAll-MatMul URMA\n";
        return 1;
    }
    int32_t deviceId = options.deviceIdList[rankId];

    CocTilingParams cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    cocTiling.transA = 0;
    cocTiling.transB = 1;
    cocTiling.m0 = 256;
    cocTiling.n0 = 256;
    cocTiling.k0 = 512;
    cocTiling.commTileM = 64;
    cocTiling.commInterval = SelectCommInterval(m, static_cast<uint32_t>(rankSize), cocTiling.m0);
    cocTiling.commNpuSplit = 1;
    cocTiling.commDataSplit = rankSize;
    cocTiling.commBlockM = 64;
    cocTiling.rankSize = rankSize;

    if (cocTiling.k % cocTiling.k0 != 0) {
        std::cerr << "[ERROR] local K (" << cocTiling.k << ") must be divisible by " << cocTiling.k0
                  << " for one-shot contiguous URMA copy\n";
        return 1;
    }

    uint32_t swizzleOffset = SelectSwizzleOffset(cocTiling);

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipPort
              << " commInterval: " << cocTiling.commInterval << " swizzleOffset: " << swizzleOffset << std::endl;

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    auto blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    if (rankSize > static_cast<int>(blockNum)) {
        std::cerr << "[ERROR] rankSize (" << rankSize << ") must be <= launch blockNum (" << blockNum
                  << ") because URMA uses one AIV core per remote destination rank\n";
        return 1;
    }
    ACL_CHECK(aclrtCreateStream(&stream));
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t default_flag_uid;
    set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, ipPort.c_str(), &attributes, &default_flag_uid);
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    auto op = OperatorRegistry::Instance().CreateOperator(SplitKUrmaExample::OPERATOR_NAME);
    if (!op) {
        std::cout << "Operator " << SplitKUrmaExample::OPERATOR_NAME << " not found!" << std::endl;
        return -1;
    }
    if (!op->ValidateCocTilingParams(rankSize, cocTiling)) {
        std::cerr << "Invalid CocTilingParams" << std::endl;
        return -1;
    }
    KernelParams kernelParams;
    op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, options.GetDataPath());
    void* symmPtr = shmem_malloc(SHMEM_BUFF_BYTES);
    uint8_t* gmSymmetric = reinterpret_cast<uint8_t*>(symmPtr);

    uint8_t* aPtr = kernelParams.ptrA;
    uint8_t* bPtr = kernelParams.ptrB;
    uint8_t* cPtr = kernelParams.ptrC;
    uint8_t* aMxScalePtr = kernelParams.customPtrs[0];
    uint8_t* bMxScalePtr = kernelParams.customPtrs[1];

    uint64_t fftsAddr = shmemx_get_ffts_config();

    RunKernelBySwizzleOffset(
        swizzleOffset, cocTiling, static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankSize), aPtr, bPtr,
        aMxScalePtr, bMxScalePtr, cPtr, gmSymmetric, stream, blockNum, fftsAddr);

    op->WriteResultFile(kernelParams, cocTiling, rankId, options.GetDataPath());
    std::printf("Rank %d test finished\n", rankId);

    shmem_free(symmPtr);

    FreeDeviceSpace(kernelParams);

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}
