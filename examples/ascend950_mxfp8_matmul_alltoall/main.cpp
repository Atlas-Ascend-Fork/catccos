

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascend950_mxfp8_matmul_alltoall_device.h"
#include "ascend950_mxfp8_matmul_alltoall_host.h"

using namespace AscendC;
using namespace Catccos;
using namespace Catlass;
using namespace tla;

using ElementA = float8_e4m3_t;
using ElementB = float8_e4m3_t;
using ElementAScale = float8_e8m0_t;
using ElementBScale = float8_e8m0_t;
using ElementC = half;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutMxScaleA = Catlass::layout::RowMajor;
using LayoutMxScaleB = Catlass::layout::RowMajor;

using Config = Ascend950MxFp8MatmulAllToAllConfig_M0_128<
    ElementA, LayoutA, ElementB, LayoutB, ElementAScale, LayoutMxScaleA, ElementBScale, LayoutMxScaleB, ElementC,
    LayoutC>;
using DeviceOp = Config::Device;

struct Options {
    static constexpr auto HELPER =
        "Usage: ascend950_mxfp8_matmul_alltoall rank_size rank_id ip_port m n k data_path [device_id_list]\n";

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

        if (argc <= static_cast<int>(ArgsIndex::DATA_PATH_INDEX) || argc > static_cast<int>(ArgsIndex::INDEX_MAX)) {
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
    int32_t deviceId = options.deviceIdList[rankId];

    CocTilingParams cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    COCMatMulInfo info{int64_t(m), int64_t(k), int64_t(n)};
    cocTiling.m0 = 128;
    cocTiling.n0 = 256;
    cocTiling.k0 = 256;
    cocTiling.commTileM = 64;
    cocTiling.commInterval = 3;
    cocTiling.commNpuSplit = 1;
    cocTiling.commDataSplit = 32;
    cocTiling.commBlockM = 64;
    cocTiling.rankSize = rankSize;
    if (!Ascend950MxFp8MatmulAllToAllOperator::CheckProblemShape(rankSize, cocTiling, Config::Kernel::L1_TILE_N)) {
        std::cerr << "[ERROR] M, N and K must be positive, and N must be divisible by rankSize * N0 ("
                  << static_cast<uint64_t>(rankSize) * Config::Kernel::L1_TILE_N << ")\n";
        return 1;
    }

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipPort << "\n";

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    auto blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    ACL_CHECK(aclrtCreateStream(&stream));
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t default_flag_uid;
    set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, ipPort.c_str(), &attributes, &default_flag_uid);
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    auto op = OperatorRegistry::Instance().CreateOperator("Ascend950MxFp8MatmulAllToAll");
    if (!op) {
        std::cout << "Operator Ascend950MxFp8MatmulAllToAll not found!" << std::endl;
        return -1;
    }

    KernelParams kernelParams;
    op->AllocateDeviceSpace(kernelParams, cocTiling, rankId, options.GetDataPath());
    void* symmPtr = aclshmem_calloc(1, SHMEM_BUFF_BYTES);
    uint8_t* symmetricPtr = reinterpret_cast<uint8_t*>(symmPtr);

    uint8_t* aPtr = kernelParams.ptrA;
    uint8_t* bPtr = kernelParams.ptrB;
    uint8_t* dPtr = kernelParams.ptrC;
    uint8_t* aScalePtr = kernelParams.customPtrs[0];
    uint8_t* bScalePtr = kernelParams.customPtrs[1];

    // Construct DeviceDGemm Arguments
    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::MatrixCoord commCoreSplit{cocTiling.commDataSplit, cocTiling.commNpuSplit};
    Catlass::MatrixCoord commBlockShape{cocTiling.commBlockM, cocTiling.n0};
    Catlass::MatrixCoord commTileShape{cocTiling.commTileM / 2, cocTiling.n0};

    DeviceOp::Arguments args{
        problemShape,
        static_cast<uint32_t>(rankId),
        static_cast<uint32_t>(rankSize),
        cocTiling.commInterval,
        aPtr,
        bPtr,
        aScalePtr,
        bScalePtr,
        dPtr,
        symmetricPtr,
        commCoreSplit,
        commBlockShape,
        commTileShape};

    DeviceOp deviceOp;
    deviceOp.Initialize(args);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "Before calling A5 MxFP8 MM_A2A kernel " << std::endl;
    uint64_t fftsAddr = shmemx_get_ffts_config();
    for (int i = 0; i < 1; i++) {
        deviceOp.Run(stream, blockNum, fftsAddr);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "After calling A5 MxFP8 MM_A2A kernel " << std::endl;

    op->WriteResultFile(kernelParams, cocTiling, rankId, options.GetDataPath());
    if (rankId == 0) {
        std::printf("test finished\n");
    }

    shmem_free(symmPtr);

    FreeDeviceSpace(kernelParams);

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}
