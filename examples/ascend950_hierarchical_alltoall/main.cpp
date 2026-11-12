/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascend950_hierarchical_alltoall_device.h"
#include "ascend950_hierarchical_alltoall_host.h"

#include <algorithm>

using namespace AscendC;
using namespace HierarchicalAllToAllHost;

using Element = float8_e4m3_t;
using ElementScale = float8_e8m0_t;
using LayoutData = Catlass::layout::RowMajor;
using LayoutScale = Catlass::layout::RowMajor;
using Config = Ascend950HierarchicalAllToAllConfig<Element, LayoutData, ElementScale, LayoutScale>;
using DeviceOp = Config::Device;

static_assert(Config::CORE_COUNT == CORE_COUNT, "Host capacity must match the communication policy");

struct Options {
    static constexpr auto HELPER =
        "Usage: ascend950_hierarchical_alltoall rank_size rank_id ip_port m n group_size data_path [device_id_list]\n"
        "Environment: QP_NUM=1|2|4 (default 1)\n";

    int rankSize;
    int rankId;
    std::string ipPort;
    std::string dataFile;
    uint32_t m{0};
    uint32_t n{0};
    uint32_t groupSize{0};
    uint32_t qpNum{1};
    std::vector<int> deviceIdList{};

    int Parse(int argc, char** argv)
    {
        enum class ArgsIndex {
            RANK_SIZE_INDEX = 1,
            RANK_ID_INDEX,
            IP_PORT_INDEX,
            M_INDEX,
            N_INDEX,
            GROUP_SIZE_INDEX,
            DATA_PATH_INDEX,
            DEVICE_LIST_INDEX,
            INDEX_MAX
        };

        if (argc <= static_cast<int>(ArgsIndex::DATA_PATH_INDEX) || argc > static_cast<int>(ArgsIndex::INDEX_MAX)) {
            printf(HELPER);
            return -1;
        }

        rankSize = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_SIZE_INDEX)]);
        if (rankSize <= 0 || rankSize > static_cast<int>(CORE_COUNT)) {
            std::cerr << "rankSize must be in [1, " << CORE_COUNT << "]" << std::endl;
            return -1;
        }
        rankId = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_ID_INDEX)]);
        if (rankId < 0 || rankId >= rankSize) {
            std::cerr << "rankId must be in [0, rankSize)" << std::endl;
            return -1;
        }
        ipPort = argv[static_cast<int>(ArgsIndex::IP_PORT_INDEX)];
        if (ipPort.compare(0, 6, "tcp://") != 0 || ipPort.size() >= sizeof(aclshmemx_init_attr_t{}.ip_port)) {
            std::cerr << "Invalid ip_port" << std::endl;
            return -1;
        }
        m = std::atoi(argv[static_cast<int>(ArgsIndex::M_INDEX)]);
        n = std::atoi(argv[static_cast<int>(ArgsIndex::N_INDEX)]);
        if (m == 0 || n == 0 || n % MX_SCALE_GROUP_SIZE != 0) {
            std::cerr << "m/n must be positive and n must be divisible by 32" << std::endl;
            return -1;
        }
        dataFile = argv[static_cast<int>(ArgsIndex::DATA_PATH_INDEX)];
        groupSize = std::atoi(argv[static_cast<int>(ArgsIndex::GROUP_SIZE_INDEX)]);
        if (groupSize == 0 || groupSize > static_cast<uint32_t>(rankSize)) {
            std::cerr << "groupSize must be in [1, rankSize]" << std::endl;
            return -1;
        }
        if (groupSize != Config::GROUP_SIZE) {
            std::cerr << "groupSize must match compiled GROUP_SIZE=" << Config::GROUP_SIZE
                      << "; rebuild with GROUP_SIZE=" << groupSize << " bash scripts/build.sh" << std::endl;
            return -1;
        }
        if (argc > static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)) {
            char* idListStr = argv[static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)];
            for (char* idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) {
                deviceIdList.push_back(std::atoi(idToken));
            }
        } else {
            for (int i = 0; i < rankSize; ++i) {
                deviceIdList.push_back(i);
            }
        }
        if (deviceIdList.empty() || deviceIdList.size() > static_cast<size_t>(rankSize) ||
            rankSize % deviceIdList.size() != 0) {
            std::cerr << "device_id_list must contain 1..rankSize local devices and split rankSize evenly" << std::endl;
            return -1;
        }

        const char* qpValue = std::getenv("QP_NUM");
        if (qpValue != nullptr) {
            std::string value(qpValue);
            if (value != "1" && value != "2" && value != "4") {
                std::cerr << "QP_NUM must be 1, 2 or 4" << std::endl;
                return -1;
            }
            qpNum = std::atoi(qpValue);
        }
        return 0;
    }
};

int main(int argc, char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }
    uint32_t rankSize = options.rankSize;
    uint32_t rankId = options.rankId;
    std::string ipPort = options.ipPort;
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t groupSize = options.groupSize;
    uint32_t qpNum = options.qpNum;
    int32_t deviceId = options.deviceIdList[rankId % options.deviceIdList.size()];

    CocTilingParams cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.rankSize = rankSize;

    Ascend950HierarchicalAllToAllOperator<Element, ElementScale, Config::BarrierDevice> op(
        rankSize, rankId, groupSize, qpNum);
    if (!op.CheckCocTilingParams(rankSize, cocTiling)) {
        std::cerr << "[ERROR] Invalid hierarchical AllToAll shape or tiling" << std::endl;
        return 1;
    }

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipPort << "\n";

    op.Initialize(deviceId, ipPort);
    auto blockNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    aclrtStream stream = op.GetStream();
    KernelParams kernelParams;
    op.AllocateDeviceSpace(kernelParams, cocTiling, rankId, options.dataFile);

    uint8_t* aPtr = kernelParams.ptrA;
    uint8_t* cPtr = kernelParams.ptrC;
    uint8_t* aScalePtr = kernelParams.customPtrs[0];
    uint8_t* cScalePtr = kernelParams.customPtrs[1];

    // Params derives the aligned data/scale peer slots from m, n and the element types.
    DeviceOp::Arguments args{m, n, rankId, rankSize, aPtr, cPtr, qpNum, aScalePtr, cScalePtr};

    DeviceOp deviceOp;
    deviceOp.Initialize(args, op.GetWorkspace());

    uint32_t groupBegin = rankId / groupSize * groupSize;
    uint32_t groupRanks = std::min(groupSize, rankSize - groupBegin);
    std::cout << "[transport] rank=" << rankId << " world=" << rankSize << " mte=1 udma=" << (groupRanks - 1)
              << " rdma=" << (rankSize - groupRanks) << " qp_num=" << qpNum << std::endl;

    op.Barrier("input-ready barrier");
    uint64_t fftsAddr = shmemx_get_ffts_config();
    deviceOp.Run(stream, blockNum, fftsAddr);
    // The data kernel waits for every source locally before returning.
    CheckStatus(aclrtSynchronizeStream(stream), "receive-ready kernel");
    op.WriteResultFile(kernelParams, cocTiling, rankId, options.dataFile);
    op.Barrier("output-ready barrier");

    return 0;
}
