
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cost_model.h"

namespace {

constexpr double A5_FP8_CUBE_FLOPS_PER_US = 27000000.0;
constexpr double A5_FP4_CUBE_FLOPS_PER_US = 54000000.0;
constexpr uint32_t MX_SCALE_GROUP_SIZE = 32;
constexpr uint32_t MX_SCALE_ELEMENT_BITS = 8;

} // namespace

CostModelHardwareType GetCostModelHardwareType(CocCommType type)
{
    switch (type) {
        case ASCEND950_ALLGATHER_MATMUL:
        case ASCEND950_MATMUL_REDUCE_SCATTER:
        case ASCEND950_GROUPED_MATMUL_ALLTOALLV:
        case ASCEND950_MXFP8_MATMUL_REDUCE_SCATTER:
        case ASCEND950_MXFP8_MATMUL_ALLTOALL:
        case ASCEND950_FP8_MX_ALLGATHER_MATMUL:
        case ASCEND950_FP4_MX_ALLGATHER_MATMUL:
        case MX_QUANT_ALLGATHER:
        case ASCEND950_FP8_MX_GROUPED_MATMUL_ALLTOALLV:
        case ASCEND950_FP4_MX_GROUPED_MATMUL_ALLTOALLV:
        case ASCEND950_FP8_MX_ALLTOALLV_GROUPED_MATMUL:
        case ASCEND950_FP4_MX_ALLTOALLV_GROUPED_MATMUL:
        case ASCEND950_ALLTOALLV_GROUPED_MATMUL:
        case ASCEND950_FP4_MX_MATMUL_REDUCE_SCATTER:
        case ASCEND950_DISPATCH_FFN_COMBINE:
        case ASCEND950_ALLGATHER_MATMUL_UDMA:
        case ASCEND950_MATMUL_REDUCE_SCATTER_UDMA:
            return CostModelHardwareType::A5;
        default:
            return CostModelHardwareType::A2;
    }
}

void ConfigureCostModelConfig(CocCommType type, CostModelConfig& config)
{
    config.hardwareType = GetCostModelHardwareType(type);
    config.inputElementBits = 0;
    config.communicationElementBits = 0;
    config.k0 = 256;
    config.mxScaleGroupSize = 0;
    config.mxScaleElementBits = 0;
    config.cubeFlopsPerUsOverride = 0.0;
    config.includeAStage = type == ALLGATHER_MATMUL_REMOTE_READ;
    config.useMxMteShape = false;
    config.allGatherBarrierPerLoop = false;

    switch (type) {
        case ALLGATHER_MATMUL:
        case ALLGATHER_MATMUL_WITH_GATHER_RESULT:
        case ALLGATHER_MATMUL_RDMA:
        case ASCEND950_ALLGATHER_MATMUL:
        case ASCEND950_ALLGATHER_MATMUL_UDMA:
            config.allGatherBarrierPerLoop = true;
            break;
        case ALLGATHER_MATMUL_DEQUANT_BIAS:
        case ALLGATHER_MATMUL_DEQUANT:
        case ALLGATHER_MATMUL_DEQUANT_PADDING:
            config.dataType = CocDataType::INT8;
            config.inputElementBits = 8;
            config.communicationElementBits = 8;
            config.allGatherBarrierPerLoop = true;
            break;
        case MATMUL_DEQUANT_REDUCE_SCATTER_WRITE:
            config.dataType = CocDataType::INT8;
            config.inputElementBits = 8;
            config.communicationElementBits = 16;
            break;
        case ASCEND950_FP8_MX_ALLGATHER_MATMUL:
            config.inputElementBits = 8;
            config.communicationElementBits = 8;
            config.mxScaleGroupSize = MX_SCALE_GROUP_SIZE;
            config.mxScaleElementBits = MX_SCALE_ELEMENT_BITS;
            config.cubeFlopsPerUsOverride = A5_FP8_CUBE_FLOPS_PER_US;
            config.useMxMteShape = true;
            config.allGatherBarrierPerLoop = true;
            break;
        case ASCEND950_FP4_MX_ALLGATHER_MATMUL:
            config.inputElementBits = 4;
            config.communicationElementBits = 4;
            config.k0 = 512;
            config.mxScaleGroupSize = MX_SCALE_GROUP_SIZE;
            config.mxScaleElementBits = MX_SCALE_ELEMENT_BITS;
            config.cubeFlopsPerUsOverride = A5_FP4_CUBE_FLOPS_PER_US;
            config.useMxMteShape = true;
            config.allGatherBarrierPerLoop = true;
            break;
        case ASCEND950_MXFP8_MATMUL_REDUCE_SCATTER:
            config.inputElementBits = 8;
            config.communicationElementBits = 16;
            config.cubeFlopsPerUsOverride = A5_FP8_CUBE_FLOPS_PER_US;
            config.useMxMteShape = true;
            break;
        case ASCEND950_FP4_MX_MATMUL_REDUCE_SCATTER:
            config.inputElementBits = 4;
            config.communicationElementBits = 16;
            config.k0 = 512;
            config.cubeFlopsPerUsOverride = A5_FP4_CUBE_FLOPS_PER_US;
            config.useMxMteShape = true;
            break;
        default:
            break;
    }
}

CostModelResult SelectCostModelTiling(
    COCMatMulInfo const& info, CocCommType type, uint32_t rankSize, CostModelConfig const& config)
{
    auto resolvedConfig = config;
    ConfigureCostModelConfig(type, resolvedConfig);
    switch (type) {
        case MATMUL_REDUCE_SCATTER:
        case ASCEND950_MATMUL_REDUCE_SCATTER:
        case MATMUL_DEQUANT_REDUCE_SCATTER_WRITE:
        case ASCEND950_MXFP8_MATMUL_REDUCE_SCATTER:
        case ASCEND950_FP4_MX_MATMUL_REDUCE_SCATTER:
        case ASCEND950_MATMUL_REDUCE_SCATTER_UDMA:
            return SelectReduceScatterTiling(info, rankSize, resolvedConfig);
        case MATMUL_ALLREDUCE:
            return SelectAllReduceTiling(info, rankSize, resolvedConfig);
        case ALLGATHER_MATMUL:
        case ASCEND950_ALLGATHER_MATMUL:
        case ALLGATHER_MATMUL_WITH_GATHER_RESULT:
        case ALLGATHER_MATMUL_RDMA:
        case ALLGATHER_MATMUL_DEQUANT_BIAS:
        case ALLGATHER_MATMUL_DEQUANT:
        case ALLGATHER_MATMUL_DEQUANT_PADDING:
        case ALLGATHER_MATMUL_REMOTE_READ:
        case ASCEND950_FP8_MX_ALLGATHER_MATMUL:
        case ASCEND950_FP4_MX_ALLGATHER_MATMUL:
        case ASCEND950_ALLGATHER_MATMUL_UDMA:
            return SelectAllGatherTiling(info, rankSize, resolvedConfig);
        default:
            return {};
    }
}

bool ApplyCostModel(
    COCMatMulInfo const& info, CocCommType type, uint32_t rankSize, CocTilingParams& tiling,
    CostModelConfig const& config)
{
    auto result = SelectCostModelTiling(info, type, rankSize, config);
    if (!result.IsSuccess() || !config.IsCandidateValid(result.tiling)) {
        return false;
    }

    tiling.m0 = result.tiling.m0;
    tiling.k0 = result.tiling.k0;
    tiling.n0 = result.tiling.n0;
    tiling.commTileM = result.tiling.commTileM;
    tiling.commInterval = result.tiling.commInterval;
    tiling.commNpuSplit = result.tiling.commNpuSplit;
    tiling.commDataSplit = result.tiling.commDataSplit;
    tiling.commBlockM = result.tiling.commBlockM;
    tiling.rankSize = rankSize;
    return true;
}
