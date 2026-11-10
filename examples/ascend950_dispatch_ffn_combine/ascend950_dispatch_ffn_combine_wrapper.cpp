/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascend950_dispatch_ffn_combine_device.h"

using namespace AscendC;
using namespace Catccos;

namespace CatccosKernel {

uint64_t catccos_ascend950_symmetric_bytes() { return SHMEM_BUFF_BYTES; }

void catccos_ascend950_dispatch_ffn_combine_wrapper(
    uint32_t blockDim, aclrtStream stream, uint64_t fftsAddr, uint8_t* xPtr, uint8_t* expertIdxPtr,
    uint8_t* gateWeightPtr, uint8_t* w1Ptr, uint8_t* w1ScalePtr, uint8_t* w2Ptr, uint8_t* w2ScalePtr,
    uint8_t* outputPtr, uint8_t* workspacePtr, uint8_t* symmetricPtr, uint32_t m, uint32_t k, uint32_t n,
    uint32_t localExperts, uint32_t topK, int rankSize)
{
    CocTilingParams cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    cocTiling.m0 = M0;
    cocTiling.n0 = N0;
    cocTiling.k0 = K0;
    cocTiling.commTileM = 64;
    cocTiling.commInterval = 10;
    cocTiling.commNpuSplit = rankSize;
    cocTiling.commDataSplit = 1;
    cocTiling.commBlockM = 64;
    cocTiling.rankSize = rankSize;
    cocTiling.epSize = rankSize;
    cocTiling.expertNum = rankSize * localExperts;
    cocTiling.topK = topK;

    optiling::MoeInitRoutingV2TilingBase moeTilingBase;
    moeTilingBase.DoTiling(
        m, k, topK, 0, cocTiling.expertNum, static_cast<int64_t>(m) * topK, 0, 2, false, sizeof(bfloat16_t), -1, 0,
        2 * blockDim, 196352);

    uint64_t initRoutingTilingKey = moeTilingBase.tilingKey_;
    if (initRoutingTilingKey == optiling::TILING_KEY_HIGH_PERFORMANCE) {
        initRoutingTilingKey = optiling::TILING_KEY_DROPLESS_SORT_ONE_CORE;
    }
    MoeInitRoutingQuantV2Tiling moeTiling{moeTilingBase.moeInitRoutingTilingData, initRoutingTilingKey};

    using ElementA = bfloat16_t;
    using ElementB = float8_e4m3_t;
    using ElementC = bfloat16_t;
    using LayoutA = Catlass::layout::RowMajor;
    using LayoutB = Catlass::layout::ColumnMajor;
    using LayoutC = Catlass::layout::RowMajor;

    DispatchFFNCombine<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC><<<blockDim, nullptr, stream>>>(
        fftsAddr, xPtr, w1Ptr, w1ScalePtr, w2Ptr, w2ScalePtr, outputPtr, expertIdxPtr, gateWeightPtr, workspacePtr,
        symmetricPtr, cocTiling, moeTiling);
}

} // namespace CatccosKernel
