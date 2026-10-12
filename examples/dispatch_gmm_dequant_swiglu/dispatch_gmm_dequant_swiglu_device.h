
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DISPATCH_GMM_DEQUANT_SWIGLU_KERNEL_H
#define DISPATCH_GMM_DEQUANT_SWIGLU_KERNEL_H

#include "info.h"

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"
#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.h"
#endif

#include "moe_init_routing_quant_v2_tiling.h"
#include "moe_init_routing_quant_v2.h"
#include "dispatch_policy_custom.h"
#include "catccos/dgemm/block/block_mmad_preload_async_fixpipe.hpp"
#include "allgather_kernel.h"
#include "group_dequant_swiglu.h"

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/tile/tile_remote_chunk_copy.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_local_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/comm/block/comm_block_scheduler_alltoallv_gmm.hpp"
#include "catccos/dgemm/kernel/alltoallv_gmm_dequant_v2.hpp"
#include "catccos/epilogue/block/block_epilogue_per_token_dequant_swiglu.hpp"

using namespace AscendC;
using namespace Catccos;
inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {
    Catlass::L2_OFFSET
};

template <class ArchTag, class ElementC, class LayoutC, class ElementD, class LayoutD>
CATLASS_DEVICE void DequantSwigluImpl(
    Catlass::GemmCoord problemShape, GM_ADDR gmC, GM_ADDR gmPerTokenScale, GM_ADDR gmD, GM_ADDR groupListPtr,
    uint32_t expertPerRank, const Callback& callback, int32_t syncInterval, Catlass::Arch::Resource<ArchTag> resource)
{
    constexpr uint32_t ubStages = 1;
    using EpilogueDispatchPolicy = Catccos::Epilogue::EpilogueAtlasA2PerTokenDequantSwiglu<ubStages>;
    using PerTokenScaleType = Catlass::Gemm::GemmType<float, Catlass::layout::VectorLayout>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;

    using TileCopy = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType, CType, DType>;
    using DequantSwigluTileScheduler = Catlass::Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using DequantSwigluBlock = Catlass::Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, CType, PerTokenScaleType, DType, TileCopy, DequantSwigluTileScheduler>;

    using DequantSwiglu = DequantSwigluKernel<ArchTag, DequantSwigluBlock>;

    LayoutC layoutC{problemShape.m(), problemShape.n()};
    Catlass::layout::VectorLayout layoutPerTokenScale{problemShape.m()};
    LayoutD layoutD{problemShape.m(), problemShape.n() / 2};

    typename DequantSwiglu::Params dequantSwigluParams{
        expertPerRank,
        problemShape.GetCoordMN(),
        gmC,
        layoutC,
        gmPerTokenScale,
        layoutPerTokenScale,
        gmD,
        layoutD,
        groupListPtr,
        callback,
        syncInterval};

    DequantSwiglu dequantSwiglu;
    dequantSwiglu(dequantSwigluParams, resource);
}

template <
    class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
    class ElementD, class LayoutD, uint32_t M0, uint32_t N0, uint32_t K0>
CATLASS_DEVICE void AllToAllVGmmDequantSwigluImpl(
    Catlass::GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmD, GM_ADDR gmScale,
    GM_ADDR gmPerTokenScale, GM_ADDR tokenPerExpert, GM_ADDR ptrCumsumMM, GM_ADDR gmSymmetric,
    Catlass::MatrixCoord const& commCoreSplit, Catlass::MatrixCoord const& commBlockShape,
    Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank, Catlass::Arch::Resource<ArchTag> resource)
{
    // =============== AllToAllv_GMM begin ===============
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;
    constexpr uint32_t workspaceStages = 2;
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;

    constexpr bool ENABLE_UNIT_FLAG = true;
    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2PreloadAsyncFixpipe<
        preloadStages, l1Stages, l0AStages, l0BStages, l0CStages, enableUnitFlag, enableShuffleK>;
    using L1TileShape = Catlass::GemmShape<M0, N0, K0>;
    using L0TileShape = Catlass::GemmShape<M0, N0, 64>;
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;

    using BlockMmad = DGemm::Block::FixpipeBlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;

    constexpr bool IS_DYNAMIC = true;

    // remote comm
    constexpr uint32_t UB_STAGES = 2;
    constexpr Catccos::detail::CopyDirect COPY_DIRECT = Catccos::detail::CopyDirect::Get;
    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using RemotePerTokenScaleUbType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using RemotePerTokenScaleType = Catlass::Gemm::GemmType<float, Catlass::layout::VectorLayout>;
    using RemoteCommDispatch = Comm::AtlasCommRemoteChunkCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Catccos::Comm::Tile::TileRemoteChunkCopy<
        ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, RemotePerTokenScaleUbType, RemotePerTokenScaleType, void,
        COPY_DIRECT, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using RemoteCommBlock = Comm::Block::CommBlock<
        RemoteCommDispatch, RemoteSrcType, RemoteDstType, RemotePerTokenScaleType, void, TileRemoteCopy, TileScheduler>;

    typename TileRemoteCopy::Params tileParams{commTileShape};

    typename RemoteCommBlock::Params remoteCommParams{commBlockShape, tileParams, problemShape.k()};

    // communication scheduler
    using BlockCommScheduler = typename Catlass::Gemm::Block::BlockCommSchedulerAllToAllVGmm;

    // kernel level
    using ElementGroupList = int64_t;
    using MatmulKernel = Catccos::DGemm::Kernel::AlltoallvGmmDequantKernel<
        BlockMmad, BlockScheduler, ElementGroupList, RemoteCommBlock, BlockCommScheduler>;
    using AicFinishSync = Catccos::DGemm::Kernel::AicFinishSync<MatmulKernel>;
    using AivWaitSync = Catccos::DGemm::Kernel::AivWaitSync<MatmulKernel>;

    LayoutA layoutA{problemShape.m(), problemShape.k()};
    LayoutB layoutB{problemShape.k(), problemShape.n()};
    LayoutC layoutC{problemShape.m(), problemShape.n()};
    Catlass::layout::VectorLayout layoutScale{problemShape.n() * expertPerRank};

    uint32_t rankId = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();
    uint32_t maxOutputSize = problemShape.m() * rankSize; // m_before_scatter * topK * rankSize

    MatmulKernel mmKernel;
    AicFinishSync aicFinishSync{&mmKernel};
    AivWaitSync aivWaitSync{&mmKernel};
    int32_t syncInterval = expertPerRank > 1 ? expertPerRank - 1 : 1;

    typename MatmulKernel::Params params{
        problemShape,
        rankSize,
        expertPerRank,
        rankId,
        rankSize,
        gmA,
        layoutA,
        gmB,
        layoutB,
        gmC,
        layoutC,
        gmScale,
        layoutScale,
        gmPerTokenScale,
        tokenPerExpert,
        ptrCumsumMM,
        gmSymmetric,
        remoteCommParams,
        MakeCallback(&aicFinishSync),
        syncInterval};
    mmKernel(params, resource);

    // =============== AllToAllv_GMM end ===============

    // =============== DequantSwiglu begin ===============

    if constexpr (g_coreType == AscendC::AIV) {
        AscendC::GlobalTensor<float> gmPerTokenScaleTensor;
        gmPerTokenScaleTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmPerTokenScale));

        AscendC::DataCacheCleanAndInvalid<
            float, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(gmPerTokenScaleTensor);

        auto groupListPtr = ptrCumsumMM + (rankSize - 1) * expertPerRank * sizeof(int32_t);

        DequantSwigluImpl<ArchTag, ElementC, LayoutC, ElementD, LayoutD>(
            problemShape, gmC, gmPerTokenScale, gmD, groupListPtr, expertPerRank, MakeCallback(&aivWaitSync),
            syncInterval, resource);
    }

    // =============== DequantSwiglu end ===============
}

template <
    class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
    class ElementD, class LayoutD>
CATLASS_DEVICE void AllToAllVGmmDequantSwigluImpl_M0_256(
    Catlass::GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmD, GM_ADDR gmScale,
    GM_ADDR gmPerTokenScale, GM_ADDR tokenPerExpert, GM_ADDR ptrCumsumMM, GM_ADDR gmSymmetric,
    Catlass::MatrixCoord const& commCoreSplit, Catlass::MatrixCoord const& commBlockShape,
    Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank, Catlass::Arch::Resource<ArchTag> resource)
{
    AllToAllVGmmDequantSwigluImpl<
        ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementD, LayoutD, 256, 128, 256>(
        problemShape, gmA, gmB, gmC, gmD, gmScale, gmPerTokenScale, tokenPerExpert, ptrCumsumMM, gmSymmetric,
        commCoreSplit, commBlockShape, commTileShape, expertPerRank, resource);
}

template <
    class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
    class ElementD, class LayoutD>
CATLASS_DEVICE void AllToAllVGmmDequantSwigluImpl_M0_128(
    Catlass::GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmD, GM_ADDR gmScale,
    GM_ADDR gmPerTokenScale, GM_ADDR tokenPerExpert, GM_ADDR ptrCumsumMM, GM_ADDR gmSymmetric,
    Catlass::MatrixCoord const& commCoreSplit, Catlass::MatrixCoord const& commBlockShape,
    Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank, Catlass::Arch::Resource<ArchTag> resource)
{
    AllToAllVGmmDequantSwigluImpl<
        ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementD, LayoutD, 128, 256, 256>(
        problemShape, gmA, gmB, gmC, gmD, gmScale, gmPerTokenScale, tokenPerExpert, ptrCumsumMM, gmSymmetric,
        commCoreSplit, commBlockShape, commTileShape, expertPerRank, resource);
}

CATLASS_DEVICE
void BarrierBetweenUpAndDown()
{
    AscendC::PipeBarrier<PIPE_ALL>();
    Arch::CrossCoreFlag gmm1AivFinished{0};
    if constexpr (g_coreType == AscendC::AIV) {
        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(gmm1AivFinished);
    } else {
        Arch::CrossCoreWaitFlag(gmm1AivFinished);
    }
}

template <class ElementA_, class ElementC_>
struct WorkspaceInfo {
    using ElementA = ElementA_;
    using ElementC = ElementC_;
    constexpr static uint64_t MB_SIZE = 1024 * 1024UL;
    // In local gm
    GM_ADDR expandedRowIdx;
    GM_ADDR ptrCumsumMM;
    GM_ADDR ptrPerTokenScale;
    GM_ADDR ptrC;
    GM_ADDR ptrA;
    // In symmetric mem
    GM_ADDR symmetricA;
    GM_ADDR peerPerTokenScale;
    GM_ADDR tokensPerExpert;

    CATLASS_DEVICE
    WorkspaceInfo(GM_ADDR ptrLocalWorkspace, GM_ADDR ptrSymmWorkspace, const CocTilingParams& cocTiling)
    {
        uint32_t rankId = shmem_my_pe();
        uint32_t rankSize = shmem_n_pes();
        uint32_t maxOutputSize = cocTiling.m * cocTiling.topK * rankSize;
        uint32_t epSize = cocTiling.epSize;
        uint32_t expertNum = cocTiling.expertNum;
        uint32_t expertPerRank = expertNum / epSize;

        // workspace in local memory
        int64_t workspaceOffset = 0;
        expandedRowIdx = ptrLocalWorkspace;
        workspaceOffset += AlignUp(cocTiling.m, 256) * cocTiling.topK * sizeof(int32_t);
        ptrCumsumMM = ptrLocalWorkspace + workspaceOffset;
        workspaceOffset += (epSize * expertPerRank) * sizeof(int32_t);
        ptrPerTokenScale = ptrLocalWorkspace + workspaceOffset;
        workspaceOffset += maxOutputSize * sizeof(float);
        ptrC = ptrLocalWorkspace + workspaceOffset;
        workspaceOffset += maxOutputSize * cocTiling.n * sizeof(ElementC);
        ptrA = ptrLocalWorkspace + workspaceOffset;
        workspaceOffset += maxOutputSize * cocTiling.k * sizeof(ElementA);

        // workspace in symmetric memory
        int64_t workspaceOffsetSymm = 0;
        symmetricA = ptrSymmWorkspace + workspaceOffsetSymm;
        workspaceOffsetSymm += cocTiling.m * cocTiling.topK * cocTiling.k * sizeof(ElementA);
        peerPerTokenScale = ptrSymmWorkspace + workspaceOffsetSymm;
        workspaceOffsetSymm += MB_SIZE;
        tokensPerExpert = ptrSymmWorkspace + SHMEM_BUFF_BYTES - 2 * MB_SIZE;
    }
};

#if defined(ENABLE_ASCENDC_DUMP)
template <
    class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC, class ElementD,
    class LayoutD>
CATLASS_GLOBAL void DispatchGmmDequantSwiglu(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD, GM_ADDR gmScale, GM_ADDR gmExpertIdx, GM_ADDR gmWorkSpace,
    GM_ADDR gmSymmetric, CocTilingParams cocTiling, MoeInitRoutingQuantV2Tiling moeTiling, GM_ADDR dump)
{
    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#else
template <
    class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC, class ElementD,
    class LayoutD>
CATLASS_GLOBAL void DispatchGmmDequantSwiglu(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD, GM_ADDR gmScale, GM_ADDR gmExpertIdx, GM_ADDR gmWorkSpace,
    GM_ADDR gmSymmetric, CocTilingParams cocTiling, MoeInitRoutingQuantV2Tiling moeTiling)
{
#endif
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Catlass::Arch::AtlasA2;
    using WorkspaceInfo = WorkspaceInfo<ElementA, ElementC>;
    using AllGather = AllGather<ArchTag, int32_t>;

    AllGather allgather;

    uint32_t m = cocTiling.m;
    uint32_t n = cocTiling.n;
    uint32_t k = cocTiling.k;
    uint32_t m0 = cocTiling.m0;
    uint32_t n0 = cocTiling.n0;
    uint32_t k0 = cocTiling.k0;
    uint32_t commInterval = cocTiling.commInterval;
    uint32_t commTileM = cocTiling.commTileM;
    uint32_t commNpuSplit = cocTiling.commNpuSplit;
    uint32_t commDataSplit = cocTiling.commDataSplit;
    uint32_t commBlockM = cocTiling.commBlockM;
    uint32_t epSize = cocTiling.epSize;
    uint32_t expertNum = cocTiling.expertNum;
    uint32_t topK = cocTiling.topK;
    uint32_t expertPerRank = expertNum / epSize;

    uint32_t rankIdx = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    uint32_t maxOutputSize = cocTiling.m * cocTiling.topK * rankSize;

    Catlass::GemmCoord problemShape{m * topK, n, k};

    Catlass::MatrixCoord commCoreSplit{commDataSplit, commNpuSplit};
    Catlass::MatrixCoord commBlockShape{commBlockM, UINT_MAX / 2};
    Catlass::MatrixCoord commTileShape{8, k + 512};

    WorkspaceInfo workspaceInfo(gmWorkSpace, gmSymmetric, cocTiling);

    auto localTokensPerExpert = workspaceInfo.tokensPerExpert + rankIdx * epSize * expertPerRank * sizeof(int32_t);

    // token + scale per line as output
    moe_init_routing_quant_v2<bfloat16_t>(
        gmA, gmExpertIdx, nullptr /*moeInitRoutingQuantV2Scale*/, nullptr /*moeInitRoutingQuantV2Offset*/,
        workspaceInfo.symmetricA, workspaceInfo.expandedRowIdx, localTokensPerExpert,
        nullptr /*expertTokensBeforeCapacity*/, workspaceInfo.peerPerTokenScale, workspaceInfo.expandedRowIdx,
        &moeTiling.moeInitRoutingQuantV2TilingData, moeTiling.initRoutingQuantTilingKey);

    aclshmemx_barrier_all_vec();

    Catlass::Arch::Resource<ArchTag> resource;

    typename AllGather::Params allgatherParams{
        epSize * expertPerRank, localTokensPerExpert, nullptr, workspaceInfo.tokensPerExpert};
    allgather(allgatherParams, resource);

    BarrierBetweenUpAndDown();

    if (m0 == 128) {
        AllToAllVGmmDequantSwigluImpl_M0_128<
            ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementD, LayoutD>(
            problemShape, workspaceInfo.ptrA, gmB, workspaceInfo.ptrC, gmD, gmScale, workspaceInfo.ptrPerTokenScale,
            workspaceInfo.tokensPerExpert, workspaceInfo.ptrCumsumMM, workspaceInfo.symmetricA, commCoreSplit,
            commBlockShape, commTileShape, expertPerRank, resource);
    } else {
        AllToAllVGmmDequantSwigluImpl_M0_256<
            ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementD, LayoutD>(
            problemShape, workspaceInfo.ptrA, gmB, workspaceInfo.ptrC, gmD, gmScale, workspaceInfo.ptrPerTokenScale,
            workspaceInfo.tokensPerExpert, workspaceInfo.ptrCumsumMM, workspaceInfo.symmetricA, commCoreSplit,
            commBlockShape, commTileShape, expertPerRank, resource);
    }
}

#endif // DISPATCH_GMM_DEQUANT_SWIGLU_KERNEL_H
