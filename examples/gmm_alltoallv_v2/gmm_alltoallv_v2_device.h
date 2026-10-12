
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GMM_ALLTOALLV_V2_KERNEL_H
#define GMM_ALLTOALLV_V2_KERNEL_H

#include "info.h"

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.h"
#endif

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_swizzle.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/comm/block/comm_block_scheduler_gmm_alltoallv.hpp"
#include "catccos/dgemm/kernel/gmm_alltoallv_v2.hpp"

using namespace AscendC;
using namespace Catccos;

template <
    class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
    uint32_t M0, uint32_t N0, uint32_t K0>
CATLASS_DEVICE void GMMAllToAllVImpl(
    Catlass::GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR tokenPerExpert,
    GM_ADDR ptrWorkspace, GM_ADDR gmSymmetric, Catlass::MatrixCoord const& commCoreSplit,
    Catlass::MatrixCoord const& commBlockShape, Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank,
    Catlass::Arch::Resource<ArchTag> resource)
{
    // =============== AllToAllv_GMM begin ===============
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    constexpr uint32_t workspaceStages = 2;
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;

    constexpr bool ENABLE_UNIT_FLAG = true;
    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2PreloadAsync<
        preloadStages, l1Stages, l0AStages, l0BStages, l0CStages, enableUnitFlag, enableShuffleK>;
    using L1TileShape = Catlass::GemmShape<M0, N0, K0>;
    using L0TileShape = Catlass::GemmShape<M0, N0, 64>;
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;

    constexpr bool IS_DYNAMIC = true;

    // remote comm
    constexpr uint32_t UB_STAGES = 2;
    constexpr Catccos::detail::CopyDirect COPY_DIRECT = Catccos::detail::CopyDirect::Put;
    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using RemoteCommDispatch = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, COPY_DIRECT, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using RemoteCommBlock =
        Comm::Block::CommBlock<RemoteCommDispatch, RemoteSrcType, RemoteDstType, void, TileRemoteCopy, TileScheduler>;

    typename TileRemoteCopy::Params tileParams{commTileShape};

    typename RemoteCommBlock::Params remoteCommParams{commBlockShape, tileParams};

    // local copy
    using CopySrcType = AType;
    using CopyDstType = AType;
    using LocalCopyBlockShape = Catlass::MatrixShape<48, UINT_MAX / 2>;
    using LocalCopyTileShape = Catlass::MatrixShape<48, 1024>;
    using TileLocalCopy = Comm::Tile::TileRemoteCopy<
        ArchTag, false, RemoteSrcType, RemoteDstType, LocalCopyTileShape, COPY_DIRECT, CopyTransport::Mte>;
    using LocalCopyDispatch = Catccos::Comm::AtlasA2CommLocalCopy<UB_STAGES>;
    using LocalCopyBlock = Catccos::Comm::Block::CommBlock<
        LocalCopyDispatch, CopySrcType, CopyDstType, LocalCopyBlockShape, TileLocalCopy, TileScheduler>;

    typename LocalCopyBlock::Params localCopyParams{};

    // communication scheduler
    using BlockCommScheduler = typename Catlass::Gemm::Block::BlockCommSchedulerGmmAllToAllV;

    // kernel level
    using ElementGroupList = int64_t;
    using MatmulKernel = Catccos::DGemm::Kernel::GMMAlltoallvKernel<
        BlockMmad, BlockScheduler, ElementGroupList, LocalCopyBlock, RemoteCommBlock, BlockCommScheduler>;
    using EmptyCallBack = Catccos::DGemm::Kernel::GmmAllToAllVEmptyCallBack<MatmulKernel>;
    // using AivWaitSync = Catccos::DGemm::Kernel::AivWaitSync<MatmulKernel>;

    LayoutA layoutA{problemShape.m(), problemShape.k()};
    LayoutB layoutB{problemShape.k(), problemShape.n()};
    LayoutC layoutC{problemShape.m(), problemShape.n()};

    uint32_t rankId = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();
    uint32_t maxOutputSize = problemShape.m() * rankSize;

    MatmulKernel mmKernel;
    EmptyCallBack emptyCallBack{&mmKernel};
    int32_t syncInterval = expertPerRank > 1 ? expertPerRank - 1 : 1;

    typename MatmulKernel::Params params{problemShape,     rankSize,
                                         expertPerRank,    maxOutputSize,
                                         rankId,           rankSize,
                                         tokenPerExpert,   gmA,
                                         layoutA,          gmB,
                                         layoutB,          gmC,
                                         layoutC,          ptrWorkspace,
                                         gmSymmetric,      localCopyParams,
                                         remoteCommParams, MakeCallback(&emptyCallBack),
                                         syncInterval};
    mmKernel(params, resource);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_DEVICE void GMMAllToAllVImpl_M0_256(
    Catlass::GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR tokenPerExpert,
    GM_ADDR ptrWorkspace, GM_ADDR gmSymmetric, Catlass::MatrixCoord const& commCoreSplit,
    Catlass::MatrixCoord const& commBlockShape, Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank,
    Catlass::Arch::Resource<ArchTag> resource)
{
    GMMAllToAllVImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 256, 128, 256>(
        problemShape, gmA, gmB, gmC, tokenPerExpert, ptrWorkspace, gmSymmetric, commCoreSplit, commBlockShape,
        commTileShape, expertPerRank, resource);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_DEVICE void GMMAllToAllVImpl_M0_128(
    Catlass::GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR tokenPerExpert,
    GM_ADDR ptrWorkspace, GM_ADDR gmSymmetric, Catlass::MatrixCoord const& commCoreSplit,
    Catlass::MatrixCoord const& commBlockShape, Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank,
    Catlass::Arch::Resource<ArchTag> resource)
{
    GMMAllToAllVImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 128, 256, 256>(
        problemShape, gmA, gmB, gmC, tokenPerExpert, ptrWorkspace, gmSymmetric, commCoreSplit, commBlockShape,
        commTileShape, expertPerRank, resource);
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
    // In local gm
    GM_ADDR expandedRowIdx;
    GM_ADDR GMMAllToAllVWorkspace;
    GM_ADDR moeInitRoutingWorkspace;
    GM_ADDR GMMAllToAllVOut;
    // In symmetric mem
    GM_ADDR symmetricC;
    GM_ADDR perTokenScale;
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
        int64_t workspaceOffsetLocal = 0;
        expandedRowIdx = ptrLocalWorkspace + workspaceOffsetLocal;
        workspaceOffsetLocal += AlignUp(cocTiling.m, 256) * cocTiling.topK * sizeof(int32_t);
        moeInitRoutingWorkspace = ptrLocalWorkspace + workspaceOffsetLocal;
        GMMAllToAllVWorkspace = ptrLocalWorkspace + workspaceOffsetLocal;
        workspaceOffsetLocal +=
            maxOutputSize * cocTiling.n * sizeof(ElementC) + epSize * epSize * expertPerRank * sizeof(int32_t);
        GMMAllToAllVOut = ptrLocalWorkspace + workspaceOffsetLocal;
        workspaceOffsetLocal += maxOutputSize * cocTiling.n * sizeof(ElementC);

        // workspace in symmetric memory
        int64_t workspaceOffsetSymm = 0;
        symmetricC = ptrSymmWorkspace + workspaceOffsetSymm;
        workspaceOffsetSymm += cocTiling.m * cocTiling.topK * cocTiling.n * rankSize * sizeof(ElementC);
        tokensPerExpert = ptrSymmWorkspace + workspaceOffsetSymm;
        workspaceOffsetSymm += epSize * epSize * expertPerRank * sizeof(int32_t);
        perTokenScale = ptrSymmWorkspace + workspaceOffsetSymm;
        // workspaceOffsetSymm += cocTiling.m * cocTiling.topK * sizeof(float);
    }
};

#if defined(ENABLE_ASCENDC_DUMP)
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_GLOBAL void GMMAllToAllVV2(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmWorkSpace, GM_ADDR gmSymmetric,
    GM_ADDR gmTokenPerExpert, CocTilingParams cocTiling, GM_ADDR dump)
{
    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#else
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_GLOBAL void GMMAllToAllVV2(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmWorkSpace, GM_ADDR gmSymmetric,
    GM_ADDR gmTokenPerExpert, CocTilingParams cocTiling)
{
#endif
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Catlass::Arch::AtlasA2;
    using WorkspaceInfo = WorkspaceInfo<ElementA, ElementC>;

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
    Catlass::MatrixCoord commTileShape{commTileM / 2, k0};

    WorkspaceInfo workspaceInfo(gmWorkSpace, gmSymmetric, cocTiling);

    auto localTokensPerExpert = gmTokenPerExpert + rankIdx * epSize * expertPerRank * sizeof(int32_t);

    shmemx_barrier_all_vec();

    Catlass::Arch::Resource<ArchTag> resource;

    if (m0 == 128) {
        GMMAllToAllVImpl_M0_128<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
            problemShape, gmA, gmB, gmC, gmTokenPerExpert, workspaceInfo.GMMAllToAllVWorkspace,
            workspaceInfo.symmetricC, commCoreSplit, commBlockShape, commTileShape, expertPerRank, resource);
    } else {
        GMMAllToAllVImpl_M0_256<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
            problemShape, gmA, gmB, gmC, gmTokenPerExpert, workspaceInfo.GMMAllToAllVWorkspace,
            workspaceInfo.symmetricC, commCoreSplit, commBlockShape, commTileShape, expertPerRank, resource);
    }
}

#endif // GMM_ALLTOALLV_V2_KERNEL_H
