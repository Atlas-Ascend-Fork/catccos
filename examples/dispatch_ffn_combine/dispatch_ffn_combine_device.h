
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DISPATCH_FFN_COMBINE_H
#define DISPATCH_FFN_COMBINE_H

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

#include "moe_init_routing_v2_tiling.h"
#include "moe_init_routing_v2.h"
#include "allgather_kernel_with_flag.h"
#include "group_swiglu.h"

#include "moe_token_unpermute_tiling.h"
#include "moe_token_unpermute.h"

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_local_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/comm/block/comm_block_scheduler_alltoallv_gmm.hpp"
#include "catccos/dgemm/kernel/alltoallv_gmm_v2.hpp"
#include "catccos/epilogue/block/block_epilogue_per_token_dequant_swiglu.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/comm/block/comm_block_scheduler_gmm_alltoallv.hpp"
#include "catccos/dgemm/kernel/gmm_alltoallv_v2.hpp"

using namespace AscendC;
using namespace Catccos;
inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {
    Catlass::L2_OFFSET
};

template <
    class ArchTag_, class ElementA_, class LayoutA_, class ElementB_, class LayoutB_, class ElementC_, class LayoutC_,
    uint32_t M0, uint32_t N0, uint32_t K0>
struct FfnPipelineTypes {
    using ArchTag = ArchTag_;
    using ElementA = ElementA_;
    using LayoutA = LayoutA_;
    using ElementB = ElementB_;
    using LayoutB = LayoutB_;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;

    static constexpr bool kEnableUnitFlag = true;
    static constexpr bool kEnableShuffleK = true;
    static constexpr bool kIsDynamic = true;

    static constexpr uint32_t kPreloadStages = 1;
    static constexpr uint32_t kL1Stages = 2;
    static constexpr uint32_t kL0AStages = 2;
    static constexpr uint32_t kL0BStages = 2;
    static constexpr uint32_t kL0CStages = 1;
    static constexpr uint32_t kCommUbStages = 2;

    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2PreloadAsync<
        kPreloadStages, kL1Stages, kL0AStages, kL0BStages, kL0CStages, kEnableUnitFlag, kEnableShuffleK>;

    using L1TileShape = Catlass::GemmShape<M0, N0, K0>;
    using L0TileShape = Catlass::GemmShape<M0, N0, 64>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
    using ElementGroupList = int64_t;

    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using LocalCopyBlockShape = Catlass::MatrixShape<48, UINT_MAX / 2>;
    using LocalCopyTileShape = Catlass::MatrixShape<48, 1024>;
    using LocalCopyDispatch = Catccos::Comm::AtlasA2CommLocalCopy<kCommUbStages>;

    // ===================== Up: AllToAllV -> GMM =====================
    static constexpr Catccos::detail::CopyDirect kUpRemoteDirect = Catccos::detail::CopyDirect::Get;

    using UpRemoteTileCopy = Comm::Tile::TileRemoteCopy<
        ArchTag, kIsDynamic, AType, AType, void, kUpRemoteDirect, Catccos::detail::CopyTransport::Mte>;

    using UpRemoteCommDispatch = Comm::AtlasCommRemoteCopy<ArchTag, kCommUbStages, kIsDynamic>;

    using UpRemoteCommBlock =
        Comm::Block::CommBlock<UpRemoteCommDispatch, AType, AType, void, UpRemoteTileCopy, TileScheduler>;

    using UpLocalTileCopy = Comm::Tile::TileRemoteCopy<
        ArchTag, false, AType, AType, LocalCopyTileShape, kUpRemoteDirect, Catccos::detail::CopyTransport::Mte>;

    using UpLocalCopyBlock = Catccos::Comm::Block::CommBlock<
        LocalCopyDispatch, AType, AType, LocalCopyBlockShape, UpLocalTileCopy, TileScheduler>;

    using UpCommScheduler = typename Catlass::Gemm::Block::BlockCommSchedulerAllToAllVGmm;

    using UpKernel = Catccos::DGemm::Kernel::AlltoallvGMMKernel<
        BlockMmad, BlockScheduler, ElementGroupList, UpLocalCopyBlock, UpRemoteCommBlock, UpCommScheduler>;

    // ===================== Swiglu =====================
    static constexpr uint32_t kSwigluUbStages = 1;

    using SwigluDispatchPolicy = Catccos::Epilogue::EpilogueAtlasA2PerTokenDequantSwiglu<kSwigluUbStages>;

    using SwigluDType = Catlass::Gemm::GemmType<ElementC, Catlass::layout::RowMajor>;

    using SwigluTileCopy = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType, CType, SwigluDType>;

    using SwigluTileScheduler = Catlass::Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using SwigluBlock = Catlass::Epilogue::Block::BlockEpilogue<
        SwigluDispatchPolicy, CType, SwigluDType, SwigluTileCopy, SwigluTileScheduler>;

    using SwigluKernel = Catccos::DGemm::Kernel::SwigluKernel<ArchTag, SwigluBlock>;

    // ===================== Down: GMM -> AllToAllV =====================
    static constexpr Catccos::detail::CopyDirect kDownRemoteDirect = Catccos::detail::CopyDirect::Put;

    // 这里保持当前实现的行为：RemoteCommBlock 使用 AType。
    // 如果后续 ElementA 和 ElementC 可能不同，建议再统一检查 GMMAlltoallvKernel 的类型约束。
    using DownRemoteType = AType;

    using DownRemoteTileCopy = Comm::Tile::TileRemoteCopy<
        ArchTag, kIsDynamic, DownRemoteType, DownRemoteType, void, kDownRemoteDirect,
        Catccos::detail::CopyTransport::Mte>;

    using DownRemoteCommDispatch = Comm::AtlasCommRemoteCopy<ArchTag, kCommUbStages, kIsDynamic>;

    using DownRemoteCommBlock = Comm::Block::CommBlock<
        DownRemoteCommDispatch, DownRemoteType, DownRemoteType, void, DownRemoteTileCopy, TileScheduler>;

    using DownLocalTileCopy = Comm::Tile::TileRemoteCopy<
        ArchTag, false, CType, CType, LocalCopyTileShape, kUpRemoteDirect, Catccos::detail::CopyTransport::Mte>;

    using DownLocalCopyBlock = Catccos::Comm::Block::CommBlock<
        LocalCopyDispatch, CType, CType, LocalCopyBlockShape, DownLocalTileCopy, TileScheduler>;

    using DownCommScheduler = typename Catlass::Gemm::Block::BlockCommSchedulerGmmAllToAllV;

    using DownKernel = Catccos::DGemm::Kernel::GMMAlltoallvKernel<
        BlockMmad, BlockScheduler, ElementGroupList, DownLocalCopyBlock, DownRemoteCommBlock, DownCommScheduler>;
};

template <class Types>
CATLASS_DEVICE void RunUpAllToAllVGmm(
    Catlass::GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmm1Output, GM_ADDR tokenPerExpert,
    GM_ADDR upWorkspace, GM_ADDR gmSymmetric, Catlass::MatrixCoord const& commBlockShape,
    Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank, uint32_t rankId, uint32_t rankSize,
    uint32_t maxOutputSize, int32_t syncInterval, typename Types::UpKernel& upKernel,
    Catccos::DGemm::Kernel::AicFinishSync<typename Types::UpKernel>& aicFinishSync,
    Catlass::Arch::Resource<typename Types::ArchTag> resource)
{
    using LayoutA = typename Types::LayoutA;
    using LayoutB = typename Types::LayoutB;
    using LayoutC = typename Types::LayoutC;

    LayoutA layoutA{problemShape.m(), problemShape.k()};
    LayoutB layoutB{problemShape.k(), problemShape.n()};
    LayoutC layoutC{problemShape.m(), problemShape.n()};

    typename Types::UpRemoteTileCopy::Params tileParams{commTileShape};
    typename Types::UpRemoteCommBlock::Params remoteCommParams{commBlockShape, tileParams};
    typename Types::UpLocalCopyBlock::Params localCopyParams{};

    typename Types::UpKernel::Params params{
        problemShape,
        rankSize,
        expertPerRank,
        maxOutputSize,

        rankId,
        rankSize,

        tokenPerExpert,

        gmA,
        layoutA,
        gmB,
        layoutB,
        gmm1Output,
        layoutC,

        upWorkspace,
        gmSymmetric,

        localCopyParams,
        remoteCommParams,

        MakeCallback(&aicFinishSync),
        syncInterval};

    upKernel(params, resource);
}

template <class Types>
CATLASS_DEVICE void RunSwigluStage(
    Catlass::GemmCoord problemShape, GM_ADDR gmm1Output, GM_ADDR swigluOutput, GM_ADDR groupListPtr,
    uint32_t expertPerRank, int32_t syncInterval, typename Types::SwigluKernel& swiglu,
    Catccos::DGemm::Kernel::AivWaitSync<typename Types::UpKernel>& aivWaitUpGmm,
    Catccos::DGemm::Kernel::AivFinishSync<typename Types::SwigluKernel>& aivFinishSwiglu,
    Catlass::Arch::Resource<typename Types::ArchTag> resource)
{
    if constexpr (g_coreType == AscendC::AIV) {
        using LayoutC = typename Types::LayoutC;
        using SwigluDType = typename Types::SwigluDType;

        LayoutC layoutC{problemShape.m(), problemShape.n()};
        typename SwigluDType::Layout layoutD{problemShape.m(), problemShape.n() / 2};

        typename Types::SwigluKernel::Params swigluParams{
            expertPerRank,
            problemShape.GetCoordMN(),
            gmm1Output,
            layoutC,
            swigluOutput,
            layoutD,
            groupListPtr,
            MakeCallback(&aivWaitUpGmm),
            MakeCallback(&aivFinishSwiglu),
            syncInterval};

        swiglu(swigluParams, resource);
    }
}

template <class Types>
CATLASS_DEVICE void RunDownGmmAllToAllV(
    Catlass::GemmCoord upProblemShape, GM_ADDR swigluOutput, GM_ADDR gmB2, GM_ADDR gmD, GM_ADDR tokenPerExpert,
    GM_ADDR downWorkspace, GM_ADDR gmSymmetric, Catlass::MatrixCoord const& commBlockShape,
    Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank, uint32_t rankId, uint32_t rankSize,
    uint32_t maxOutputSize, int32_t syncInterval,
    Catccos::DGemm::Kernel::AicWaitSync<typename Types::SwigluKernel>& aicWaitSwiglu,
    Catlass::Arch::Resource<typename Types::ArchTag> resource)
{
    using LayoutA = typename Types::LayoutA;
    using LayoutB = typename Types::LayoutB;
    using LayoutC = typename Types::LayoutC;

    uint32_t k2 = upProblemShape.n() / 2;
    uint32_t n2 = upProblemShape.k();

    Catlass::GemmCoord downProblemShape{
        upProblemShape.m(), // M = m * topK
        n2,                 // N = k
        k2                  // K = n / 2
    };

    LayoutA layoutA{downProblemShape.m(), downProblemShape.k()};
    LayoutB layoutB{downProblemShape.k(), downProblemShape.n()};
    LayoutC layoutC{downProblemShape.m(), downProblemShape.n()};

    typename Types::DownRemoteTileCopy::Params tileParams{commTileShape};
    typename Types::DownRemoteCommBlock::Params remoteCommParams{commBlockShape, tileParams};
    typename Types::DownLocalCopyBlock::Params localCopyParams{};

    typename Types::DownKernel downKernel;

    typename Types::DownKernel::Params params{
        downProblemShape,
        rankSize,
        expertPerRank,
        maxOutputSize,
        rankId,
        rankSize,
        tokenPerExpert,
        swigluOutput,
        layoutA,
        gmB2,
        layoutB,
        gmD,
        layoutC,
        downWorkspace,
        gmSymmetric,
        localCopyParams,
        remoteCommParams,

        MakeCallback(&aicWaitSwiglu),
        syncInterval};

    downKernel(params, resource);
}

template <
    class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
    uint32_t M0, uint32_t N0, uint32_t K0>
CATLASS_DEVICE void DispatchFFNCombineImpl(
    Catlass::GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmB2, GM_ADDR gmD, GM_ADDR tokenPerExpert,
    GM_ADDR ptrWorkspace, GM_ADDR gmSymmetric, GM_ADDR gmmAllToAllWorkspace, Catlass::MatrixCoord const& commCoreSplit,
    Catlass::MatrixCoord const& commBlockShape, Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank,
    uint32_t topK, Catlass::Arch::Resource<ArchTag> resource)
{
    (void)commCoreSplit;
    (void)topK;

    using Types = FfnPipelineTypes<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, M0, N0, K0>;

    uint32_t rankId = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();
    uint32_t maxOutputSize = problemShape.m() * rankSize;

    int32_t syncInterval = expertPerRank > 1 ? expertPerRank - 1 : 1;

    // Workspace layout inside ptrWorkspace:
    //   [0, maxOutputSize * K): AllToAllV_GMM temp input
    //   next meta: cumsum/token metadata
    //   next: GMM1 output
    GM_ADDR upWorkspace = ptrWorkspace;
    GM_ADDR upCumsumBase = ptrWorkspace + maxOutputSize * problemShape.k() * sizeof(ElementA);

    GM_ADDR gmm1Output = upCumsumBase + rankSize * rankSize * expertPerRank * sizeof(int32_t);

    GM_ADDR groupListPtr = upCumsumBase + (rankSize - 1) * expertPerRank * sizeof(int32_t);

    GM_ADDR swigluOutput = gmm1Output + maxOutputSize * problemShape.k() * sizeof(ElementC);

    // ===================== Up: AllToAllV -> GMM =====================
    typename Types::UpKernel upKernel;
    Catccos::DGemm::Kernel::AicFinishSync<typename Types::UpKernel> aicFinishUpGmm{&upKernel};
    Catccos::DGemm::Kernel::AivWaitSync<typename Types::UpKernel> aivWaitUpGmm{&upKernel};

    RunUpAllToAllVGmm<Types>(
        problemShape, gmA, gmB, gmm1Output, tokenPerExpert, upWorkspace, gmSymmetric, commBlockShape, commTileShape,
        expertPerRank, rankId, rankSize, maxOutputSize, syncInterval, upKernel, aicFinishUpGmm, resource);

    // ===================== Swiglu =====================
    typename Types::SwigluKernel swiglu;
    Catccos::DGemm::Kernel::AivFinishSync<typename Types::SwigluKernel> aivFinishSwiglu{&swiglu};
    Catccos::DGemm::Kernel::AicWaitSync<typename Types::SwigluKernel> aicWaitSwiglu{&swiglu};

    RunSwigluStage<Types>(
        problemShape, gmm1Output, swigluOutput, groupListPtr, expertPerRank, syncInterval, swiglu, aivWaitUpGmm,
        aivFinishSwiglu, resource);

    // ===================== Down: GMM -> AllToAllV =====================
    RunDownGmmAllToAllV<Types>(
        problemShape, swigluOutput, gmB2, gmD, tokenPerExpert, gmmAllToAllWorkspace, gmSymmetric, commBlockShape,
        commTileShape, expertPerRank, rankId, rankSize, maxOutputSize, syncInterval, aicWaitSwiglu, resource);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_DEVICE void DispatchFFNCombineImpl_M0_256(
    Catlass::GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmB2, GM_ADDR gmD, GM_ADDR tokenPerExpert,
    GM_ADDR ptrWorkspace, GM_ADDR gmSymmetric, GM_ADDR gmmAllToAllWorkspace, Catlass::MatrixCoord const& commCoreSplit,
    Catlass::MatrixCoord const& commBlockShape, Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank,
    uint32_t topK, Catlass::Arch::Resource<ArchTag> resource)
{
    DispatchFFNCombineImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 256, 128, 256>(
        problemShape, gmA, gmB, gmB2, gmD, tokenPerExpert, ptrWorkspace, gmSymmetric, gmmAllToAllWorkspace,
        commCoreSplit, commBlockShape, commTileShape, expertPerRank, topK, resource);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_DEVICE void DispatchFFNCombineImpl_M0_128(
    Catlass::GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmB2, GM_ADDR gmD, GM_ADDR tokenPerExpert,
    GM_ADDR ptrWorkspace, GM_ADDR gmSymmetric, GM_ADDR gmmAllToAllWorkspace, Catlass::MatrixCoord const& commCoreSplit,
    Catlass::MatrixCoord const& commBlockShape, Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank,
    uint32_t topK, Catlass::Arch::Resource<ArchTag> resource)
{
    DispatchFFNCombineImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 128, 256, 256>(
        problemShape, gmA, gmB, gmB2, gmD, tokenPerExpert, ptrWorkspace, gmSymmetric, gmmAllToAllWorkspace,
        commCoreSplit, commBlockShape, commTileShape, expertPerRank, topK, resource);
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

    // Local GM workspace
    GM_ADDR expandedRowIdx;
    GM_ADDR moeInitRoutingWorkspace;
    GM_ADDR allToAllVGmmWorkspace;
    GM_ADDR allToAllVGmmOut;
    GM_ADDR swigluOutput;
    GM_ADDR gmmAllToAllWorkspace;

    // Symmetric GM workspace
    GM_ADDR symmetricA;
    GM_ADDR perTokenScale;
    GM_ADDR tokensPerExpert;

    int64_t localWorkspaceBytes;
    int64_t symmetricWorkspaceBytes;

    CATLASS_HOST_DEVICE
    static int64_t AlignBytes(int64_t bytes, int64_t align = 512) { return (bytes + align - 1) / align * align; }

    CATLASS_DEVICE
    WorkspaceInfo(GM_ADDR ptrLocalWorkspace, GM_ADDR ptrSymmWorkspace, const CocTilingParams& cocTiling)
    {
        uint32_t rankSize = shmem_n_pes();
        uint32_t epSize = cocTiling.epSize;
        uint32_t expertNum = cocTiling.expertNum;
        uint32_t expertPerRank = expertNum / epSize;

        int64_t expandedM = static_cast<int64_t>(cocTiling.m) * cocTiling.topK;
        int64_t maxOutputSize = expandedM * rankSize;

        // ================= local workspace =================
        int64_t workspaceOffsetLocal = 0;

        // expandedRowIdx: [AlignUp(M, 256), topK]
        expandedRowIdx = ptrLocalWorkspace + workspaceOffsetLocal;
        workspaceOffsetLocal +=
            AlignBytes(static_cast<int64_t>(AlignUp(cocTiling.m, 256)) * cocTiling.topK * sizeof(int32_t));

        // moe_init_routing and first AllToAllV_GMM are executed sequentially,
        // so their workspace can start from the same offset.
        moeInitRoutingWorkspace = ptrLocalWorkspace + workspaceOffsetLocal;
        allToAllVGmmWorkspace = moeInitRoutingWorkspace;

        // AllToAllV_GMM workspace:
        // temp input: [maxOutputSize, K]
        // cumsum/meta: [epSize, epSize, expertPerRank]
        workspaceOffsetLocal += AlignBytes(maxOutputSize * cocTiling.k * sizeof(ElementA));
        workspaceOffsetLocal += AlignBytes(static_cast<int64_t>(epSize) * epSize * expertPerRank * sizeof(int32_t));

        // GMM1 output: [maxOutputSize, N]. Swiglu output reuses the second half of
        // this buffer in the current implementation, so no extra local workspace is
        // required for swigluOutput.
        allToAllVGmmOut = ptrLocalWorkspace + workspaceOffsetLocal;
        workspaceOffsetLocal += AlignBytes(maxOutputSize * cocTiling.n * sizeof(ElementC));
        swigluOutput = allToAllVGmmOut + maxOutputSize * cocTiling.k * sizeof(ElementC);

        // GMM2 + AllToAllV workspace:
        // temp output: [maxOutputSize, K]
        // cumsum/meta: [epSize, epSize, expertPerRank]
        gmmAllToAllWorkspace = ptrLocalWorkspace + workspaceOffsetLocal;
        workspaceOffsetLocal += AlignBytes(maxOutputSize * cocTiling.k * sizeof(ElementC));
        workspaceOffsetLocal += AlignBytes(static_cast<int64_t>(epSize) * epSize * expertPerRank * sizeof(int32_t));

        localWorkspaceBytes = workspaceOffsetLocal;

        // ================= symmetric workspace =================
        int64_t workspaceOffsetSymm = 0;

        // symmetricA:
        // 1. routing output A: [M * topK, K]
        // 2. GMM2 alltoallv routed-back output: [M * topK, K]
        // Use the larger element size to make the reuse safe when ElementA/ElementC differ.
        constexpr int64_t SymmetricElementBytes =
            sizeof(ElementA) > sizeof(ElementC) ? sizeof(ElementA) : sizeof(ElementC);

        symmetricA = ptrSymmWorkspace + workspaceOffsetSymm;
        workspaceOffsetSymm += AlignBytes(expandedM * cocTiling.k * SymmetricElementBytes);

        tokensPerExpert = ptrSymmWorkspace + workspaceOffsetSymm;
        workspaceOffsetSymm += AlignBytes(static_cast<int64_t>(epSize) * epSize * expertPerRank * sizeof(int32_t));

        perTokenScale = ptrSymmWorkspace + workspaceOffsetSymm;
        // If quant routing scale is enabled later, reserve this region:
        // workspaceOffsetSymm += AlignBytes(expandedM * sizeof(float));

        symmetricWorkspaceBytes = workspaceOffsetSymm;
    }
};

#if defined(ENABLE_ASCENDC_DUMP)
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_GLOBAL void DispatchFFNCombine(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmB2, GM_ADDR gmD, GM_ADDR gmExpertIdx, GM_ADDR gmProbs,
    GM_ADDR gmWorkSpace, GM_ADDR gmSymmetric, CocTilingParams cocTiling, MoeInitRoutingQuantV2Tiling moeTiling,
    GM_ADDR dump)
{
    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#else
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_GLOBAL void DispatchFFNCombine(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmB2, GM_ADDR gmD, GM_ADDR gmExpertIdx, GM_ADDR gmProbs,
    GM_ADDR gmWorkSpace, GM_ADDR gmSymmetric, CocTilingParams cocTiling, MoeInitRoutingQuantV2Tiling moeTiling)
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
    Catlass::MatrixCoord commTileShape{commTileM / 2, k0};

    WorkspaceInfo workspaceInfo(gmWorkSpace, gmSymmetric, cocTiling);

    auto localTokensPerExpert = workspaceInfo.tokensPerExpert + rankIdx * epSize * expertPerRank * sizeof(int32_t);

    moe_init_routing_v2<ElementA>(
        gmA, gmExpertIdx, workspaceInfo.symmetricA, workspaceInfo.expandedRowIdx, localTokensPerExpert,
        nullptr /*expertTokensBeforeCapacity*/, workspaceInfo.moeInitRoutingWorkspace,
        &moeTiling.moeInitRoutingQuantV2TilingData, moeTiling.initRoutingQuantTilingKey);

    shmemx_barrier_all_vec();

    Catlass::Arch::Resource<ArchTag> resource;

    typename AllGather::Params allgatherParams{
        epSize * expertPerRank, localTokensPerExpert, nullptr, workspaceInfo.tokensPerExpert};
    allgather(allgatherParams, resource);

    BarrierBetweenUpAndDown();

    if (m0 == 128) {
        DispatchFFNCombineImpl_M0_128<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
            problemShape, nullptr, gmB, gmB2, nullptr, workspaceInfo.tokensPerExpert,
            workspaceInfo.allToAllVGmmWorkspace, workspaceInfo.symmetricA, workspaceInfo.gmmAllToAllWorkspace,
            commCoreSplit, commBlockShape, commTileShape, expertPerRank, topK, resource);
    } else {
        DispatchFFNCombineImpl_M0_256<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
            problemShape, nullptr, gmB, gmB2, nullptr, workspaceInfo.tokensPerExpert,
            workspaceInfo.allToAllVGmmWorkspace, workspaceInfo.symmetricA, workspaceInfo.gmmAllToAllWorkspace,
            commCoreSplit, commBlockShape, commTileShape, expertPerRank, topK, resource);
    }

    shmemx_barrier_all_vec();

    // =============== token upermute start ===============
    if ASCEND_IS_AIV {
        MoeTokenUnpermuteTilingData tilingData;
        MoeTokenUnpermuteTiling(m * topK, k, topK, tilingData, get_block_num() * get_subblockdim());

        KernelMoeTokenUnpermute<ElementC, int32_t, float, true> kernelMoeTokenUnpermuteOp;

        kernelMoeTokenUnpermuteOp.Init(
            workspaceInfo.symmetricA, workspaceInfo.expandedRowIdx, gmProbs, gmD, &tilingData);
        kernelMoeTokenUnpermuteOp.Process();
    }
}

#endif
