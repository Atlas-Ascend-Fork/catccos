

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_DISPATCH_FFN_COMBINE_H
#define ASCEND950_DISPATCH_FFN_COMBINE_H

#include "info.h"
#include "catccos/detail/ascend950_moe_ready.hpp"

// from catlass
#include "catlass/arch/arch.hpp"
#include "catlass/arch/local_tensor_buffer.hpp"
#include "catlass/catlass.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/layout/layout.hpp"
#if defined(ENABLE_ASCENDC_DUMP)
#include "debug.h"
#endif

using AscendC::AIC;
using AscendC::TPosition;

#include "allgather_kernel_with_flag.h"
#include "ascend950_swiglu_from_ub.h"
#include "catccos/catccos.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_local_copy.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "moe_init_routing_v2.h"
#include "moe_init_routing_v2_tiling.h"
#include "moe_token_unpermute.h"
#include "moe_token_unpermute_tiling.h"
#define CATCCOS_COMM_SCHEDULER_ARCH Catlass::Arch::Ascend950
#include "catccos/comm/block/comm_block_scheduler_alltoallv_gmm.hpp"
#include "catccos/comm/block/comm_block_scheduler_gmm_alltoallv.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"
#include "catccos/dgemm/kernel/ascend950_alltoall_gmm.hpp"
#include "catccos/epilogue/block/block_epilogue_regbase_swiglu.hpp"
#undef CATCCOS_COMM_SCHEDULER_ARCH
#include "catccos/dgemm/kernel/ascend950_gmm_alltoall.hpp"

using namespace AscendC;
using namespace Catccos;

inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {
    Catlass::L2_OFFSET
};

template <
    class ArchTag_, class ElementA_, class LayoutA_, class ElementB_, class LayoutB_, class ElementC_, class LayoutC_,
    uint32_t Gmm1M0>
struct FfnPipelineTypes {
    using ArchTag = ArchTag_;
    using ElementDispatch = ElementA_;
    using ElementA = ElementB_;
    using LayoutA = LayoutA_;
    using ElementB = ElementB_;
    using LayoutB = LayoutB_;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using ElementScale = float8_e8m0_t;

    static constexpr bool kEnableUnitFlag = true;
    static constexpr bool kUseHF32 = false;
    static constexpr bool kIsDynamic = true;
    static constexpr uint32_t kCommUbStages = 2;
    static constexpr uint32_t kGmm1M0 = Gmm1M0;
    static constexpr uint32_t kGmm1N0 = 256;
    static constexpr uint32_t kGmm1K0 = 256;
    static constexpr uint32_t kGmm2M0 = 256;
    static constexpr uint32_t kGmm2N0 = 256;
    static constexpr uint32_t kGmm2K0 = 256;
    static constexpr uint32_t kPhysicalUbBytes = 256 * 1024;

    using DispatchPolicy = Catlass::Gemm::MmadMx<ArchTag, kEnableUnitFlag, 16>;

    using Gmm1L1TileShape = tla::Shape<tla::Int<kGmm1M0>, tla::Int<kGmm1N0>, tla::Int<kGmm1K0>>;
    using Gmm1L0TileShape = tla::Shape<tla::Int<kGmm1M0>, tla::Int<kGmm1N0>, tla::Int<128>>;
    using Gmm2L1TileShape = tla::Shape<tla::Int<kGmm2M0>, tla::Int<kGmm2N0>, tla::Int<kGmm2K0>>;
    using Gmm2L0TileShape = tla::Shape<tla::Int<kGmm2M0>, tla::Int<kGmm2N0>, tla::Int<128>>;

    static_assert(kGmm1M0 == 128 || kGmm1M0 == 256, "GMM1 M tile must be 128 or 256");
    static_assert(
        kGmm1M0 != 256 || std::is_same_v<ArchTag, Catlass::Arch::Ascend950>,
        "GMM1 M256 full-UB path is supported only on Ascend950");
    static_assert(
        kGmm1M0 != 256 || std::is_same_v<ElementC, bfloat16_t>,
        "GMM1 M256 UB aliasing is validated only for BF16 output");
    static_assert(
        kGmm1M0 != 256 || 2 * kGmm1M0 * kGmm1N0 * sizeof(ElementC) == 256 * 1024,
        "GMM1 M256 must occupy two exact 128 KiB BF16 UB planes");
    static_assert(
        std::is_same_v<ArchTag, Catlass::Arch::Ascend950>, "GMM2 M256 full-UB path is supported only on Ascend950");
    static_assert(std::is_same_v<ElementC, bfloat16_t>, "GMM2 M256 UB ping-pong is validated only for BF16 output");
    static_assert(
        2 * kGmm2M0 * kGmm2N0 * sizeof(ElementC) == kPhysicalUbBytes,
        "GMM2 M256 must occupy two exact 128 KiB BF16 UB planes");

    using AType = Catlass::Gemm::GemmType<ElementDispatch, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;

    using LayoutScaleA = decltype(tla::MakeMxScaleLayout<ElementScale, LayoutA, false>(0U, 0U));
    using LayoutScaleB = decltype(tla::MakeMxScaleLayout<ElementScale, LayoutB, true>(0U, 0U));
    using TileCopyToUB = Catlass::Gemm::Tile::PackedMxTileCopyTlaToUB<
        ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementScale, LayoutScaleA, ElementScale, LayoutScaleB, ElementC,
        LayoutC, void, Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;

    using Gmm1BlockMmadToUB = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy, Gmm1L1TileShape, Gmm1L0TileShape, ElementA, ElementB, ElementC, void, TileCopyToUB>;

    using Gmm1BlockScheduler = std::conditional_t<
        kGmm1M0 == 256, typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>,
        typename Gemm::Block::GemmIdentityBlockSwizzle<9, 1>>;
    using Gmm2BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using LocalCopyBlockShape = Catlass::MatrixShape<48, UINT_MAX / 2>;
    using LocalCopyTileShape = Catlass::MatrixShape<48, 1024>;
    using LocalCopyDispatch = Catccos::Comm::AtlasCommLocalCopy<ArchTag, kCommUbStages>;

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

    using UpKernel = Catccos::DGemm::Kernel::Ascend950AllToAllGmmKernel<
        Gmm1BlockMmadToUB, Gmm1BlockScheduler, UpLocalCopyBlock, UpRemoteCommBlock, UpCommScheduler, ElementDispatch>;

    // ===================== Swiglu =====================
    static constexpr uint32_t kRegBaseSwigluUbStages = 2;
    static constexpr uint32_t kRegBaseSwigluTileM = 8;

    using RegBaseSwigluDispatchPolicy =
        Catccos::Epilogue::EpilogueAscend950RegBaseSwiglu<kRegBaseSwigluUbStages, kRegBaseSwigluTileM>;

    using SwigluDispatchPolicy = RegBaseSwigluDispatchPolicy;

    using SwigluDType = Catlass::Gemm::GemmType<ElementC, Catlass::layout::RowMajor>;
    using SwigluTileCopy = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType, CType, SwigluDType>;
    using SwigluTileScheduler = Catlass::Epilogue::Tile::EpilogueHorizontalTileSwizzle;
    using SwigluBlock = Catlass::Epilogue::Block::BlockEpilogue<
        SwigluDispatchPolicy, CType, SwigluDType, SwigluTileCopy, SwigluTileScheduler>;
    using SwigluKernel = Catccos::DGemm::Kernel::Ascend950SwigluMxQuantFromUbKernel<
        ArchTag, SwigluBlock, ElementA, ElementScale, Gmm1BlockScheduler, kGmm1M0, kGmm1N0>;

    static_assert(
        UpKernel::UB_TILE_M == SwigluKernel::UB_TILE_M && UpKernel::UB_TILE_N == SwigluKernel::UB_TILE_N,
        "GMM producer and SwiGLU consumer must use the same UB tile shape");

    // ===================== Down: GMM -> AIV1 UB -> remote symmetric GM =====================
    using DownTileCopyToAiv1UB = Catccos::DGemm::Kernel::PackedMxTileCopyTlaToTargetUb<
        ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementScale, LayoutScaleA, ElementScale, LayoutScaleB, ElementC,
        LayoutC, 1>;

    using DownBlockMmadToAiv1UB = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy, Gmm2L1TileShape, Gmm2L0TileShape, ElementA, ElementB, ElementC, void, DownTileCopyToAiv1UB>;

    using DownMetadataScheduler = typename Catlass::Gemm::Block::BlockCommSchedulerGmmAllToAllV;
    // The top-level fused operator owns the world barrier immediately before unpermute.
    static constexpr bool kDownWorldBarrierAtEnd = false;

    using DownKernel = Catccos::DGemm::Kernel::Ascend950GmmAllToAllKernel<
        DownBlockMmadToAiv1UB, Gmm2BlockScheduler, DownMetadataScheduler, kGmm1M0, kGmm1N0, kPhysicalUbBytes,
        kDownWorldBarrierAtEnd>;
};

template <class Types>
CATLASS_DEVICE void RunUpAllToAllVGmm(
    Catlass::GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmScaleB, GM_ADDR tokenPerExpert,
    GM_ADDR upWorkspace, GM_ADDR quantAWorkspace, GM_ADDR quantAScaleWorkspace, GM_ADDR dispatchReady,
    GM_ADDR swigluReady, GM_ADDR gmSymmetric, Catlass::MatrixCoord const& commBlockShape,
    Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank, uint32_t rankId, uint32_t rankSize,
    uint32_t maxOutputSize, typename Types::UpKernel& upKernel,
    Catccos::DGemm::Kernel::Ascend950AicWaitUbFree<typename Types::SwigluKernel>& aicWaitUbFree,
    Catccos::DGemm::Kernel::Ascend950AicNotifyUbReady<typename Types::SwigluKernel>& aicNotifyUbReady,
    Catlass::Arch::Resource<typename Types::ArchTag> resource)
{
    using LayoutA = typename Types::LayoutA;
    using LayoutB = typename Types::LayoutB;

    LayoutA layoutA{problemShape.m(), problemShape.k()};
    LayoutB layoutB{problemShape.k(), problemShape.n()};

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

        quantAWorkspace,
        quantAScaleWorkspace,
        gmScaleB,

        upWorkspace,
        dispatchReady,
        swigluReady,
        gmSymmetric,

        localCopyParams,
        remoteCommParams,

        MakeCallback(&aicWaitUbFree),
        MakeCallback(&aicNotifyUbReady)};

    upKernel(params, resource);
}

template <class Types>
CATLASS_DEVICE void RunSwigluStage(
    Catlass::GemmCoord problemShape, GM_ADDR swigluBf16Output, GM_ADDR swigluOutput, GM_ADDR swigluScaleOutput,
    GM_ADDR groupListPtr, GM_ADDR swigluReady, uint32_t expertPerRank, typename Types::SwigluKernel& swiglu,
    Catlass::Arch::Resource<typename Types::ArchTag> resource)
{
    if constexpr (g_coreType == AscendC::AIV) {
        typename Types::SwigluKernel::Params swigluParams{expertPerRank, problemShape.GetCoordMN(), swigluBf16Output,
                                                          swigluOutput,  swigluScaleOutput,         groupListPtr,
                                                          swigluReady};

        swiglu(swigluParams, resource);
    }
}

template <class Types>
CATLASS_DEVICE void RunDownGmmAllToAllV(
    Catlass::GemmCoord upProblemShape, GM_ADDR swigluOutput, GM_ADDR swigluScaleOutput, GM_ADDR gmB2, GM_ADDR gmB2Scale,
    GM_ADDR tokenPerExpert, GM_ADDR cumsumMM, GM_ADDR downWorkspace, GM_ADDR swigluReady, GM_ADDR gmSymmetric,
    uint32_t expertPerRank, uint32_t rankId, uint32_t rankSize, uint32_t maxOutputSize,
    Catlass::Arch::Resource<typename Types::ArchTag> resource)
{
    using LayoutA = typename Types::LayoutA;
    using LayoutB = typename Types::LayoutB;

    uint32_t k2 = upProblemShape.n() / 2;
    uint32_t n2 = upProblemShape.k();

    Catlass::GemmCoord downProblemShape{
        upProblemShape.m(), // M = m * topK
        n2,                 // N = k
        k2                  // K = n / 2
    };

    LayoutA layoutA{downProblemShape.m(), downProblemShape.k()};
    LayoutB layoutB{downProblemShape.k(), downProblemShape.n()};

    typename Types::DownKernel downKernel;

    typename Types::DownKernel::Params params{
        downProblemShape,  rankSize,  expertPerRank, maxOutputSize, rankId,     rankSize,
        tokenPerExpert,    cumsumMM,  swigluOutput,  layoutA,       gmB2,       layoutB,
        swigluScaleOutput, gmB2Scale, downWorkspace, swigluReady,   gmSymmetric};

    downKernel(params, resource);
}

template <
    class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
    uint32_t Gmm1M0>
CATLASS_DEVICE void DispatchFFNCombineImpl(
    Catlass::GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmBScale, GM_ADDR gmB2, GM_ADDR gmB2Scale,
    GM_ADDR gmD, GM_ADDR tokenPerExpert, GM_ADDR ptrWorkspace, GM_ADDR quantAWorkspace, GM_ADDR quantAScaleWorkspace,
    GM_ADDR swigluOutput, GM_ADDR swigluBf16Output, GM_ADDR swigluScaleOutput, GM_ADDR gmSymmetric,
    GM_ADDR gmmAllToAllWorkspace, Catlass::MatrixCoord const& commCoreSplit, Catlass::MatrixCoord const& commBlockShape,
    Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank, uint32_t topK,
    Catlass::Arch::Resource<ArchTag> resource)
{
    (void)commCoreSplit;
    (void)topK;
    (void)gmD;

    using Types = FfnPipelineTypes<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, Gmm1M0>;

    uint32_t rankId = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();
    uint32_t maxOutputSize = problemShape.m() * rankSize;

    // Packed dispatch writes directly into the quantized GMM input regions, so
    // ptrWorkspace begins with cumsum/token metadata (and is shared with the
    // earlier routing scratch lifetime).
    GM_ADDR upWorkspace = ptrWorkspace;
    GM_ADDR upCumsumBase = ptrWorkspace;

    GM_ADDR groupListPtr = upCumsumBase + (rankSize - 1) * expertPerRank * sizeof(int32_t);

    // GMM2 direct mode only uses the prefix of its output staging area for
    // destination offsets. Reuse the following cache-line-aligned bytes for the
    // per-expert Dispatch -> GMM1 readiness counters.
    GM_ADDR dispatchReady =
        gmmAllToAllWorkspace + Catccos::Detail::Ascend950MoeReady::DispatchOffset(rankSize, expertPerRank);
    // The direct MX path no longer uses the BF16 SwiGLU staging area. Reuse its
    // aligned prefix for one cache line per expert readiness counter without
    // changing the public workspace size or ABI.
    GM_ADDR swigluReady = swigluBf16Output;

    // ===================== Up: AllToAllV -> GMM =====================
    typename Types::UpKernel upKernel;

    typename Types::SwigluKernel swiglu;
    Catccos::DGemm::Kernel::Ascend950AicUbSyncState aicUbSyncState{};
    Catccos::DGemm::Kernel::Ascend950AicWaitUbFree<typename Types::SwigluKernel> aicWaitUbFree{aicUbSyncState};
    Catccos::DGemm::Kernel::Ascend950AicNotifyUbReady<typename Types::SwigluKernel> aicNotifyUbReady{aicUbSyncState};
    RunUpAllToAllVGmm<Types>(
        problemShape, gmA, gmB, gmBScale, tokenPerExpert, upWorkspace, quantAWorkspace, quantAScaleWorkspace,
        dispatchReady, swigluReady, gmA, commBlockShape, commTileShape, expertPerRank, rankId, rankSize, maxOutputSize,
        upKernel, aicWaitUbFree, aicNotifyUbReady, resource);

    // ===================== Swiglu =====================
    RunSwigluStage<Types>(
        problemShape, swigluBf16Output, swigluOutput, swigluScaleOutput, groupListPtr, swigluReady, expertPerRank,
        swiglu, resource);

    // ===================== Down: GMM -> AllToAllV =====================
    RunDownGmmAllToAllV<Types>(
        problemShape, swigluOutput, swigluScaleOutput, gmB2, gmB2Scale, tokenPerExpert, upCumsumBase,
        gmmAllToAllWorkspace, swigluReady, gmSymmetric, expertPerRank, rankId, rankSize, maxOutputSize, resource);

    // GMM2 is allowed to overlap AIV0's last SwiGLU/MX tile. Consume the
    // remaining acknowledgement only after the down path has completed, when
    // it is normally already available. The stateful callback is a no-op on
    // cores that never produced a GMM1 tile.
    if constexpr (g_coreType == AscendC::AIC) {
        aicWaitUbFree();
    }
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_DEVICE void DispatchFFNCombineImpl_M0_256(
    Catlass::GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmBScale, GM_ADDR gmB2, GM_ADDR gmB2Scale,
    GM_ADDR gmD, GM_ADDR tokenPerExpert, GM_ADDR ptrWorkspace, GM_ADDR quantAWorkspace, GM_ADDR quantAScaleWorkspace,
    GM_ADDR swigluOutput, GM_ADDR swigluBf16Output, GM_ADDR swigluScaleOutput, GM_ADDR gmSymmetric,
    GM_ADDR gmmAllToAllWorkspace, Catlass::MatrixCoord const& commCoreSplit, Catlass::MatrixCoord const& commBlockShape,
    Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank, uint32_t topK,
    Catlass::Arch::Resource<ArchTag> resource)
{
    DispatchFFNCombineImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 256>(
        problemShape, gmA, gmB, gmBScale, gmB2, gmB2Scale, gmD, tokenPerExpert, ptrWorkspace, quantAWorkspace,
        quantAScaleWorkspace, swigluOutput, swigluBf16Output, swigluScaleOutput, gmSymmetric, gmmAllToAllWorkspace,
        commCoreSplit, commBlockShape, commTileShape, expertPerRank, topK, resource);
}

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_DEVICE void DispatchFFNCombineImpl_M0_128(
    Catlass::GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmBScale, GM_ADDR gmB2, GM_ADDR gmB2Scale,
    GM_ADDR gmD, GM_ADDR tokenPerExpert, GM_ADDR ptrWorkspace, GM_ADDR quantAWorkspace, GM_ADDR quantAScaleWorkspace,
    GM_ADDR swigluOutput, GM_ADDR swigluBf16Output, GM_ADDR swigluScaleOutput, GM_ADDR gmSymmetric,
    GM_ADDR gmmAllToAllWorkspace, Catlass::MatrixCoord const& commCoreSplit, Catlass::MatrixCoord const& commBlockShape,
    Catlass::MatrixCoord const& commTileShape, uint32_t expertPerRank, uint32_t topK,
    Catlass::Arch::Resource<ArchTag> resource)
{
    DispatchFFNCombineImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 128>(
        problemShape, gmA, gmB, gmBScale, gmB2, gmB2Scale, gmD, tokenPerExpert, ptrWorkspace, quantAWorkspace,
        quantAScaleWorkspace, swigluOutput, swigluBf16Output, swigluScaleOutput, gmSymmetric, gmmAllToAllWorkspace,
        commCoreSplit, commBlockShape, commTileShape, expertPerRank, topK, resource);
}

template <class ArchTag>
CATLASS_DEVICE void PrepareDstExpertOffsets(
    GM_ADDR tokenPerExpertPtr, GM_ADDR gmmAllToAllWorkspace, uint32_t rank, uint32_t rankSize, uint32_t expertPerRank,
    Catlass::Arch::Resource<ArchTag> resource)
{
    if constexpr (g_coreType == AscendC::AIV) {
        AscendC::GlobalTensor<int32_t> tokenPerExpert;
        tokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenPerExpertPtr));
        AscendC::GlobalTensor<int32_t> dstExpertOffset;
        dstExpertOffset.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(gmmAllToAllWorkspace));
        auto offsetsUb = resource.ubBuf.template GetBufferByByte<int32_t>(0);

        uint32_t coreIdx = get_block_idx() + get_subblockid() * get_block_num();
        uint32_t coreNum = get_block_num() * get_subblockdim();
        coreNum = coreNum == 0 ? 1 : coreNum;
        uint32_t rankExpertBase = rank * expertPerRank;
        uint32_t rankExpertEnd = rankExpertBase + expertPerRank;

        for (uint32_t dstRank = coreIdx; dstRank < rankSize; dstRank += coreNum) {
            uint32_t tokenBase = dstRank * rankSize * expertPerRank;
            uint32_t running = 0;
            for (uint32_t globalExpert = 0; globalExpert < rankExpertEnd; ++globalExpert) {
                if (globalExpert >= rankExpertBase) {
                    offsetsUb.SetValue(globalExpert - rankExpertBase, running);
                }
                running += tokenPerExpert(tokenBase + globalExpert);
            }

            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(expertPerRank * sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPad(dstExpertOffset[dstRank * expertPerRank], offsetsUb, copyParams);
            // Fence this rank's write before the UB is reused by another rank.
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }
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

template <class ElementDispatch_, class ElementQuant_, class ElementC_>
struct WorkspaceInfo {
    using ElementDispatch = ElementDispatch_;
    using ElementQuant = ElementQuant_;
    using ElementC = ElementC_;
    using ElementScale = float8_e8m0_t;

    // Local GM workspace
    GM_ADDR expandedRowIdx;
    GM_ADDR moeInitRoutingWorkspace;
    GM_ADDR allToAllVGmmWorkspace;
    GM_ADDR quantAWorkspace;
    GM_ADDR quantAScaleWorkspace;
    GM_ADDR swigluBf16Output;
    GM_ADDR swigluOutput;
    GM_ADDR swigluScaleOutput;
    GM_ADDR gmmAllToAllWorkspace;

    // Symmetric GM workspace
    GM_ADDR symmetricA;
    GM_ADDR symmetricPackedA;
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
        // so sorting scratch and dispatch metadata share the same base. Packed
        // MX dispatch writes directly to quantAWorkspace/quantAScaleWorkspace;
        // no expanded BF16 dispatch buffer is reserved here.
        moeInitRoutingWorkspace = ptrLocalWorkspace + workspaceOffsetLocal;
        allToAllVGmmWorkspace = moeInitRoutingWorkspace;

        // AllToAllV_GMM cumsum/meta: [epSize, epSize, expertPerRank]
        workspaceOffsetLocal += AlignBytes(static_cast<int64_t>(epSize) * epSize * expertPerRank * sizeof(int32_t));

        int64_t scaleKRaw = CeilDiv<int64_t>(cocTiling.k, Catlass::MX_SCALE_GROUP_NUM);
        int64_t scaleK = (scaleKRaw + 1) / 2 * 2;
        quantAWorkspace = ptrLocalWorkspace + workspaceOffsetLocal;
        workspaceOffsetLocal += AlignBytes(maxOutputSize * cocTiling.k * sizeof(ElementQuant));
        quantAScaleWorkspace = ptrLocalWorkspace + workspaceOffsetLocal;
        workspaceOffsetLocal += AlignBytes(maxOutputSize * scaleK * sizeof(ElementScale));

        int64_t swigluN = cocTiling.n / 2;
        int64_t swigluScaleNRaw = CeilDiv<int64_t>(swigluN, Catlass::MX_SCALE_GROUP_NUM);
        int64_t swigluScaleN = (swigluScaleNRaw + 1) / 2 * 2;
        swigluBf16Output = ptrLocalWorkspace + workspaceOffsetLocal;
        workspaceOffsetLocal += Catccos::Detail::Ascend950MoeReady::SwigluRegionBytes(
            maxOutputSize * swigluN * sizeof(ElementC), expertPerRank);
        swigluOutput = ptrLocalWorkspace + workspaceOffsetLocal;
        workspaceOffsetLocal += AlignBytes(maxOutputSize * swigluN * sizeof(ElementQuant));
        swigluScaleOutput = ptrLocalWorkspace + workspaceOffsetLocal;
        workspaceOffsetLocal += AlignBytes(maxOutputSize * swigluScaleN * sizeof(ElementScale));

        // GMM2 + AllToAllV workspace:
        // temp output: [maxOutputSize, K]
        // cumsum/meta: [epSize, epSize, expertPerRank]
        gmmAllToAllWorkspace = ptrLocalWorkspace + workspaceOffsetLocal;
        workspaceOffsetLocal += Catccos::Detail::Ascend950MoeReady::DispatchRegionBytes(
            maxOutputSize * cocTiling.k * sizeof(ElementC), rankSize, expertPerRank);
        workspaceOffsetLocal += AlignBytes(static_cast<int64_t>(epSize) * epSize * expertPerRank * sizeof(int32_t));

        localWorkspaceBytes = workspaceOffsetLocal;

        // ================= symmetric workspace =================
        int64_t workspaceOffsetSymm = 0;

        // GMM2 may return rows while slower peers still read packed inputs.
        // These two distributed lifetimes must own separate symmetric regions.
        constexpr int64_t SymmetricElementBytes =
            sizeof(ElementDispatch) > sizeof(ElementC) ? sizeof(ElementDispatch) : sizeof(ElementC);

        symmetricA = ptrSymmWorkspace + workspaceOffsetSymm;
        workspaceOffsetSymm += AlignBytes(expandedM * cocTiling.k * SymmetricElementBytes);

        symmetricPackedA = ptrSymmWorkspace + workspaceOffsetSymm;
        workspaceOffsetSymm += AlignBytes(expandedM * RoundUp<int64_t>(cocTiling.k + scaleK, Catlass::BYTE_PER_BLK));

        tokensPerExpert = ptrSymmWorkspace + workspaceOffsetSymm;
        workspaceOffsetSymm += AlignBytes(static_cast<int64_t>(epSize) * epSize * expertPerRank * sizeof(int32_t));

        perTokenScale = ptrSymmWorkspace + workspaceOffsetSymm;
        // If quant routing scale is enabled later, reserve this region:
        // workspaceOffsetSymm += AlignBytes(expandedM * sizeof(float));

        symmetricWorkspaceBytes = workspaceOffsetSymm;
    }
};

CATLASS_DEVICE void ResetInt32Region(GM_ADDR ptr, uint32_t elementCount, AscendC::LocalTensor<int32_t> zeroUb)
{
    constexpr uint32_t elementsPerCacheLine = 64 / sizeof(int32_t);
    uint32_t coreIdx = get_block_idx() + get_subblockid() * get_block_num();
    uint32_t coreNum = get_block_num() * get_subblockdim();
    coreNum = coreNum == 0 ? 1 : coreNum;

    AscendC::GlobalTensor<int32_t> region;
    region.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ptr));
    uint32_t cacheLineCount = (elementCount + elementsPerCacheLine - 1) / elementsPerCacheLine;
    for (uint32_t line = coreIdx; line < cacheLineCount; line += coreNum) {
        uint32_t offset = line * elementsPerCacheLine;
        uint32_t remaining = elementCount - offset;
        uint32_t count = remaining < elementsPerCacheLine ? remaining : elementsPerCacheLine;
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(region[offset], zeroUb, copyParams);
    }
}

template <class ArchTag>
CATLASS_DEVICE void ResetPersistentState(
    GM_ADDR tokensPerExpert, GM_ADDR dispatchReady, GM_ADDR swigluReady, uint32_t rankSize, uint32_t expertPerRank,
    Catlass::Arch::Resource<ArchTag> resource)
{
    if constexpr (g_coreType == AscendC::AIV) {
        constexpr uint32_t readyStride = Catccos::Detail::Ascend950MoeReady::kStride;
        auto zeroUb = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::Duplicate(zeroUb, static_cast<int32_t>(0), readyStride);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);

        ResetInt32Region(tokensPerExpert, rankSize * rankSize * expertPerRank, zeroUb);
        ResetInt32Region(dispatchReady, expertPerRank * readyStride, zeroUb);
        ResetInt32Region(swigluReady, expertPerRank * readyStride, zeroUb);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::SyncAll<true>();
}

template <class ArchTag, class ElementInput, class ElementQuant>
CATLASS_DEVICE void QuantScatterRoutingMxFp8(
    GM_ADDR gmInput, GM_ADDR expandedRowIdx, GM_ADDR packedSymmetric, uint32_t tokenCount, uint32_t hiddenSize,
    uint32_t topK, Catlass::Arch::Resource<ArchTag>& resource)
{
    if constexpr (g_coreType != AscendC::AIV) {
        return;
    }

    using InputType = Catlass::Gemm::GemmType<ElementInput, Catlass::layout::RowMajor>;
    using QuantType = Catlass::Gemm::GemmType<ElementQuant, Catlass::layout::RowMajor>;
    using BlockMxQuant =
        Catccos::Comm::Block::CommBlockMxQuant<1, Catlass::MX_SCALE_GROUP_NUM, 2, InputType, QuantType>;

    uint32_t coreIdx = get_block_idx() + get_subblockid() * get_block_num();
    uint32_t coreNum = get_block_num() * get_subblockdim();
    coreNum = coreNum == 0 ? 1 : coreNum;
    uint32_t tokensPerCore = CeilDiv<uint32_t>(tokenCount, coreNum);
    uint32_t tokenBegin = coreIdx * tokensPerCore;
    uint32_t remainingTokens = tokenBegin < tokenCount ? tokenCount - tokenBegin : 0;
    uint32_t localTokenCount = remainingTokens < tokensPerCore ? remainingTokens : tokensPerCore;

    AscendC::GlobalTensor<ElementInput> input;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ ElementInput*>(gmInput));
    AscendC::GlobalTensor<int32_t> rowIdx;
    rowIdx.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expandedRowIdx));
    AscendC::GlobalTensor<uint8_t> packedOutput;
    packedOutput.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(packedSymmetric));

    typename BlockMxQuant::Params quantParams{hiddenSize};
    BlockMxQuant mxQuant(resource, quantParams);
    mxQuant.InitBlockLoop();
    mxQuant.QuantScatterPacked(input, rowIdx, packedOutput, tokenBegin, localTokenCount, topK);
    mxQuant.FinalizeBlockLoop();
}

#if defined(ENABLE_ASCENDC_DUMP)
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_GLOBAL void DispatchFFNCombine(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmBScale, GM_ADDR gmB2, GM_ADDR gmB2Scale, GM_ADDR gmD,
    GM_ADDR gmExpertIdx, GM_ADDR gmProbs, GM_ADDR gmWorkSpace, GM_ADDR gmSymmetric, CocTilingParams cocTiling,
    MoeInitRoutingQuantV2Tiling moeTiling, GM_ADDR dump)
{
    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#else
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_GLOBAL void DispatchFFNCombine(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmBScale, GM_ADDR gmB2, GM_ADDR gmB2Scale, GM_ADDR gmD,
    GM_ADDR gmExpertIdx, GM_ADDR gmProbs, GM_ADDR gmWorkSpace, GM_ADDR gmSymmetric, CocTilingParams cocTiling,
    MoeInitRoutingQuantV2Tiling moeTiling)
{
#endif
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Catlass::Arch::Ascend950;
    using WorkspaceInfo = WorkspaceInfo<ElementA, ElementB, ElementC>;
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

    Catlass::Arch::Resource<ArchTag> resource;
    GM_ADDR dispatchReady = workspaceInfo.gmmAllToAllWorkspace +
                            Catccos::Detail::Ascend950MoeReady::DispatchOffset(rankSize, expertPerRank);
    ResetPersistentState<ArchTag>(
        workspaceInfo.tokensPerExpert, dispatchReady, workspaceInfo.swigluBf16Output, rankSize, expertPerRank,
        resource);

    // Keep the v2 sort/count/inverse-index contract, but replace its expanded
    // BF16 gather with one MXFP8 quantization per original token followed by a
    // top-k scatter into packed symmetric rows.
    moe_init_routing_v2<ElementA, true>(
        gmA, gmExpertIdx, workspaceInfo.symmetricPackedA, workspaceInfo.expandedRowIdx, localTokensPerExpert,
        nullptr /*expertTokensBeforeCapacity*/, workspaceInfo.moeInitRoutingWorkspace,
        &moeTiling.moeInitRoutingQuantV2TilingData, moeTiling.initRoutingQuantTilingKey);

    QuantScatterRoutingMxFp8<ArchTag, ElementA, ElementB>(
        gmA, workspaceInfo.expandedRowIdx, workspaceInfo.symmetricPackedA, m, k, topK, resource);

    shmemx_barrier_all_vec();
    typename AllGather::Params allgatherParams{
        epSize * expertPerRank, localTokensPerExpert, nullptr, workspaceInfo.tokensPerExpert};
    allgather(allgatherParams, resource);
    PrepareDstExpertOffsets<ArchTag>(
        workspaceInfo.tokensPerExpert, workspaceInfo.gmmAllToAllWorkspace, rankIdx, rankSize, expertPerRank, resource);

    BarrierBetweenUpAndDown();

    if (m0 == 128) {
        DispatchFFNCombineImpl_M0_128<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
            problemShape, workspaceInfo.symmetricPackedA, gmB, gmBScale, gmB2, gmB2Scale, nullptr,
            workspaceInfo.tokensPerExpert, workspaceInfo.allToAllVGmmWorkspace, workspaceInfo.quantAWorkspace,
            workspaceInfo.quantAScaleWorkspace, workspaceInfo.swigluOutput, workspaceInfo.swigluBf16Output,
            workspaceInfo.swigluScaleOutput, workspaceInfo.symmetricA, workspaceInfo.gmmAllToAllWorkspace,
            commCoreSplit, commBlockShape, commTileShape, expertPerRank, topK, resource);
    } else {
        DispatchFFNCombineImpl_M0_256<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
            problemShape, workspaceInfo.symmetricPackedA, gmB, gmBScale, gmB2, gmB2Scale, nullptr,
            workspaceInfo.tokensPerExpert, workspaceInfo.allToAllVGmmWorkspace, workspaceInfo.quantAWorkspace,
            workspaceInfo.quantAScaleWorkspace, workspaceInfo.swigluOutput, workspaceInfo.swigluBf16Output,
            workspaceInfo.swigluScaleOutput, workspaceInfo.symmetricA, workspaceInfo.gmmAllToAllWorkspace,
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
    // Retire AIV unpermute reads on every rank before the next reusable launch.
    shmemx_barrier_all_vec();
}

#endif
