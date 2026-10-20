/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALL_MATMUL_SPLIT_K_URMA_LOCAL_ATOMIC_H
#define CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALL_MATMUL_SPLIT_K_URMA_LOCAL_ATOMIC_H

#include "catccos/catccos.hpp"

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/numeric_size.hpp"

#include "shmem.h"
#include "tla/tensor.hpp"

namespace Catccos::DGemm::Kernel {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

/// MXFP8 AllToAll + MatMul split-K using remote writes and direct local MMAD.
///
/// AIC first computes the complete local source-rank contribution directly from
/// local GM and writes C normally before entering the communication consume loop.
/// AIV only sends remote-rank chunks. After each remote URMA stage finishes, AIC
/// computes remote source-rank contributions and atomically adds them into C.
template <
    class BlockMmad_, class BlockUdmaAllToAll_, class BlockUdmaAllToAllScale_, class BlockScheduler_,
    uint32_t WORKSPACE_STAGES_>
class Ascend950AllToAllMatmulSplitKUrmaLocalAtomic {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutTagA = typename BlockMmad::TileCopy::LayoutTagA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using ElementMxScaleA = typename BlockMmad::TileCopy::ElementMxScaleA;
    using LayoutMxScaleA = typename BlockMmad::TileCopy::LayoutMxScaleA;
    using ElementMxScaleB = typename BlockMmad::TileCopy::ElementMxScaleB;
    using LayoutMxScaleB = typename BlockMmad::TileCopy::LayoutMxScaleB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockUdmaAllToAll = BlockUdmaAllToAll_;
    using LayoutAllToAllSrc = typename BlockUdmaAllToAll::LayoutSrc;

    using BlockUdmaAllToAllScale = BlockUdmaAllToAllScale_;
    using LayoutAllToAllScaleSrc = typename BlockUdmaAllToAllScale::LayoutSrc;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static_assert(
        WORKSPACE_STAGES * 2 <= Catlass::Arch::FFTS_MAX_FLAG + 1,
        "split-K URMA kernel uses two cross-core flags per workspace stage.");

    struct Params {
        GemmCoord problemShape;

        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;

        __gm__ ElementA* ptrA;
        LayoutA layoutA;
        __gm__ ElementB* ptrB;
        LayoutB layoutB;
        __gm__ ElementMxScaleA* ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        __gm__ ElementMxScaleB* ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        __gm__ ElementC* ptrC;
        LayoutC layoutC;
        LayoutAllToAllSrc layoutAllToAllSrc;
        LayoutAllToAllScaleSrc layoutAllToAllScaleSrc;
        GM_ADDR ptrSymmetric;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, uint32_t rank_, uint32_t rankSize_, uint32_t commInterval_,
            LayoutAllToAllSrc layoutAllToAllSrc_, LayoutAllToAllScaleSrc layoutAllToAllScaleSrc_, GM_ADDR ptrA_,
            LayoutA const& layoutA_, GM_ADDR ptrB_, LayoutB const& layoutB_, GM_ADDR ptrMxScaleA_,
            LayoutMxScaleA const& layoutMxScaleA_, GM_ADDR ptrMxScaleB_, LayoutMxScaleB const& layoutMxScaleB_,
            GM_ADDR ptrC_, LayoutC const& layoutC_, GM_ADDR ptrSymmetric_)
            : problemShape(problemShape_),
              rankIdx(rank_),
              rankSize(rankSize_),
              commInterval(commInterval_),
              ptrA(reinterpret_cast<__gm__ ElementA*>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB*>(ptrB_)),
              layoutB(layoutB_),
              ptrMxScaleA(reinterpret_cast<__gm__ ElementMxScaleA*>(ptrMxScaleA_)),
              layoutMxScaleA(layoutMxScaleA_),
              ptrMxScaleB(reinterpret_cast<__gm__ ElementMxScaleB*>(ptrMxScaleB_)),
              layoutMxScaleB(layoutMxScaleB_),
              ptrC(reinterpret_cast<__gm__ ElementC*>(ptrC_)),
              layoutC(layoutC_),
              layoutAllToAllSrc(layoutAllToAllSrc_),
              layoutAllToAllScaleSrc(layoutAllToAllScaleSrc_),
              ptrSymmetric(ptrSymmetric_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrMxScaleA;
        GM_ADDR ptrMxScaleB;
        GM_ADDR ptrC;
        GM_ADDR ptrSymmetric;
    };

    static size_t GetWorkspaceSize(Arguments const& args) { return 0; }

    static Params ToUnderlyingArguments(Arguments const& args, uint8_t* workspace = nullptr)
    {
        (void)workspace;
        uint32_t m = args.problemShape.m();
        uint32_t n = args.problemShape.n();
        uint32_t localK = args.problemShape.k();
        uint32_t chunkM = m / args.rankSize;
        uint32_t fullK = localK * args.rankSize;
        uint32_t localScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(localK);
        uint32_t alignedLocalScaleK = RoundUp<2>(localScaleK);
        uint32_t fullScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(fullK);

        LayoutAllToAllSrc layoutAllToAllSrc{m, localK, localK};
        LayoutAllToAllScaleSrc layoutAllToAllScaleSrc{m, localScaleK, alignedLocalScaleK};
        LayoutTagB layoutTagB{fullK, n};
        Catlass::layout::RowMajor layoutTagC{chunkM, n, n};

        auto layoutA = tla::MakeLayoutFromTag(layoutAllToAllSrc);
        auto layoutB = tla::MakeLayoutFromTag(layoutTagB);
        auto layoutC = tla::MakeLayoutFromTag(layoutTagC);
        auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagA, false>(m, localScaleK);
        auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagB, true>(fullScaleK, n);

        return Params(
            args.problemShape, args.rankIdx, args.rankSize, args.commInterval, layoutAllToAllSrc,
            layoutAllToAllScaleSrc, args.ptrA, layoutA, args.ptrB, layoutB, args.ptrMxScaleA, layoutMxScaleA,
            args.ptrMxScaleB, layoutMxScaleB, args.ptrC, layoutC, args.ptrSymmetric);
    }

    CATLASS_DEVICE
    Ascend950AllToAllMatmulSplitKUrmaLocalAtomic()
    {
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            flagAivFinishCompute[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx + WORKSPACE_STAGES);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params& params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        GemmCoord blockShape = GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K};

        uint32_t chunkM = params.problemShape.m() / params.rankSize;
        uint32_t localK = params.problemShape.k();
        uint32_t localScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(localK);
        uint32_t alignedLocalScaleK = RoundUp<2>(localScaleK);
        uint32_t commSizeM = params.commInterval * L1_TILE_M;
        uint32_t commLoops = CeilDiv(chunkM, commSizeM);

        constexpr uint32_t ELE_NUM_PER_FRACTAL_A =
            Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<ElementA>::value;
        int64_t alignedLocalK = RoundUp<int64_t>(localK, ELE_NUM_PER_FRACTAL_A);
        int64_t aStageStride = static_cast<int64_t>(params.rankSize) * commSizeM * alignedLocalK;
        int64_t scaleStageStride =
            static_cast<int64_t>(params.rankSize) * commSizeM * static_cast<int64_t>(alignedLocalScaleK);
        uint64_t aWorkspaceBytes = static_cast<uint64_t>(WORKSPACE_STAGES) * aStageStride *
                                   Catlass::SizeOfBits<ElementA>::value / Catlass::SizeOfBits<uint8_t>::value;
        auto ptrScaleSymmetricBase = reinterpret_cast<__gm__ ElementMxScaleA*>(params.ptrSymmetric + aWorkspaceBytes);

        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
        gmMxScaleB.SetGlobalBuffer(params.ptrMxScaleB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);

        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Catlass::Arch::PositionGM{});
        auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutMxScaleB, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Catlass::Arch::PositionGM{});

        auto localProblemShape = Catlass::MakeCoord<uint32_t>(chunkM, params.problemShape.n(), localK, 1);
        BlockScheduler localScheduler(localProblemShape, blockShape.GetCoordMN(), chunkM, alignedLocalK);
        uint32_t localCoreLoops = localScheduler.GetCoreLoops();

        MatrixCoord localSrcOffset{params.rankIdx * chunkM, 0};
        AscendC::GlobalTensor<ElementA> gmLocalA;
        gmLocalA.SetGlobalBuffer(params.ptrA + params.layoutAllToAllSrc.GetOffset(localSrcOffset));
        AscendC::GlobalTensor<ElementMxScaleA> gmLocalMxScaleA;
        gmLocalMxScaleA.SetGlobalBuffer(params.ptrMxScaleA + params.layoutAllToAllScaleSrc.GetOffset(localSrcOffset));

        for (uint32_t loopIdx = aicoreIdx; loopIdx < localCoreLoops; loopIdx += aicoreNum) {
            auto blockOffset = localScheduler.GetBlockOffset(loopIdx);
            auto actualBlockShape = localScheduler.GetActualBlockShapeByOffset(blockOffset);

            MatrixCoord offsetBInLocal = blockOffset.GetCoordKN();
            MatrixCoord offsetB{params.rankIdx * localK + offsetBInLocal.row(), offsetBInLocal.column()};
            MatrixCoord offsetC{blockOffset.m(), blockOffset.n()};

            auto tensorBlockB = GetTile(
                tensorB, tla::MakeCoord(offsetB.row(), offsetB.column()),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockMxScaleB = GetTile(
                tensorMxScaleB, tla::MakeCoord(offsetB.row() / Catlass::MX_SCALE_GROUP_NUM, offsetB.column()),
                tla::MakeShape(CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));
            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord(offsetC.row(), offsetC.column()),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            blockMmad(
                gmLocalA, gmLocalMxScaleA, tensorBlockB, tensorBlockC, actualBlockShape.GetCoordMNK(), blockOffset.m(),
                chunkM, localK, alignedLocalK, localScaleK, alignedLocalScaleK, tensorBlockMxScaleB);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        uint32_t remoteK = (params.rankSize > 1) ? ((params.rankSize - 1) * localK) : localK;
        auto remoteProblemShape = Catlass::MakeCoord<uint32_t>(chunkM, params.problemShape.n(), remoteK, 1);
        BlockScheduler remoteScheduler(remoteProblemShape, blockShape.GetCoordMN(), chunkM, alignedLocalK);
        uint32_t remoteCoreLoops = remoteScheduler.GetCoreLoops();

        // gmC.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;
            uint32_t actualCommSizeM = Min(commSizeM, chunkM - commIdx * commSizeM);
            uint32_t commMStart = commIdx * commSizeM;
            uint32_t commMEnd = commMStart + actualCommSizeM;

            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]);
            AscendC::PipeBarrier<PIPE_ALL>();

            if (params.rankSize > 1) {
                AscendC::SetAtomicAdd<ElementC>();
                AscendC::PipeBarrier<PIPE_ALL>();
                for (uint32_t loopIdx = aicoreIdx; loopIdx < remoteCoreLoops; loopIdx += aicoreNum) {
                    auto blockOffset = remoteScheduler.GetBlockOffset(loopIdx);
                    if (blockOffset.m() < commMStart || blockOffset.m() >= commMEnd) {
                        continue;
                    }

                    auto actualBlockShape = remoteScheduler.GetActualBlockShapeByOffset(blockOffset);
                    uint32_t blockMOffsetInStage = blockOffset.m() - commMStart;
                    MatrixCoord offsetC{blockOffset.m(), blockOffset.n()};
                    auto tensorBlockC = GetTile(
                        tensorC, tla::MakeCoord(offsetC.row(), offsetC.column()),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                    uint32_t beforeRankNum = params.rankIdx;
                    uint32_t afterRankStart = params.rankIdx + 1;
                    uint32_t afterRankNum = params.rankSize - afterRankStart;
                    uint32_t beforeK = beforeRankNum * localK;
                    uint32_t afterK = afterRankNum * localK;

                    if (beforeK > 0) {
                        AscendC::GlobalTensor<ElementA> gmSymmetricBeforeStage;
                        gmSymmetricBeforeStage.SetGlobalBuffer(
                            reinterpret_cast<__gm__ ElementA*>(params.ptrSymmetric) + stageId * aStageStride);
                        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleSymmetricBeforeStage;
                        gmMxScaleSymmetricBeforeStage.SetGlobalBuffer(
                            ptrScaleSymmetricBase + stageId * scaleStageStride);

                        MatrixCoord offsetB{0, blockOffset.n()};
                        auto tensorBlockB = GetTile(
                            tensorB, tla::MakeCoord(offsetB.row(), offsetB.column()),
                            tla::MakeShape(beforeK, actualBlockShape.n()));
                        auto tensorBlockMxScaleB = GetTile(
                            tensorMxScaleB,
                            tla::MakeCoord(offsetB.row() / Catlass::MX_SCALE_GROUP_NUM, offsetB.column()),
                            tla::MakeShape(CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(beforeK), actualBlockShape.n()));
                        GemmCoord groupShape{actualBlockShape.m(), actualBlockShape.n(), beforeK};
                        bool storeResult = afterK == 0;
                        blockMmad(
                            gmSymmetricBeforeStage, gmMxScaleSymmetricBeforeStage, tensorBlockB, tensorBlockC,
                            groupShape, blockMOffsetInStage, commSizeM, localK, alignedLocalK, localScaleK,
                            alignedLocalScaleK, tensorBlockMxScaleB, Catlass::EmptyClass{}, true, storeResult,
                            storeResult);
                    }

                    if (afterK > 0) {
                        AscendC::GlobalTensor<ElementA> gmSymmetricAfterStage;
                        gmSymmetricAfterStage.SetGlobalBuffer(
                            reinterpret_cast<__gm__ ElementA*>(params.ptrSymmetric) + stageId * aStageStride +
                            static_cast<int64_t>(afterRankStart) * commSizeM * alignedLocalK);
                        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleSymmetricAfterStage;
                        gmMxScaleSymmetricAfterStage.SetGlobalBuffer(
                            ptrScaleSymmetricBase + stageId * scaleStageStride +
                            static_cast<int64_t>(afterRankStart) * commSizeM * alignedLocalScaleK);

                        MatrixCoord offsetB{afterRankStart * localK, blockOffset.n()};
                        auto tensorBlockB = GetTile(
                            tensorB, tla::MakeCoord(offsetB.row(), offsetB.column()),
                            tla::MakeShape(afterK, actualBlockShape.n()));
                        auto tensorBlockMxScaleB = GetTile(
                            tensorMxScaleB,
                            tla::MakeCoord(offsetB.row() / Catlass::MX_SCALE_GROUP_NUM, offsetB.column()),
                            tla::MakeShape(CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(afterK), actualBlockShape.n()));
                        GemmCoord groupShape{actualBlockShape.m(), actualBlockShape.n(), afterK};
                        bool initAccumulator = beforeK == 0;
                        blockMmad(
                            gmSymmetricAfterStage, gmMxScaleSymmetricAfterStage, tensorBlockB, tensorBlockC, groupShape,
                            blockMOffsetInStage, commSizeM, localK, alignedLocalK, localScaleK, alignedLocalScaleK,
                            tensorBlockMxScaleB, Catlass::EmptyClass{}, initAccumulator, true, true);
                    }
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SetAtomicNone();
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            if (commIdx + WORKSPACE_STAGES < commLoops) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params& params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        uint32_t chunkM = params.problemShape.m() / params.rankSize;
        uint32_t localK = params.problemShape.k();
        uint32_t localScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(localK);
        uint32_t alignedLocalScaleK = RoundUp<2>(localScaleK);
        uint32_t commSizeM = params.commInterval * L1_TILE_M;
        uint32_t commLoops = CeilDiv(chunkM, commSizeM);

        constexpr uint32_t ELE_NUM_PER_FRACTAL_A =
            Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<ElementA>::value;
        int64_t alignedLocalK = RoundUp<int64_t>(localK, ELE_NUM_PER_FRACTAL_A);
        int64_t aStageStride = static_cast<int64_t>(params.rankSize) * commSizeM * alignedLocalK;
        uint64_t aWorkspaceBytes = static_cast<uint64_t>(WORKSPACE_STAGES) * aStageStride *
                                   Catlass::SizeOfBits<ElementA>::value / Catlass::SizeOfBits<uint8_t>::value;

        BlockUdmaAllToAll udmaAllToAll(resource);
        BlockUdmaAllToAllScale udmaAllToAllScale(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
        gmMxScaleA.SetGlobalBuffer(params.ptrMxScaleA);
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleSymmetric;
        gmMxScaleSymmetric.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementMxScaleA*>(params.ptrSymmetric + aWorkspaceBytes));

        auto layoutSymmetric =
            Catlass::layout::RowMajor(WORKSPACE_STAGES * params.rankSize * commSizeM, localK, alignedLocalK);
        auto layoutScaleSymmetric =
            Catlass::layout::RowMajor(WORKSPACE_STAGES * params.rankSize * commSizeM, localScaleK, alignedLocalScaleK);
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
                aclshmemx_barrier_all_vec();
            }

            uint32_t actualCommSizeM = Min(commSizeM, chunkM - commIdx * commSizeM);
            MatrixCoord actualACommShape{actualCommSizeM, localK};
            MatrixCoord actualScaleCommShape{actualCommSizeM, localScaleK};
            MatrixCoord commDstOffset{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, params.rankIdx, 0)), 0};

            if (subcoreIdx == 0 && aicoreIdx < params.rankSize && aicoreIdx != params.rankIdx) {
                uint32_t remoteRankIdx = aicoreIdx;
                MatrixCoord commSrcOffset{remoteRankIdx * chunkM + commIdx * commSizeM, 0};

                auto gmBlockSrc = gmA[params.layoutAllToAllSrc.GetOffset(commSrcOffset)];
                auto layoutBlockSrc = params.layoutAllToAllSrc.GetTileLayout(actualACommShape);
                auto gmBlockDst = gmSymmetric[layoutSymmetric.GetOffset(commDstOffset)];
                auto layoutBlockDst = layoutSymmetric.GetTileLayout(actualACommShape);

                udmaAllToAll(gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst, actualACommShape, remoteRankIdx);

                MatrixCoord commScaleSrcOffset{commSrcOffset.row(), 0};
                MatrixCoord commScaleDstOffset{commDstOffset.row(), 0};
                auto gmScaleBlockSrc = gmMxScaleA[params.layoutAllToAllScaleSrc.GetOffset(commScaleSrcOffset)];
                auto layoutScaleBlockSrc = params.layoutAllToAllScaleSrc.GetTileLayout(actualScaleCommShape);
                auto gmScaleBlockDst = gmMxScaleSymmetric[layoutScaleSymmetric.GetOffset(commScaleDstOffset)];
                auto layoutScaleBlockDst = layoutScaleSymmetric.GetTileLayout(actualScaleCommShape);

                udmaAllToAllScale(
                    gmScaleBlockSrc, layoutScaleBlockSrc, gmScaleBlockDst, layoutScaleBlockDst, actualScaleCommShape,
                    remoteRankIdx);
                // AIC consumes A and its scales together after the rank barrier.
                // Both requests use the same peer queue, so one quiet completes them.
                aclshmemx_udma_quiet(remoteRankIdx);
            }

            aclshmemx_barrier_all_vec();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALL_MATMUL_SPLIT_K_URMA_LOCAL_ATOMIC_H
