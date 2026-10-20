
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_ALLGATHER_MATMUL_WITH_UDMA_HPP
#define CATCCOS_DGEMM_KERNEL_ASCEND950_ALLGATHER_MATMUL_WITH_UDMA_HPP

#include "catccos/catccos.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"

// UDMA headers
#include "shmem.h"

namespace Catccos::DGemm::Kernel {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;
using namespace AscendC;

template <
    class BlockMmad_, class BlockAllGather_, class BlockUdmaAllGather_, class BlockScheduler_,
    class BlockAllGatherScheduler_, uint32_t WORKSPACE_STAGES_>
class Ascend950AllGatherMatmulWithUdma {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using LayoutTagA = typename BlockMmad::TileCopy::LayoutTagA;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using LayoutTagC = typename BlockMmad::TileCopy::LayoutTagC;

    using BlockAllGather = BlockAllGather_;
    using AllGatherParams = typename BlockAllGather::Params;

    using BlockUdmaAllGather = BlockUdmaAllGather_;

    using BlockScheduler = BlockScheduler_;
    using BlockAllGatherScheduler = BlockAllGatherScheduler_;
    using BlockCommParams = typename BlockAllGatherScheduler::Params;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;

        __gm__ ElementA* ptrA;
        LayoutA layoutA;
        __gm__ ElementB* ptrB;
        LayoutB layoutB;
        __gm__ ElementC* ptrC;
        LayoutC layoutC;
        LayoutTagA layoutTagA;
        GM_ADDR ptrSymmetric;

        AllGatherParams allGatherParams;
        BlockCommParams commParams;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, uint32_t rank_, uint32_t rankSize_, uint32_t commInterval_,
            LayoutTagA layoutTagA_, GM_ADDR ptrA_, LayoutA const& layoutA_, GM_ADDR ptrB_, LayoutB const& layoutB_,
            GM_ADDR ptrC_, LayoutC const& layoutC_, GM_ADDR ptrSymmetric_, AllGatherParams const& allGatherParams_,
            BlockCommParams const& commParams_)
            : problemShape(problemShape_),
              rankIdx(rank_),
              rankSize(rankSize_),
              commInterval(commInterval_),
              layoutTagA(layoutTagA_),
              ptrA(reinterpret_cast<__gm__ ElementA*>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB*>(ptrB_)),
              layoutB(layoutB_),
              ptrC(reinterpret_cast<__gm__ ElementC*>(ptrC_)),
              layoutC(layoutC_),
              ptrSymmetric(ptrSymmetric_),
              allGatherParams(allGatherParams_),
              commParams(commParams_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrSymmetric;
        MatrixCoord commCoreSplit;
        MatrixCoord commBlockShape;
        MatrixCoord commTileShape;
    };

    static Params ToUnderlyingArguments(Arguments const& args, uint8_t* workspace = nullptr)
    {
        LayoutTagA tagA{args.problemShape.m(), args.problemShape.k()};
        LayoutTagB tagB{args.problemShape.k(), args.problemShape.n()};
        LayoutTagC tagC{args.problemShape.m() * args.rankSize, args.problemShape.n()};

        auto layoutA = tla::MakeLayoutFromTag(tagA);
        auto layoutB = tla::MakeLayoutFromTag(tagB);
        auto layoutC = tla::MakeLayoutFromTag(tagC);

        typename BlockAllGather::TileRemoteCopy::Params tileParams{args.commTileShape};
        AllGatherParams allGatherParams{args.commBlockShape, tileParams};
        BlockCommParams commParams{args.commCoreSplit};

        return Params(
            args.problemShape, args.rankIdx, args.rankSize, args.commInterval, tagA, args.ptrA, layoutA, args.ptrB,
            layoutB, args.ptrC, layoutC, args.ptrSymmetric, allGatherParams, commParams);
    }

    CATLASS_DEVICE
    Ascend950AllGatherMatmulWithUdma()
    {
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishCompute[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
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
        uint32_t commSizeM = params.commInterval * L1_TILE_M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        BlockMmad mmad(resource);

        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);

        auto layoutTagSymmetric =
            Catlass::layout::RowMajor(WORKSPACE_STAGES * params.rankSize * commSizeM, params.problemShape.k());
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        auto layoutSymmetric = tla::MakeLayoutFromTag(layoutTagSymmetric);

        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Catlass::Arch::PositionGM{});
        auto tensorPeerMem = tla::MakeTensor(gmSymmetric, layoutSymmetric, Catlass::Arch::PositionGM{});

        auto layoutCRowLogicStride = Catlass::MakeCoord<int64_t>(params.problemShape.m(), commSizeM, 1);
        auto layoutCRow = layout::AffineRankN<3>(layoutCRowLogicStride);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualProblemShape = Catlass::MakeCoord<uint32_t>(
                actualCommSizeM, params.problemShape.n(), params.problemShape.k(), params.rankSize);
            BlockScheduler mmadScheduler(actualProblemShape, blockShape.GetCoordMN());
            uint32_t coreLoops = mmadScheduler.GetCoreLoops();

            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]);

            for (uint32_t loopIdx = aicoreIdx; loopIdx < coreLoops; loopIdx += aicoreNum) {
                auto blockOffset = mmadScheduler.GetBlockOffset(loopIdx);
                auto actualBlockShape = mmadScheduler.GetActualBlockShapeByOffset(blockOffset);

                uint32_t srcRankIdx = blockOffset.rank();
                MatrixCoord commOffsetA{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, srcRankIdx, 0)), 0};
                MatrixCoord commOffsetC{layoutCRow(Catlass::MakeCoord<int>(srcRankIdx, commIdx, 0)), 0};

                MatrixCoord offsetA = commOffsetA + blockOffset.GetCoordMK();
                MatrixCoord offsetB = blockOffset.GetCoordKN();
                MatrixCoord offsetC = commOffsetC + blockOffset.GetCoordMN();

                auto tensorBlockA = GetTile(
                    tensorPeerMem, tla::MakeCoord(offsetA.row(), offsetA.column()),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB = GetTile(
                    tensorB, tla::MakeCoord(offsetB.row(), offsetB.column()),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockC = GetTile(
                    tensorC, tla::MakeCoord(offsetC.row(), offsetC.column()),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                mmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape.GetCoordMNK());
            }

            if (commIdx < commLoops - WORKSPACE_STAGES && commLoops >= WORKSPACE_STAGES) {
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

        uint32_t commSizeM = params.commInterval * L1_TILE_M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        BlockAllGather allGather(resource, params.allGatherParams);
        BlockUdmaAllGather udmaAllGather(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrSymmetric));

        auto layoutSymmetric =
            Catlass::layout::RowMajor(WORKSPACE_STAGES * params.rankSize * commSizeM, params.problemShape.k());
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        MatrixCoord commBlockShape = params.allGatherParams.BlockShape();
        MatrixCoord commCoreSplit = params.commParams.CoreSplit();
        BlockAllGatherScheduler commScheduler(commBlockShape, commCoreSplit);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualCommShape = DistMatrixCoord(actualCommSizeM, params.problemShape.k(), params.rankSize);
            MatrixCoord loopsInRank = CeilDiv(MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);
            commScheduler.UpdateProblem(actualCommShape, loopsInRank);
            auto commAicoreNum = commScheduler.GetRealCore();
            auto commCoreLoops = commScheduler.GetCoreLoop();

            MatrixCoord commSrcOffset{commIdx * commSizeM, 0};
            MatrixCoord commDstOffset{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, params.rankIdx, 0)), 0};

            // Wait for AIC to finish reading previous stage
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
            }

            aclshmemx_barrier_all_vec();

            // --- Subcore 0: Local copy (A matrix GM → local symmetric workspace) ---
            allGather.InitBlockLoop();
            if (subcoreIdx == 0 && aicoreIdx < commAicoreNum) {
                for (uint32_t commLoopIdx = aicoreIdx; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                    DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(commLoopIdx);
                    MatrixCoord blockOffsetInRank = commScheduler.GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                    MatrixCoord actualCommBlockShape = commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);

                    uint32_t remoteRankIdx = commBlockCoord.rank() % params.rankSize;

                    if (remoteRankIdx != params.rankIdx)
                        continue;

                    auto offsetSrc = commSrcOffset + blockOffsetInRank;
                    auto offsetDst = commDstOffset + blockOffsetInRank;

                    auto gmBlockSrc = gmA[params.layoutTagA.GetOffset(offsetSrc)];
                    auto layoutBlockSrc = params.layoutTagA.GetTileLayout(actualCommBlockShape);

                    auto gmBlockDst = gmSymmetric[layoutSymmetric.GetOffset(offsetDst)];
                    auto layoutBlockDst = layoutSymmetric.GetTileLayout(actualCommBlockShape);

                    allGather(
                        gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst, actualCommBlockShape, remoteRankIdx);
                }
            }
            allGather.FinalizeBlockLoop();

            // --- Subcore 1: UDMA put (A matrix GM → remote symmetric) ---
            // Each AIV core handles data to one remote PE (distributed by aicoreIdx)
            if (subcoreIdx == 1 && aicoreIdx < params.rankSize) {
                uint32_t udmaCoreLoops = params.rankSize;
                uint32_t udmaAicoreNum = params.rankSize;

                auto udmaActualShape = MatrixCoord{actualCommShape.row(), actualCommShape.column()};

                for (uint32_t loopIdx = aicoreIdx; loopIdx < udmaCoreLoops; loopIdx += udmaAicoreNum) {
                    uint32_t remoteRankIdx = loopIdx;

                    if (remoteRankIdx == params.rankIdx)
                        continue;

                    MatrixCoord offsetSrc{commIdx * commSizeM, 0};
                    MatrixCoord offsetDst{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, params.rankIdx, 0)), 0};

                    auto gmBlockSrc = gmA[params.layoutTagA.GetOffset(offsetSrc)];
                    auto layoutBlockSrc = params.layoutTagA.GetTileLayout(udmaActualShape);

                    auto gmBlockDst = gmSymmetric[layoutSymmetric.GetOffset(offsetDst)];
                    auto layoutBlockDst = layoutSymmetric.GetTileLayout(udmaActualShape);

                    udmaAllGather(
                        gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst, udmaActualShape, remoteRankIdx);
                    // Complete this stage before the rank barrier publishes it to AIC.
                    aclshmemx_udma_quiet(remoteRankIdx);
                }
            }

            aclshmemx_barrier_all_vec();

            // Notify AIC that AllGather + UDMA writes for this stage are done
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

#endif // CATCCOS_DGEMM_KERNEL_ASCEND950_ALLGATHER_MATMUL_WITH_UDMA_HPP
