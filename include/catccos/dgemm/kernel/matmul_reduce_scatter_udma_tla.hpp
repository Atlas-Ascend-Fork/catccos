
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_UDMA_TLA_HPP
#define CATCCOS_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_UDMA_TLA_HPP

#include "catccos/catccos.hpp"
#include "catccos/layout/dist_matrix.hpp"

#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif

namespace Catccos::DGemm::Kernel {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

template <
    class BlockMmad_, class BlockLocalCopy_, class BlockUdmaReduceScatter_, class BlockMmadScheduler_,
    class BlockCommScheduler_, uint32_t WORKSPACE_STAGES_>
class MatmulReduceScatterUdmaTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementD = typename BlockMmad::ElementC;
    using LayoutTagA = typename BlockMmad::TileCopy::LayoutTagA;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using LayoutTagD = typename BlockMmad::TileCopy::LayoutTagC;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutD = typename BlockMmad::LayoutC;

    using BlockComm = BlockLocalCopy_;
    using BlockCommParams = typename BlockComm::Params;

    using BlockUdmaReduceScatter = BlockUdmaReduceScatter_;

    using ElementComm = typename BlockComm::ElementDst;
    using LayoutTagComm = typename BlockComm::LayoutDst;

    using BlockMmadScheduler = BlockMmadScheduler_;
    using BlockCommScheduler = BlockCommScheduler_;
    using BlockCommSchedulerParams = typename BlockCommScheduler::Params;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        DistGemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;

        __gm__ ElementA* ptrA;
        LayoutA layoutA;
        __gm__ ElementB* ptrB;
        LayoutB layoutB;
        __gm__ ElementD* ptrD;
        LayoutD layoutD;
        LayoutTagComm layoutTagComm;
        GM_ADDR ptrSymmetric;

        BlockCommParams blockCommParams;
        BlockCommSchedulerParams blockCommSchedulerParams;

        GM_ADDR ptrScratch;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(
            DistGemmCoord const& problemShape_, uint32_t rankIdx_, uint32_t rankSize_, uint32_t commInterval_,
            LayoutTagComm layoutTagComm_, GM_ADDR ptrA_, LayoutA const& layoutA_, GM_ADDR ptrB_,
            LayoutB const& layoutB_, GM_ADDR ptrD_, LayoutD const& layoutD_, GM_ADDR ptrSymmetric_,
            BlockCommParams const& blockCommParams_, BlockCommSchedulerParams const& blockCommSchedulerParams_,
            GM_ADDR ptrScratch_)
            : problemShape(problemShape_),
              rankIdx(rankIdx_),
              rankSize(rankSize_),
              commInterval(commInterval_),
              layoutTagComm(layoutTagComm_),
              ptrA(reinterpret_cast<__gm__ ElementA*>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB*>(ptrB_)),
              layoutB(layoutB_),
              ptrD(reinterpret_cast<__gm__ ElementD*>(ptrD_)),
              layoutD(layoutD_),
              ptrSymmetric(ptrSymmetric_),
              blockCommParams(blockCommParams_),
              blockCommSchedulerParams(blockCommSchedulerParams_),
              ptrScratch(ptrScratch_)
        {}
    };

    /// User API arguments
    struct Arguments {
        GemmCoord problemShape; // m (total, not per-rank), n, k
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrD;
        GM_ADDR ptrSymmetric;
        MatrixCoord commCoreSplit;
        MatrixCoord commBlockShape;
        MatrixCoord commTileShape;
        GM_ADDR ptrScratch;
    };

    static size_t GetWorkspaceSize(Arguments const& args) { return 0; }

    static Params ToUnderlyingArguments(Arguments const& args, uint8_t* workspace = nullptr)
    {
        LayoutTagA layoutTagA{args.problemShape.m(), args.problemShape.k()};
        LayoutTagB layoutTagB{args.problemShape.k(), args.problemShape.n()};
        LayoutTagD layoutTagD{args.problemShape.m() / args.rankSize, args.problemShape.n()};

        auto layoutA = tla::MakeLayoutFromTag(layoutTagA);
        auto layoutB = tla::MakeLayoutFromTag(layoutTagB);
        auto layoutD = tla::MakeLayoutFromTag(layoutTagD);

        DistGemmCoord distProblemShape{
            args.problemShape.m() / args.rankSize, args.problemShape.n(), args.problemShape.k(), args.rankSize};

        typename BlockComm::TileRemoteCopy::Params tileParams{args.commTileShape};
        BlockCommParams blockCommParams{args.commBlockShape, tileParams};
        BlockCommSchedulerParams blockCommSchedulerParams{args.commCoreSplit};

        return Params{distProblemShape, args.rankIdx,
                      args.rankSize,    args.commInterval,
                      layoutTagD,       args.ptrA,
                      layoutA,          args.ptrB,
                      layoutB,          args.ptrD,
                      layoutD,          args.ptrSymmetric,
                      blockCommParams,  blockCommSchedulerParams,
                      args.ptrScratch};
    }

    CATLASS_DEVICE
    MatmulReduceScatterUdmaTla()
    {
#ifdef ENABLE_TIMER
        __gm__ uint8_t* timer_buffer = GetTimerBuffer();
        if (timer_buffer != nullptr) {
            timer.Init(timer_buffer);
            timer.Tik();
        }
#endif
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishCompute[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }

    CATLASS_DEVICE
    ~MatmulReduceScatterUdmaTla()
    {
#ifdef ENABLE_TIMER
        timer.Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
#endif
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t blockPerComm = aicoreNum * params.commInterval;
        uint32_t blockPerCommInRank = blockPerComm / params.rankSize;

        GemmCoord mmadBlockShape = GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        GemmCoord problemShapeInRank = params.problemShape.GetCoordMNK();
        BlockMmadScheduler mmadScheduler(problemShapeInRank, mmadBlockShape.GetCoordMN());
        uint32_t coreLoops = mmadScheduler.GetCoreLoops() * params.rankSize;
        uint32_t commLoops = CeilDiv(coreLoops, blockPerComm);

        uint32_t commShapeM = blockPerCommInRank * mmadBlockShape.m();
        auto layoutTagSymmetric =
            Catlass::layout::RowMajor::MakeLayout<ElementD>(commShapeM * params.rankSize, mmadBlockShape.n());
        AscendC::GlobalTensor<ElementD> gmSymmetricList[WORKSPACE_STAGES];
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementD*>(params.ptrSymmetric);
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmSymmetricList[stageIdx].SetGlobalBuffer(ptrSymmetric + stageIdx * layoutTagSymmetric.Capacity());
        }
        auto layoutSymmetric = tla::MakeLayoutFromTag(layoutTagSymmetric);

        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Catlass::Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Catlass::Arch::PositionGM{});
        auto tensorD = tla::MakeTensor(gmD, params.layoutD, Catlass::Arch::PositionGM{});

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const& gmSymmetric = gmSymmetricList[stageIdx];

            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageIdx]);
            }
#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIC);
#endif

            uint32_t actualBlockPerComm =
                (commIdx == commLoops - 1) ? (coreLoops - blockPerComm * commIdx) : blockPerComm;
            uint32_t actualBlockPerCommInRank = actualBlockPerComm / params.rankSize;

            auto tensorSymmetric = tla::MakeTensor(gmSymmetric, layoutSymmetric, Catlass::Arch::PositionGM{});

            uint32_t commBlockOffsetInRank = commIdx * blockPerCommInRank;
            for (uint32_t blockIdxInComm = aicoreIdx; blockIdxInComm < actualBlockPerComm;
                 blockIdxInComm += aicoreNum) {
                uint32_t targetRankIdx = blockIdxInComm / actualBlockPerCommInRank;
                uint32_t blockIdxInRank = blockIdxInComm - targetRankIdx * actualBlockPerCommInRank;
                uint32_t loopIdxInRank = commBlockOffsetInRank + blockIdxInRank;

                MatrixCoord commOffsetSymm{targetRankIdx * commShapeM, 0};

                GemmCoord mmadBlockCoord = mmadScheduler.GetBlockCoord(loopIdxInRank);
                GemmCoord actualBlockShape = mmadScheduler.GetActualBlockShape(mmadBlockCoord);

                GemmCoord offsetCoord = mmadBlockCoord * mmadBlockShape;
                auto rankOffsetA = problemShapeInRank.GetCoordMK() * Catlass::MakeCoord<uint32_t>(targetRankIdx, 0);
                auto blockOffsetA = offsetCoord.GetCoordMK() + rankOffsetA;
                auto blockOffsetB = offsetCoord.GetCoordKN();
                auto blockOffsetD = offsetCoord.GetCoordMN();

                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(blockOffsetA[0], blockOffsetA[1]),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB = GetTile(
                    tensorB, tla::MakeCoord(blockOffsetB[0], blockOffsetB[1]),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockC = GetTile(
                    tensorD, tla::MakeCoord(blockOffsetD[0], blockOffsetD[1]),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                if (targetRankIdx != params.rankIdx) {
                    MatrixCoord blockOffsetSymm{blockIdxInRank * L1_TILE_M, 0};
                    tensorBlockC = GetTile(
                        tensorSymmetric,
                        tla::MakeCoord(blockOffsetSymm.row() + commOffsetSymm.row(), blockOffsetSymm.column()),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                }
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
            }
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad.SynchronizeBlock();
            }
#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIC);
#endif
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageIdx]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
        uint32_t blockPerComm = aicoreNum * params.commInterval;
        uint32_t blockPerCommInRank = blockPerComm / params.rankSize;

        BlockComm blockLocalCopy(resource, params.blockCommParams);
        BlockUdmaReduceScatter udmaReduceScatter(resource);

        AscendC::GlobalTensor<ElementD> gmSymmetricList[WORKSPACE_STAGES];
        auto layoutSymmetric = layout::DistRowMajor(blockPerCommInRank * L1_TILE_M, L1_TILE_N, params.rankSize);
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementD*>(params.ptrSymmetric);
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmSymmetricList[stageIdx].SetGlobalBuffer(ptrSymmetric + stageIdx * layout::Capacity(layoutSymmetric));
        }

        AscendC::GlobalTensor<ElementD> gmScratchList[WORKSPACE_STAGES];
        auto layoutScratch = layout::DistRowMajor(blockPerCommInRank * L1_TILE_M, L1_TILE_N, params.rankSize);
        auto ptrScratch = reinterpret_cast<__gm__ ElementD*>(params.ptrScratch);
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmScratchList[stageIdx].SetGlobalBuffer(ptrScratch + stageIdx * layout::Capacity(layoutScratch));
        }

        auto const& layoutD = params.layoutTagComm;
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        MatrixCoord commBlockShape = params.blockCommParams.BlockShape();
        MatrixCoord commCoreSplit = params.blockCommSchedulerParams.CoreSplit();
        GemmCoord mmadBlockShape = GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        BlockCommScheduler commScheduler(
            commBlockShape, commCoreSplit, params.rankSize, params.rankIdx, blockPerCommInRank,
            {params.problemShape, mmadBlockShape.GetCoordMN()});

        uint32_t commLoops = commScheduler.GetCommLoops();

        uint32_t commShapeM = blockPerCommInRank * mmadBlockShape.m();

        GemmCoord problemShapeInRank = params.problemShape.GetCoordMNK();
        BlockMmadScheduler mmadScheduler(problemShapeInRank, mmadBlockShape.GetCoordMN());
        uint32_t coreLoops = mmadScheduler.GetCoreLoops() * params.rankSize;

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const& gmSymmetric = gmSymmetricList[stageIdx];
            auto remapperDst = commScheduler.GetRemapperDst(commIdx);

            auto const& gmScratch = gmScratchList[stageIdx];

            uint32_t actualBlockPerComm =
                (commIdx == commLoops - 1) ? (coreLoops - blockPerComm * commIdx) : blockPerComm;
            uint32_t actualBlockPerCommInRank = actualBlockPerComm / params.rankSize;
            uint32_t actualCommSizeM = actualBlockPerCommInRank * L1_TILE_M;
            auto actualCommShape = DistMatrixCoord(actualCommSizeM, mmadBlockShape.n(), params.rankSize);

            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);

            aclshmemx_barrier_all_vec();

#ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIV);
#endif
            if (subcoreIdx == 0 && aicoreIdx < params.rankSize) {
                uint32_t udmaCoreLoops = params.rankSize;
                uint32_t udmaAicoreNum = params.rankSize;

                auto udmaActualShape = MatrixCoord{actualCommShape.row(), actualCommShape.column()};

                for (uint32_t loopIdx = aicoreIdx; loopIdx < udmaCoreLoops; loopIdx += udmaAicoreNum) {
                    uint32_t remoteRankIdx = loopIdx;

                    if (remoteRankIdx == params.rankIdx) {
                        continue;
                    }

                    DistMatrixCoord blockOffsetSrc = {params.rankIdx * commShapeM, 0, 0};
                    DistMatrixCoord blockOffsetDst = {remoteRankIdx * commShapeM, 0, 0};

                    auto gmBlockSrc = gmSymmetric[layoutSymmetric.GetOffset(blockOffsetSrc)];
                    auto layoutBlockSrc = layoutSymmetric.GetTileLayout(udmaActualShape);

                    auto gmBlockDst = gmScratch[layoutScratch.GetOffset(blockOffsetDst)];
                    auto layoutBlockDst = layoutScratch.GetTileLayout(udmaActualShape);

                    udmaReduceScatter(
                        gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst, udmaActualShape, remoteRankIdx);
                    aclshmemx_udma_quiet(remoteRankIdx);
                }
            }
            AscendC::SyncAll<true>();

            AscendC::SetAtomicAdd<ElementD>();
            AscendC::PipeBarrier<PIPE_ALL>();
            blockLocalCopy.InitBlockLoop();
            for (auto iter = commScheduler.Begin(commIdx); !iter.End(); iter.Next()) {
                auto blockOffset = commScheduler.GetBlockOffset(iter.taskIdx);

                uint32_t remoteRankIdx = blockOffset.remote();
                if (remoteRankIdx == params.rankIdx) {
                    continue;
                }

                remapperDst.UpdateMmadContext(blockOffset);

                auto blockOffsetSrc = DistMatrixCoord{blockOffset.GetMatrixCoord(), remoteRankIdx};
                auto blockOffsetDst = remapperDst(blockOffset);

                auto actualCommBlockShape = commScheduler.RemapActualBlockShape(blockOffset, remapperDst);

                auto gmBlockSrc = gmScratch[layoutScratch.GetOffset(blockOffsetSrc)];
                auto layoutBlockSrc = layoutScratch.GetTileLayout(actualCommBlockShape);

                auto gmBlockDst = gmD[layoutD.GetOffset(blockOffsetDst)];
                auto layoutBlockDst = layoutD.GetTileLayout(actualCommBlockShape);

                blockLocalCopy(gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst, actualCommBlockShape);
            }
            blockLocalCopy.FinalizeBlockLoop();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetAtomicNone();
            AscendC::PipeBarrier<PIPE_ALL>();

            aclshmemx_barrier_all_vec();

#ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIV);
#endif
            if (commIdx < commLoops - WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageIdx]);
            }
        }
    }

private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
#ifdef ENABLE_TIMER
    AscendTimerDevice timer;
#endif
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_UDMA_TLA_HPP
