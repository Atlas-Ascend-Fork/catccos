
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_MATMUL_ALLTOALL_MX_SLICE_M_HPP
#define CATCCOS_DGEMM_KERNEL_ASCEND950_MATMUL_ALLTOALL_MX_SLICE_M_HPP

#include "catccos/catccos.hpp"
#include "catccos/layout/dist_matrix.hpp"
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

/// MatMul + AllToAll kernel for Ascend950 with MX FP8 inputs.
///
/// Compute: C = A (M x K) x B (K x N) where A/B are MX FP8 with E8M0 scales,
/// then redistribute C via uniform AllToAll (pure redistribution, no reduction).
///
/// Each rank computes the full M x N result (in chunks destined for each rank),
/// then sends chunk[j] of C (rows [j*chunkM : (j+1)*chunkM]) to rank j. After
/// AllToAll, each rank holds an M x N output assembled from all ranks' different
/// row-chunks.
///
/// AIC (Cube): Computes MX FP8 MatMul. For blocks destined for remote ranks,
///   writes to symmetric memory. For local-rank blocks, writes directly to D at
///   [rankIdx*chunkM + offset.m, offset.n].
/// AIV (Vector): Subcore 0 handles the local/self case (already in D from AIC,
///   skipped). Subcore 1 uses UDMA PUT to push local symmetric chunks to remote
///   ranks' D. Data from local rank i lands in [i*chunkM, :] on remote rank j's D.
template <
    class BlockMmad_, class BlockComm_, class BlockMmadScheduler_, class BlockCommScheduler_,
    uint32_t WORKSPACE_STAGES_>
class MatmulAllToAllMxSliceM {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;
    using ElementD = typename BlockMmad::ElementC; // Same as ElementC for MX path
    using LayoutTagA = typename BlockMmad::TileCopy::LayoutTagA;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using LayoutTagC = typename BlockMmad::TileCopy::LayoutTagC;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementMxScaleA = typename BlockMmad::ElementMxScaleA;
    using LayoutMxScaleA = typename BlockMmad::TileCopy::LayoutMxScaleA;
    using ElementMxScaleB = typename BlockMmad::ElementMxScaleB;
    using LayoutMxScaleB = typename BlockMmad::TileCopy::LayoutMxScaleB;

    using BlockComm = BlockComm_;

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
        __gm__ ElementMxScaleA* ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        __gm__ ElementMxScaleB* ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        __gm__ ElementD* ptrD;
        LayoutC layoutD;
        LayoutTagComm layoutTagComm;
        GM_ADDR ptrSymmetric;

        BlockCommSchedulerParams blockCommSchedulerParams;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(
            DistGemmCoord const& problemShape_, uint32_t rankIdx_, uint32_t rankSize_, uint32_t commInterval_,
            LayoutTagComm layoutTagComm_, GM_ADDR ptrA_, LayoutA const& layoutA_, GM_ADDR ptrB_,
            LayoutB const& layoutB_, GM_ADDR ptrMxScaleA_, LayoutMxScaleA const& layoutMxScaleA_, GM_ADDR ptrMxScaleB_,
            LayoutMxScaleB const& layoutMxScaleB_, GM_ADDR ptrD_, LayoutC const& layoutD_, GM_ADDR ptrSymmetric_,
            BlockCommSchedulerParams const& blockCommSchedulerParams_)
            : problemShape(problemShape_),
              rankIdx(rankIdx_),
              rankSize(rankSize_),
              commInterval(commInterval_),
              layoutTagComm(layoutTagComm_),
              ptrA(reinterpret_cast<__gm__ ElementA*>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB*>(ptrB_)),
              layoutB(layoutB_),
              ptrMxScaleA(reinterpret_cast<__gm__ ElementMxScaleA*>(ptrMxScaleA_)),
              layoutMxScaleA(layoutMxScaleA_),
              ptrMxScaleB(reinterpret_cast<__gm__ ElementMxScaleB*>(ptrMxScaleB_)),
              layoutMxScaleB(layoutMxScaleB_),
              ptrD(reinterpret_cast<__gm__ ElementD*>(ptrD_)),
              layoutD(layoutD_),
              ptrSymmetric(ptrSymmetric_),
              blockCommSchedulerParams(blockCommSchedulerParams_)
        {}
    };

    /// User API arguments
    struct Arguments {
        GemmCoord problemShape; // m (total = chunkM * rankSize), n, k
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrMxScaleA;
        GM_ADDR ptrMxScaleB;
        GM_ADDR ptrD;
        GM_ADDR ptrSymmetric;
        MatrixCoord commCoreSplit;
        MatrixCoord commBlockShape;
        MatrixCoord commTileShape;
    };

    static size_t GetWorkspaceSize(Arguments const& args) { return 0; }

    static Params ToUnderlyingArguments(Arguments const& args, uint8_t* workspace = nullptr)
    {
        // AllToAll: each rank's output D is the full M x N (assembled from R chunks).
        LayoutTagC layoutTagC{args.problemShape.m(), args.problemShape.n()};
        LayoutTagA layoutTagA{args.problemShape.m(), args.problemShape.k()};
        LayoutTagB layoutTagB{args.problemShape.k(), args.problemShape.n()};

        auto layoutA = tla::MakeLayoutFromTag(layoutTagA);
        auto layoutB = tla::MakeLayoutFromTag(layoutTagB);
        auto layoutD = tla::MakeLayoutFromTag(layoutTagC);

        uint32_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(args.problemShape.k());
        auto layoutMxScaleA =
            tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagA, false>(args.problemShape.m(), mxScaleK);
        auto layoutMxScaleB =
            tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagB, true>(mxScaleK, args.problemShape.n());

        DistGemmCoord distProblemShape{
            args.problemShape.m() / args.rankSize, args.problemShape.n(), args.problemShape.k(), args.rankSize};

        typename BlockComm::TileRemoteCopy::Params tileParams{args.commTileShape};
        BlockCommSchedulerParams blockCommSchedulerParams{args.commCoreSplit};

        return Params{
            distProblemShape,
            args.rankIdx,
            args.rankSize,
            args.commInterval,
            layoutTagC,
            args.ptrA,
            layoutA,
            args.ptrB,
            layoutB,
            args.ptrMxScaleA,
            layoutMxScaleA,
            args.ptrMxScaleB,
            layoutMxScaleB,
            args.ptrD,
            layoutD,
            args.ptrSymmetric,
            blockCommSchedulerParams};
    }

    CATLASS_DEVICE
    MatmulAllToAllMxSliceM()
    {
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishCompute[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }

    CATLASS_DEVICE
    ~MatmulAllToAllMxSliceM() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    /// AIC: Compute MX FP8 MatMul. Local-rank results go to D (shifted by rankIdx*chunkM),
    /// remote-rank results go to symmetric memory.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t blockPerComm = aicoreNum * params.commInterval;
        uint32_t blockPerCommInRank = blockPerComm / params.rankSize;

        GemmCoord mmadBlockShape = GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        GemmCoord problemShapeInRank = params.problemShape.GetCoordMNK();

        uint32_t commShapeM = params.commInterval * L1_TILE_M;
        auto layoutTagSymmetric =
            Catlass::layout::RowMajor::MakeLayout<ElementC>(commShapeM * params.rankSize, problemShapeInRank.n());
        AscendC::GlobalTensor<ElementC> gmSymmetricList[WORKSPACE_STAGES];
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementC*>(params.ptrSymmetric);
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmSymmetricList[stageIdx].SetGlobalBuffer(
                ptrSymmetric + stageIdx * layoutTagSymmetric.Capacity() * sizeof(ElementC));
        }
        auto layoutSymmetric = tla::MakeLayoutFromTag(layoutTagSymmetric);

        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
        gmMxScaleA.SetGlobalBuffer(params.ptrMxScaleA);
        AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
        gmMxScaleB.SetGlobalBuffer(params.ptrMxScaleB);
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        // Matrix A or Matrix B does not have duplicate data reads. Setting L2 Cache to Disable,
        // data reads will bypass L2 Cache.
        if (CeilDiv(problemShapeInRank.m() * params.rankSize, L1_TILE_M) == 1) {
            gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        } else {
            gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        }
        if (CeilDiv(problemShapeInRank.n() * params.rankSize, L1_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        } else {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        }

        auto const& layoutD = params.layoutTagComm;

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Catlass::Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Catlass::Arch::PositionGM{});
        auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, params.layoutMxScaleA, Catlass::Arch::PositionGM{});
        auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, params.layoutMxScaleB, Catlass::Arch::PositionGM{});
        auto tensorD = tla::MakeTensor(gmD, params.layoutD, Catlass::Arch::PositionGM{});

        uint32_t commLoops = CeilDiv(problemShapeInRank.m(), commShapeM);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const& gmSymmetric = gmSymmetricList[stageIdx];

            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageIdx]);
            }

            uint32_t actualCommSizeM = Min(commShapeM, problemShapeInRank.m() - commIdx * commShapeM);
            auto actualProblemShape =
                Catlass::MakeCoord<uint32_t>(actualCommSizeM, problemShapeInRank.n(), problemShapeInRank.k());
            BlockMmadScheduler mmadScheduler(actualProblemShape, mmadBlockShape.GetCoordMN());

            uint32_t actualBlockPerComm = mmadScheduler.GetCoreLoops() * params.rankSize;
            uint32_t actualBlockPerCommInRank = actualBlockPerComm / params.rankSize;

            auto tensorSymmetric = tla::MakeTensor(gmSymmetric, layoutSymmetric, Catlass::Arch::PositionGM{});

            uint32_t commBlockOffsetInRank = commIdx * blockPerCommInRank;
            for (uint32_t blockIdxInComm = aicoreIdx; blockIdxInComm < actualBlockPerComm;
                 blockIdxInComm += aicoreNum) {
                uint32_t targetRankIdx = blockIdxInComm / actualBlockPerCommInRank;
                uint32_t blockIdxInRank = blockIdxInComm - targetRankIdx * actualBlockPerCommInRank;
                uint32_t loopIdxInRank = commBlockOffsetInRank + blockIdxInRank;

                MatrixCoord commOffsetSymm{targetRankIdx * commShapeM, 0};
                MatrixCoord commOffsetGM{commIdx * commShapeM, 0};

                GemmCoord mmadBlockCoord = mmadScheduler.GetBlockCoord(loopIdxInRank);
                GemmCoord actualBlockShape = mmadScheduler.GetActualBlockShape(mmadBlockCoord);

                GemmCoord offsetCoord = mmadBlockCoord * mmadBlockShape;
                // A is M_total x K; for targetRank, read A[targetRank*chunkM + offset.m, offset.k]
                auto rankOffsetA = problemShapeInRank.GetCoordMK() * Catlass::MakeCoord<uint32_t>(targetRankIdx, 0);
                auto blockOffsetA = offsetCoord.GetCoordMK() + rankOffsetA + commOffsetGM;
                auto blockOffsetB = offsetCoord.GetCoordKN();

                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(blockOffsetA[0], blockOffsetA[1]),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockMxScaleA = GetTile(
                    tensorMxScaleA, tla::MakeCoord(blockOffsetA[0], blockOffsetA[1] / Catlass::MX_SCALE_GROUP_NUM),
                    tla::MakeShape(actualBlockShape.m(), CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualBlockShape.k())));
                auto tensorBlockB = GetTile(
                    tensorB, tla::MakeCoord(blockOffsetB[0], blockOffsetB[1]),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockMxScaleB = GetTile(
                    tensorMxScaleB, tla::MakeCoord(blockOffsetB[0] / Catlass::MX_SCALE_GROUP_NUM, blockOffsetB[1]),
                    tla::MakeShape(CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));

                if (targetRankIdx == params.rankIdx) {
                    // AllToAll local rank: write directly to GM D
                    auto blockOffsetD = offsetCoord.GetCoordMN() +
                                        Catlass::MakeCoord<uint32_t>(params.rankIdx * problemShapeInRank.m(), 0) +
                                        commOffsetGM;
                    auto tensorBlockD = GetTile(
                        tensorD, tla::MakeCoord(blockOffsetD[0], blockOffsetD[1]),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    blockMmad(
                        tensorBlockA, tensorBlockB, tensorBlockD, actualBlockShape, tensorBlockMxScaleA,
                        tensorBlockMxScaleB);
                } else {
                    // Remote rank: write to symmetric memory
                    MatrixCoord blockOffsetSymm = offsetCoord.GetCoordMN();
                    auto tensorBlockC = GetTile(
                        tensorSymmetric,
                        tla::MakeCoord(blockOffsetSymm.row() + commOffsetSymm.row(), blockOffsetSymm.column()),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    blockMmad(
                        tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockMxScaleA,
                        tensorBlockMxScaleB);
                }
            }
            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad.SynchronizeBlock();
            }

            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageIdx]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// AIV: Subcore 0 handles the local/self case (already in D from AIC, skipped).
    /// Subcore 1 uses UDMA PUT to push local symmetric chunks to remote ranks' D.
    /// Data from local rank i lands in [i*chunkM, :] on remote rank j's D.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
        GemmCoord mmadBlockShape = GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        GemmCoord problemShapeInRank = params.problemShape.GetCoordMNK();

        BlockComm blockRemoteCopy(resource);
        uint32_t commShapeM = params.commInterval * L1_TILE_M;
        auto layoutTagSymmetric =
            Catlass::layout::RowMajor::MakeLayout<ElementC>(commShapeM * params.rankSize, problemShapeInRank.n());
        AscendC::GlobalTensor<ElementC> gmSymmetricList[WORKSPACE_STAGES];
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementC*>(params.ptrSymmetric);
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmSymmetricList[stageIdx].SetGlobalBuffer(
                ptrSymmetric + stageIdx * layoutTagSymmetric.Capacity() * sizeof(ElementC));
        }

        auto const& layoutD = params.layoutTagComm;
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        uint32_t chunkM = params.problemShape.GetCoordMNK().At(0);
        uint32_t commLoops = CeilDiv(chunkM, commShapeM);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const& gmSymmetric = gmSymmetricList[stageIdx];

            uint32_t actualCommM = Min<uint32_t>(commShapeM, chunkM - commIdx * commShapeM);

            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);

            aclshmemx_barrier_all_vec();

            if (subcoreIdx == 0 && aicoreIdx < params.rankSize) {
                uint32_t udmaCoreLoops = params.rankSize;
                uint32_t udmaAicoreNum = params.rankSize;

                auto actualCommBlockShape = MatrixCoord{actualCommM, problemShapeInRank.n()};

                for (uint32_t remoteRankIdx = aicoreIdx; remoteRankIdx < udmaCoreLoops;
                     remoteRankIdx += udmaAicoreNum) {
                    if (remoteRankIdx == params.rankIdx) {
                        continue;
                    }

                    MatrixCoord blockOffsetSrc = MatrixCoord{params.rankIdx * commShapeM, 0};
                    MatrixCoord blockOffsetDst = MatrixCoord{remoteRankIdx * chunkM + commIdx * commShapeM, 0};

                    auto gmBlockSrc = gmSymmetric[layoutTagSymmetric.GetOffset(blockOffsetSrc)];
                    auto layoutBlockSrc = layoutTagSymmetric.GetTileLayout(actualCommBlockShape);

                    // AllToAll: GET from remote rank's symmetric → local D.
                    auto gmBlockDst = gmD[layoutD.GetOffset(blockOffsetDst)];
                    auto layoutBlockDst = layoutD.GetTileLayout(actualCommBlockShape);

                    blockRemoteCopy(
                        gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst, actualCommBlockShape, remoteRankIdx);
                }
            }

            if (commIdx + WORKSPACE_STAGES < commLoops) {
                // Finish remote reads before any AIC can overwrite this source stage.
                if (subcoreIdx == 0 && aicoreIdx < params.rankSize && aicoreIdx != params.rankIdx) {
                    aclshmemx_udma_quiet(aicoreIdx);
                }
                aclshmemx_barrier_all_vec();
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageIdx]);
            }
        }

        // The remaining stages are not reused and write disjoint output rows.
        // Batch their completion here (all rounds when commLoops <= WORKSPACE_STAGES).
        if (commLoops > 0) {
            if (subcoreIdx == 0 && aicoreIdx < params.rankSize && aicoreIdx != params.rankIdx) {
                aclshmemx_udma_quiet(aicoreIdx);
            }
            // Keep every rank's source workspace alive until all remote reads finish.
            aclshmemx_barrier_all_vec();
        }
    }

private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_ASCEND950_MATMUL_ALLTOALL_MX_SLICE_M_HPP
