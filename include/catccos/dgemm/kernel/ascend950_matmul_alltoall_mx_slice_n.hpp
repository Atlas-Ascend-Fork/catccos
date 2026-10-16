
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_MATMUL_ALLTOALL_MX_SLICE_N_HPP
#define CATCCOS_DGEMM_KERNEL_ASCEND950_MATMUL_ALLTOALL_MX_SLICE_N_HPP

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

/// MatMul + AllToAll (along N) kernel for Ascend950 with MX FP8 inputs.
///
/// Compute: C = A (M x K) x B (K x N) where A/B are MX FP8 with E8M0 scales,
/// then redistribute C via uniform AllToAll along the N dimension.
///
/// Semantic (matches the production MatmulAlltoAll "split along N" formula):
///   - Each rank i holds its own A_i of shape (M, K) (row-sharded, different rows
///     per rank) and a shared B of shape (K, N) (replicated).
///   - MatMul: C_i = dequant(A_i) @ dequant(B), shape (M, N), full N.
///   - AllToAll along N: chunkN = N / rankSize. Rank i sends C_i[:, j*chunkN]
///     (the j-th N-slice, shape (M, chunkN)) to rank j, and receives from each
///     rank j the N-slice destined for itself (C_j[:, i*chunkN]).
///   - After AllToAll, each rank holds D of shape (rankSize*M, chunkN), where the
///     row-block [j*M : (j+1)*M] comes from rank j.
///
/// AIC (Cube): Computes MX FP8 MatMul. For the N-slice destined for the local
///   rank, writes directly to D at [rankIdx*M + offset.m, offset.n_local]. For
///   N-slices destined for remote ranks, writes to symmetric memory.
/// AIV (Vector): Subcore 0 uses UDMA GET to pull the local rank's N-slice from
///   each remote rank's symmetric memory into the remote rank's row-block of D.
///   Data from remote rank j lands in [j*M, :] on local D.
///
/// Staging: the M dimension is staged (commShapeM = commInterval * L1_TILE_M rows
/// per stage). Within one stage the AIC computes (commShapeM, N) once and routes
/// each (L1_TILE_M, L1_TILE_N) output tile to D (local N-slice) or symmetric
/// (remote N-slices). Constraint: chunkN must be a multiple of L1_TILE_N so that
/// tile boundaries align with N-slice boundaries (no tile straddles local/remote).
template <
    class BlockMmad_, class BlockComm_, class BlockMmadScheduler_, class BlockCommScheduler_,
    uint32_t WORKSPACE_STAGES_>
class MatmulAllToAllMxSliceN {
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
        DistGemmCoord problemShape; // (M, N, K, rankSize): M = per-rank rows, N = full
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
        GemmCoord problemShape; // (M, N, K): M = per-rank rows, N = full, K
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
        // A: (M, K) per rank; B: (K, N) shared; D: (rankSize*M, N/rankSize) per rank.
        LayoutTagC layoutTagC{args.problemShape.m() * args.rankSize, args.problemShape.n() / args.rankSize};
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

        // distProblemShape: (M, N, K, rankSize). M is NOT divided (each rank computes
        // its own full (M, N)); the N-slice (chunkN = N/rankSize) is the scatter unit.
        DistGemmCoord distProblemShape{
            args.problemShape.m(), args.problemShape.n(), args.problemShape.k(), args.rankSize};

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
    MatmulAllToAllMxSliceN()
    {
        for (uint32_t stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            flagAicFinishStore[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
            flagAivFinishCompute[stageIdx] = Catlass::Arch::CrossCoreFlag(stageIdx);
        }
    }

    CATLASS_DEVICE
    ~MatmulAllToAllMxSliceN() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    /// AIC: Compute MX FP8 MatMul staged along M. For each output tile, the
    /// destination rank is derived from the tile's N-position. Local-rank
    /// N-slices go to D; remote-rank N-slices go to symmetric memory.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        GemmCoord mmadBlockShape = GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        GemmCoord problemShapeInRank = params.problemShape.GetCoordMNK(); // (M, N, K)

        uint32_t chunkN = problemShapeInRank.n() / params.rankSize; // per-rank N-slice
        uint32_t commShapeM = params.commInterval * L1_TILE_M;
        uint32_t commLoops = CeilDiv(problemShapeInRank.m(), commShapeM);

        // Symmetric per stage: (commShapeM, N). The rankSize N-slices are implicit
        // column blocks of width chunkN; only remote slices are written by AIC.
        auto layoutTagSymmetric = Catlass::layout::RowMajor::MakeLayout<ElementC>(commShapeM * params.rankSize, chunkN);
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

        // L2 cache hints: A is reused across N-tiles, B is reused across M-tiles.
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

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const& gmSymmetric = gmSymmetricList[stageIdx];
            auto tensorSymmetric = tla::MakeTensor(gmSymmetric, layoutSymmetric, Catlass::Arch::PositionGM{});

            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageIdx]);
            }

            uint32_t actualCommM = Min(commShapeM, problemShapeInRank.m() - commIdx * commShapeM);
            auto actualProblemShape =
                Catlass::MakeCoord<uint32_t>(actualCommM, problemShapeInRank.n(), problemShapeInRank.k());

            // Per-stage scheduler over (actualCommM, N); each core handles a subset
            // of the tiles of this stage's grid.
            BlockMmadScheduler mmadScheduler(actualProblemShape, mmadBlockShape.GetCoordMN());
            uint32_t coreLoops = mmadScheduler.GetCoreLoops();

            MatrixCoord commOffsetGM{commIdx * commShapeM, 0};

            for (uint32_t blockIdx = aicoreIdx; blockIdx < coreLoops; blockIdx += aicoreNum) {
                GemmCoord mmadBlockCoord = mmadScheduler.GetBlockCoord(blockIdx);
                GemmCoord actualBlockShape = mmadScheduler.GetActualBlockShape(mmadBlockCoord);

                GemmCoord offsetCoord = mmadBlockCoord * mmadBlockShape;
                MatrixCoord offsetMN = offsetCoord.GetCoordMN();

                // Target rank from the tile's N-position (requires chunkN % L1_TILE_N == 0).
                uint32_t targetRankIdx = offsetMN.column() / chunkN;

                // A read: full M of this rank, staged along M (no target-rank row offset).
                auto blockOffsetA = offsetCoord.GetCoordMK() + commOffsetGM;
                // B read: full N (the tile's N-position selects the target rank).
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
                    // Local N-slice: write directly to D at [rankIdx*M + commOffset, n - rankIdx*chunkN].
                    auto blockOffsetD = tla::MakeCoord(
                        offsetMN.row() + params.rankIdx * problemShapeInRank.m() + commOffsetGM.row(),
                        offsetMN.column() - params.rankIdx * chunkN);
                    auto tensorBlockD =
                        GetTile(tensorD, blockOffsetD, tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                    blockMmad(
                        tensorBlockA, tensorBlockB, tensorBlockD, actualBlockShape, tensorBlockMxScaleA,
                        tensorBlockMxScaleB);
                } else {
                    // Remote N-slice: write to symmetric memory at [offset.m, offset.n].
                    auto blockOffsetSymm = tla::MakeCoord(
                        offsetMN.row() + targetRankIdx * commShapeM, offsetMN.column() - targetRankIdx * chunkN);
                    auto tensorBlockC = GetTile(
                        tensorSymmetric, blockOffsetSymm, tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
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

    /// AIV: Pull the local rank's N-slice from each remote rank's symmetric memory
    /// into D. Data from remote rank j lands in [j*M, :] on local D.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
        GemmCoord mmadBlockShape = GemmCoord{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        GemmCoord problemShapeInRank = params.problemShape.GetCoordMNK(); // (M, N, K)

        BlockComm blockRemoteCopy(resource);
        uint32_t chunkN = problemShapeInRank.n() / params.rankSize; // per-rank N-slice
        uint32_t commShapeM = params.commInterval * L1_TILE_M;
        uint32_t commLoops = CeilDiv(problemShapeInRank.m(), commShapeM);

        // Symmetric per stage: (commShapeM, N). Read the local rank's column block
        // [rankIdx*chunkN : (rankIdx+1)*chunkN] from each remote rank.
        auto layoutTagSymmetric = Catlass::layout::RowMajor::MakeLayout<ElementC>(commShapeM * params.rankSize, chunkN);
        AscendC::GlobalTensor<ElementC> gmSymmetricList[WORKSPACE_STAGES];
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementC*>(params.ptrSymmetric);
        for (int stageIdx = 0; stageIdx < WORKSPACE_STAGES; ++stageIdx) {
            gmSymmetricList[stageIdx].SetGlobalBuffer(
                ptrSymmetric + stageIdx * layoutTagSymmetric.Capacity() * sizeof(ElementC));
        }

        auto const& layoutD = params.layoutTagComm;
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
            auto const& gmSymmetric = gmSymmetricList[stageIdx];

            uint32_t actualCommM = Min<uint32_t>(commShapeM, problemShapeInRank.m() - commIdx * commShapeM);

            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);

            aclshmemx_barrier_all_vec();

            if (subcoreIdx == 0 && aicoreIdx < params.rankSize) {
                uint32_t udmaCoreLoops = params.rankSize;
                uint32_t udmaAicoreNum = params.rankSize;

                auto actualCommBlockShape = MatrixCoord{actualCommM, chunkN};

                for (uint32_t remoteRankIdx = aicoreIdx; remoteRankIdx < udmaCoreLoops;
                     remoteRankIdx += udmaAicoreNum) {
                    if (remoteRankIdx == params.rankIdx) {
                        continue;
                    }

                    // Source: remote rank's symmetric, local rank's column block.
                    MatrixCoord blockOffsetSrc = MatrixCoord{params.rankIdx * commShapeM, 0};
                    // Destination: D row-block for remoteRankIdx, this stage's rows.
                    MatrixCoord blockOffsetDst =
                        MatrixCoord{remoteRankIdx * problemShapeInRank.m() + commIdx * commShapeM, 0};

                    auto gmBlockSrc = gmSymmetric[layoutTagSymmetric.GetOffset(blockOffsetSrc)];
                    auto layoutBlockSrc = layoutTagSymmetric.GetTileLayout(actualCommBlockShape);

                    // AllToAll along N: GET from remote rank's symmetric -> local D.
                    auto gmBlockDst = gmD[layoutD.GetOffset(blockOffsetDst)];
                    auto layoutBlockDst = layoutD.GetTileLayout(actualCommBlockShape);

                    blockRemoteCopy(
                        gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst, actualCommBlockShape,
                        remoteRankIdx % params.rankSize);
                }
            }

            if (commIdx < commLoops - WORKSPACE_STAGES && commLoops >= WORKSPACE_STAGES) {
                aclshmemx_barrier_all_vec();
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageIdx]);
            }
        }
    }

private:
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_ASCEND950_MATMUL_ALLTOALL_MX_SLICE_N_HPP
