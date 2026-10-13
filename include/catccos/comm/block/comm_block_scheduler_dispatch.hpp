/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_BLOCK_COMM_BLOCK_SCHEDULER_DISPATCH_HPP
#define CATCCOS_COMM_BLOCK_COMM_BLOCK_SCHEDULER_DISPATCH_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

#include "catccos/catccos.hpp"
#include "catccos/symm_coord.hpp"

namespace Catccos::Comm::Block {

using Catlass::MatrixCoord;

/**
 * Problem shape of one Dispatch call.
 *
 * Dispatch has no channel dimension: the compact receive layout is the
 * concatenation over source ranks of the rows addressed to each rank, so the
 * per-destination split is simply numTokensPerRank and the receive base of a
 * source is a prefix sum over sources.
 */
struct DispatchProblemShape {
    using UnderlyingProblemShape = Catlass::GemmCoord;

    CATLASS_HOST_DEVICE
    DispatchProblemShape() = default;

    CATLASS_HOST_DEVICE
    DispatchProblemShape(
        UnderlyingProblemShape const originProblemShape, uint32_t rankSize, uint32_t rankIdx,
        GM_ADDR ptrNumTokensPerRank, GM_ADDR ptrRankPrefixMatrix)
        : originProblemShape_(originProblemShape),
          rankSize_(rankSize),
          rankIdx_(rankIdx),
          ptrNumTokensPerRank_(reinterpret_cast<__gm__ int32_t*>(ptrNumTokensPerRank)),
          ptrRankPrefixMatrix_(reinterpret_cast<__gm__ int32_t*>(ptrRankPrefixMatrix))
    {}

    CATLASS_HOST_DEVICE
    uint32_t m() const { return originProblemShape_.m(); }

    CATLASS_HOST_DEVICE
    uint32_t k() const { return originProblemShape_.k(); }

    CATLASS_DEVICE
    uint32_t rankSize() const { return rankSize_; }

    CATLASS_DEVICE
    uint32_t rankIdx() const { return rankIdx_; }

    /// Rows this rank sends to `dstRankIdx`; the length of that staging run.
    CATLASS_DEVICE
    uint32_t numTokensPerRank(uint32_t dstRankIdx) const
    {
        return static_cast<uint32_t>(ptrNumTokensPerRank_[dstRankIdx]);
    }

    /// First row of this source's run inside `dstRankIdx`'s receive region.
    CATLASS_DEVICE
    uint32_t rankPrefix(uint32_t srcRankIdx, uint32_t dstRankIdx) const
    {
        return static_cast<uint32_t>(ptrRankPrefixMatrix_[srcRankIdx * rankSize_ + dstRankIdx]);
    }

private:
    UnderlyingProblemShape originProblemShape_;
    uint32_t rankSize_{0};
    uint32_t rankIdx_{0};
    __gm__ int32_t* ptrNumTokensPerRank_{nullptr};
    __gm__ int32_t* ptrRankPrefixMatrix_{nullptr};
};

/**
 * Round and block schedule for the Dispatch AllToAllV phase.
 *
 * Dispatch repacks its tokens into a destination-major staging buffer first, so
 * the rows owned by destination `d` start at the prefix sum of
 * `inputSplitList[d]` -- the same inputSplitList / inputOffsetList contract
 * BlockCommSchedulerAllToAllVAllGather exposes.  The one addition is
 * `outputOffsetList[d]`: the row inside destination `d` where this source's run
 * begins, so the sender derives both endpoints from the same split lists.
 *
 * Tasks are enumerated block-major across destinations (`taskIdx % rankSize`
 * picks the destination) so the first sweep opens one link per destination instead of
 * hammering a single peer, and `dstOrder` carries the group-aware permutation on
 * top of it. The round window is wider than one communication block when
 * enough rows are available, allowing several AIVs to work on each destination.
 * Each round assigns tasks across all AIVs with a
 * GetBlockNum() * GetSubBlockNum() stride.
 *
 * Dispatch has no in-kernel consumer of the received rows, so rounds need no
 * completion handshake and there is deliberately no GetActualReceiveAccum():
 * every (source, destination) row range has a single writer and is only read
 * after the rank-wide barrier.
 */
struct BlockCommSchedulerDispatch {
    using ProblemShape = DispatchProblemShape;

    /// Group-aware permutation, evaluated once per destination in InitRouting.
    CATLASS_HOST_DEVICE
    static uint32_t GetDestinationRank(
        uint32_t rankSize, uint32_t rankIdx, uint32_t groupSize, uint32_t destinationStep)
    {
        groupSize = groupSize == 0 ? rankSize : Min(groupSize, rankSize);
        const uint32_t groupCount = CeilDiv(rankSize, groupSize);
        // Production group topologies are normally rectangular.  This path is
        // equivalent to the generic mapping below but removes the binary
        // search and group scan from the hot loop.
        if (rankSize % groupSize == 0) {
            auto rankRound = destinationStep / groupCount;
            auto groupOrder = destinationStep - rankRound * groupCount;
            auto sourceGroup = rankIdx / groupSize;
            auto sourceLocalRank = rankIdx % groupSize;
            auto firstGroup = (sourceGroup + sourceLocalRank) % groupCount;
            auto groupIdx = (firstGroup + groupOrder) % groupCount;
            auto firstRank = sourceLocalRank;
            auto rankInGroup = (firstRank + rankRound) % groupSize;
            return groupIdx * groupSize + rankInGroup;
        }

        // Completing R rank rounds visits sum(min(groupWidth[g], R)) ranks.
        // Locate the round with a small binary search; this also supports a
        // partially populated final group.
        uint32_t roundBegin = 0;
        uint32_t roundEnd = groupSize;
        while (roundBegin < roundEnd) {
            auto roundMid = (roundBegin + roundEnd + 1) / 2;
            uint32_t completed = 0;
            for (uint32_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
                completed += Min(Min(groupSize, ClipSub(rankSize, groupIdx * groupSize)), roundMid);
            }
            if (completed <= destinationStep) {
                roundBegin = roundMid;
            } else {
                roundEnd = roundMid - 1;
            }
        }

        auto rankRound = roundBegin;
        uint32_t emitted = 0;
        for (uint32_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
            emitted += Min(Min(groupSize, ClipSub(rankSize, groupIdx * groupSize)), rankRound);
        }

        auto sourceGroup = rankIdx / groupSize;
        auto sourceLocalRank = rankIdx % groupSize;
        auto firstGroup = (sourceGroup + sourceLocalRank) % groupCount;
        for (uint32_t groupOrder = 0; groupOrder < groupCount; ++groupOrder) {
            auto groupIdx = (firstGroup + groupOrder) % groupCount;
            auto ranksInGroup = Min(groupSize, ClipSub(rankSize, groupIdx * groupSize));
            if (rankRound >= ranksInGroup) {
                continue;
            }
            if (emitted == destinationStep) {
                // Adjacent steps start from different peers inside the
                // selected group, spreading MTE links within the group too.
                auto firstRank = sourceLocalRank % ranksInGroup;
                auto rankInGroup = (firstRank + rankRound) % ranksInGroup;
                return groupIdx * groupSize + rankInGroup;
            }
            ++emitted;
        }
        return 0;
    }

    /// UB views the block fills in InitRouting before the AllToAllV phase.
    struct RoutingLists {
        AscendC::LocalTensor<int32_t> inputSplitList;
        AscendC::LocalTensor<int32_t> inputOffsetList;
        AscendC::LocalTensor<int32_t> outputOffsetList;
        AscendC::LocalTensor<int32_t> dstOrder;
    };

    ProblemShape problemShape;
    MatrixCoord commShape;
    MatrixCoord blockShape;
    MatrixCoord blockGrid;

    uint32_t commLoops{1};
    uint32_t coreLoops{0};

    RoutingLists routing;

    CATLASS_DEVICE
    BlockCommSchedulerDispatch() = default;

    CATLASS_DEVICE
    BlockCommSchedulerDispatch(
        ProblemShape const& problemShape_, uint32_t commShapeM_, MatrixCoord const& blockShape_,
        RoutingLists const& routing_)
        : problemShape(problemShape_), blockShape(blockShape_), routing(routing_)
    {
        commShape = Catlass::MakeCoord<uint32_t>(commShapeM_, problemShape.k());
        blockGrid = CeilDiv(commShape, blockShape);
        coreLoops = Numel(blockGrid) * problemShape.rankSize();
        commLoops = 1;
        for (uint32_t dstRankIdx = 0; dstRankIdx < problemShape.rankSize(); ++dstRankIdx) {
            auto sendLoops = CeilDiv(static_cast<uint32_t>(routing.inputSplitList.GetValue(dstRankIdx)), commShapeM_);
            commLoops = Max(commLoops, sendLoops);
        }
    }

    CATLASS_DEVICE
    uint32_t GetCommLoops() const { return commLoops; }

    CATLASS_DEVICE
    uint32_t GetCoreLoops(uint32_t commIdx) const
    {
        (void)commIdx;
        return coreLoops; // an upper bound, independent of commIdx
    }

    /// Destination of a task slot, after the group-aware permutation.
    CATLASS_DEVICE
    uint32_t GetDstRankIdx(uint32_t taskIdx) const
    {
        return static_cast<uint32_t>(routing.dstOrder.GetValue(taskIdx % problemShape.rankSize()));
    }

    struct Iter {
        uint32_t taskIdx{0};
        uint32_t coreLoops{0};

        CATLASS_DEVICE
        void Next() { taskIdx += AscendC::GetBlockNum() * AscendC::GetSubBlockNum(); }

        CATLASS_DEVICE
        bool End() { return taskIdx >= coreLoops; }
    };

    CATLASS_DEVICE
    Iter Begin(uint32_t commIdx) const
    {
        uint32_t taskIdx = AscendC::GetBlockIdx();
        return {taskIdx, GetCoreLoops(commIdx)};
    }

    CATLASS_DEVICE
    SymmMatrixCoord GetBlockOffset(Iter const& iter) const
    {
        auto dstRankIdx = GetDstRankIdx(iter.taskIdx);
        auto blockIdx = iter.taskIdx / problemShape.rankSize();
        auto blockCoord = Catlass::MakeCoord(blockIdx / blockGrid.column(), blockIdx % blockGrid.column());
        return SymmMatrixCoord{blockCoord * blockShape, problemShape.rankIdx(), dstRankIdx};
    }

    struct RemapperSrc {
        using Scheduler = BlockCommSchedulerDispatch;

        Scheduler const* scheduler;
        uint32_t commIdx;

        CATLASS_DEVICE
        MatrixCoord operator()(SymmMatrixCoord const& blockOffset) const
        {
            auto dstRankIdx = blockOffset.remote();
            auto runOffset = Catlass::MakeCoord<uint32_t>(scheduler->routing.inputOffsetList.GetValue(dstRankIdx), 0);
            auto commOffset = Catlass::MakeCoord<uint32_t>(commIdx, 0) * scheduler->commShape;
            return runOffset + commOffset + blockOffset.GetMatrixCoord();
        }

        CATLASS_DEVICE
        MatrixCoord GetResidueShape(SymmMatrixCoord const& blockOffset) const
        {
            auto dstRankIdx = blockOffset.remote();
            auto splitShape = Catlass::MakeCoord<uint32_t>(
                scheduler->routing.inputSplitList.GetValue(dstRankIdx), scheduler->problemShape.k());
            auto commOffset = Catlass::MakeCoord<uint32_t>(commIdx, 0) * scheduler->commShape;
            auto actualCommShape = Min(scheduler->commShape, ClipSub(splitShape, commOffset));
            return ClipSub(actualCommShape, blockOffset.GetMatrixCoord());
        }
    };

    /// Destination endpoint: this source's run inside the destination's receive
    /// region.  BlockCommSchedulerAllToAllVAllGather addresses a
    /// `[stage][src][commShapeM]` window instead, which is why it needs no
    /// equivalent remapper.
    struct RemapperDst {
        using Scheduler = BlockCommSchedulerDispatch;

        Scheduler const* scheduler;
        uint32_t commIdx;

        CATLASS_DEVICE
        MatrixCoord operator()(SymmMatrixCoord const& blockOffset) const
        {
            auto dstRankIdx = blockOffset.remote();
            auto runOffset = Catlass::MakeCoord<uint32_t>(scheduler->routing.outputOffsetList.GetValue(dstRankIdx), 0);
            auto commOffset = Catlass::MakeCoord<uint32_t>(commIdx, 0) * scheduler->commShape;
            return runOffset + commOffset + blockOffset.GetMatrixCoord();
        }
    };

    CATLASS_DEVICE
    const RemapperSrc GetRemapperSrc(uint32_t commIdx) { return {this, commIdx}; }

    CATLASS_DEVICE
    const RemapperDst GetRemapperDst(uint32_t commIdx) { return {this, commIdx}; }

    CATLASS_DEVICE
    MatrixCoord RemapActualBlockShape(SymmMatrixCoord const& blockOffset, RemapperSrc const& remapperSrc) const
    {
        return Min(blockShape, remapperSrc.GetResidueShape(blockOffset));
    }
};

} // namespace Catccos::Comm::Block

#endif // CATCCOS_COMM_BLOCK_COMM_BLOCK_SCHEDULER_DISPATCH_HPP
