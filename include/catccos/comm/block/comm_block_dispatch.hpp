/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_BLOCK_COMM_BLOCK_DISPATCH_HPP
#define CATCCOS_COMM_BLOCK_COMM_BLOCK_DISPATCH_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/layout/layout.hpp"
#include "catccos/catccos.hpp"
#include "catccos/comm/block/comm_block_scheduler_dispatch.hpp"
#include "shmem.h"

namespace Catccos::Comm::Block {

/**
 * MTE communication block shared by staged and direct MoE Dispatch.
 *
 * Like the other AllToAllV CommBlock implementations this class owns the UB
 * tiles, the event lifecycle and local/remote copy scheduling; the kernel owns the
 * round loop, the scheduler and the barrier.
 *
 * RepackToStage permutes the local tokens into a destination-major staging
 * buffer that lives in this rank's own symmetric heap: the run owned by
 * destination `d` holds every token addressed to `d`, in input token order.
 * Workers own disjoint row intervals of the staging tensor, so no atomic cursor
 * is needed even when several workers handle the same destination.
 * SendDirect uses the same UB tiles but writes to the final remote receive
 * region, partitioning valid rows in group-aware destination order.
 *
 * operator() then moves one scheduled block of one destination's run to that
 * rank's receive region in 2D tiles, for the activation and for
 * every metadata field.  Because the destination's per-source region is
 * contiguous, both endpoints come from the split lists and no completion
 * handshake is required inside a round.
 *
 * FP8 and INT8 specializations also carry one ElementScale activation scale per token.
 */
template <
    class DispatchPolicy_, class ElementToken_, class ElementScale_, class ElementTopkIdx_, class ElementTopkW_,
    class ElementIdx_, template <class> class TileRemoteCopy_, class CopyGmToUb_, class CopyUbToGm_, class TileSwizzle_,
    uint32_t MAX_RANKS_ = 1024, uint32_t MAX_TOPK_ = 128>
class CommBlockDispatch {
public:
    using DispatchPolicy = DispatchPolicy_;
    using ElementToken = ElementToken_;
    using ElementScale = ElementScale_;
    using ElementTopkIdx = ElementTopkIdx_;
    using ElementTopkW = ElementTopkW_;
    using ElementIdx = ElementIdx_;
    static constexpr bool HAS_SCALES =
        std::is_same_v<ElementToken, float8_e4m3_t> || std::is_same_v<ElementToken, int8_t>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ProblemShape = DispatchProblemShape;
    using CopyGmToUb = CopyGmToUb_;
    using CopyUbToGm = CopyUbToGm_;
    using Layout = typename CopyGmToUb::LayoutSrc;
    using TileSwizzle = TileSwizzle_;
    template <class Element>
    using TileRemoteCopy = TileRemoteCopy_<Element>;

    static constexpr uint32_t MAX_RANKS = MAX_RANKS_;
    static constexpr uint32_t MAX_TOPK = MAX_TOPK_;
    static constexpr uint32_t UB_STAGES = DispatchPolicy::UB_STAGES;
    // Reserved bytes at UB start for hardware flags, matching the other
    // CommBlock implementations.
    static constexpr uint32_t flagSize = 2 * 1024;
    static constexpr uint32_t ubAlignSize = 32;
    static constexpr uint32_t tileAlignSize = 512;
    // Metadata chunks are accumulated across rows, so they are carved once.
    static constexpr uint32_t metaRegionBytes = 16 * 1024 + 128;
    static constexpr uint32_t metaRoundSlackBytes = HAS_SCALES ? 128 : 96;
    // Scalar -> MTE3 handshake for the metadata chunks.
    static constexpr uint32_t metaEventId = UB_STAGES;
    // inputSplitList / inputOffsetList / outputOffsetList / dstOrder.
    static constexpr uint32_t routingListBytes =
        (MAX_RANKS * sizeof(int32_t) + ubAlignSize - 1) / ubAlignSize * ubAlignSize;
    static constexpr uint32_t routingBytes = 4 * routingListBytes;

    static_assert(UB_STAGES >= 1, "Dispatch requires at least one UB stage");
    static_assert(
        std::is_same_v<ElementTopkIdx, int32_t> || std::is_same_v<ElementTopkIdx, int64_t>,
        "Dispatch top-k index must be int32-compatible or int64-compatible");
    static_assert(
        std::is_same_v<ElementIdx, int32_t> || std::is_same_v<ElementIdx, int64_t>,
        "Dispatch source index must be int32_t or int64_t");
    static_assert(
        MAX_TOPK * (sizeof(ElementTopkIdx) + sizeof(ElementTopkW)) + sizeof(ElementIdx) + sizeof(ElementScale) +
                metaRoundSlackBytes <=
            metaRegionBytes,
        "Dispatch metadata chunk cannot hold one worst-case row");
    static_assert(
        flagSize + metaRegionBytes + UB_STAGES * tileAlignSize + routingBytes <= ArchTag::UB_SIZE,
        "Dispatch staging and metadata exceed the Ascend950 UB budget");

    struct Params {
        uint32_t rankIdx{0};
        uint32_t rankSize{0};
        uint32_t groupSize{1};
        uint32_t numTokens{0};
        uint32_t hidden{0};
        uint32_t numTopk{0};
        uint32_t numExperts{0};
        bool hasTopkWeights{false};
        __gm__ ElementScale* ptrQuantScales{nullptr};
        __gm__ ElementScale* ptrRecvScales{nullptr};
        uint64_t stageScalesOffset{0};
        uint64_t recvScalesOffset{0};
        MatrixCoord commBlockShape{};

        GM_ADDR ptrSymmetric{nullptr};
        uint64_t recvXOffset{0};
        uint64_t recvTopkIdxOffset{0};
        uint64_t recvTopkWeightsOffset{0};
        uint64_t recvSrcIdxOffset{0};
        uint64_t stageXOffset{0};
        uint64_t stageTopkIdxOffset{0};
        uint64_t stageTopkWeightsOffset{0};
        uint64_t stageSrcIdxOffset{0};

        __gm__ ElementToken* ptrRecvX{nullptr};
        __gm__ ElementTopkIdx* ptrRecvTopkIdx{nullptr};
        __gm__ ElementTopkW* ptrRecvTopkWeights{nullptr};
        __gm__ ElementIdx* ptrRecvSrcIdx{nullptr};

        CATLASS_DEVICE
        MatrixCoord BlockShape() const { return commBlockShape; }
    };

    CATLASS_DEVICE
    CommBlockDispatch(Catlass::Arch::Resource<ArchTag>& resource, Params const& params_) : params(params_)
    {
        gmSymmetric.SetGlobalBuffer(GetLocalBase());
        topkIdxRowBytes = params.numTopk * sizeof(ElementTopkIdx);
        topkWeightRowBytes = params.numTopk * sizeof(ElementTopkW);
        uint32_t minMetaFieldBytes = sizeof(ElementIdx);
        if constexpr (HAS_SCALES) {
            minMetaFieldBytes = Min(minMetaFieldBytes, static_cast<uint32_t>(sizeof(ElementScale)));
        }
        if (params.numTopk > 0) {
            minMetaFieldBytes = Min(minMetaFieldBytes, topkIdxRowBytes);
            if (params.hasTopkWeights) {
                minMetaFieldBytes = Min(minMetaFieldBytes, topkWeightRowBytes);
            }
        }
        minTransferRows = CeilDiv(ubAlignSize, minMetaFieldBytes);
        const uint32_t metaRowBytes =
            topkIdxRowBytes + topkWeightRowBytes + sizeof(ElementIdx) + (HAS_SCALES ? sizeof(ElementScale) : 0);
        const uint32_t carveRows = (metaRegionBytes - metaRoundSlackBytes) / (metaRowBytes == 0 ? 1 : metaRowBytes);
        metaRows = Max(1U, carveRows);

        const uint32_t idxBytes = AlignUpBytes(metaRows * topkIdxRowBytes);
        const uint32_t weightBytes = AlignUpBytes(metaRows * topkWeightRowBytes);
        const uint32_t srcBytes = AlignUpBytes(metaRows * sizeof(ElementIdx));
        const uint32_t scaleBytes = HAS_SCALES ? AlignUpBytes(metaRows * sizeof(ElementScale)) : 0;
        metaByteSize = idxBytes + weightBytes + srcBytes + scaleBytes;

        size_t ubOffset = flagSize;
        ubMetaTopkIdx = resource.ubBuf.template GetBufferByByte<ElementTopkIdx>(ubOffset);
        ubMetaTopkWeight = resource.ubBuf.template GetBufferByByte<ElementTopkW>(ubOffset + idxBytes);
        ubMetaSrcIdx = resource.ubBuf.template GetBufferByByte<ElementIdx>(ubOffset + idxBytes + weightBytes);
        ubMetaScales =
            resource.ubBuf.template GetBufferByByte<ElementScale>(ubOffset + idxBytes + weightBytes + srcBytes);
        ubOffset += metaByteSize;

        tileBytes = static_cast<uint32_t>(ArchTag::UB_SIZE - ubOffset - routingBytes) / UB_STAGES / tileAlignSize *
                    tileAlignSize;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubList[i] = resource.ubBuf.template GetBufferByByte<uint8_t>(ubOffset);
            ubOffset += tileBytes;
        }

        ubInputSplitList = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += routingListBytes;
        ubInputOffsetList = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += routingListBytes;
        ubOutputOffsetList = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
        ubOffset += routingListBytes;
        ubDstOrder = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
    }

    CATLASS_DEVICE
    void InitBlockLoop()
    {
        uint32_t copyEventId = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            copyEventIdList[i] = copyEventId++;
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[i]);
        }
    }

    CATLASS_DEVICE
    void FinalizeBlockLoop()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[i]);
        }
        ubListId = 0;
    }

    /// Fills the split lists and group-aware destination order, and counts
    /// destination-local tasks available to the direct path.
    CATLASS_DEVICE
    BlockCommSchedulerDispatch::RoutingLists InitRouting(ProblemShape const& problemShape)
    {
        activeDestinations = 0;
        directTaskCapacity = 0;
        int32_t runOffset = 0;
        for (uint32_t dstRankIdx = 0; dstRankIdx < problemShape.rankSize(); ++dstRankIdx) {
            const int32_t split = static_cast<int32_t>(problemShape.numTokensPerRank(dstRankIdx));
            ubInputSplitList.SetValue(dstRankIdx, split);
            ubInputOffsetList.SetValue(dstRankIdx, runOffset);
            runOffset += split;
            if (split > 0) {
                ++activeDestinations;
                directTaskCapacity += CeilDiv(static_cast<uint32_t>(split), minTransferRows);
            }
            const int32_t recvOffset =
                problemShape.rankIdx() == 0 ?
                    0 :
                    static_cast<int32_t>(problemShape.rankPrefix(problemShape.rankIdx() - 1, dstRankIdx));
            ubOutputOffsetList.SetValue(dstRankIdx, recvOffset);
            ubDstOrder.SetValue(
                dstRankIdx, static_cast<int32_t>(BlockCommSchedulerDispatch::GetDestinationRank(
                                problemShape.rankSize(), problemShape.rankIdx(), params.groupSize, dstRankIdx)));
        }
        return {ubInputSplitList, ubInputOffsetList, ubOutputOffsetList, ubDstOrder};
    }

    /**
     * Permutes the local tokens into the destination-major staging run.
     *
     * Workers own disjoint, balanced row intervals of the flattened staging
     * tensor. Each interval may intersect several destination runs; scanning
     * their routing masks locates the input tokens without changing row order.
     */
    CATLASS_DEVICE
    void RepackToStage(
        ProblemShape const& problemShape, AscendC::GlobalTensor<uint8_t> const& gmX,
        AscendC::GlobalTensor<ElementTopkIdx> const& gmTopkIdx, AscendC::GlobalTensor<ElementTopkW> const& gmTopkWeight,
        AscendC::GlobalTensor<uint8_t> const& gmIsTokenInRank)
    {
        CopyTokens<false>(problemShape, gmX, gmTopkIdx, gmTopkWeight, gmIsTokenInRank);
    }

    /// Gather into UB and put directly into the final receive region; no GM staging.
    CATLASS_DEVICE
    void SendDirect(
        ProblemShape const& problemShape, AscendC::GlobalTensor<uint8_t> const& gmX,
        AscendC::GlobalTensor<ElementTopkIdx> const& gmTopkIdx, AscendC::GlobalTensor<ElementTopkW> const& gmTopkWeight,
        AscendC::GlobalTensor<uint8_t> const& gmIsTokenInRank)
    {
        CopyTokens<true>(problemShape, gmX, gmTopkIdx, gmTopkWeight, gmIsTokenInRank);
    }

private:
    // Both paths use the same token selection, metadata conversion and tile
    // copies. DIRECT changes only row ownership order and the write endpoint.
    template <bool DIRECT>
    CATLASS_DEVICE void CopyTokens(
        ProblemShape const& problemShape, AscendC::GlobalTensor<uint8_t> const& gmX,
        AscendC::GlobalTensor<ElementTopkIdx> const& gmTopkIdx, AscendC::GlobalTensor<ElementTopkW> const& gmTopkWeight,
        AscendC::GlobalTensor<uint8_t> const& gmIsTokenInRank)
    {
        const uint32_t aicoreIdx = AscendC::GetBlockIdx();
        const uint32_t aicoreNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        const uint32_t xRowBytes = params.hidden * sizeof(ElementToken);
        const uint32_t numTokens = problemShape.m();
        const uint32_t rankSize = problemShape.rankSize();
        if (xRowBytes == 0 || aicoreNum == 0) {
            return;
        }
        const uint32_t expertsPerRank = params.numExperts / rankSize;
        const uint32_t tileN = Min(xRowBytes, tileBytes);
        const uint32_t tileM = Min(params.commBlockShape.row(), tileBytes / AlignUpBytes(tileN));
        const uint32_t numSendRows =
            static_cast<uint32_t>(ubInputOffsetList.GetValue(rankSize - 1) + ubInputSplitList.GetValue(rankSize - 1));
        const uint32_t firstRow = static_cast<uint64_t>(numSendRows) * aicoreIdx / aicoreNum;
        const uint32_t lastRow = static_cast<uint64_t>(numSendRows) * (aicoreIdx + 1) / aicoreNum;
        if (numSendRows == 0 || (!DIRECT && firstRow == lastRow)) {
            return;
        }

        // Give each nonempty destination one task, then distribute the remaining
        // AIVs in proportion to destination-local chunk capacity. The capacity
        // uses a 32-byte target for the smallest metadata field, while even a
        // shorter nonempty destination retains one task.
        const uint32_t plannedTasks = Min(aicoreNum, directTaskCapacity);
        const uint32_t extraCapacity = directTaskCapacity - activeDestinations;
        const uint32_t extraTasks = plannedTasks > activeDestinations ? plannedTasks - activeDestinations : 0;
        uint32_t taskBase = 0;
        uint32_t capacityPrefix = 0;

        uint32_t orderedRows = 0;
        for (uint32_t dstStep = 0; dstStep < rankSize; ++dstStep) {
            const uint32_t dstRankIdx = static_cast<uint32_t>(ubDstOrder.GetValue(dstStep));
            const uint32_t dstStartRow =
                DIRECT ? orderedRows : static_cast<uint32_t>(ubInputOffsetList.GetValue(dstRankIdx));
            const uint32_t dstRows = static_cast<uint32_t>(ubInputSplitList.GetValue(dstRankIdx));
            const uint32_t dstEndRow = dstStartRow + dstRows;
            orderedRows = dstEndRow;
            uint32_t runStartRow = Max(firstRow, dstStartRow);
            uint32_t runEndRow = Min(lastRow, dstEndRow);
            if constexpr (DIRECT) {
                if (dstRows == 0) {
                    continue;
                }
                const uint32_t capacity = CeilDiv(dstRows, minTransferRows) - 1;
                const uint32_t nextPrefix = capacityPrefix + capacity;
                uint32_t additionalTasks = 0;
                if (extraCapacity > 0) {
                    const uint32_t nextAssigned = static_cast<uint64_t>(extraTasks) * nextPrefix / extraCapacity;
                    const uint32_t previousAssigned =
                        static_cast<uint64_t>(extraTasks) * capacityPrefix / extraCapacity;
                    additionalTasks = nextAssigned - previousAssigned;
                }
                capacityPrefix = nextPrefix;
                const uint32_t taskCount = 1 + additionalTasks;
                const uint32_t localTask = (aicoreIdx + aicoreNum - taskBase % aicoreNum) % aicoreNum;
                taskBase += taskCount;
                if (localTask >= taskCount) {
                    continue;
                }
                runStartRow = dstStartRow + static_cast<uint64_t>(dstRows) * localTask / taskCount;
                runEndRow = dstStartRow + static_cast<uint64_t>(dstRows) * (localTask + 1) / taskCount;
            }
            if (runStartRow >= runEndRow) {
                continue;
            }
            AscendC::GlobalTensor<uint8_t> gmTarget = gmSymmetric;
            uint32_t targetRow = runStartRow;
            if constexpr (DIRECT) {
                gmTarget.SetGlobalBuffer(
                    reinterpret_cast<__gm__ uint8_t*>(aclshmem_ptr(params.ptrSymmetric, dstRankIdx)));
                targetRow = static_cast<uint32_t>(ubOutputOffsetList.GetValue(dstRankIdx)) + runStartRow - dstStartRow;
            }
            const uint64_t xOffset = DIRECT ? params.recvXOffset : params.stageXOffset;
            uint32_t selectedRow = dstStartRow;
            uint32_t stagedRows = 0;
            uint32_t prefetchedRows = 0;
            uint32_t runRow = 0;
            uint32_t metaStaged = 0;
            uint32_t metaFlushedRows = 0;

            for (uint32_t tokenIdx = 0; tokenIdx < numTokens; ++tokenIdx) {
                if (prefetchedRows == 0 &&
                    gmIsTokenInRank.GetValue(static_cast<uint64_t>(tokenIdx) * rankSize + dstRankIdx) == 0) {
                    continue;
                }
                if (selectedRow++ < runStartRow) {
                    continue;
                }
                // Gather selected rows into a TileM x TileN UB tile. Only UB
                // rows are padded; both GM endpoints keep their packed stride.
                uint32_t copied = 0;
                while (copied < xRowBytes) {
                    const uint32_t bytes = Min(tileN, xRowBytes - copied);
                    const uint32_t ubStride = AlignUpBytes(bytes);
                    if (prefetchedRows == 0) {
                        if (stagedRows == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);
                        }
                        prefetchedRows = 1;
                        if (tileN == xRowBytes) {
                            // Load adjacent selected tokens together, without
                            // crossing the tile, worker run or copy block-count limit.
                            constexpr uint32_t maxCopyRows = 4095;
                            const uint32_t maxRows = Min(
                                maxCopyRows,
                                Min(tileM - stagedRows, Min(numTokens - tokenIdx, runEndRow - runStartRow - runRow)));
                            while (prefetchedRows < maxRows &&
                                   gmIsTokenInRank.GetValue(
                                       static_cast<uint64_t>(tokenIdx + prefetchedRows) * rankSize + dstRankIdx) != 0) {
                                ++prefetchedRows;
                            }
                        }
                        auto ubTile = ubList[ubListId][stagedRows * ubStride];
                        auto gmTileSrc = gmX[static_cast<uint64_t>(tokenIdx) * xRowBytes + copied];
                        const Layout layoutUb{prefetchedRows, bytes, ubStride};
                        const Layout layoutTileSrc{prefetchedRows, bytes, xRowBytes};
                        copyGmToUb(ubTile, gmTileSrc, layoutUb, layoutTileSrc);
                    }
                    --prefetchedRows;
                    ++stagedRows;
                    if (stagedRows == tileM) {
                        const uint64_t targetOffset =
                            xOffset + static_cast<uint64_t>(targetRow + runRow + 1 - stagedRows) * xRowBytes + copied;
                        const Layout layoutStage{stagedRows, bytes, xRowBytes};
                        const Layout layoutStagedUb{stagedRows, bytes, ubStride};
                        FlushX(gmTarget, targetOffset, layoutStage, layoutStagedUb);
                        stagedRows = 0;
                    }
                    copied += bytes;
                }
                ++runRow;

                if (params.numTopk > 0) {
                    const int32_t localBegin = static_cast<int32_t>(dstRankIdx * expertsPerRank);
                    const int32_t localEnd = localBegin + static_cast<int32_t>(expertsPerRank);
                    for (uint32_t topkIdx = 0; topkIdx < params.numTopk; ++topkIdx) {
                        const ElementTopkIdx expert =
                            gmTopkIdx.GetValue(static_cast<uint64_t>(tokenIdx) * params.numTopk + topkIdx);
                        const bool owned = expert >= static_cast<ElementTopkIdx>(localBegin) &&
                                           expert < static_cast<ElementTopkIdx>(localEnd);
                        ubMetaTopkIdx.SetValue(
                            metaStaged * params.numTopk + topkIdx,
                            owned ? expert - static_cast<ElementTopkIdx>(localBegin) : static_cast<ElementTopkIdx>(-1));
                        if (params.hasTopkWeights) {
                            ubMetaTopkWeight.SetValue(
                                metaStaged * params.numTopk + topkIdx,
                                owned ?
                                    gmTopkWeight.GetValue(static_cast<uint64_t>(tokenIdx) * params.numTopk + topkIdx) :
                                    static_cast<ElementTopkW>(0));
                        }
                    }
                }
                ubMetaSrcIdx.SetValue(metaStaged, static_cast<ElementIdx>(tokenIdx));
                if constexpr (HAS_SCALES) {
                    AscendC::GlobalTensor<ElementScale> gmScales;
                    gmScales.SetGlobalBuffer(params.ptrQuantScales);
                    ubMetaScales.SetValue(metaStaged, gmScales.GetValue(tokenIdx));
                }
                ++metaStaged;
                if (metaStaged == metaRows) {
                    FlushMeta<DIRECT>(gmTarget, targetRow + metaFlushedRows, metaStaged);
                    metaFlushedRows += metaStaged;
                    metaStaged = 0;
                }
                if (runStartRow + runRow == runEndRow) {
                    break;
                }
            }

            if (stagedRows > 0) {
                const uint64_t targetOffset =
                    xOffset + static_cast<uint64_t>(targetRow + runRow - stagedRows) * xRowBytes;
                const Layout layoutStage{stagedRows, xRowBytes};
                const Layout layoutUb{stagedRows, xRowBytes, AlignUpBytes(xRowBytes)};
                FlushX(gmTarget, targetOffset, layoutStage, layoutUb);
            }
            if (metaStaged > 0) {
                FlushMeta<DIRECT>(gmTarget, targetRow + metaFlushedRows, metaStaged);
            }
        }
    }

public:
    /// Splits a typed field into 2D tiles; the tile layer owns the DMA details.
    template <class ElementSrc, class LayoutSrc, class ElementDst, class LayoutDst>
    CATLASS_DEVICE void operator()(
        AscendC::GlobalTensor<ElementSrc> const& gmSrc, LayoutSrc const& layoutSrc,
        AscendC::GlobalTensor<ElementDst> const& gmDst, LayoutDst const& layoutDst,
        MatrixCoord const& actualCommBlockShape, uint32_t rankIdx)
    {
        static_assert(std::is_same_v<ElementSrc, ElementDst>, "Dispatch remote copy does not convert types");
        static_assert(
            std::is_same_v<LayoutSrc, Catlass::layout::RowMajor> &&
                std::is_same_v<LayoutDst, Catlass::layout::RowMajor>,
            "Dispatch field copy supports RowMajor layouts");
        const uint32_t rows = actualCommBlockShape.row();
        const uint32_t columns = actualCommBlockShape.column();
        if (rows == 0 || columns == 0) {
            return;
        }
        const uint32_t tileN = Min(columns, tileBytes / static_cast<uint32_t>(sizeof(ElementSrc)));
        const uint32_t tileM = Min(params.commBlockShape.row(), tileBytes / AlignUpBytes(tileN * sizeof(ElementSrc)));
        const MatrixCoord tileShape{tileM, tileN};
        TileSwizzle tileSwizzle{actualCommBlockShape, tileShape};
        TileRemoteCopy<ElementSrc> tileRemoteCopy;
        const uint32_t tileLoops = tileSwizzle.GetLoops();
        for (uint32_t tileIdx = 0; tileIdx < tileLoops; ++tileIdx) {
            const auto tileCoord = tileSwizzle.GetTileCoord(tileIdx);
            const auto actualTileShape = tileSwizzle.GetActualTileShape(tileCoord);
            const auto tileOffsetInBlock = tileCoord * tileShape;
            auto gmTileSrc = gmSrc[layoutSrc.GetOffset(tileOffsetInBlock)];
            auto layoutTileSrc = layoutSrc.GetTileLayout(actualTileShape);
            auto gmTileDst = gmDst[layoutDst.GetOffset(tileOffsetInBlock)];
            auto layoutTileDst = layoutDst.GetTileLayout(actualTileShape);
            auto ubTile = ubList[ubListId].template ReinterpretCast<ElementSrc>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);
            tileRemoteCopy(
                gmTileDst, layoutTileDst, gmTileSrc, layoutTileSrc, actualTileShape, ubTile, copyEventIdList[ubListId],
                rankIdx);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);
            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

    /// Copies the valid receive rows to the public
    /// outputs when they are not aliased to the symmetric buffer.
    CATLASS_DEVICE
    void Complete(uint32_t actualRecvTokens, uint32_t aicoreIdx, uint32_t aicoreNum)
    {
        const uint32_t xRowBytes = params.hidden * sizeof(ElementToken);
        CopyReceiveResult(actualRecvTokens, aicoreIdx, aicoreNum, xRowBytes);
    }

private:
    static constexpr uint32_t AlignUpBytes(uint32_t value)
    {
        return (value + ubAlignSize - 1) / ubAlignSize * ubAlignSize;
    }

    CATLASS_DEVICE
    __gm__ uint8_t* GetLocalBase() const { return reinterpret_cast<__gm__ uint8_t*>(params.ptrSymmetric); }

    CATLASS_DEVICE
    void FlushX(
        AscendC::GlobalTensor<uint8_t> const& gmTarget, uint64_t destByteOffset, Layout const& layoutGm,
        Layout const& layoutUb)
    {
        auto gmTileDst = gmTarget[destByteOffset];
        auto& ubTile = ubList[ubListId];
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(copyEventIdList[ubListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(copyEventIdList[ubListId]);
        copyUbToGm(gmTileDst, ubTile, layoutGm, layoutUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);
        ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
    }

    template <bool DIRECT>
    CATLASS_DEVICE void FlushMeta(AscendC::GlobalTensor<uint8_t> const& gmTarget, uint32_t destRow, uint32_t rows)
    {
        if (rows == 0) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(metaEventId);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(metaEventId);
        if constexpr (HAS_SCALES) {
            const uint64_t offset = DIRECT ? params.recvScalesOffset : params.stageScalesOffset;
            auto gmTileDst = gmTarget[offset + static_cast<uint64_t>(destRow) * sizeof(ElementScale)];
            auto ubTile = ubMetaScales.template ReinterpretCast<uint8_t>();
            const Layout layoutTile{1, rows * static_cast<uint32_t>(sizeof(ElementScale))};
            copyUbToGm(gmTileDst, ubTile, layoutTile, layoutTile);
        }
        if (params.numTopk > 0) {
            const uint64_t offset = DIRECT ? params.recvTopkIdxOffset : params.stageTopkIdxOffset;
            auto gmTileDst = gmTarget[offset + static_cast<uint64_t>(destRow) * topkIdxRowBytes];
            auto ubTile = ubMetaTopkIdx.template ReinterpretCast<uint8_t>();
            const Layout layoutTile{1, rows * topkIdxRowBytes};
            copyUbToGm(gmTileDst, ubTile, layoutTile, layoutTile);
            if (params.hasTopkWeights) {
                const uint64_t weightOffset = DIRECT ? params.recvTopkWeightsOffset : params.stageTopkWeightsOffset;
                auto gmWeightTileDst = gmTarget[weightOffset + static_cast<uint64_t>(destRow) * topkWeightRowBytes];
                auto ubWeightTile = ubMetaTopkWeight.template ReinterpretCast<uint8_t>();
                const Layout layoutWeightTile{1, rows * topkWeightRowBytes};
                copyUbToGm(gmWeightTileDst, ubWeightTile, layoutWeightTile, layoutWeightTile);
            }
        }
        const uint64_t srcOffset = DIRECT ? params.recvSrcIdxOffset : params.stageSrcIdxOffset;
        auto gmTileDst = gmTarget[srcOffset + static_cast<uint64_t>(destRow) * sizeof(ElementIdx)];
        auto ubTile = ubMetaSrcIdx.template ReinterpretCast<uint8_t>();
        const Layout layoutTile{1, rows * static_cast<uint32_t>(sizeof(ElementIdx))};
        copyUbToGm(gmTileDst, ubTile, layoutTile, layoutTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(metaEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(metaEventId);
    }

    CATLASS_DEVICE
    void CopyReceiveResult(uint32_t actualRecvTokens, uint32_t aicoreIdx, uint32_t aicoreNum, uint32_t xRowBytes)
    {
        if constexpr (HAS_SCALES) {
            CopyRows(
                params.recvScalesOffset, reinterpret_cast<__gm__ uint8_t*>(params.ptrRecvScales), sizeof(ElementScale),
                actualRecvTokens, aicoreIdx, aicoreNum);
        }
        CopyRows(
            params.recvXOffset, reinterpret_cast<__gm__ uint8_t*>(params.ptrRecvX), xRowBytes, actualRecvTokens,
            aicoreIdx, aicoreNum);
        if (params.numTopk > 0) {
            CopyRows(
                params.recvTopkIdxOffset, reinterpret_cast<__gm__ uint8_t*>(params.ptrRecvTopkIdx), topkIdxRowBytes,
                actualRecvTokens, aicoreIdx, aicoreNum);
            if (params.hasTopkWeights) {
                CopyRows(
                    params.recvTopkWeightsOffset, reinterpret_cast<__gm__ uint8_t*>(params.ptrRecvTopkWeights),
                    topkWeightRowBytes, actualRecvTokens, aicoreIdx, aicoreNum);
            }
        }
        CopyRows(
            params.recvSrcIdxOffset, reinterpret_cast<__gm__ uint8_t*>(params.ptrRecvSrcIdx), sizeof(ElementIdx),
            actualRecvTokens, aicoreIdx, aicoreNum);
    }

    CATLASS_DEVICE
    void CopyRows(
        uint64_t recvRegionOffset, __gm__ uint8_t* gmDst, uint32_t rowBytes, uint32_t actualRecvTokens,
        uint32_t aicoreIdx, uint32_t aicoreNum)
    {
        if (gmDst == nullptr || rowBytes == 0) {
            return;
        }
        __gm__ uint8_t* gmLocal = GetLocalBase();
        if (gmDst == gmLocal + recvRegionOffset) {
            return;
        }
        AscendC::GlobalTensor<uint8_t> gmSrc;
        gmSrc.SetGlobalBuffer(gmLocal);
        AscendC::GlobalTensor<uint8_t> gmOutput;
        gmOutput.SetGlobalBuffer(gmDst);
        InitBlockLoop();
        const uint32_t blockM = params.commBlockShape.row();
        const uint32_t tileN = Min(rowBytes, tileBytes);
        const uint32_t ubStride = AlignUpBytes(tileN);
        const uint32_t tileM = Min(blockM, tileBytes / ubStride);
        const MatrixCoord tileShape{tileM, tileN};
        for (uint64_t row = aicoreIdx; row < actualRecvTokens; row += static_cast<uint64_t>(aicoreNum) * blockM) {
            const uint32_t rows =
                static_cast<uint32_t>(Min<uint64_t>(blockM, (actualRecvTokens - row - 1) / aicoreNum + 1));
            const Layout layoutBlock{rows, rowBytes, static_cast<int64_t>(aicoreNum) * rowBytes};
            auto gmBlockSrc = gmSrc[recvRegionOffset + row * rowBytes];
            auto gmBlockDst = gmOutput[row * rowBytes];
            TileSwizzle tileSwizzle{MatrixCoord{rows, rowBytes}, tileShape};
            const uint32_t tileLoops = tileSwizzle.GetLoops();
            for (uint32_t tileIdx = 0; tileIdx < tileLoops; ++tileIdx) {
                const auto tileCoord = tileSwizzle.GetTileCoord(tileIdx);
                const auto actualTileShape = tileSwizzle.GetActualTileShape(tileCoord);
                const auto tileOffsetInBlock = tileCoord * tileShape;
                auto gmTileSrc = gmBlockSrc[layoutBlock.GetOffset(tileOffsetInBlock)];
                auto layoutTileSrc = layoutBlock.GetTileLayout(actualTileShape);
                auto gmTileDst = gmBlockDst[layoutBlock.GetOffset(tileOffsetInBlock)];
                auto layoutTileDst = layoutBlock.GetTileLayout(actualTileShape);
                auto& ubTile = ubList[ubListId];
                const Layout layoutUb{actualTileShape.row(), actualTileShape.column(), ubStride};

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);
                copyGmToUb(ubTile, gmTileSrc, layoutUb, layoutTileSrc);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(copyEventIdList[ubListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(copyEventIdList[ubListId]);
                copyUbToGm(gmTileDst, ubTile, layoutTileDst, layoutUb);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyEventIdList[ubListId]);
                ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
            }
        }
        FinalizeBlockLoop();
    }

private:
    Params params;
    AscendC::GlobalTensor<uint8_t> gmSymmetric;
    AscendC::LocalTensor<uint8_t> ubList[UB_STAGES];
    uint32_t copyEventIdList[UB_STAGES];
    uint32_t ubListId{0};
    uint32_t tileBytes{0};
    CopyGmToUb copyGmToUb;
    CopyUbToGm copyUbToGm;
    AscendC::LocalTensor<ElementTopkIdx> ubMetaTopkIdx;
    AscendC::LocalTensor<ElementTopkW> ubMetaTopkWeight;
    AscendC::LocalTensor<ElementScale> ubMetaScales;
    AscendC::LocalTensor<ElementIdx> ubMetaSrcIdx;
    AscendC::LocalTensor<int32_t> ubInputSplitList;
    AscendC::LocalTensor<int32_t> ubInputOffsetList;
    AscendC::LocalTensor<int32_t> ubOutputOffsetList;
    AscendC::LocalTensor<int32_t> ubDstOrder;
    uint32_t metaRows{1};
    uint32_t metaByteSize{0};
    uint32_t topkIdxRowBytes{0};
    uint32_t topkWeightRowBytes{0};
    uint32_t minTransferRows{1};
    uint32_t activeDestinations{0};
    uint32_t directTaskCapacity{0};
};

} // namespace Catccos::Comm::Block

#endif // CATCCOS_COMM_BLOCK_COMM_BLOCK_DISPATCH_HPP
