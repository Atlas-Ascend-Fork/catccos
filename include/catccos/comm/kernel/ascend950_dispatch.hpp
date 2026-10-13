/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_KERNEL_ASCEND950_DISPATCH_HPP
#define CATCCOS_COMM_KERNEL_ASCEND950_DISPATCH_HPP

#include <cstddef>
#include <type_traits>

#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catccos/catccos.hpp"
#include "catccos/comm/block/comm_block_dispatch.hpp"
#include "catccos/comm/block/comm_block_scheduler_dispatch.hpp"
#include "catccos/comm/block/block_quantize.hpp"
#include "shmem.h"

namespace Catccos::Comm::Kernel {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

/**
 * MoE Dispatch as a local repack followed by an AllToAllV.
 *
 * This path separates local repacking from remote copies, in the same shape as
 * the other Ascend950 AllToAllV kernels: the kernel owns the round loop, the
 * scheduler and the barrier, while the block owns the UB tiles and every copy.
 * Ascend950DispatchDirect provides the alternative without GM staging.
 *
 *   RepackToStage gathers each destination's tokens into this rank's own
 *   symmetric heap, destination-major, so the run owned by destination `d`
 *   starts at the prefix sum of numTokensPerRank -- the inputSplitList /
 *   inputOffsetList contract the AllToAllV scheduler expects.
 *
 *   The commIdx loop then walks the AllToAllV round and block schedule and moves
 *   each run to its destination with one contiguous transfer per block.
 *
 * The compact receive layout keeps its meaning (the concatenation over sources
 * of those runs) but loses the channel dimension: a source's rows are contiguous
 * on the destination, so channelPrefixMatrix and the recvChannelPrefixMatrix
 * output are gone.
 *
 * BF16 input is always quantized before repacking. Local AIV syncs publish the
 * quantized input to repack workers and the repacked staging to senders. The
 * final rank-wide barrier publishes every remote write before a rank reads its
 * own receive region. The caller must finish consuming the previous receive
 * region before reusing it across invocations. No per-round completion handshake
 * is needed because dispatch has no in-kernel consumer during communication.
 */
template <
    class BlockDispatch_, class BlockCommScheduler_ = Block::BlockCommSchedulerDispatch,
    class BlockQuantize_ = Block::BlockQuantizeBf16Int8>
class Ascend950Dispatch {
public:
    using BlockDispatch = BlockDispatch_;
    using BlockCommScheduler = BlockCommScheduler_;
    using ProblemShape = typename BlockCommScheduler::ProblemShape;
    using ArchTag = typename BlockDispatch::ArchTag;
    using ElementToken = typename BlockDispatch::ElementToken;
    using ElementScale = typename BlockDispatch::ElementScale;
    using ElementTopkIdx = typename BlockDispatch::ElementTopkIdx;
    using ElementTopkW = typename BlockDispatch::ElementTopkW;
    using ElementIdx = typename BlockDispatch::ElementIdx;
    using BlockQuantize = BlockQuantize_;
    using InputElement = bfloat16_t;
    static_assert(std::is_same_v<ElementToken, int8_t>, "Dispatch quantization requires INT8 output");
    static_assert(std::is_same_v<ElementScale, float>, "BF16 quantization requires FP32 scales");
    using BlockDispatchParams = typename BlockDispatch::Params;

    static constexpr uint32_t MAX_RANKS = BlockDispatch::MAX_RANKS;
    static constexpr uint32_t MAX_TOPK = BlockDispatch::MAX_TOPK;

    struct Params {
        ProblemShape problemShape;
        uint32_t commInterval{1};
        uint32_t numTopk{0};
        uint32_t numExperts{0};
        uint32_t numOutputTokens{0};
        bool hasTopkWeights{false};

        __gm__ InputElement* ptrX{nullptr};
        GM_ADDR ptrQuantX{nullptr};
        GM_ADDR ptrQuantScales{nullptr};
        __gm__ ElementTopkIdx* ptrTopkIdx{nullptr};
        __gm__ ElementTopkW* ptrTopkWeights{nullptr};
        __gm__ uint8_t* ptrIsTokenInRank{nullptr};

        BlockDispatchParams blockDispatchParams{};
    };

    /// User-facing arguments, kept type-compatible with the DeepEP dispatch ABI
    /// except for the dropped channel prefix inputs and handle outputs.
    struct Arguments {
        GemmCoord gemmShape;
        MatrixCoord commBlockShape;
        uint32_t rankIdx{0};
        uint32_t rankSize{0};
        uint32_t groupSize{1};
        uint32_t commInterval{1};
        uint32_t numTopk{0};
        uint32_t numExperts{0};
        uint32_t numOutputTokens{0};
        bool hasTopkWeights{false};

        GM_ADDR ptrX{nullptr};
        GM_ADDR ptrTopkIdx{nullptr};
        GM_ADDR ptrTopkWeights{nullptr};
        GM_ADDR ptrNumTokensPerRank{nullptr};
        GM_ADDR ptrIsTokenInRank{nullptr};
        GM_ADDR ptrRankPrefixMatrix{nullptr};
        GM_ADDR ptrSymmetric{nullptr};
        uint64_t symmetricRecvXOffset{0};
        uint64_t symmetricRecvTopkIdxOffset{0};
        uint64_t symmetricRecvTopkWeightsOffset{0};
        uint64_t symmetricRecvSrcIdxOffset{0};
        uint64_t symmetricStageXOffset{0};
        uint64_t symmetricStageTopkIdxOffset{0};
        uint64_t symmetricStageTopkWeightsOffset{0};
        uint64_t symmetricStageSrcIdxOffset{0};
        GM_ADDR ptrRecvX{nullptr};
        GM_ADDR ptrRecvTopkIdx{nullptr};
        GM_ADDR ptrRecvTopkWeights{nullptr};
        GM_ADDR ptrRecvSrcIdx{nullptr};
        GM_ADDR ptrQuantX{nullptr};
        GM_ADDR ptrQuantScales{nullptr};
        uint64_t symmetricStageScalesOffset{0};
        uint64_t symmetricRecvScalesOffset{0};
        GM_ADDR ptrRecvScales{nullptr};
    };

    CATLASS_HOST_DEVICE
    static Params ToUnderlyingArguments(Arguments const& args, uint8_t* workspace = nullptr)
    {
        (void)workspace;
        Params params{};
        params.problemShape = ProblemShape{
            args.gemmShape, args.rankSize, args.rankIdx, args.ptrNumTokensPerRank, args.ptrRankPrefixMatrix};
        params.commInterval = args.commInterval;
        params.numTopk = args.numTopk;
        params.numExperts = args.numExperts;
        params.numOutputTokens = args.numOutputTokens;
        params.hasTopkWeights = args.hasTopkWeights;
        params.ptrX = reinterpret_cast<__gm__ InputElement*>(args.ptrX);
        params.ptrQuantX = args.ptrQuantX;
        params.ptrQuantScales = args.ptrQuantScales;
        params.ptrTopkIdx = reinterpret_cast<__gm__ ElementTopkIdx*>(args.ptrTopkIdx);
        params.ptrTopkWeights = reinterpret_cast<__gm__ ElementTopkW*>(args.ptrTopkWeights);
        params.ptrIsTokenInRank = reinterpret_cast<__gm__ uint8_t*>(args.ptrIsTokenInRank);

        auto& block = params.blockDispatchParams;
        block.rankIdx = args.rankIdx;
        block.rankSize = args.rankSize;
        block.groupSize = args.groupSize;
        block.numTokens = args.gemmShape.m();
        block.hidden = args.gemmShape.k();
        block.numTopk = args.numTopk;
        block.numExperts = args.numExperts;
        block.hasTopkWeights = args.hasTopkWeights;
        block.ptrQuantScales = reinterpret_cast<__gm__ ElementScale*>(args.ptrQuantScales);
        block.stageScalesOffset = args.symmetricStageScalesOffset;
        block.recvScalesOffset = args.symmetricRecvScalesOffset;
        block.ptrRecvScales = reinterpret_cast<__gm__ ElementScale*>(args.ptrRecvScales);
        block.commBlockShape = args.commBlockShape;
        block.ptrSymmetric = args.ptrSymmetric;
        block.recvXOffset = args.symmetricRecvXOffset;
        block.recvTopkIdxOffset = args.symmetricRecvTopkIdxOffset;
        block.recvTopkWeightsOffset = args.symmetricRecvTopkWeightsOffset;
        block.recvSrcIdxOffset = args.symmetricRecvSrcIdxOffset;
        block.stageXOffset = args.symmetricStageXOffset;
        block.stageTopkIdxOffset = args.symmetricStageTopkIdxOffset;
        block.stageTopkWeightsOffset = args.symmetricStageTopkWeightsOffset;
        block.stageSrcIdxOffset = args.symmetricStageSrcIdxOffset;
        block.ptrRecvX = reinterpret_cast<__gm__ ElementToken*>(args.ptrRecvX);
        block.ptrRecvTopkIdx = reinterpret_cast<__gm__ ElementTopkIdx*>(args.ptrRecvTopkIdx);
        block.ptrRecvTopkWeights = reinterpret_cast<__gm__ ElementTopkW*>(args.ptrRecvTopkWeights);
        block.ptrRecvSrcIdx = reinterpret_cast<__gm__ ElementIdx*>(args.ptrRecvSrcIdx);
        return params;
    }

    CATLASS_HOST_DEVICE
    static size_t GetWorkspaceSize(Arguments const&) { return 0; }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const&)
    {}

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        if (!CanImplement(params)) {
            return;
        }

        auto const& problemShape = params.problemShape;
        AscendC::GlobalTensor<uint8_t> gmX;
        AscendC::GlobalTensor<ElementTopkIdx> gmTopkIdx;
        AscendC::GlobalTensor<ElementTopkW> gmTopkWeight;
        AscendC::GlobalTensor<uint8_t> gmIsTokenInRank;
        gmX.SetGlobalBuffer(params.ptrQuantX);
        if (params.numTopk > 0) {
            gmTopkIdx.SetGlobalBuffer(params.ptrTopkIdx);
        }
        if (params.hasTopkWeights) {
            gmTopkWeight.SetGlobalBuffer(params.ptrTopkWeights);
        }
        gmIsTokenInRank.SetGlobalBuffer(params.ptrIsTokenInRank);

        Catlass::Arch::Resource<ArchTag> resource;
        BlockQuantize quantize;
        const uint32_t aicoreNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        quantize(
            resource, reinterpret_cast<GM_ADDR>(params.ptrX), params.ptrQuantX, params.ptrQuantScales, problemShape.m(),
            problemShape.k(), AscendC::GetBlockIdx(), aicoreNum);
        // Quantized rows are consumed only by repack workers on this rank.
        aclshmem_fence();
        AscendC::SyncAll<true>();
        BlockDispatch blockDispatch(resource, params.blockDispatchParams);
        auto routing = blockDispatch.InitRouting(problemShape);

        // Phase 1: local repack, consumed only by this rank's AllToAllV workers.
        blockDispatch.InitBlockLoop();
        blockDispatch.RepackToStage(problemShape, gmX, gmTopkIdx, gmTopkWeight, gmIsTokenInRank);
        blockDispatch.FinalizeBlockLoop();
        aclshmem_fence();
        AscendC::SyncAll<true>();

        // Phase 2: AllToAllV rounds.
        MatrixCoord blockShape = params.blockDispatchParams.BlockShape();
        // Unlike the per-task block, the round window holds enough row blocks
        // per destination to expose work to every AIV when the split permits.
        const uint32_t blocksPerRank = CeilDiv(aicoreNum, problemShape.rankSize());
        const uint32_t maxBlocksPerRank = CeilDiv(problemShape.m(), blockShape.row());
        const uint32_t commBlocksPerRank = static_cast<uint32_t>(
            Min<uint64_t>(static_cast<uint64_t>(params.commInterval) * blocksPerRank, maxBlocksPerRank));
        const uint32_t commShapeM = static_cast<uint32_t>(
            Min<uint64_t>(static_cast<uint64_t>(commBlocksPerRank) * blockShape.row(), problemShape.m()));
        BlockCommScheduler scheduler{problemShape, commShapeM, blockShape, routing};
        blockDispatch.InitBlockLoop();
        auto commLoops = scheduler.GetCommLoops();
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            auto remapperSrc = scheduler.GetRemapperSrc(commIdx);
            auto remapperDst = scheduler.GetRemapperDst(commIdx);
            for (auto iter = scheduler.Begin(commIdx); !iter.End(); iter.Next()) {
                auto blockOffset = scheduler.GetBlockOffset(iter);
                auto blockOffsetSrc = remapperSrc(blockOffset);
                auto blockOffsetDst = remapperDst(blockOffset);
                auto actualBlockShape = scheduler.RemapActualBlockShape(blockOffset, remapperSrc);
                if (Numel(actualBlockShape) == 0) {
                    continue;
                }
                const auto& commParams = params.blockDispatchParams;
                const uint32_t srcRow = blockOffsetSrc.row();
                const uint32_t dstRow = blockOffsetDst.row();
                const uint32_t rows = actualBlockShape.row();
                const uint32_t dstRank = blockOffset.remote();
                BlockCopy<ElementToken>(
                    blockDispatch, commParams.ptrSymmetric, commParams.stageXOffset, commParams.recvXOffset, srcRow,
                    dstRow, rows, problemShape.k(), dstRank);
                BlockCopy<ElementScale>(
                    blockDispatch, commParams.ptrSymmetric, commParams.stageScalesOffset, commParams.recvScalesOffset,
                    srcRow, dstRow, rows, 1, dstRank);
                if (params.numTopk > 0) {
                    BlockCopy<ElementTopkIdx>(
                        blockDispatch, commParams.ptrSymmetric, commParams.stageTopkIdxOffset,
                        commParams.recvTopkIdxOffset, srcRow, dstRow, rows, params.numTopk, dstRank);
                    if (params.hasTopkWeights) {
                        BlockCopy<ElementTopkW>(
                            blockDispatch, commParams.ptrSymmetric, commParams.stageTopkWeightsOffset,
                            commParams.recvTopkWeightsOffset, srcRow, dstRow, rows, params.numTopk, dstRank);
                    }
                }
                BlockCopy<ElementIdx>(
                    blockDispatch, commParams.ptrSymmetric, commParams.stageSrcIdxOffset, commParams.recvSrcIdxOffset,
                    srcRow, dstRow, rows, 1, dstRank);
            }
        }
        blockDispatch.FinalizeBlockLoop();

        // Publish every remote write before any rank reads its receive region.
        aclshmem_fence();
        aclshmemx_barrier_all_vec();

        const uint32_t actualRecvTokens = problemShape.rankPrefix(problemShape.rankSize() - 1, problemShape.rankIdx());
        const uint32_t aicoreIdx = AscendC::GetBlockIdx();
        blockDispatch.Complete(actualRecvTokens, aicoreIdx, aicoreNum);
    }

private:
    /// Bind a field's source staging run and destination receive run.
    template <class Element>
    CATLASS_DEVICE static void BlockCopy(
        BlockDispatch& blockDispatch, GM_ADDR symmetric, uint64_t stageOffset, uint64_t recvOffset, uint32_t srcRow,
        uint32_t dstRow, uint32_t rows, uint32_t columns, uint32_t dstRank)
    {
        AscendC::GlobalTensor<Element> gmSrc, gmDst;
        gmSrc.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(
            symmetric + stageOffset + static_cast<uint64_t>(srcRow) * columns * sizeof(Element)));
        gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(
            symmetric + recvOffset + static_cast<uint64_t>(dstRow) * columns * sizeof(Element)));
        Catlass::layout::RowMajor layoutSrc(rows, columns);
        Catlass::layout::RowMajor layoutDst(rows, columns);
        blockDispatch(gmSrc, layoutSrc, gmDst, layoutDst, MatrixCoord{rows, columns}, dstRank);
    }

public:
    CATLASS_DEVICE
    static bool CanImplement(Params const& params)
    {
        const uint32_t rankSize = params.problemShape.rankSize();
        if (!BlockQuantize::CanImplement(params.problemShape.k()) || params.ptrQuantX == nullptr ||
            params.ptrQuantScales == nullptr || params.blockDispatchParams.ptrRecvScales == nullptr) {
            return false;
        }
        return rankSize > 0 && rankSize <= MAX_RANKS && params.commInterval > 0 && params.problemShape.m() > 0 &&
               params.problemShape.k() > 0 && params.numTopk <= MAX_TOPK &&
               (!params.hasTopkWeights || params.numTopk > 0) &&
               (params.numTopk == 0 || (params.numExperts > 0 && params.numExperts % rankSize == 0));
    }
};

} // namespace Catccos::Comm::Kernel

#endif // CATCCOS_COMM_KERNEL_ASCEND950_DISPATCH_HPP
