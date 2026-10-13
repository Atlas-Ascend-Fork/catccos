/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_KERNEL_ASCEND950_DISPATCH_DIRECT_HPP
#define CATCCOS_COMM_KERNEL_ASCEND950_DISPATCH_DIRECT_HPP

#include <cstddef>
#include <type_traits>

#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catccos/catccos.hpp"
#include "catccos/comm/block/block_quantize.hpp"
#include "catccos/comm/block/comm_block_dispatch.hpp"
#include "shmem.h"

namespace Catccos::Comm::Kernel {

/**
 * Quantize each input token once, then send selected rows through UB directly
 * to the final receive region. There is no destination-major GM staging and
 * no repack-to-AllToAllV publication barrier.
 *
 * The block assigns destination-local row chunks to AIV workers in group-aware
 * order. rankPrefix and the token's ordinal within its destination determine
 * the receive row, preserving the repack kernel's output contract.
 * Kernel selection belongs to the caller, not to the device algorithm.
 */
template <class BlockDispatch_, class BlockQuantize_ = Block::BlockQuantizeBf16Int8>
class Ascend950DispatchDirect {
public:
    using BlockDispatch = BlockDispatch_;
    using ProblemShape = typename BlockDispatch::ProblemShape;
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

    struct Arguments {
        Catlass::GemmCoord gemmShape;
        Catlass::MatrixCoord commBlockShape;
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
        Catlass::Arch::Resource<ArchTag> resource;
        BlockQuantize quantize;
        quantize(
            resource, reinterpret_cast<GM_ADDR>(params.ptrX), params.ptrQuantX, params.ptrQuantScales, problemShape.m(),
            problemShape.k(), AscendC::GetBlockIdx(), AscendC::GetBlockNum() * AscendC::GetSubBlockNum());
        // Quantized rows are consumed only by AIV workers on this rank.
        // Complete their GM writes and synchronize local AIV workers before sending.
        aclshmem_fence();
        AscendC::SyncAll<true>();

        BlockDispatch blockDispatch(resource, params.blockDispatchParams);
        blockDispatch.InitRouting(problemShape);
        AscendC::GlobalTensor<uint8_t> gmX, gmIsTokenInRank;
        AscendC::GlobalTensor<ElementTopkIdx> gmTopkIdx;
        AscendC::GlobalTensor<ElementTopkW> gmTopkWeight;
        gmX.SetGlobalBuffer(params.ptrQuantX);
        gmIsTokenInRank.SetGlobalBuffer(params.ptrIsTokenInRank);
        if (params.numTopk > 0) {
            gmTopkIdx.SetGlobalBuffer(params.ptrTopkIdx);
        }
        if (params.hasTopkWeights) {
            gmTopkWeight.SetGlobalBuffer(params.ptrTopkWeights);
        }
        blockDispatch.InitBlockLoop();
        blockDispatch.SendDirect(problemShape, gmX, gmTopkIdx, gmTopkWeight, gmIsTokenInRank);
        blockDispatch.FinalizeBlockLoop();
        // Every source has a disjoint receive interval. One final collective
        // publishes X and all metadata, including ranks with no outgoing rows.
        aclshmem_fence();
        aclshmemx_barrier_all_vec();

        const uint32_t actualRecvTokens = problemShape.rankPrefix(problemShape.rankSize() - 1, problemShape.rankIdx());
        const uint32_t aicoreIdx = AscendC::GetBlockIdx();
        const uint32_t aicoreNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        blockDispatch.Complete(actualRecvTokens, aicoreIdx, aicoreNum);
    }

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

#endif // CATCCOS_COMM_KERNEL_ASCEND950_DISPATCH_DIRECT_HPP
