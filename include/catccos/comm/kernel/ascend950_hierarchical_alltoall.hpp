/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_COMM_KERNEL_ASCEND950_HIERARCHICAL_ALLTOALL_HPP
#define CATCCOS_COMM_KERNEL_ASCEND950_HIERARCHICAL_ALLTOALL_HPP

#include <cstddef>
#include <cstdint>

#include "catccos/catccos.hpp"
#include "catccos/comm/tile/tile_hierarchical_ready.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catccos::Comm::Kernel {

/// Uniform, dense peer-to-peer AllToAll for Ascend950.
///
/// Source rank r sends input[p] to rank p's output[r]. Every AIV owns one peer and uses that
/// peer's QPs. MTE walks row blocks (commIdx); UDMA/RDMA send each whole peer tensor
/// in one CommBlock call, partitioned across QPs inside the block without adding AIVs.
/// The self peer is copied with MTE; no transport fallback is performed.
///
/// One full tensor is one ready epoch in this functional example. AIV 0 also coordinates
/// local receive readiness; AIC 0 exercises the consumer wait without doing computation.
template <class BlockComm_, class BlockCommScale_, class BlockCommScheduler_>
class HierarchicalAllToAll {
public:
    using BlockComm = BlockComm_;
    using BlockCommScale = BlockCommScale_;
    using BlockCommScheduler = BlockCommScheduler_;
    using ArchTag = typename BlockComm::ArchTag;
    using Element = typename BlockComm::ElementSrc;
    using ElementScale = typename BlockCommScale::ElementSrc;
    using MatrixCoord = Catlass::MatrixCoord;
    using Ready = Tile::TileHierarchicalReady;

    template <class CommBlock>
    using TensorElement = typename CommBlock::ElementSrc;
    template <class CommBlock>
    using TensorLayout = typename CommBlock::LayoutSrc;

    struct Arguments {
        uint32_t m;         // rows per peer (one row block of the AllToAll matrix)
        uint32_t n;         // columns per peer
        uint32_t rankIdx;   // this rank
        uint32_t rankSize;  // world size
        GM_ADDR ptrInput;   // input[peer] base
        GM_ADDR ptrOutput;  // output[rank] base
        uint32_t qpNum = 1; // QPs per network peer; must match SHMEM initialization (1, 2 or 4)
        GM_ADDR ptrInputScale;
        GM_ADDR ptrOutputScale;
        uint32_t epoch = 1; // increase when reusing the same ready workspace across launches
    };

    struct Params {
        uint32_t m;
        uint32_t n;
        uint32_t rankIdx;
        uint32_t rankSize;
        __gm__ Element* ptrInput;
        __gm__ Element* ptrOutput;
        uint64_t peerStrideBytes;
        uint32_t qpNum = 1;
        uint32_t scaleN;
        __gm__ ElementScale* ptrInputScale;
        __gm__ ElementScale* ptrOutputScale;
        uint64_t scalePeerStrideBytes;
        GM_ADDR ptrWorkspace;
        uint32_t epoch;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(Arguments const& args, uint8_t* workspace)
            : m(args.m),
              n(args.n),
              rankIdx(args.rankIdx),
              rankSize(args.rankSize),
              ptrInput(reinterpret_cast<__gm__ Element*>(args.ptrInput)),
              ptrOutput(reinterpret_cast<__gm__ Element*>(args.ptrOutput)),
              peerStrideBytes(GetPeerStrideBytes(static_cast<uint64_t>(args.m) * args.n * sizeof(Element))),
              qpNum(args.qpNum),
              scaleN(args.n / 32),
              ptrInputScale(reinterpret_cast<__gm__ ElementScale*>(args.ptrInputScale)),
              ptrOutputScale(reinterpret_cast<__gm__ ElementScale*>(args.ptrOutputScale)),
              scalePeerStrideBytes(GetPeerStrideBytes(static_cast<uint64_t>(args.m) * scaleN * sizeof(ElementScale))),
              ptrWorkspace(reinterpret_cast<GM_ADDR>(reinterpret_cast<uintptr_t>(workspace))),
              epoch(args.epoch)
        {}

    private:
        // Positive m/n and n divisible by 32 are checked by the host.
        CATLASS_HOST_DEVICE
        static uint64_t GetPeerStrideBytes(uint64_t bytes) { return (bytes + 31) / 32 * 32; }
    };

    static size_t GetWorkspaceSize(Arguments const& args) { return Ready::GetWorkspaceSize(args.rankSize); }

    // workspace must be a zero-initialized symmetric allocation on every rank.
    static Params ToUnderlyingArguments(Arguments const& args, uint8_t* workspace) { return Params{args, workspace}; }

    CATLASS_DEVICE
    HierarchicalAllToAll() = default;

    CATLASS_DEVICE
    ~HierarchicalAllToAll() = default;

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    // Exercise the same frontier wait that a fused AIC consumer uses before GM loads.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        if (AscendC::GetBlockIdx() == 0) {
            Ready ready(params.ptrWorkspace, params.rankIdx, params.rankSize);
            ready.WaitReady(params.epoch);
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        // 1:2 MIX launch: each AIC has two AIV subcores. GetBlockIdx() is the physical AIV
        // index (0 .. 2*blockDim-1); the scheduler skips AIVs without a peer.
        const uint32_t aicoreIdx = AscendC::GetBlockIdx();
        BlockCommScheduler commScheduler({params.m, params.n, params.rankSize}, params.rankIdx, aicoreIdx);
        if (!commScheduler.IsActiveCore() || commScheduler.GetCoreLoop() == 0) {
            return;
        }

        Catlass::Arch::Resource<ArchTag> resource;
        // The two blocks reuse the same UB and event IDs sequentially. MTE/RDMA
        // complete in the block; UDMA NBI consumes its scratch before returning.
        CopyTensor<BlockComm>(
            resource, commScheduler, params, params.n, params.ptrInput, params.ptrOutput, params.peerStrideBytes);
        BlockCommScheduler scaleScheduler({params.m, params.scaleN, params.rankSize}, params.rankIdx, aicoreIdx);
        CopyTensor<BlockCommScale>(
            resource, scaleScheduler, params, params.scaleN, params.ptrInputScale, params.ptrOutputScale,
            params.scalePeerStrideBytes);
        auto ub = resource.ubBuf.template GetBufferByByte<uint32_t>(0);
        Ready ready(params.ptrWorkspace, params.rankIdx, params.rankSize);
        ready.Notify(
            commScheduler.GetPeer(), commScheduler.GetTransport(), params.epoch, params.qpNum,
            reinterpret_cast<__ubuf__ uint32_t*>(ub.GetPhyAddr()), 0);
        // Send first, then wait: no extra AIV is required, and peer 0's QPs are
        // already idle before AIV 0 becomes this rank's receive coordinator.
        if (aicoreIdx == 0) {
            ready.WaitAll(params.epoch);
        }
    }

private:
    template <class CommBlock>
    CATLASS_DEVICE void CopyTensor(
        Catlass::Arch::Resource<ArchTag>& resource, const BlockCommScheduler& commScheduler, const Params& params,
        uint32_t n, __gm__ TensorElement<CommBlock>* ptrSrc, __gm__ TensorElement<CommBlock>* ptrDst,
        uint64_t peerStrideBytes)
    {
        typename CommBlock::Params blockParams{};
        if constexpr (CommBlock::IsDynamic) {
            blockParams.blockShape = commScheduler.GetActualBlockShape({0U, 0U});
        }
        CommBlock blockCopy(resource, blockParams);
        AscendC::GlobalTensor<TensorElement<CommBlock>> gmSrc;
        AscendC::GlobalTensor<TensorElement<CommBlock>> gmDst;
        const uint64_t peerStride = peerStrideBytes / sizeof(TensorElement<CommBlock>);
        gmSrc.SetGlobalBuffer(ptrSrc + static_cast<uint64_t>(commScheduler.GetPeer()) * peerStride);
        gmDst.SetGlobalBuffer(ptrDst + static_cast<uint64_t>(params.rankIdx) * peerStride);
        const TensorLayout<CommBlock> layoutSrc(params.m, n);
        const TensorLayout<CommBlock> layoutDst(params.m, n);
        for (uint32_t commIdx = 0; commIdx < commScheduler.GetCoreLoop(); ++commIdx) {
            const auto commBlockCoord = commScheduler.GetBlockCoord(commIdx);
            const uint32_t remoteRankIdx = commBlockCoord.rank();
            const MatrixCoord commBlockCoordInRank{commBlockCoord.row(), commBlockCoord.column()};
            const auto blockOffsetSrc = commScheduler.GetBlockOffsetInRank(commBlockCoordInRank);
            const auto blockOffsetDst = blockOffsetSrc;
            const auto actualCommBlockShape = commScheduler.GetActualBlockShape(commBlockCoordInRank);

            const auto gmBlockSrc = gmSrc[layoutSrc.GetOffset(blockOffsetSrc)];
            const auto layoutBlockSrc = layoutSrc.GetTileLayout(actualCommBlockShape);
            const auto gmBlockDst = gmDst[layoutDst.GetOffset(blockOffsetDst)];
            const auto layoutBlockDst = layoutDst.GetTileLayout(actualCommBlockShape);

            blockCopy(
                gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst, actualCommBlockShape, remoteRankIdx,
                commScheduler.GetTransport(), params.qpNum);
        }
    }
};

} // namespace Catccos::Comm::Kernel
#endif
