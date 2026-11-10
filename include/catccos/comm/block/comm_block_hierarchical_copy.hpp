/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_COMM_BLOCK_HIERARCHICAL_COPY_HPP
#define CATCCOS_COMM_BLOCK_HIERARCHICAL_COPY_HPP

#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catccos::Comm::Block {

// Communication block for the hierarchical transport: one block is one communication round (commIdx)
// of one peer: an MTE row block or the entire current UDMA/RDMA range. There is no in-block tile scheduler.
template <
    class ArchTag_, uint32_t UB_STAGES_, bool IsDynamic_, class SrcType_, class DstType_, class BlockShape_,
    class TileRemoteCopy_>
class CommBlock<
    AtlasCommHierarchicalCopy<ArchTag_, UB_STAGES_, IsDynamic_>, SrcType_, DstType_, BlockShape_, TileRemoteCopy_> {
public:
    using DispatchPolicy = AtlasCommHierarchicalCopy<ArchTag_, UB_STAGES_, IsDynamic_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;
    using BlockShape = BlockShape_;
    using TileRemoteCopy = TileRemoteCopy_;
    using MatrixCoord = Catlass::MatrixCoord;
    static constexpr bool IsDynamic = IsDynamic_;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr uint32_t UB_STAGE_BYTES = TileRemoteCopy::UB_BYTES;
    static constexpr detail::CopyDirect RemoteCopyDirect = TileRemoteCopy::RemoteCopyDirect;
    static_assert(UB_STAGES > 0 && UB_STAGES <= 8, "Use 1 to 8 UB stages/event IDs");
    static_assert(UB_STAGE_BYTES % 32 == 0, "UB stages must be aligned");
    static_assert(UB_STAGES <= ArchTag::UB_SIZE / UB_STAGE_BYTES, "Communication scratch exceeds UB");

    template <bool IsDynamicParams_>
    struct ParamsBase {};

    template <>
    struct ParamsBase<false> {
        CATLASS_DEVICE
        static MatrixCoord BlockShape() { return BlockShape_::ToCoord(); }
    };

    template <>
    struct ParamsBase<true> {
        MatrixCoord blockShape;
        CATLASS_HOST_DEVICE
        ParamsBase() = default;
        CATLASS_HOST_DEVICE
        explicit ParamsBase(MatrixCoord shape) : blockShape(shape) {}
        CATLASS_DEVICE
        MatrixCoord BlockShape() const { return blockShape; }
    };

    using Params = ParamsBase<IsDynamic>;

    CATLASS_DEVICE
    CommBlock(Catlass::Arch::Resource<ArchTag>& resource, const Params& params) : params(params)
    {
        for (uint32_t stage = 0; stage < UB_STAGES; ++stage) {
            ubSrcList[stage] = resource.ubBuf.template GetBufferByByte<ElementSrc>(stage * UB_STAGE_BYTES);
        }
    }

    // Execute ONE communication round across the peer's QPs. The caller owns the commIdx loop.
    // gmSrc/gmDst already include outer window and block offsets; actualBlockShape defines this call's
    // transfer extent. This block neither selects fusion windows nor advances their base addresses.
    // Returns true when network requests were submitted: the caller must quiet UDMA peers before
    // consuming Get destinations, reusing Put sources or releasing its completion signal.
    // RDMA and MTE waits are still completed inside the tile.
    CATLASS_DEVICE
    bool operator()(
        const AscendC::GlobalTensor<ElementSrc>& gmSrc, const LayoutSrc& layoutSrc,
        const AscendC::GlobalTensor<ElementDst>& gmDst, const LayoutDst& layoutDst, const MatrixCoord& actualBlockShape,
        uint32_t peer, detail::CopyTransport transport, uint32_t qpNum = 1)
    {
        if (actualBlockShape.row() == 0 || actualBlockShape.column() == 0 || transport == detail::CopyTransport::None) {
            return false;
        }
        const bool submitted = tileRemoteCopy(
            gmDst, layoutDst, gmSrc, layoutSrc, actualBlockShape, ubSrcList[ubListId], ubListId, peer, transport,
            qpNum);
        ubListId = (ubListId + 1) % UB_STAGES;
        return submitted;
    }

    CATLASS_DEVICE void PutWithSignal(
        const AscendC::GlobalTensor<ElementSrc>& gmSrc, const AscendC::GlobalTensor<ElementDst>& gmDst,
        uint32_t elements, __gm__ uint64_t* signal, uint64_t value, uint32_t peer)
    {
        tileRemoteCopy.PutWithSignal(gmDst, gmSrc, elements, signal, value, ubSrcList[ubListId], ubListId, peer);
        ubListId = (ubListId + 1) % UB_STAGES;
    }

    // Only the owning AIV may submit and quiet this peer/QP. The caller supplies
    // the payload/ready ordering protocol and keeps GM buffers valid until quiet.
    CATLASS_DEVICE bool CopyRdmaNbi(
        const AscendC::GlobalTensor<ElementSrc>& gmSrc, const LayoutSrc& layoutSrc,
        const AscendC::GlobalTensor<ElementDst>& gmDst, const LayoutDst& layoutDst, const MatrixCoord& actualBlockShape,
        uint32_t peer, uint32_t qp = 0)
    {
        const bool submitted = tileRemoteCopy.CopyRdmaNbi(
            gmDst, layoutDst, gmSrc, layoutSrc, actualBlockShape, ubSrcList[ubListId], ubListId, peer, qp);
        ubListId = (ubListId + 1) % UB_STAGES;
        return submitted;
    }

    CATLASS_DEVICE void QuietRdma(uint32_t peer, uint32_t qp = 0)
    {
        tileRemoteCopy.QuietRdma(peer, ubSrcList[ubListId], ubListId, qp);
        ubListId = (ubListId + 1) % UB_STAGES;
    }

private:
    Params params;
    AscendC::LocalTensor<ElementSrc> ubSrcList[UB_STAGES];
    uint32_t ubListId = 0;
    TileRemoteCopy tileRemoteCopy;
};

} // namespace Catccos::Comm::Block
#endif
