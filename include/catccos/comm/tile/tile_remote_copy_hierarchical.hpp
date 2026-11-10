/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_TILE_REMOTE_COPY_HIERARCHICAL_HPP
#define CATCCOS_TILE_REMOTE_COPY_HIERARCHICAL_HPP

#include <type_traits>

#include "catccos/detail/remote_copy_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"

namespace Catccos::Comm::Tile {

// Latest-SHMEM QP APIs are isolated in this explicitly included tile module.
// Existing fixed-transport TileRemoteCopy specializations keep their original SDK requirements and behavior.
//
// One AIV owns every QP of a peer. The caller-provided communication range is partitioned into disjoint
// contiguous QP slices, with 32-byte boundaries relative to the span and the tail on the last QP.
// Each slice is split only at the 256 MiB request limit. MTE uses 32 KiB UB segments.
//
// Following the upstream tile/kernel split, UDMA only submits requests here: the NBI call has consumed
// the UB scratch on return, but the GM payload is still in flight, so the kernel must quiet this peer
// before consuming the destination, reusing either GM buffer, or publishing completion.
// RDMA keeps its in-tile quiet because it needs the tile's UB scratch and sync id.
template <
    class ArchTag, bool IsDynamic_, class SrcType_, class DstType_, class TileShape_, detail::CopyDirect Direction_>
struct TileHierarchicalRemoteCopy {
    using ElementSrc = typename SrcType_::Element;
    using ElementDst = typename DstType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using LayoutDst = typename DstType_::Layout;
    using TileShape = TileShape_;
    using CopyTransport = detail::CopyTransport;
    using MteCopy = TileRemoteCopy<ArchTag, IsDynamic_, SrcType_, DstType_, TileShape_, Direction_, CopyTransport::Mte>;
    using Params = typename MteCopy::Params;
    static constexpr bool IsDynamic = IsDynamic_;
    static constexpr detail::CopyDirect RemoteCopyDirect = Direction_;
    static constexpr uint32_t UB_BYTES = 32 * 1024;
    static constexpr uint32_t WQE_BYTES = 256;
    static constexpr uint32_t MAX_QP_NUM = 4;
    static constexpr uint64_t MAX_NETWORK_BYTES = 256ULL * 1024 * 1024;
    static_assert(std::is_same_v<ElementSrc, ElementDst>, "Source and destination types must match");
    static_assert(
        std::is_same_v<LayoutSrc, Catlass::layout::RowMajor> && std::is_same_v<LayoutDst, Catlass::layout::RowMajor>,
        "Hierarchical copy requires row-major layouts");
    static_assert(
        Catlass::SizeOfBits<ElementSrc>::value == sizeof(ElementSrc) * 8, "Packed sub-byte types are unsupported");
    static_assert(
        UB_BYTES >= MAX_QP_NUM * WQE_BYTES && WQE_BYTES >= ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE,
        "Insufficient WQE scratch");

    // dst/src already point at the first element of this communication round.
    // shape describes only this call's range, which may be a caller-selected fusion window.
    // QP slices and request-size chunks are transport subdivisions, not fusion window scheduling.
    // Returns true when network requests were submitted; UDMA callers must quiet this peer.
    CATLASS_DEVICE
    bool operator()(
        const AscendC::GlobalTensor<ElementDst>& dst, const LayoutDst& dstLayout,
        const AscendC::GlobalTensor<ElementSrc>& src, const LayoutSrc& srcLayout, const MatrixCoord& shape,
        const AscendC::LocalTensor<ElementSrc>& ub, uint32_t eventId, uint32_t peer, CopyTransport transport,
        uint32_t qpNum = 1)
    {
        if (shape.row() == 0 || shape.column() == 0) {
            return false;
        }
        switch (transport) {
            case CopyTransport::Mte:
                CopyMte(dst, dstLayout, src, srcLayout, shape, ub, eventId, peer);
                return false;
            case CopyTransport::Udma:
                return CopyNetwork<CopyTransport::Udma>(
                    dst, dstLayout, src, srcLayout, shape, ub, eventId, peer, qpNum);
            case CopyTransport::Rdma:
                return CopyNetwork<CopyTransport::Rdma>(
                    dst, dstLayout, src, srcLayout, shape, ub, eventId, peer, qpNum);
            default:
                return false;
        }
    }

    // Submit on exactly one caller-owned peer/QP without waiting for network completion.
    // Source/destination GM must remain valid until QuietRdma returns. NBI consumes
    // its WQE UB scratch before returning; later submissions may reuse that scratch.
    CATLASS_DEVICE bool CopyRdmaNbi(
        const AscendC::GlobalTensor<ElementDst>& dst, const LayoutDst& dstLayout,
        const AscendC::GlobalTensor<ElementSrc>& src, const LayoutSrc& srcLayout, const MatrixCoord& shape,
        const AscendC::LocalTensor<ElementSrc>& ub, uint32_t eventId, uint32_t peer, uint32_t qp = 0)
    {
        if (shape.row() == 0 || shape.column() == 0) {
            return false;
        }
        if (srcLayout.stride(0) == shape.column() && dstLayout.stride(0) == shape.column()) {
            CopyNetworkQp<CopyTransport::Rdma>(
                dst, src, 0, static_cast<uint32_t>(static_cast<uint64_t>(shape.row()) * shape.column()), ub, eventId,
                peer, qp);
        } else {
            for (uint32_t row = 0; row < shape.row(); ++row) {
                CopyNetworkQp<CopyTransport::Rdma>(
                    dst[static_cast<uint64_t>(row) * dstLayout.stride(0)],
                    src[static_cast<uint64_t>(row) * srcLayout.stride(0)], 0, shape.column(), ub, eventId, peer, qp);
            }
        }
        return true;
    }

    CATLASS_DEVICE void QuietRdma(
        uint32_t peer, const AscendC::LocalTensor<ElementSrc>& ub, uint32_t eventId, uint32_t qp = 0)
    {
        aclshmemx_roce_qp_quiet(
            peer, qp, reinterpret_cast<__ubuf__ uint8_t*>(ub.GetPhyAddr()) + qp * WQE_BYTES, eventId);
    }

    // Contiguous UDMA Put with a completion notification. Prior payloads must have
    // completed before this call; only the final WQE publishes the ready value.
    CATLASS_DEVICE void PutWithSignal(
        const AscendC::GlobalTensor<ElementDst>& dst, const AscendC::GlobalTensor<ElementSrc>& src, uint32_t elements,
        __gm__ uint64_t* signal, uint64_t value, const AscendC::LocalTensor<ElementSrc>& ub, uint32_t eventId,
        uint32_t peer)
    {
        static_assert(Direction_ == detail::CopyDirect::Put, "PutWithSignal requires Put direction");
        constexpr uint32_t maxElements = static_cast<uint32_t>(MAX_NETWORK_BYTES / sizeof(ElementSrc));
        uint32_t offset = 0;
        while (elements - offset > maxElements) {
            CopyNetworkQp<CopyTransport::Udma>(dst, src, offset, maxElements, ub, eventId, peer, 0);
            offset += maxElements;
        }
        if (offset != 0) {
            aclshmemx_udma_qp_quiet(static_cast<int>(peer), 0);
        }
        auto dstPtr = const_cast<__gm__ uint8_t*>(reinterpret_cast<const __gm__ uint8_t*>(dst.GetPhyAddr() + offset));
        auto srcPtr = const_cast<__gm__ uint8_t*>(reinterpret_cast<const __gm__ uint8_t*>(src.GetPhyAddr() + offset));
        aclshmemx_udma_qp_put_signal_nbi<uint8_t>(
            dstPtr, srcPtr, (elements - offset) * sizeof(ElementSrc), signal, value, static_cast<int>(peer), 0,
            reinterpret_cast<__ubuf__ uint8_t*>(ub.GetPhyAddr()), eventId);
    }

private:
    // MTE round copy: merge contiguous rows and split only at the 32 KiB UB boundary.
    CATLASS_DEVICE
    void CopyMte(
        const AscendC::GlobalTensor<ElementDst>& dst, const LayoutDst& dstLayout,
        const AscendC::GlobalTensor<ElementSrc>& src, const LayoutSrc& srcLayout, const MatrixCoord& shape,
        const AscendC::LocalTensor<ElementSrc>& ub, uint32_t eventId, uint32_t peer)
    {
        if (srcLayout.stride(0) == shape.column() && dstLayout.stride(0) == shape.column()) {
            CopyMteSpan(dst, src, static_cast<uint64_t>(shape.row()) * shape.column(), ub, eventId, peer);
            return;
        }
        // Copy logical columns only; row padding must not overwrite adjacent output regions.
        for (uint32_t row = 0; row < shape.row(); ++row) {
            CopyMteSpan(
                dst[static_cast<uint64_t>(row) * dstLayout.stride(0)],
                src[static_cast<uint64_t>(row) * srcLayout.stride(0)], shape.column(), ub, eventId, peer);
        }
    }

    CATLASS_DEVICE
    void CopyMteSpan(
        AscendC::GlobalTensor<ElementDst> dst, AscendC::GlobalTensor<ElementSrc> src, uint64_t elements,
        const AscendC::LocalTensor<ElementSrc>& ub, uint32_t eventId, uint32_t peer)
    {
        constexpr uint32_t maxElements = UB_BYTES / sizeof(ElementSrc);
        uint64_t offset = 0;
        while (offset < elements) {
            const uint32_t count =
                static_cast<uint32_t>(elements - offset < maxElements ? elements - offset : maxElements);
            const MatrixCoord shape{1U, count};
            const Catlass::layout::RowMajor layout(1, count);
            mteCopy_(dst[offset], layout, src[offset], layout, shape, ub, eventId, peer);
            // The existing MTE tile pairs MTE2_MTE3. Finish its store before UB reuse or returning.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            offset += count;
        }
    }

    // UDMA/RDMA copy: partition the current contiguous communication range across QPs. For padded layouts,
    // partition each logical row separately so padding remains untouched.
    // Returns true when at least one request was submitted.
    template <CopyTransport Transport>
    CATLASS_DEVICE bool CopyNetwork(
        const AscendC::GlobalTensor<ElementDst>& dst, const LayoutDst& dstLayout,
        const AscendC::GlobalTensor<ElementSrc>& src, const LayoutSrc& srcLayout, const MatrixCoord& shape,
        const AscendC::LocalTensor<ElementSrc>& ub, uint32_t eventId, uint32_t peer, uint32_t qpNum)
    {
        if (srcLayout.stride(0) == shape.column() && dstLayout.stride(0) == shape.column()) {
            CopyNetworkSpan<Transport>(
                dst, src, static_cast<uint32_t>(static_cast<uint64_t>(shape.row()) * shape.column()), ub, eventId, peer,
                qpNum);
        } else {
            for (uint32_t row = 0; row < shape.row(); ++row) {
                CopyNetworkSpan<Transport>(
                    dst[static_cast<uint64_t>(row) * dstLayout.stride(0)],
                    src[static_cast<uint64_t>(row) * srcLayout.stride(0)], shape.column(), ub, eventId, peer, qpNum);
            }
        }
        // RDMA keeps its completion wait here; UDMA completion is the kernel's responsibility.
        if constexpr (Transport == CopyTransport::Rdma) {
            for (uint32_t qp = 0; qp < qpNum; ++qp) {
                aclshmemx_roce_qp_quiet(
                    peer, qp, reinterpret_cast<__ubuf__ uint8_t*>(ub.GetPhyAddr()) + qp * WQE_BYTES, eventId);
            }
        }
        return true;
    }

    template <CopyTransport Transport>
    CATLASS_DEVICE void CopyNetworkSpan(
        AscendC::GlobalTensor<ElementDst> dst, AscendC::GlobalTensor<ElementSrc> src, uint32_t elements,
        const AscendC::LocalTensor<ElementSrc>& ub, uint32_t eventId, uint32_t peer, uint32_t qpNum)
    {
        // Keep boundaries element-aligned for generic element types, as well as 32-byte aligned
        // for FP8 data and scales. Small spans may leave some QPs empty.
        constexpr uint32_t alignElements = sizeof(ElementSrc) < 32 ? 32 / sizeof(ElementSrc) : 1;
        const uint64_t units = elements / alignElements;
        for (uint32_t qp = 0; qp < qpNum; ++qp) {
            const uint64_t begin = units * qp / qpNum * alignElements;
            const uint64_t end = qp + 1 == qpNum ? elements : units * (qp + 1) / qpNum * alignElements;
            CopyNetworkQp<Transport>(dst, src, begin, static_cast<uint32_t>(end - begin), ub, eventId, peer, qp);
        }
    }

    template <CopyTransport Transport>
    CATLASS_DEVICE void CopyNetworkQp(
        AscendC::GlobalTensor<ElementDst> dst, AscendC::GlobalTensor<ElementSrc> src, uint64_t offset,
        uint32_t elements, const AscendC::LocalTensor<ElementSrc>& ub, uint32_t eventId, uint32_t peer, uint32_t qp)
    {
        constexpr uint32_t maxElements = static_cast<uint32_t>(MAX_NETWORK_BYTES / sizeof(ElementSrc));
        auto scratch = reinterpret_cast<__ubuf__ uint8_t*>(ub.GetPhyAddr()) + qp * WQE_BYTES;
        uint32_t done = 0;
        while (done < elements) {
            const uint32_t count = elements - done < maxElements ? elements - done : maxElements;
            // GlobalTensor exposes a const view; SHMEM's Get/Put pointer overloads use mutable pointers.
            auto dstPtr =
                const_cast<__gm__ uint8_t*>(reinterpret_cast<const __gm__ uint8_t*>(dst.GetPhyAddr() + offset + done));
            auto srcPtr =
                const_cast<__gm__ uint8_t*>(reinterpret_cast<const __gm__ uint8_t*>(src.GetPhyAddr() + offset + done));
            const uint32_t bytes = count * static_cast<uint32_t>(sizeof(ElementSrc));
            if constexpr (Transport == CopyTransport::Udma) {
                if constexpr (Direction_ == detail::CopyDirect::Put) {
                    aclshmemx_udma_qp_put_nbi<uint8_t>(dstPtr, srcPtr, scratch, bytes, peer, qp, eventId);
                } else {
                    aclshmemx_udma_qp_get_nbi<uint8_t>(dstPtr, srcPtr, scratch, bytes, peer, qp, eventId);
                }
            } else {
                if constexpr (Direction_ == detail::CopyDirect::Put) {
                    aclshmemx_roce_qp_put_nbi<uint8_t>(dstPtr, srcPtr, scratch, bytes, peer, qp, eventId);
                } else {
                    aclshmemx_roce_qp_get_nbi<uint8_t>(dstPtr, srcPtr, scratch, bytes, peer, qp, eventId);
                }
            }
            done += count;
        }
    }

    MteCopy mteCopy_;
};

} // namespace Catccos::Comm::Tile
#endif
