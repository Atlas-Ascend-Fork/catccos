/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_COMM_BLOCK_HIERARCHICAL_SWIZZLE_HPP
#define CATCCOS_COMM_BLOCK_HIERARCHICAL_SWIZZLE_HPP

#include <cstdint>
#include "catccos/dist_coord.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catccos::Comm::Block {

// Consecutive ranks form a UDMA domain; the launcher may override GROUP_SIZE.
// One AIV owns one peer; the tile partitions
// the current peer tensor across that peer's QPs. The self peer is copied in blocks with MTE;
// every other same-group peer uses UDMA and every peer in another group uses RDMA.
// There is no automatic fallback: the host verifies the transport routes
// before launching the kernel.
template <uint32_t CoreCount_ = 32, uint32_t GroupSize_ = 16>
class HierarchicalCommSwizzle {
public:
    static_assert(CoreCount_ > 0, "Core count must be positive");
    static constexpr uint32_t CORE_COUNT = CoreCount_;
    static constexpr uint32_t GROUP_SIZE = GroupSize_;
    static_assert(GROUP_SIZE > 0 && GROUP_SIZE <= CORE_COUNT, "Group size must be in [1, CORE_COUNT]");
    using CopyTransport = detail::CopyTransport;
    using MatrixCoord = Catlass::MatrixCoord;

    // problemShape = {rows per peer, columns per peer, world size}.
    // Rows and columns describe the current communication range: a complete peer tensor or
    // a caller-selected fusion window. The caller owns the outer window loop and its base addresses.
    // aicoreIdx is the physical AIV index; with one AIV per peer it equals the peer rank.
    // commBlockM is the number of rows moved by one MTE round (commIdx).
    // UDMA/RDMA schedule the entire current range in one round, regardless of commBlockM.
    // groupSize is the number of participating ranks per UDMA domain.
    CATLASS_HOST_DEVICE
    HierarchicalCommSwizzle(
        DistMatrixCoord problemShape, uint32_t rank, uint32_t aicoreIdx, uint32_t commBlockM,
        uint32_t groupSize = GROUP_SIZE)
        : problemShape(problemShape),
          rank(rank),
          aicoreIdx(aicoreIdx),
          commBlockM(commBlockM),
          peer(aicoreIdx),
          groupSize(groupSize)
    {
        const bool ownsPeer = IsValidProblem() && peer < problemShape.rank() && peer < CORE_COUNT;
        if (!ownsPeer) {
            transport = CopyTransport::None;
        } else if (peer == rank) {
            transport = CopyTransport::Mte;
        } else if (peer / groupSize == rank / groupSize) {
            transport = CopyTransport::Udma;
        } else {
            transport = CopyTransport::Rdma;
        }
        const bool hasWork = transport != CopyTransport::None && problemShape.column() != 0 &&
                             problemShape.row() != 0 && commBlockM != 0;
        if (!hasWork) {
            coreLoops = 0;
        } else if (transport == CopyTransport::Mte) {
            coreLoops = problemShape.row() / commBlockM + (problemShape.row() % commBlockM != 0);
        } else {
            coreLoops = 1;
        }
    }

    CATLASS_HOST_DEVICE
    bool IsValidProblem() const
    {
        return problemShape.rank() > 0 && problemShape.rank() <= CORE_COUNT && rank < problemShape.rank() &&
               commBlockM > 0 && groupSize > 0 && groupSize <= problemShape.rank() &&
               problemShape.rank() % groupSize == 0;
    }

    CATLASS_HOST_DEVICE
    bool IsActiveCore() const
    {
        return IsValidProblem() && aicoreIdx < problemShape.rank() && peer < problemShape.rank() && peer < CORE_COUNT &&
               transport != CopyTransport::None;
    }

    CATLASS_HOST_DEVICE
    CopyTransport GetTransport() const { return transport; }

    CATLASS_HOST_DEVICE
    uint32_t GetPeer() const { return peer; }

    // First QP of this peer context; additional QPs are managed by the tile.
    CATLASS_HOST_DEVICE
    uint32_t GetQpIndex() const { return 0; }

    CATLASS_HOST_DEVICE
    uint32_t GetCommBlockM() const { return commBlockM; }

    CATLASS_HOST_DEVICE
    uint32_t GetGroupSize() const { return groupSize; }

    CATLASS_HOST_DEVICE
    uint32_t GetAicoreIdx() const { return aicoreIdx; }

    CATLASS_HOST_DEVICE
    uint32_t GetRealCore() const { return CORE_COUNT; }

    // Number of communication rounds (commIdx) for this AIV's current problemShape, not fusion windows.
    // For valid nonempty work: ceil(rows / commBlockM) for MTE, one for UDMA/RDMA; otherwise zero.
    CATLASS_HOST_DEVICE
    uint32_t GetCoreLoop() const { return coreLoops; }

    CATLASS_HOST_DEVICE
    DistMatrixCoord GetBlockCoord(uint32_t commIdx) const { return {commIdx, 0, peer}; }

    CATLASS_HOST_DEVICE
    bool IsValidBlock(DistMatrixCoord block) const
    {
        return block.rank() == peer && block.column() == 0 && block.row() < GetCoreLoop();
    }

    // Offset relative to the current communication range; the caller adds any outer window offset.
    CATLASS_HOST_DEVICE
    MatrixCoord GetBlockOffsetInRank(MatrixCoord block) const
    {
        if (block.column() != 0 || block.row() >= GetCoreLoop()) {
            return {UINT32_MAX, UINT32_MAX};
        }
        return {transport == CopyTransport::Mte ? block.row() * commBlockM : 0U, 0};
    }

    CATLASS_HOST_DEVICE
    MatrixCoord GetActualBlockShape(MatrixCoord block) const
    {
        if (block.column() != 0 || block.row() >= GetCoreLoop()) {
            return {0U, 0U};
        }
        if (transport != CopyTransport::Mte) {
            return {problemShape.row(), problemShape.column()};
        }
        const uint32_t remaining = problemShape.row() - block.row() * commBlockM;
        return {remaining < commBlockM ? remaining : commBlockM, problemShape.column()};
    }

private:
    DistMatrixCoord problemShape;
    uint32_t rank;
    uint32_t aicoreIdx;
    uint32_t commBlockM = 1;
    uint32_t peer = 0;
    uint32_t groupSize = GROUP_SIZE;
    CopyTransport transport = CopyTransport::None;
    uint32_t coreLoops = 0;
};

} // namespace Catccos::Comm::Block
#endif
