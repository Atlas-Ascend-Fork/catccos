/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_COMM_TILE_HIERARCHICAL_READY_HPP
#define CATCCOS_COMM_TILE_HIERARCHICAL_READY_HPP

#include <cstddef>
#include <cstdint>

#include "catccos/catccos.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "shmem.h"

namespace Catccos::Comm::Tile {

// Receiver readiness for Put windows: each source publishes ready[source] at its
// destination after completed or ordered writes of both data and scales. One local coordinator merges
// all sources into a frontier; consumers wait that frontier before loading data.
// This is not a global barrier and does not acknowledge consumption.
class TileHierarchicalReady {
public:
    using CopyTransport = detail::CopyTransport;
    static constexpr uint32_t SLOT_BYTES = 64;
    static constexpr uint32_t SLOT_WORDS = SLOT_BYTES / sizeof(uint32_t);
    static constexpr uint32_t UB_BYTES = 256;
    static_assert(UB_BYTES >= ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE, "Insufficient signal scratch");

    // Allocate symmetrically with aclshmem_calloc before the first launch.
    // Layout: ready[source], sendValue[peer], local frontier; every slot owns a
    // cache line. Separate send slots prevent concurrent peer AIVs sharing a source.
    static size_t GetWorkspaceSize(uint32_t rankSize)
    {
        return (2 * static_cast<size_t>(rankSize) + 1) * SLOT_BYTES + SLOT_BYTES - 1;
    }

    CATLASS_DEVICE
    TileHierarchicalReady(GM_ADDR workspace, uint32_t rankIdx, uint32_t rankSize)
        : ptrWorkspace(reinterpret_cast<__gm__ uint32_t*>(
              (reinterpret_cast<uintptr_t>(workspace) + SLOT_BYTES - 1) & ~(uintptr_t(SLOT_BYTES) - 1))),
          rankIdx(rankIdx),
          rankSize(rankSize)
    {}

    // Call once after ALL Put calls for this peer's window (including scales).
    // The calling AIV exclusively owns the peer's QPs and send slot. scratch is
    // a private UB_BYTES-byte buffer; eventId and scratch must be idle on entry.
    // MTE is supported only for self; network transport must match the scheduler.
    // epoch starts at 1 and increases without wrapping, identically on all ranks.
    // Do not reset slots between windows or overwrite unconsumed payload buffers.
    // immutableEpoch optionally points to a preinitialized symmetric value equal to epoch.
    // On a single ordered RDMA QP, it permits notification without quiet; keep it and
    // the payload unchanged until the caller drains QP0. Other paths retain completion waits.
    CATLASS_DEVICE
    void Notify(
        uint32_t peer, CopyTransport transport, uint32_t epoch, uint32_t qpNum, __ubuf__ uint32_t* scratch,
        uint32_t eventId, __gm__ uint32_t* immutableEpoch = nullptr)
    {
        auto dst = ptrWorkspace + rankIdx * SLOT_WORDS;
        if (transport == CopyTransport::Rdma && qpNum == 1 && immutableEpoch != nullptr) {
            aclshmemx_roce_qp_put_nbi<uint32_t>(dst, immutableEpoch, scratch, 1, peer, 0, eventId);
            return;
        }
        if (transport == CopyTransport::Mte) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
            Store(dst, epoch);
            return;
        }
        if (transport != CopyTransport::Udma && transport != CopyTransport::Rdma) {
            return;
        }
        // QP0 carries the notification only AFTER every payload QP is complete.
        // A QP0 signal alone cannot order payload sent through QP1/2/3.
        for (uint32_t qp = 0; qp < qpNum; ++qp) {
            if (transport == CopyTransport::Udma) {
                aclshmemx_udma_qp_quiet(peer, qp);
            } else {
                aclshmemx_roce_qp_quiet(peer, qp, scratch, eventId);
            }
        }
        auto src = ptrWorkspace + (rankSize + peer) * SLOT_WORDS;
        Store(src, epoch);
        if (transport == CopyTransport::Udma) {
            aclshmemx_udma_qp_put_nbi<uint32_t>(dst, src, scratch, 1, peer, 0, eventId);
            aclshmemx_udma_qp_quiet(peer, 0);
        } else {
            aclshmemx_roce_qp_put_nbi<uint32_t>(dst, src, scratch, 1, peer, 0, eventId);
            aclshmemx_roce_qp_quiet(peer, 0, scratch, eventId);
        }
        // The final quiet also makes sendValue[peer] reusable for the next epoch.
    }

    // Exactly one coordinator per rank calls this in increasing epoch order.
    // A fast source may already have published a later window: GE preserves that
    // progress. Payload windows must remain valid until their consumers finish.
    CATLASS_DEVICE
    void WaitAll(uint32_t epoch)
    {
        for (uint32_t source = 0; source < rankSize; ++source) {
            aclshmem_uint32_wait_until(ptrWorkspace + source * SLOT_WORDS, ACLSHMEM_CMP_GE, epoch);
        }
        Store(ptrWorkspace + 2 * rankSize * SLOT_WORDS, epoch);
    }

    // AIC/AIV consumers call this before reading this window's received tensors.
    // It waits only for this rank's inputs, not for other ranks to finish receiving.
    CATLASS_DEVICE
    void WaitReady(uint32_t epoch) const
    {
        aclshmem_uint32_wait_until(ptrWorkspace + 2 * rankSize * SLOT_WORDS, ACLSHMEM_CMP_GE, epoch);
    }

private:
    CATLASS_DEVICE
    static void Store(__gm__ uint32_t* ptr, uint32_t epoch)
    {
        AscendC::GlobalTensor<uint32_t> value;
        value.SetGlobalBuffer(ptr);
        value.SetValue(0, epoch);
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<
            uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(value);
        __asm__ __volatile__("");
    }

    __gm__ uint32_t* ptrWorkspace;
    uint32_t rankIdx;
    uint32_t rankSize;
};

} // namespace Catccos::Comm::Tile
#endif
