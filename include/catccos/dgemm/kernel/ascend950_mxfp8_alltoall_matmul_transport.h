/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_TRANSPORT_H
#define CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_TRANSPORT_H

#include <cstddef>
#include <cstdint>

#include "catccos/catccos.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/kernel/ascend950_mxfp8_alltoall_matmul_layout.h"
#include "catlass/arch/resource.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "shmem.h"
#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif

namespace Catccos::DGemm::Kernel {

// One sender AIV per peer; one coordinator publishes the all-source frontier.
template <class BlockComm_, class BlockCommScheduler_>
class Ascend950MxFp8AllToAllMatmulTransport {
public:
    using BlockComm = BlockComm_;
    using BlockCommScheduler = BlockCommScheduler_;
    using ArchTag = typename BlockComm::ArchTag;
    using Element = typename BlockComm::ElementSrc;
    static_assert(sizeof(Element) == 1, "Transport offsets use byte units");

    struct Arguments {
        uint32_t chunkM;
        uint32_t recvSlotRows;
        uint32_t windowRows;
        uint32_t localK;
        uint32_t rankIdx;
        uint32_t rankSize;  // runtime number of source K segments
        uint32_t groupSize; // participating ranks per node
        GM_ADDR ptrSourceA;
        GM_ADDR ptrSourceScaleA;
        GM_ADDR ptrRecvA;
        GM_ADDR ptrRecvScaleA;
        GM_ADDR ptrReady = nullptr;
        GM_ADDR ptrReadyFrontier = nullptr;
        GM_ADDR ptrReadyValues = nullptr;
    };

    struct Params {
        uint32_t chunkM;
        uint32_t recvSlotRows;
        uint32_t windowRows;
        uint32_t localK;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t groupSize;
        __gm__ Element* ptrSourceA;
        __gm__ Element* ptrSourceScaleA;
        __gm__ Element* ptrRecvA;
        __gm__ Element* ptrRecvScaleA;
        __gm__ int32_t* ptrReady;
        __gm__ int32_t* ptrReadyFrontier;
        __gm__ Element* ptrReadyValues;

        CATLASS_HOST_DEVICE Params() = default;
        CATLASS_HOST_DEVICE explicit Params(const Arguments& args)
            : chunkM(args.chunkM),
              recvSlotRows(args.recvSlotRows),
              windowRows(args.windowRows),
              localK(args.localK),
              rankIdx(args.rankIdx),
              rankSize(args.rankSize),
              groupSize(args.groupSize),
              ptrSourceA(reinterpret_cast<__gm__ Element*>(args.ptrSourceA)),
              ptrSourceScaleA(reinterpret_cast<__gm__ Element*>(args.ptrSourceScaleA)),
              ptrRecvA(reinterpret_cast<__gm__ Element*>(args.ptrRecvA)),
              ptrRecvScaleA(reinterpret_cast<__gm__ Element*>(args.ptrRecvScaleA)),
              ptrReady(reinterpret_cast<__gm__ int32_t*>(args.ptrReady)),
              ptrReadyFrontier(reinterpret_cast<__gm__ int32_t*>(args.ptrReadyFrontier)),
              ptrReadyValues(reinterpret_cast<__gm__ Element*>(args.ptrReadyValues))
        {}
    };

    static size_t GetWorkspaceSize(const Arguments&) { return 0; }
    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* = nullptr) { return Params(args); }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        (void)params;
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
#ifdef ENABLE_TIMER
        const int64_t kernelBegin = AscendC::GetSystemCycle();
#endif
        const uint32_t aElements = params.localK;
        const uint32_t scaleElements = params.localK / 32;
        const uint32_t core = AscendC::GetBlockIdx();
        // Use qp0 per peer; two QPs showed no bandwidth gain in the 32-rank tests.
        if (params.rankSize == 0 || params.rankSize > MxFp8AllToAllMatmul::MAX_RANKS ||
            params.rankIdx >= params.rankSize || params.recvSlotRows < params.chunkM || params.windowRows == 0 ||
            params.groupSize == 0 || params.groupSize > params.rankSize || params.ptrReady == nullptr) {
            return;
        }
        const uint32_t coordinator = params.rankSize;
        if (core > coordinator) {
            return;
        }
#ifdef ENABLE_TIMER
        AscendTimerDevice timer;
        const bool trace = GetTimerBuffer() != nullptr;
        if (trace) {
            timer.Init(GetTimerBuffer(), kernelBegin);
        }
#endif
        if (core == coordinator) {
            // Publish the frontier only after all sources have completed this window.
            const uint32_t windows = (params.chunkM + params.windowRows - 1) / params.windowRows;
            constexpr uint32_t readyStride = MxFp8AllToAllMatmul::CONTROL_STRIDE / sizeof(int32_t);
            for (uint32_t window = 0; window < windows; ++window) {
#ifdef ENABLE_TIMER
                if (trace) {
                    timer.Tik(AscendTimer::AIV_WAIT);
                }
#endif
                for (uint32_t source = 0; source < params.rankSize; ++source) {
                    aclshmem_uint64_wait_until(
                        reinterpret_cast<__gm__ uint64_t*>(params.ptrReady + source * readyStride), ACLSHMEM_CMP_GE,
                        static_cast<uint64_t>(window + 1));
                }
                aclshmemx_signal_op(
                    params.ptrReadyFrontier, static_cast<int32_t>(window + 1), ACLSHMEM_SIGNAL_SET,
                    static_cast<int>(params.rankIdx));
#ifdef ENABLE_TIMER
                if (trace) {
                    timer.Tok<Overwrite>(AscendTimer::AIV_WAIT);
                }
#endif
            }
#ifdef ENABLE_TIMER
            if (trace) {
                timer.Tok();
            }
#endif
            return;
        }
        const uint32_t peer = core;
        constexpr uint32_t readyStride = MxFp8AllToAllMatmul::CONTROL_STRIDE / sizeof(int32_t);
        const uint32_t readyIndex = params.rankIdx * readyStride;

        BlockCommScheduler commScheduler(
            {params.chunkM, aElements, params.rankSize}, params.rankIdx, peer, params.windowRows, params.groupSize);
        if (!commScheduler.IsActiveCore()) {
            return;
        }
        const auto transport = commScheduler.GetTransport();
        // Route self copies through MTE, local peers through UDMA, and remote peers through RDMA.
        Catlass::Arch::Resource<ArchTag> resource;
        typename BlockComm::Params blockParams{};
        blockParams.blockShape = Catlass::MatrixCoord{params.windowRows, aElements};
        BlockComm commBlock(resource, blockParams);
        AscendC::GlobalTensor<Element> sourceA, sourceScaleA, recvA, recvScaleA, readyValues, ready;
        sourceA.SetGlobalBuffer(params.ptrSourceA);
        sourceScaleA.SetGlobalBuffer(params.ptrSourceScaleA);
        recvA.SetGlobalBuffer(params.ptrRecvA);
        recvScaleA.SetGlobalBuffer(params.ptrRecvScaleA);
        readyValues.SetGlobalBuffer(params.ptrReadyValues);
        ready.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrReady));

        // Fusion windows are independent of the transport scheduler's rounds.
        const uint32_t windows = (params.chunkM + params.windowRows - 1) / params.windowRows;
        for (uint32_t window = 0; window < windows; ++window) {
#ifdef ENABLE_TIMER
            if (trace) {
                timer.Tik(AscendTimer::AIV);
            }
#endif
            const uint32_t begin = window * params.windowRows;
            const uint32_t remaining = params.chunkM - begin;
            const uint32_t rows = remaining < params.windowRows ? remaining : params.windowRows;
            const uint64_t srcRow = static_cast<uint64_t>(peer) * params.chunkM + begin;
            const uint64_t dstRow = static_cast<uint64_t>(params.rankIdx) * params.recvSlotRows + begin;
            // Source slots are indexed by destination; receive slots by source rank.
            const Catlass::MatrixCoord aShape{rows, aElements};
            const Catlass::layout::RowMajor srcALayout{rows, aElements, aElements};
            const Catlass::layout::RowMajor recvALayout{rows, aElements, aElements};

            if (transport == detail::CopyTransport::Mte) {
                commBlock(
                    sourceA[srcRow * aElements], srcALayout, recvA[dstRow * aElements], recvALayout, aShape, peer,
                    transport);
                const Catlass::MatrixCoord scaleShape{rows, scaleElements};
                const Catlass::layout::RowMajor scaleLayout{rows, scaleElements, scaleElements};
                commBlock(
                    sourceScaleA[srcRow * scaleElements], scaleLayout, recvScaleA[dstRow * scaleElements], scaleLayout,
                    scaleShape, peer, transport);
                // Wait only for this sender's local stores before publishing ready.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
                aclshmemx_signal_op(
                    params.ptrReady + readyIndex, static_cast<int32_t>(window + 1), ACLSHMEM_SIGNAL_SET,
                    static_cast<int>(peer));
            } else if (transport == detail::CopyTransport::Udma) {
                // UDMA WQEs are unordered: complete AScale before submitting A with ready.
                const Catlass::MatrixCoord scaleShape{rows, scaleElements};
                const Catlass::layout::RowMajor scaleLayout{rows, scaleElements, scaleElements};
                commBlock(
                    sourceScaleA[srcRow * scaleElements], scaleLayout, recvScaleA[dstRow * scaleElements], scaleLayout,
                    scaleShape, peer, transport);
                aclshmemx_udma_qp_quiet(static_cast<int>(peer), 0);
                commBlock.PutWithSignal(
                    sourceA[srcRow * aElements], recvA[dstRow * aElements], rows * aElements,
                    reinterpret_cast<__gm__ uint64_t*>(params.ptrReady + readyIndex), static_cast<uint64_t>(window + 1),
                    peer);
            } else if (transport == detail::CopyTransport::Rdma) {
                // Requires ordered A -> AScale -> ready writes on one QP (verified on HNS_1825).
                // Keep readyValues unchanged until the final quiet completes.
                commBlock.CopyRdmaNbi(
                    sourceA[srcRow * aElements], srcALayout, recvA[dstRow * aElements], recvALayout, aShape, peer);
                const Catlass::MatrixCoord scaleShape{rows, scaleElements};
                const Catlass::layout::RowMajor scaleLayout{rows, scaleElements, scaleElements};
                commBlock.CopyRdmaNbi(
                    sourceScaleA[srcRow * scaleElements], scaleLayout, recvScaleA[dstRow * scaleElements], scaleLayout,
                    scaleShape, peer);
                constexpr uint32_t signalElements = sizeof(uint64_t) / sizeof(Element);
                const Catlass::MatrixCoord signalShape{1U, signalElements};
                const Catlass::layout::RowMajor signalLayout{1U, signalElements};
                commBlock.CopyRdmaNbi(
                    readyValues[window * signalElements], signalLayout, ready[readyIndex * sizeof(int32_t)],
                    signalLayout, signalShape, peer);
            }
#ifdef ENABLE_TIMER
            if (trace) {
                timer.Tok<Overwrite>(AscendTimer::AIV);
            }
#endif
        }
        if (transport == detail::CopyTransport::Udma || transport == detail::CopyTransport::Rdma) {
#ifdef ENABLE_TIMER
            if (trace) {
                timer.Tik(AscendTimer::AIV_WAIT);
            }
#endif
            if (transport == detail::CopyTransport::Udma) {
                // Earlier windows are drained by the next scale quiet; finish the last A+ready Put.
                aclshmemx_udma_qp_quiet(static_cast<int>(peer), 0);
            } else {
                // Drain all remaining RDMA payload/ready writes before the source can be reused.
                commBlock.QuietRdma(peer);
            }
#ifdef ENABLE_TIMER
            if (trace) {
                timer.Tok<Overwrite>(AscendTimer::AIV_WAIT);
            }
#endif
        }
#ifdef ENABLE_TIMER
        if (trace) {
            timer.Tok();
        }
#endif
    }
};

} // namespace Catccos::DGemm::Kernel
#endif // CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_TRANSPORT_H
