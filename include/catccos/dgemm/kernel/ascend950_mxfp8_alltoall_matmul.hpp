/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_HPP
#define CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_HPP

#include <cstddef>
#include <cstdint>

#include "catccos/catccos.hpp"
#include "catccos/comm/tile/tile_hierarchical_ready.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "shmem.h"
#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif

namespace Catccos::DGemm::Kernel::MxFp8AllToAllMatmul {

constexpr uint32_t MAX_RANKS = 32; // Communication scheduler capacity, not the runtime world size.
constexpr uint32_t MAX_CHUNK_M = 300000;
constexpr uint32_t MAX_LOCAL_DEVICES = 16;
// Select a precompiled K tile that never crosses a source's local-K segment.
constexpr uint32_t SelectTileK(uint32_t localK) { return localK % 256 == 0 ? 256 : (localK % 128 == 0 ? 128 : 64); }

constexpr uint32_t Gcd(uint32_t a, uint32_t b)
{
    while (b != 0) {
        const uint32_t remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}
using Ready = Comm::Tile::TileHierarchicalReady;
constexpr uint64_t ALIGNMENT = 512;

constexpr uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

struct Layout {
    uint32_t rankSize = 0;
    uint32_t fullK = 0;
    uint32_t localK = 0;
    uint32_t scaleK = 0;
    uint32_t n = 0;
    uint32_t m = 0;
    uint32_t chunkM = 0;
    uint32_t recvSlotRows = 0;
    uint32_t windowRows = 0;
    uint32_t windowCount = 0;

    uint64_t sourceA = 0;
    uint64_t sourceScaleA = 0;
    uint64_t recvA = 0;
    uint64_t recvScaleA = 0;
    uint64_t ready = 0;
    uint64_t readyBytes = 0;
    uint64_t readyValues = 0; // Immutable RDMA notification values, shared by all peers.
    uint64_t symmetricBytes = 0;

    constexpr uint64_t SourceRow(uint32_t destination, uint32_t row) const
    {
        return static_cast<uint64_t>(destination) * chunkM + row;
    }

    constexpr uint64_t SourceAOffset(uint32_t destination, uint32_t row) const
    {
        return sourceA + SourceRow(destination, row) * localK;
    }

    constexpr uint64_t SourceScaleAOffset(uint32_t destination, uint32_t row) const
    {
        return sourceScaleA + SourceRow(destination, row) * scaleK;
    }

    constexpr uint64_t RecvAOffset(uint32_t source, uint32_t row) const
    {
        return recvA + (static_cast<uint64_t>(source) * recvSlotRows + row) * localK;
    }

    constexpr uint64_t RecvScaleAOffset(uint32_t source, uint32_t row) const
    {
        return recvScaleA + (static_cast<uint64_t>(source) * recvSlotRows + row) * scaleK;
    }

    constexpr uint64_t ReadyOffset(uint32_t source) const
    {
        return ready + static_cast<uint64_t>(source) * Ready::SLOT_BYTES;
    }

    constexpr uint64_t FrontierOffset() const { return ready + 2ULL * rankSize * Ready::SLOT_BYTES; }

    constexpr uint32_t RowsInWindow(uint32_t window) const
    {
        const uint32_t begin = window * windowRows;
        return begin >= chunkM ? 0 : ((chunkM - begin) < windowRows ? (chunkM - begin) : windowRows);
    }

    constexpr uint64_t BBytes() const { return static_cast<uint64_t>(fullK) * n; }
    constexpr uint64_t ScaleBBytes() const { return static_cast<uint64_t>(fullK / 32) * n; }
    constexpr uint64_t CBytes() const { return static_cast<uint64_t>(chunkM) * n * sizeof(uint16_t); }
};

// Byte offsets: compact source slots and aligned receive slots.
inline bool MakeLayout(uint32_t m, uint32_t n, uint32_t localK, uint32_t rankSize, uint32_t windowRows, Layout& out)
{
    out = {};
    if (rankSize == 0 || rankSize > MAX_RANKS || m == 0 || m % rankSize != 0 || windowRows == 0 || n == 0 ||
        n % 16 != 0 || localK == 0 || localK % 64 != 0 || localK > UINT32_MAX / rankSize ||
        localK > UINT32_MAX / windowRows) {
        return false;
    }
    const uint64_t chunkM = m / rankSize;
    if (chunkM > MAX_CHUNK_M) {
        return false;
    }
    const uint32_t scaleK = localK / 32;
    // Each source slot starts on 512 B, including scaleK values such as 6 B/row.
    const uint64_t recvSlotRows = AlignUp(chunkM, ALIGNMENT / Gcd(ALIGNMENT, scaleK));
    if (((chunkM + 255) / 256) * ((static_cast<uint64_t>(n) + 255) / 256) > UINT32_MAX) {
        return false;
    }
    const uint64_t windows = (chunkM + windowRows - 1) / windowRows;
    out.rankSize = rankSize;
    out.fullK = rankSize * localK;
    out.localK = localK;
    out.scaleK = scaleK;
    out.n = n;
    out.m = m;
    out.chunkM = static_cast<uint32_t>(chunkM);
    out.recvSlotRows = static_cast<uint32_t>(recvSlotRows);
    out.windowRows = windowRows;
    out.windowCount = static_cast<uint32_t>(windows);
    out.sourceA = 0;
    out.sourceScaleA = AlignUp(static_cast<uint64_t>(m) * localK, ALIGNMENT);
    out.recvA = AlignUp(out.sourceScaleA + static_cast<uint64_t>(m) * scaleK, ALIGNMENT);
    out.recvScaleA = AlignUp(out.recvA + static_cast<uint64_t>(rankSize) * recvSlotRows * localK, ALIGNMENT);
    out.ready = AlignUp(out.recvScaleA + static_cast<uint64_t>(rankSize) * recvSlotRows * scaleK, ALIGNMENT);
    out.readyBytes = Ready::GetWorkspaceSize(rankSize);
    out.readyValues = AlignUp(out.ready + out.readyBytes, ALIGNMENT);
    out.symmetricBytes = AlignUp(out.readyValues + windows * sizeof(uint32_t), ALIGNMENT);
    return true;
}

} // namespace Catccos::DGemm::Kernel::MxFp8AllToAllMatmul

namespace Catccos::DGemm::Kernel {

// AICs own disjoint C tiles; AIVs send each destination's windows and one
// coordinator advances the local receive frontier. No global window barrier.
template <class BlockMmad_, class BlockComm_, class BlockCommScheduler_>
class Ascend950MxFp8AllToAllMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;
    using ElementMxScaleA = typename BlockMmad::TileCopy::ElementMxScaleA;
    using ElementMxScaleB = typename BlockMmad::TileCopy::ElementMxScaleB;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;

    using BlockComm = BlockComm_;
    using BlockCommScheduler = BlockCommScheduler_;
    using ElementComm = typename BlockComm::ElementSrc;
    using Ready = MxFp8AllToAllMatmul::Ready;
    static_assert(sizeof(ElementComm) == 1, "Transport offsets use byte units");

    static constexpr uint32_t TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t TILE_N = tla::get<1>(L1TileShape{});
    // Host dispatch guarantees localK is divisible by the selected L1 K tile.

    struct Arguments {
        Catlass::GemmCoord problemShape; // M, N, localK; M = rankSize * chunkM
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t windowRows;
        GM_ADDR ptrA; // read-only SHMEM source
        GM_ADDR ptrB;
        GM_ADDR ptrScaleA; // read-only SHMEM source scales
        GM_ADDR ptrScaleB;
        GM_ADDR ptrC;
        GM_ADDR ptrSymmetric;
    };

    struct Params {
        uint32_t chunkM;
        uint32_t recvSlotRows; // Physical rows per source slot, >= chunkM.
        uint32_t fullK;
        uint32_t localK;
        uint32_t n;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t windowRows;

        __gm__ ElementComm* ptrSourceA;
        __gm__ ElementComm* ptrSourceScaleA;
        __gm__ ElementA* ptrRecvA;
        __gm__ ElementMxScaleA* ptrRecvScaleA;
        __gm__ ElementB* ptrB;
        __gm__ ElementMxScaleB* ptrScaleB;
        __gm__ ElementC* ptrC;
        GM_ADDR ptrReady;
        __gm__ uint32_t* ptrReadyValues;
        LayoutB layoutB;
        decltype(tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagB, true>(0U, 0U)) layoutScaleB;
        decltype(tla::MakeLayoutFromTag(Catlass::layout::RowMajor(0U, 0U, 0U))) layoutC;

        CATLASS_HOST_DEVICE Params() = default;
        CATLASS_HOST_DEVICE Params(const Arguments& args, const MxFp8AllToAllMatmul::Layout& layout)
            : chunkM(layout.chunkM),
              recvSlotRows(layout.recvSlotRows),
              fullK(layout.fullK),
              localK(layout.localK),
              n(layout.n),
              rankIdx(args.rankIdx),
              rankSize(args.rankSize),
              windowRows(layout.windowRows),
              ptrSourceA(reinterpret_cast<__gm__ ElementComm*>(args.ptrA)),
              ptrSourceScaleA(reinterpret_cast<__gm__ ElementComm*>(args.ptrScaleA)),
              ptrRecvA(reinterpret_cast<__gm__ ElementA*>(args.ptrSymmetric + layout.recvA)),
              ptrRecvScaleA(reinterpret_cast<__gm__ ElementMxScaleA*>(args.ptrSymmetric + layout.recvScaleA)),
              ptrB(reinterpret_cast<__gm__ ElementB*>(args.ptrB)),
              ptrScaleB(reinterpret_cast<__gm__ ElementMxScaleB*>(args.ptrScaleB)),
              ptrC(reinterpret_cast<__gm__ ElementC*>(args.ptrC)),
              ptrReady(args.ptrSymmetric + layout.ready),
              ptrReadyValues(reinterpret_cast<__gm__ uint32_t*>(args.ptrSymmetric + layout.readyValues)),
              layoutB(tla::MakeLayoutFromTag(LayoutTagB{fullK, n})),
              layoutScaleB(tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagB, true>(fullK / 32, n)),
              layoutC(tla::MakeLayoutFromTag(Catlass::layout::RowMajor{chunkM, n, n}))
        {}
    };

    static size_t GetWorkspaceSize(const Arguments&) { return 0; }
    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* = nullptr)
    {
        MxFp8AllToAllMatmul::Layout layout;
        MxFp8AllToAllMatmul::MakeLayout(
            args.problemShape.m(), args.problemShape.n(), args.problemShape.k(), args.rankSize, args.windowRows,
            layout);
        return Params(args, layout);
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params& params);

    // AIC: wait for each tile's window, then compute its complete K reduction.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params& params)
    {
#ifdef ENABLE_TIMER
        const int64_t kernelBegin = AscendC::GetSystemCycle();
        AscendTimerDevice timer;
        const bool trace = GetTimerBuffer() != nullptr;
        if (trace) {
            timer.Init(GetTimerBuffer(), kernelBegin);
        }
#endif
        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        Ready ready(params.ptrReady, params.rankIdx, params.rankSize);
        AscendC::GlobalTensor<ElementA> recvA;
        AscendC::GlobalTensor<ElementMxScaleA> recvScaleA;
        AscendC::GlobalTensor<ElementB> gmB;
        AscendC::GlobalTensor<ElementMxScaleB> gmScaleB;
        AscendC::GlobalTensor<ElementC> gmC;
        recvA.SetGlobalBuffer(params.ptrRecvA);
        recvScaleA.SetGlobalBuffer(params.ptrRecvScaleA);
        gmB.SetGlobalBuffer(params.ptrB);
        gmScaleB.SetGlobalBuffer(params.ptrScaleB);
        gmC.SetGlobalBuffer(params.ptrC);

        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Catlass::Arch::PositionGM{});
        auto tensorScaleB = tla::MakeTensor(gmScaleB, params.layoutScaleB, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Catlass::Arch::PositionGM{});
        const uint32_t tileRows = (params.chunkM + TILE_M - 1) / TILE_M;
        const uint32_t tileCols = (params.n - 1) / TILE_N + 1;
        const uint32_t tileCount = tileRows * tileCols;

        for (uint32_t tileId = AscendC::GetBlockIdx(); tileId < tileCount; tileId += AscendC::GetBlockNum()) {
            const uint32_t row = (tileId / tileCols) * TILE_M;
            const uint32_t col = (tileId % tileCols) * TILE_N;
            const uint32_t rows = row + TILE_M <= params.chunkM ? TILE_M : params.chunkM - row;
            const uint32_t cols = params.n - col >= TILE_N ? TILE_N : params.n - col;
#ifdef ENABLE_TIMER
            if (trace) {
                timer.Tik(AscendTimer::AIC_WAIT);
            }
#endif
            ready.WaitReady((row + rows - 1) / params.windowRows + 1);
#ifdef ENABLE_TIMER
            if (trace) {
                timer.Tok<Overwrite>(AscendTimer::AIC_WAIT);
                timer.Tik(AscendTimer::AIC);
            }
#endif
            auto bTile = GetTile(tensorB, tla::MakeCoord(0U, col), tla::MakeShape(params.fullK, cols));
            auto scaleBTile = GetTile(tensorScaleB, tla::MakeCoord(0U, col), tla::MakeShape(params.fullK / 32, cols));
            auto cTile = GetTile(tensorC, tla::MakeCoord(row, col), tla::MakeShape(rows, cols));
            blockMmad(
                recvA, recvScaleA, bTile, cTile, Catlass::GemmCoord{rows, cols, params.fullK}, row, params.recvSlotRows,
                params.localK, params.localK, params.localK / 32, params.localK / 32, scaleBTile, Catlass::EmptyClass{},
                true, true, true);
#ifdef ENABLE_TIMER
            if (trace) {
                timer.Tok<Overwrite>(AscendTimer::AIC);
            }
#endif
        }
#ifdef ENABLE_TIMER
        if (trace) {
            // Tok already drains this core's pipelines.
            timer.Tok();
            return;
        }
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // AIV: send peer windows or publish the all-source frontier.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params& params)
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
            params.rankSize % BlockCommScheduler::GROUP_SIZE != 0 || params.ptrReady == nullptr) {
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
        Ready ready(params.ptrReady, params.rankIdx, params.rankSize);
        if (core == coordinator) {
            // Publish the frontier only after all sources have completed this window.
            const uint32_t windows = (params.chunkM + params.windowRows - 1) / params.windowRows;
            for (uint32_t window = 0; window < windows; ++window) {
#ifdef ENABLE_TIMER
                if (trace) {
                    timer.Tik(AscendTimer::AIV_WAIT);
                }
#endif
                ready.WaitAll(window + 1);
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
        Catlass::Arch::Resource<typename BlockComm::ArchTag> resource;
        auto scratch = resource.ubBuf.template GetBufferByByte<uint32_t>(0);

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
            BlockCommScheduler commScheduler({rows, aElements, params.rankSize}, params.rankIdx, core);
            if (!commScheduler.IsActiveCore()) {
                return;
            }
            BlockCommScheduler scaleScheduler({rows, scaleElements, params.rankSize}, params.rankIdx, core);
            const uint64_t srcRow = static_cast<uint64_t>(commScheduler.GetPeer()) * params.chunkM + begin;
            const uint64_t dstRow = static_cast<uint64_t>(params.rankIdx) * params.recvSlotRows + begin;
            CopyWindow(
                resource, commScheduler, rows, aElements, params.ptrSourceA + srcRow * aElements,
                reinterpret_cast<__gm__ ElementComm*>(params.ptrRecvA) + dstRow * aElements);
            CopyWindow(
                resource, scaleScheduler, rows, scaleElements, params.ptrSourceScaleA + srcRow * scaleElements,
                reinterpret_cast<__gm__ ElementComm*>(params.ptrRecvScaleA) + dstRow * scaleElements);
            // HNS_1825 orders A, scales and the immutable epoch on the same QP0.
            ready.Notify(
                commScheduler.GetPeer(), commScheduler.GetTransport(), window + 1, 1,
                reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()), 0, params.ptrReadyValues + window);
#ifdef ENABLE_TIMER
            if (trace) {
                timer.Tok<Overwrite>(AscendTimer::AIV);
            }
#endif
            if (window + 1 == windows && commScheduler.GetTransport() == detail::CopyTransport::Rdma) {
#ifdef ENABLE_TIMER
                if (trace) {
                    timer.Tik(AscendTimer::AIV_WAIT);
                }
#endif
                typename BlockComm::Params blockParams{};
                BlockComm commBlock(resource, blockParams);
                commBlock.QuietRdma(commScheduler.GetPeer());
#ifdef ENABLE_TIMER
                if (trace) {
                    timer.Tok<Overwrite>(AscendTimer::AIV_WAIT);
                }
#endif
            }
        }

#ifdef ENABLE_TIMER
        if (trace) {
            timer.Tok();
        }
#endif
    }

private:
    CATLASS_DEVICE void CopyWindow(
        Catlass::Arch::Resource<typename BlockComm::ArchTag>& resource, const BlockCommScheduler& scheduler,
        uint32_t rows, uint32_t columns, __gm__ ElementComm* src, __gm__ ElementComm* dst)
    {
        typename BlockComm::Params blockParams{};
        if constexpr (BlockComm::IsDynamic) {
            blockParams.blockShape = scheduler.GetActualBlockShape({0U, 0U});
        }
        BlockComm commBlock(resource, blockParams);
        AscendC::GlobalTensor<ElementComm> source, destination;
        source.SetGlobalBuffer(src);
        destination.SetGlobalBuffer(dst);
        const Catlass::layout::RowMajor layout{rows, columns};
        // MTE traverses COMM_BLOCK_M rows; network peers send the current window.
        for (uint32_t index = 0; index < scheduler.GetCoreLoop(); ++index) {
            const auto coord = scheduler.GetBlockCoord(index);
            const Catlass::MatrixCoord blockCoord{coord.row(), coord.column()};
            const auto offset = scheduler.GetBlockOffsetInRank(blockCoord);
            const auto shape = scheduler.GetActualBlockShape(blockCoord);
            const auto blockLayout = layout.GetTileLayout(shape);
            commBlock(
                source[layout.GetOffset(offset)], blockLayout, destination[layout.GetOffset(offset)], blockLayout,
                shape, coord.rank(), scheduler.GetTransport(), 1, BlockComm::RdmaCompletion::Deferred);
        }
    }
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_HPP
