/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALL_MATMUL_FUSED_HPP
#define CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALL_MATMUL_FUSED_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

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

namespace Catccos::DGemm::Kernel::AllToAllMatmul {

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
    uint32_t rowBytes = 0;
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

    constexpr uint64_t ReadyOffset(uint32_t source) const
    {
        return ready + static_cast<uint64_t>(source) * Ready::SLOT_BYTES;
    }

    constexpr uint64_t FrontierOffset() const { return ready + 2ULL * rankSize * Ready::SLOT_BYTES; }

    constexpr uint64_t BBytes() const { return static_cast<uint64_t>(rankSize) * rowBytes * n; }
    constexpr uint64_t ScaleBBytes() const { return static_cast<uint64_t>(rankSize) * scaleK * n; }
    constexpr uint64_t CBytes() const { return static_cast<uint64_t>(chunkM) * n * sizeof(uint16_t); }
};

// Byte offsets: compact source slots and aligned receive slots.
template <bool IsQuantized>
inline bool MakeLayout(uint32_t m, uint32_t n, uint32_t localK, uint32_t rankSize, uint32_t windowRows, Layout& out)
{
    out = {};
    constexpr uint32_t elementBytes = IsQuantized ? sizeof(uint8_t) : sizeof(uint16_t);
    if (rankSize == 0 || rankSize > MAX_RANKS || m == 0 || m % rankSize != 0 || windowRows == 0 || n == 0 ||
        n % 16 != 0 || localK == 0 || localK > UINT32_MAX / rankSize ||
        localK > UINT32_MAX / windowRows / elementBytes) {
        return false;
    }
    if constexpr (IsQuantized) {
        if (localK % 64 != 0) {
            return false;
        }
    }

    const uint64_t chunkM = m / rankSize;
    constexpr uint32_t tileRows = IsQuantized ? 256 : 128;
    if (chunkM > MAX_CHUNK_M ||
        ((chunkM + tileRows - 1) / tileRows) * ((static_cast<uint64_t>(n) + 255) / 256) > UINT32_MAX) {
        return false;
    }

    const uint32_t rowBytes = localK * elementBytes;
    const uint32_t scaleK = IsQuantized ? localK / 32 : 0;
    // MXFP8's scale stride also divides its A row stride, so it sets the slot alignment.
    const uint32_t alignmentStride = IsQuantized ? scaleK : rowBytes;
    const uint64_t recvSlotRows = AlignUp(chunkM, ALIGNMENT / Gcd(ALIGNMENT, alignmentStride));
    const uint64_t windows = (chunkM + windowRows - 1) / windowRows;
    out.rankSize = rankSize;
    out.fullK = rankSize * localK;
    out.localK = localK;
    out.rowBytes = rowBytes;
    out.scaleK = scaleK;
    out.n = n;
    out.m = m;
    out.chunkM = static_cast<uint32_t>(chunkM);
    out.recvSlotRows = static_cast<uint32_t>(recvSlotRows);
    out.windowRows = windowRows;
    out.windowCount = static_cast<uint32_t>(windows);

    uint64_t sourceEnd = static_cast<uint64_t>(m) * rowBytes;
    if constexpr (IsQuantized) {
        out.sourceScaleA = AlignUp(sourceEnd, ALIGNMENT);
        sourceEnd = out.sourceScaleA + static_cast<uint64_t>(m) * scaleK;
    }
    out.recvA = AlignUp(sourceEnd, ALIGNMENT);
    uint64_t recvEnd = out.recvA + static_cast<uint64_t>(rankSize) * recvSlotRows * rowBytes;
    if constexpr (IsQuantized) {
        out.recvScaleA = AlignUp(recvEnd, ALIGNMENT);
        recvEnd = out.recvScaleA + static_cast<uint64_t>(rankSize) * recvSlotRows * scaleK;
    }
    out.ready = AlignUp(recvEnd, ALIGNMENT);
    out.readyBytes = Ready::GetWorkspaceSize(rankSize);
    out.readyValues = AlignUp(out.ready + out.readyBytes, ALIGNMENT);
    out.symmetricBytes = AlignUp(out.readyValues + windows * sizeof(uint32_t), ALIGNMENT);
    return true;
}

} // namespace Catccos::DGemm::Kernel::AllToAllMatmul

namespace Catccos::DGemm::Kernel {

struct Bf16AllToAllMatmulArguments {
    Catlass::GemmCoord problemShape; // M = rankSize * rows per destination, K = localK.
    uint32_t rankIdx;
    uint32_t rankSize;
    uint32_t windowRows;
    GM_ADDR ptrA; // Read-only SHMEM source.
    GM_ADDR ptrB;
    GM_ADDR ptrC;
    GM_ADDR ptrSymmetric;
};

struct MxFp8AllToAllMatmulArguments {
    Catlass::GemmCoord problemShape;
    uint32_t rankIdx;
    uint32_t rankSize;
    uint32_t windowRows;
    GM_ADDR ptrA;
    GM_ADDR ptrB;
    GM_ADDR ptrScaleA; // Read-only SHMEM source scales.
    GM_ADDR ptrScaleB;
    GM_ADDR ptrC;
    GM_ADDR ptrSymmetric;
};

template <class BlockMmad, bool IsQuantized>
struct AllToAllMatmulScaleParams {
    using TensorA = Catlass::EmptyClass;
    using TensorB = Catlass::EmptyClass;
};

template <class BlockMmad>
struct AllToAllMatmulScaleParams<BlockMmad, true> {
    using ElementA = typename BlockMmad::TileCopy::ElementMxScaleA;
    using ElementB = typename BlockMmad::TileCopy::ElementMxScaleB;
    using TensorA = AscendC::GlobalTensor<ElementA>;
    using TensorB = AscendC::GlobalTensor<ElementB>;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;

    GM_ADDR ptrSourceA;
    __gm__ ElementA* ptrRecvA;
    __gm__ ElementB* ptrB;
    decltype(tla::MakeMxScaleLayout<ElementB, LayoutTagB, true>(0U, 0U)) layoutB;
};

// AICs wait for local receive windows and own disjoint C tiles.
template <class BlockMmad_, class BlockComm_, class BlockCommScheduler_, bool IsQuantized>
class Ascend950AllToAllMatmul {
public:
    using BlockMmad = BlockMmad_;
    using BlockComm = BlockComm_;
    using BlockCommScheduler = BlockCommScheduler_;
    using Arguments = std::conditional_t<IsQuantized, MxFp8AllToAllMatmulArguments, Bf16AllToAllMatmulArguments>;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;
    using ElementComm = typename BlockComm::ElementSrc;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using Ready = Comm::Tile::TileHierarchicalReady;
    static_assert(sizeof(ElementComm) == 1, "Transport offsets use byte units");

    static constexpr uint32_t TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t TILE_N = tla::get<1>(L1TileShape{});

    enum class Mode : uint32_t { Fused, Communication, Matmul };

private:
    using Layout = AllToAllMatmul::Layout;
    using ScaleParams = AllToAllMatmulScaleParams<BlockMmad, IsQuantized>;

    CATLASS_HOST_DEVICE static ScaleParams MakeScaleParams(const Arguments& args, const Layout& layout)
    {
        if constexpr (IsQuantized) {
            using ScaleA = typename ScaleParams::ElementA;
            using ScaleB = typename ScaleParams::ElementB;
            return {
                args.ptrScaleA, reinterpret_cast<__gm__ ScaleA*>(args.ptrSymmetric + layout.recvScaleA),
                reinterpret_cast<__gm__ ScaleB*>(args.ptrScaleB),
                tla::MakeMxScaleLayout<ScaleB, LayoutTagB, true>(layout.fullK / 32, layout.n)};
        } else {
            return {};
        }
    }

    CATLASS_DEVICE static auto MakeScaleBTensor(typename ScaleParams::TensorB& gmScaleB, ScaleParams& scales)
    {
        if constexpr (IsQuantized) {
            gmScaleB.SetGlobalBuffer(scales.ptrB);
            return tla::MakeTensor(gmScaleB, scales.layoutB, Catlass::Arch::PositionGM{});
        } else {
            return Catlass::EmptyClass{};
        }
    }

public:
    struct Params {
        uint32_t chunkM;
        uint32_t recvSlotRows; // Physical rows per source slot, >= chunkM.
        uint32_t fullK;
        uint32_t localK;
        uint32_t n;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t windowRows;
        Mode mode = Mode::Fused; // Benchmarks may also launch standalone stages.

        __gm__ ElementComm* ptrSourceA;
        __gm__ ElementA* ptrRecvA;
        __gm__ ElementB* ptrB;
        __gm__ ElementC* ptrC;
        GM_ADDR ptrReady;
        __gm__ uint32_t* ptrReadyValues;
        LayoutB layoutB;
        decltype(tla::MakeLayoutFromTag(Catlass::layout::RowMajor(0U, 0U, 0U))) layoutC;
        ScaleParams scales;

        CATLASS_HOST_DEVICE Params() = default;
        CATLASS_HOST_DEVICE Params(const Arguments& args, const Layout& layout)
            : chunkM(layout.chunkM),
              recvSlotRows(layout.recvSlotRows),
              fullK(layout.fullK),
              localK(layout.localK),
              n(layout.n),
              rankIdx(args.rankIdx),
              rankSize(args.rankSize),
              windowRows(layout.windowRows),
              ptrSourceA(reinterpret_cast<__gm__ ElementComm*>(args.ptrA)),
              ptrRecvA(reinterpret_cast<__gm__ ElementA*>(args.ptrSymmetric + layout.recvA)),
              ptrB(reinterpret_cast<__gm__ ElementB*>(args.ptrB)),
              ptrC(reinterpret_cast<__gm__ ElementC*>(args.ptrC)),
              ptrReady(args.ptrSymmetric + layout.ready),
              ptrReadyValues(reinterpret_cast<__gm__ uint32_t*>(args.ptrSymmetric + layout.readyValues)),
              layoutB(tla::MakeLayoutFromTag(LayoutTagB{fullK, n})),
              layoutC(tla::MakeLayoutFromTag(Catlass::layout::RowMajor{chunkM, n, n})),
              scales(MakeScaleParams(args, layout))
        {}
    };

    static size_t GetWorkspaceSize(const Arguments&) { return 0; }
    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* = nullptr)
    {
        Layout layout;
        if (!AllToAllMatmul::MakeLayout<IsQuantized>(
                args.problemShape.m(), args.problemShape.n(), args.problemShape.k(), args.rankSize, args.windowRows,
                layout)) {
            throw std::invalid_argument("invalid AllToAll MatMul layout: check M, N, localK, rankSize and windowRows");
        }
        return Params(args, layout);
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params& params);

    // AICs wait for the required windows, then compute their C tiles.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params& params)
    {
        if (params.mode == Mode::Communication) {
            return;
        }
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
        AscendC::GlobalTensor<ElementB> gmB;
        AscendC::GlobalTensor<ElementC> gmC;
        typename ScaleParams::TensorA recvScaleA;
        typename ScaleParams::TensorB gmScaleB;
        recvA.SetGlobalBuffer(params.ptrRecvA);
        gmB.SetGlobalBuffer(params.ptrB);
        gmC.SetGlobalBuffer(params.ptrC);
        if constexpr (IsQuantized) {
            recvScaleA.SetGlobalBuffer(params.scales.ptrRecvA);
        }

        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Catlass::Arch::PositionGM{});
        auto tensorScaleB = MakeScaleBTensor(gmScaleB, params.scales);
        const uint32_t tileCols = (params.n - 1) / TILE_N + 1;
        const uint32_t tileCount = ((params.chunkM - 1) / TILE_M + 1) * tileCols;
        for (uint32_t tileId = AscendC::GetBlockIdx(); tileId < tileCount; tileId += AscendC::GetBlockNum()) {
            const uint32_t row = tileId / tileCols * TILE_M;
            const uint32_t col = tileId % tileCols * TILE_N;
            const uint32_t rows = ::Min(params.chunkM - row, TILE_M);
            const uint32_t cols = ::Min(params.n - col, TILE_N);
#ifdef ENABLE_TIMER
            if (trace) {
                timer.Tik(AscendTimer::AIC_WAIT);
            }
#endif
            if (params.mode == Mode::Fused) {
                // A tile can cross windows; wait until its last row is ready.
                ready.WaitReady((row + rows - 1) / params.windowRows + 1);
            }
#ifdef ENABLE_TIMER
            if (trace) {
                timer.Tok<Overwrite>(AscendTimer::AIC_WAIT);
                timer.Tik(AscendTimer::AIC);
            }
#endif
            auto bTile = GetTile(tensorB, tla::MakeCoord(0U, col), tla::MakeShape(params.fullK, cols));
            auto cTile = GetTile(tensorC, tla::MakeCoord(row, col), tla::MakeShape(rows, cols));
            const Catlass::GemmCoord shape{rows, cols, params.fullK};
            if constexpr (IsQuantized) {
                auto scaleBTile =
                    GetTile(tensorScaleB, tla::MakeCoord(0U, col), tla::MakeShape(params.fullK / 32, cols));
                blockMmad(
                    recvA, recvScaleA, bTile, cTile, shape, row, params.recvSlotRows, params.localK, params.localK,
                    params.localK / 32, params.localK / 32, scaleBTile, Catlass::EmptyClass{}, true, true, true);
            } else {
                blockMmad(recvA, bTile, cTile, shape, row, params.recvSlotRows, params.localK);
            }
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

    // AIVs send one peer each; the extra AIV advances the local ready frontier.
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params& params)
    {
        if (params.mode == Mode::Matmul) {
            return;
        }
#ifdef ENABLE_TIMER
        const int64_t kernelBegin = AscendC::GetSystemCycle();
#endif
        const uint32_t core = AscendC::GetBlockIdx();
        if (params.rankSize == 0 || params.rankSize > BlockCommScheduler::CORE_COUNT ||
            params.rankIdx >= params.rankSize || params.recvSlotRows < params.chunkM || params.windowRows == 0 ||
            params.rankSize % BlockCommScheduler::GROUP_SIZE != 0 || core > params.rankSize ||
            params.ptrReady == nullptr) {
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
        const uint32_t aRowBytes = params.localK * static_cast<uint32_t>(sizeof(ElementA));
        const uint32_t windows = (params.chunkM - 1) / params.windowRows + 1;
        if (core == params.rankSize) {
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
        } else {
            Catlass::Arch::Resource<typename BlockComm::ArchTag> resource;
            auto scratch = resource.ubBuf.template GetBufferByByte<uint32_t>(0);
            for (uint32_t window = 0; window < windows; ++window) {
#ifdef ENABLE_TIMER
                if (trace) {
                    timer.Tik(AscendTimer::AIV);
                }
#endif
                const uint32_t begin = window * params.windowRows;
                const uint32_t rows = ::Min(params.chunkM - begin, params.windowRows);
                const uint64_t recvRow = static_cast<uint64_t>(params.rankIdx) * params.recvSlotRows + begin;
                const BlockCommScheduler aScheduler({rows, aRowBytes, params.rankSize}, params.rankIdx, core);
                const uint64_t sourceRow = static_cast<uint64_t>(aScheduler.GetPeer()) * params.chunkM + begin;
                // Copy A first; MXFP8 also copies this window's scales before notifying ready.
                CopyWindow(
                    resource, aScheduler, rows, aRowBytes, params.ptrSourceA + sourceRow * aRowBytes,
                    reinterpret_cast<__gm__ ElementComm*>(params.ptrRecvA) + recvRow * aRowBytes);
                if constexpr (IsQuantized) {
                    const uint32_t scaleRowBytes = params.localK / 32;
                    const BlockCommScheduler scaleScheduler(
                        {rows, scaleRowBytes, params.rankSize}, params.rankIdx, core);
                    CopyWindow(
                        resource, scaleScheduler, rows, scaleRowBytes,
                        reinterpret_cast<__gm__ ElementComm*>(params.scales.ptrSourceA) + sourceRow * scaleRowBytes,
                        reinterpret_cast<__gm__ ElementComm*>(params.scales.ptrRecvA) + recvRow * scaleRowBytes);
                }
                // A, scales and the immutable ready value are submitted in order on RDMA QP0.
                ready.Notify(
                    aScheduler.GetPeer(), aScheduler.GetTransport(), window + 1, 1,
                    reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()), 0, params.ptrReadyValues + window);
#ifdef ENABLE_TIMER
                if (trace) {
                    timer.Tok<Overwrite>(AscendTimer::AIV);
                }
#endif
                if (window + 1 == windows && aScheduler.GetTransport() == detail::CopyTransport::Rdma) {
#ifdef ENABLE_TIMER
                    if (trace) {
                        timer.Tik(AscendTimer::AIV_WAIT);
                    }
#endif
                    typename BlockComm::Params blockParams{};
                    BlockComm commBlock(resource, blockParams);
                    commBlock.QuietRdma(aScheduler.GetPeer());
#ifdef ENABLE_TIMER
                    if (trace) {
                        timer.Tok<Overwrite>(AscendTimer::AIV_WAIT);
                    }
#endif
                }
            }
        }
#ifdef ENABLE_TIMER
        if (trace) {
            timer.Tok();
        }
#endif
    }

private:
    CATLASS_DEVICE static void CopyWindow(
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

template <class BlockMmad, class BlockComm, class BlockCommScheduler>
using Ascend950MxFp8AllToAllMatmul = Ascend950AllToAllMatmul<BlockMmad, BlockComm, BlockCommScheduler, true>;

template <class BlockMmad, class BlockComm, class BlockCommScheduler>
using Ascend950Bf16AllToAllMatmul = Ascend950AllToAllMatmul<BlockMmad, BlockComm, BlockCommScheduler, false>;

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALL_MATMUL_FUSED_HPP
