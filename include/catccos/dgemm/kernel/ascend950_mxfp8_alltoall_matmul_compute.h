/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_COMPUTE_H
#define CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_COMPUTE_H

#include <cstddef>
#include <cstdint>

#include "catccos/catccos.hpp"
#include "catccos/dgemm/kernel/ascend950_mxfp8_alltoall_matmul_layout.h"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "shmem.h"
#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif

namespace Catccos::DGemm::Kernel {

// AIC scheduling path of the fused kernel.
// A non-null frontier gates each output tile until its complete window arrives.
template <class BlockMmad_>
class Ascend950MxFp8AllToAllMatmulCompute {
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

    static constexpr uint32_t TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t TILE_N = tla::get<1>(L1TileShape{});
    // Host dispatch guarantees localK is divisible by the selected L1 K tile.

    struct Arguments {
        uint32_t chunkM;
        uint32_t recvRows; // physical rows per source slot, >= chunkM
        uint32_t rankSize;
        uint32_t localK;
        uint32_t n;
        GM_ADDR ptrRecvA;                   // [rankSize, recvRows, localK], compact FP8 rows
        GM_ADDR ptrRecvScaleA;              // [rankSize, recvRows, localK/32], E8M0
        GM_ADDR ptrB;                       // logical [rankSize * localK, n], ColumnMajor
        GM_ADDR ptrScaleB;                  // packed E8M0 layout for B
        GM_ADDR ptrC;                       // [chunkM, n], BF16
        GM_ADDR ptrReadyFrontier = nullptr; // optional fused-window gate
        uint32_t windowRows = 0;
    };

    struct Params {
        uint32_t chunkM;
        uint32_t recvRows;
        uint32_t fullK;
        uint32_t localK;
        uint32_t n;
        __gm__ ElementA* ptrRecvA;
        __gm__ ElementMxScaleA* ptrRecvScaleA;
        __gm__ ElementB* ptrB;
        __gm__ ElementMxScaleB* ptrScaleB;
        __gm__ ElementC* ptrC;
        __gm__ int32_t* ptrReadyFrontier;
        uint32_t windowRows;
        LayoutB layoutB;
        decltype(tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagB, true>(0U, 0U)) layoutScaleB;
        decltype(tla::MakeLayoutFromTag(Catlass::layout::RowMajor(0U, 0U, 0U))) layoutC;

        CATLASS_HOST_DEVICE Params() = default;
        CATLASS_HOST_DEVICE explicit Params(const Arguments& args)
            : chunkM(args.chunkM),
              recvRows(args.recvRows),
              fullK(args.rankSize * args.localK),
              localK(args.localK),
              n(args.n),
              ptrRecvA(reinterpret_cast<__gm__ ElementA*>(args.ptrRecvA)),
              ptrRecvScaleA(reinterpret_cast<__gm__ ElementMxScaleA*>(args.ptrRecvScaleA)),
              ptrB(reinterpret_cast<__gm__ ElementB*>(args.ptrB)),
              ptrScaleB(reinterpret_cast<__gm__ ElementMxScaleB*>(args.ptrScaleB)),
              ptrC(reinterpret_cast<__gm__ ElementC*>(args.ptrC)),
              ptrReadyFrontier(reinterpret_cast<__gm__ int32_t*>(args.ptrReadyFrontier)),
              windowRows(args.windowRows),
              layoutB(tla::MakeLayoutFromTag(LayoutTagB{fullK, args.n})),
              layoutScaleB(tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagB, true>(fullK / 32, args.n)),
              layoutC(tla::MakeLayoutFromTag(Catlass::layout::RowMajor{args.chunkM, args.n, args.n}))
        {}
    };

    static size_t GetWorkspaceSize(const Arguments&) { return 0; }
    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* = nullptr) { return Params(args); }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params& params)
    {
        (void)params;
    }

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
            if (params.ptrReadyFrontier != nullptr) {
                aclshmem_signal_wait_until(
                    params.ptrReadyFrontier, ACLSHMEM_CMP_GE,
                    static_cast<int32_t>((row + rows - 1) / params.windowRows + 1));
            }
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
                recvA, recvScaleA, bTile, cTile, Catlass::GemmCoord{rows, cols, params.fullK}, row, params.recvRows,
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
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_COMPUTE_H
