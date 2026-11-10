
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_EXAMPLES_AUX_OPS_ASCEND950_SWIGLU_FROM_UB_H
#define CATCCOS_EXAMPLES_AUX_OPS_ASCEND950_SWIGLU_FROM_UB_H

#include <type_traits>
#include "catccos/detail/ascend950_moe_ready.hpp"

#include "catccos/comm/block/comm_block_mx_quant.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/catlass.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "kernel_operator.h"

namespace Catccos::DGemm::Kernel {

struct Ascend950AicUbSyncState {
    bool outstanding{false};
};

template <class SwigluKernel>
struct Ascend950AicWaitUbFree {
    Ascend950AicUbSyncState* state{nullptr};

    CATLASS_DEVICE Ascend950AicWaitUbFree() = default;

    CATLASS_DEVICE
    explicit Ascend950AicWaitUbFree(Ascend950AicUbSyncState& state_) : state(&state_) {}

    CATLASS_DEVICE
    void operator()() const
    {
        if (state != nullptr && !state->outstanding) {
            return;
        }
        AscendC::CrossCoreWaitFlag<SwigluKernel::kUbSyncMode, PIPE_FIX>(SwigluKernel::kAivDoneFlag);
        if (state != nullptr) {
            state->outstanding = false;
        }
    }
};

template <class SwigluKernel>
struct Ascend950AicNotifyUbReady {
    Ascend950AicUbSyncState* state{nullptr};

    CATLASS_DEVICE Ascend950AicNotifyUbReady() = default;

    CATLASS_DEVICE
    explicit Ascend950AicNotifyUbReady(Ascend950AicUbSyncState& state_) : state(&state_) {}

    CATLASS_DEVICE
    void operator()() const
    {
        AscendC::CrossCoreSetFlag<SwigluKernel::kUbSyncMode, PIPE_FIX>(SwigluKernel::kAicReadyFlag);
        if (state != nullptr) {
            state->outstanding = true;
        }
    }
};

template <class ArchTag_, class SwigluBlock_, class BlockScheduler_, uint32_t UbTileM_, uint32_t UbTileN_>
class Ascend950SwigluFromUbKernel {
public:
    using ArchTag = ArchTag_;
    using SwigluBlock = SwigluBlock_;
    using BlockScheduler = BlockScheduler_;
    using ElementC = typename SwigluBlock::ElementC;
    using ElementD = typename SwigluBlock::ElementD;

    static constexpr uint32_t UB_TILE_M = UbTileM_;
    static constexpr uint32_t UB_TILE_N = UbTileN_;
    static constexpr uint16_t kUbSyncMode = 0x4;
    static constexpr int16_t kAivDoneFlag = 5;
    static constexpr int16_t kAicReadyFlag = 6;
    static constexpr int16_t kDownReadyFlag = 4;
    static constexpr uint32_t kElementPerBlock = Catlass::BYTE_PER_BLK / sizeof(ElementC);
    static constexpr uint32_t kUbTileNAlign =
        ((UB_TILE_N + kElementPerBlock - 1) / kElementPerBlock) * kElementPerBlock;
    static constexpr uint32_t kUbTileElem = UB_TILE_M * kUbTileNAlign;

    struct Params {
        uint32_t problemCount;
        Catlass::MatrixCoord problemShape;
        __gm__ ElementD* ptrD;
        GM_ADDR ptrGroupList;
        Callback notifyCallback;
        uint32_t syncInterval;

        CATLASS_DEVICE
        Params() = default;

        CATLASS_DEVICE
        Params(
            uint32_t problemCount_, Catlass::MatrixCoord problemShape_, GM_ADDR ptrD_, GM_ADDR ptrGroupList_,
            const Callback& notifyCallback_, uint32_t syncInterval_)
            : problemCount(problemCount_),
              problemShape(problemShape_),
              ptrD(reinterpret_cast<__gm__ ElementD*>(ptrD_)),
              ptrGroupList(ptrGroupList_),
              notifyCallback(notifyCallback_),
              syncInterval(syncInterval_)
        {}
    };

    CATLASS_DEVICE
    void operator()(Params const& params, Catlass::Arch::Resource<ArchTag> resource)
    {
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }

        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);
        AscendC::GlobalTensor<int32_t> groupList;
        groupList.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrGroupList));

        BlockScheduler blockScheduler;
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t nOut = params.problemShape.column() / 2;
        int64_t gmGroupOffsetD = 0;
        uint32_t startCoreIdx = 0;

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = groupList(groupIdx);
            Catlass::GemmCoord inGroupProblemShape{currentM, nOut, 1};

            blockScheduler.Update(inGroupProblemShape, Catlass::MakeCoord(UB_TILE_M, UB_TILE_N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;

            if (startLoopIdx < coreLoops) {
            }

            bool firstTile = true;
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                Catlass::GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                Catlass::GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                Catlass::MatrixCoord tileOffset{blockCoord.m() * UB_TILE_M, blockCoord.n() * UB_TILE_N};
                Catlass::MatrixCoord actualTileShape{actualBlockShape.m(), actualBlockShape.n()};

                AscendC::CrossCoreWaitFlag<kUbSyncMode, PIPE_V>(kAicReadyFlag);
                ComputeUbTile(resource, gmD, gmGroupOffsetD, tileOffset, actualTileShape, nOut);
                AscendC::CrossCoreSetFlag<kUbSyncMode, PIPE_MTE3>(kAivDoneFlag);
                firstTile = false;
            }

            gmGroupOffsetD += static_cast<int64_t>(currentM) * nOut;
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;

            if ((groupIdx + 1) % params.syncInterval == 0 || groupIdx == params.problemCount - 1) {
                AscendC::PipeBarrier<PIPE_MTE3>();
                params.notifyCallback();
            }
        }
    }

private:
    CATLASS_DEVICE
    void ComputeUbTile(
        Catlass::Arch::Resource<ArchTag> resource, AscendC::GlobalTensor<ElementD>& gmD, int64_t gmGroupOffsetD,
        Catlass::MatrixCoord const& tileOffset, Catlass::MatrixCoord const& actualTileShape, uint32_t nOut)
    {
        static_assert(
            std::is_same_v<ElementC, ElementD>,
            "Direct UB SwiGLU currently requires identical BF16 input and output types");

        auto ubGate = resource.ubBuf.template GetBufferByByte<ElementC>(0);
        auto ubUp = resource.ubBuf.template GetBufferByByte<ElementC>(kUbTileElem * sizeof(ElementC));

        // SwiGLU consumes gate before storing the corresponding result, so the output can reuse gate UB.
        auto ubOut = ubGate;
        uint16_t tileM = static_cast<uint16_t>(actualTileShape.row());
        uint32_t tileN = static_cast<uint32_t>(actualTileShape.column());
        SwigluBlock::ComputeFromUb(ubOut, ubGate, ubUp, tileM, tileN, kUbTileNAlign, kUbTileNAlign);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(tileN * sizeof(ElementD)), 0, 0, 0};
        for (uint32_t row = 0; row < actualTileShape.row(); ++row) {
            uint32_t ubOffset = row * kUbTileNAlign;
            int64_t gmDOffset =
                gmGroupOffsetD + static_cast<int64_t>(tileOffset.row() + row) * nOut + tileOffset.column();
            AscendC::DataCopyPad(gmD[gmDOffset], ubOut[ubOffset], copyParams);
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }
};

template <
    class ArchTag_, class SwigluBlock_, class ElementD_, class ElementScale_, class BlockScheduler_, uint32_t UbTileM_,
    uint32_t UbTileN_>
class Ascend950SwigluMxQuantFromUbKernel {
public:
    using ArchTag = ArchTag_;
    using SwigluBlock = SwigluBlock_;
    using ElementC = typename SwigluBlock::ElementC;
    using ElementD = ElementD_;
    using ElementScale = ElementScale_;
    using BlockScheduler = BlockScheduler_;
    using InputType = Catlass::Gemm::GemmType<ElementC, Catlass::layout::RowMajor>;
    using OutputType = Catlass::Gemm::GemmType<ElementD, Catlass::layout::RowMajor>;
    using BlockMxQuant =
        Catccos::Comm::Block::CommBlockMxQuant<1, Catlass::MX_SCALE_GROUP_NUM, 2, InputType, OutputType>;

    static constexpr uint32_t UB_TILE_M = UbTileM_;
    static constexpr uint32_t UB_TILE_N = UbTileN_;
    static constexpr uint16_t kUbSyncMode = 0x4;
    static constexpr int16_t kAivDoneFlag = 5;
    static constexpr int16_t kAicReadyFlag = 6;
    static constexpr uint32_t kMxBlockSize = Catlass::MX_SCALE_GROUP_NUM;
    static constexpr uint32_t kElementPerBlock = Catlass::BYTE_PER_BLK / sizeof(ElementC);
    static constexpr uint32_t kUbTileNAlign =
        ((UB_TILE_N + kElementPerBlock - 1) / kElementPerBlock) * kElementPerBlock;
    static constexpr uint32_t kUbTileElem = UB_TILE_M * kUbTileNAlign;
    static constexpr uint32_t kUbPlaneBytes = kUbTileElem * sizeof(ElementC);
    static constexpr uint32_t kReadyStride = Catccos::Detail::Ascend950MoeReady::kStride;
    static constexpr bool kUseTileWideMxQuant = UB_TILE_M == 256;
    static_assert(UB_TILE_M == 128 || UB_TILE_M == 256, "Unsupported GMM1 UB tile height");
    static_assert(std::is_same_v<ElementC, bfloat16_t>, "SwiGLU MX quant currently consumes BF16 GMM output");
    static_assert(2 * kUbPlaneBytes <= 256 * 1024, "SwiGLU gate/up input planes exceed the physical Ascend950 UB");
    static_assert(kReadyStride * sizeof(int32_t) == 64, "SwiGLU readiness counters must occupy separate cache lines");

    struct Params {
        uint32_t problemCount;
        Catlass::MatrixCoord problemShape;
        __gm__ ElementC* ptrBf16;
        __gm__ ElementD* ptrD;
        __gm__ ElementScale* ptrScale;
        GM_ADDR ptrGroupList;
        __gm__ int32_t* ptrReady;

        CATLASS_DEVICE Params() = default;

        CATLASS_DEVICE
        Params(
            uint32_t problemCount_, Catlass::MatrixCoord problemShape_, GM_ADDR ptrBf16_, GM_ADDR ptrD_,
            GM_ADDR ptrScale_, GM_ADDR ptrGroupList_, GM_ADDR ptrReady_)
            : problemCount(problemCount_),
              problemShape(problemShape_),
              ptrBf16(reinterpret_cast<__gm__ ElementC*>(ptrBf16_)),
              ptrD(reinterpret_cast<__gm__ ElementD*>(ptrD_)),
              ptrScale(reinterpret_cast<__gm__ ElementScale*>(ptrScale_)),
              ptrGroupList(ptrGroupList_),
              ptrReady(reinterpret_cast<__gm__ int32_t*>(ptrReady_))
        {}
    };

    CATLASS_DEVICE
    void operator()(Params const& params, Catlass::Arch::Resource<ArchTag> resource)
    {
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }

        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);
        AscendC::GlobalTensor<ElementScale> gmScale;
        gmScale.SetGlobalBuffer(params.ptrScale);
        AscendC::GlobalTensor<int32_t> groupList;
        groupList.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrGroupList));

        typename BlockMxQuant::Params tileQuantParams{UB_TILE_N};
        BlockMxQuant tileMxQuant(resource, tileQuantParams);
        if constexpr (kUseTileWideMxQuant) {
            tileMxQuant.InitBlockLoop();
        }

        BlockScheduler blockScheduler;
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t nOut = params.problemShape.column() / 2;
        uint32_t scaleN = RoundUp<2>(CeilDiv<kMxBlockSize>(nOut));
        int64_t gmGroupOffsetD = 0;
        int64_t gmGroupOffsetScale = 0;
        uint32_t startCoreIdx = 0;
        (void)params.ptrBf16;

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = groupList(groupIdx);
            Catlass::GemmCoord inGroupProblemShape{currentM, nOut, 1};
            blockScheduler.Update(inGroupProblemShape, Catlass::MakeCoord(UB_TILE_M, UB_TILE_N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;

            if (startLoopIdx < coreLoops) {
            }

            bool firstTile = true;
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                Catlass::GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                Catlass::GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                Catlass::MatrixCoord tileOffset{blockCoord.m() * UB_TILE_M, blockCoord.n() * UB_TILE_N};
                Catlass::MatrixCoord actualTileShape{actualBlockShape.m(), actualBlockShape.n()};

                AscendC::CrossCoreWaitFlag<kUbSyncMode, PIPE_V>(kAicReadyFlag);
                ComputeUbTile(
                    resource, gmD, gmScale, gmGroupOffsetD, gmGroupOffsetScale, tileOffset, actualTileShape, nOut,
                    scaleN, tileMxQuant);
                // Publish this tile only after both quant data and scale MTE3
                // stores are globally visible. GMM2 waits on the exact expert
                // tile count and no longer needs a batched cross-core barrier.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
                AscendC::AtomicAdd(params.ptrReady + groupIdx * kReadyStride, static_cast<int32_t>(1));
                AscendC::CrossCoreSetFlag<kUbSyncMode, PIPE_MTE3>(kAivDoneFlag);
                firstTile = false;
            }

            gmGroupOffsetD += static_cast<int64_t>(currentM) * nOut;
            gmGroupOffsetScale += static_cast<int64_t>(currentM) * scaleN;
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        if constexpr (kUseTileWideMxQuant) {
            tileMxQuant.FinalizeBlockLoop();
        }
    }

private:
    CATLASS_DEVICE
    void ComputeUbTile(
        Catlass::Arch::Resource<ArchTag> resource, AscendC::GlobalTensor<ElementD>& gmD,
        AscendC::GlobalTensor<ElementScale>& gmScale, int64_t gmGroupOffsetD, int64_t gmGroupOffsetScale,
        Catlass::MatrixCoord const& tileOffset, Catlass::MatrixCoord const& actualTileShape, uint32_t nOut,
        uint32_t scaleN, BlockMxQuant& tileMxQuant)
    {
        uint16_t tileM = static_cast<uint16_t>(actualTileShape.row());
        uint32_t tileN = static_cast<uint32_t>(actualTileShape.column());
        auto ubGate = resource.ubBuf.template GetBufferByByte<ElementC>(0);
        auto ubUp = resource.ubBuf.template GetBufferByByte<ElementC>(kUbPlaneBytes);

        // Both BF16 input planes are consumed before CommBlockMxQuant reuses
        // their storage for its output/scale scratch. The AIC/AIV ready/free
        // handshake prevents the producer from overwriting either plane.
        // Keep the GMM1 result in UB and feed the proven CommBlock MX quantizer
        // directly.  This removes the old BF16 GM store + MTE2 reload roundtrip.
        // The FFN host contract requires N % 512 == 0, so every N tile is full
        // width and the SwiGLU output is contiguous in UB.
        auto ubOut = ubGate;
        SwigluBlock::ComputeFromUb(ubOut, ubGate, ubUp, tileM, tileN, kUbTileNAlign, kUbTileNAlign);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        int64_t outRow = tileOffset.row();
        int64_t gmOffset = gmGroupOffsetD + outRow * nOut + tileOffset.column();
        int64_t gmScaleOffset = gmGroupOffsetScale + outRow * scaleN + tileOffset.column() / kMxBlockSize;

        auto quantLayout = Catlass::layout::RowMajor(actualTileShape.row(), nOut);
        auto scaleLayout = Catlass::layout::RowMajor(actualTileShape.row(), scaleN);
        if constexpr (kUseTileWideMxQuant) {
            tileMxQuant.QuantFromUb2D(
                ubOut, gmD[gmOffset], quantLayout, gmScale[gmScaleOffset], scaleLayout, actualTileShape);
        } else {
            typename BlockMxQuant::Params quantParams{tileN};
            BlockMxQuant rowMxQuant(resource, quantParams);
            rowMxQuant.InitBlockLoop();
            rowMxQuant.QuantFromUb(
                ubOut, gmD[gmOffset], quantLayout, gmScale[gmScaleOffset], scaleLayout, actualTileShape);
            rowMxQuant.FinalizeBlockLoop();
        }
    }
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_EXAMPLES_AUX_OPS_ASCEND950_SWIGLU_FROM_UB_H
