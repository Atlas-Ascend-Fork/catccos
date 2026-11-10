
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_GMM_ALLTOALL_HPP
#define CATCCOS_DGEMM_KERNEL_ASCEND950_GMM_ALLTOALL_HPP

#include <type_traits>
#include "catccos/detail/ascend950_moe_ready.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_operator.h"
#include "shmem.h"
#include "tla/tensor.hpp"

namespace Catccos::DGemm::Kernel {

template <class ArchTag_, class TensorSrc_, class TensorDst_, uint32_t TargetUbIdx_>
struct CopyL0CToTargetUbTla {
    using ArchTag = ArchTag_;
    using TensorSrc = TensorSrc_;
    using TensorDst = TensorDst_;
    using ElementSrc = typename TensorSrc::Element;
    using ElementDst = typename TensorDst::Element;

    static constexpr auto kQuantMode = Catlass::Gemm::Tile::CopyL0CToDstQuantMode<
        ArchTag, ElementSrc, ElementDst, Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT>::VALUE;

    template <class DstTensor, class SrcTensor>
    CATLASS_DEVICE void operator()(DstTensor const& dstTensor, SrcTensor const& srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            std::is_same_v<ArchTag, Catlass::Arch::Ascend950>, "Target UB copy is only implemented for Ascend950");
        static_assert(
            tla::detail::isRowMajor<typename DstTensor::Layout>::value &&
                SrcTensor::position == AscendC::TPosition::CO1 && DstTensor::position == AscendC::TPosition::VECCALC,
            "Source must be L0C and destination must be row-major VECCALC UB");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());
        intriParams.quantPre = kQuantMode;
        intriParams.reluEn = false;
        intriParams.unitFlag = unitFlag;
        intriParams.subBlockId = TargetUbIdx_;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <
    class ArchTag, class ElementA, class LayoutTagA, class ElementB, class LayoutTagB, class ElementC, class LayoutTagC,
    uint32_t TargetUbIdx, class ElementBias = void>
struct PackedTileCopyTlaToTargetUb
    : public Catlass::Gemm::Tile::PackedTileCopyTla<
          ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias> {
    using Base = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, ElementBias>;
    using TensorL0C = typename Base::TensorL0C;

    template <class TensorC>
    using CopyL0CToDst = CopyL0CToTargetUbTla<ArchTag, TensorL0C, TensorC, TargetUbIdx>;
};

template <
    class ArchTag, class ElementA, class LayoutTagA, class ElementB, class LayoutTagB, class ElementMxScaleA,
    class LayoutMxScaleA, class ElementMxScaleB, class LayoutMxScaleB, class ElementC, class LayoutTagC,
    uint32_t TargetUbIdx, class ElementBias = void>
struct PackedMxTileCopyTlaToTargetUb
    : public Catlass::Gemm::Tile::PackedMxTileCopyTla<
          ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScaleA, LayoutMxScaleA, ElementMxScaleB,
          LayoutMxScaleB, ElementC, LayoutTagC, ElementBias> {
    using Base = Catlass::Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScaleA, LayoutMxScaleA, ElementMxScaleB,
        LayoutMxScaleB, ElementC, LayoutTagC, ElementBias>;
    using TensorL0C = typename Base::TensorL0C;

    template <class TensorC>
    using CopyL0CToDst = CopyL0CToTargetUbTla<ArchTag, TensorL0C, TensorC, TargetUbIdx>;
};

template <
    class BlockMmadToUb_, class BlockScheduler_, class MetadataScheduler_, uint32_t ReadinessTileM_,
    uint32_t ReadinessTileN_, uint32_t UbCapacityBytes_ = Catlass::Arch::Ascend950::UB_SIZE,
    bool WorldBarrierAtEnd_ = true>
class Ascend950GmmAllToAllKernel {
public:
    using BlockMmadToUb = BlockMmadToUb_;
    using BlockScheduler = BlockScheduler_;
    using MetadataScheduler = MetadataScheduler_;
    using ArchTag = typename BlockMmadToUb::ArchTag;
    using L1TileShape = typename BlockMmadToUb::L1TileShape;
    using ElementA = typename BlockMmadToUb::ElementA;
    using LayoutA = typename BlockMmadToUb::TileCopy::LayoutTagA;
    using ElementB = typename BlockMmadToUb::ElementB;
    using LayoutB = typename BlockMmadToUb::TileCopy::LayoutTagB;
    using ElementC = typename BlockMmadToUb::ElementC;
    using LayoutC = typename BlockMmadToUb::TileCopy::LayoutTagC;
    using ElementScaleA = typename BlockMmadToUb::TileCopy::ElementMxScaleA;
    using ElementScaleB = typename BlockMmadToUb::TileCopy::ElementMxScaleB;

    static constexpr uint32_t kTileM = tla::get<0>(L1TileShape{});
    static constexpr uint32_t kTileN = tla::get<1>(L1TileShape{});
    static constexpr uint32_t kTileK = tla::get<2>(L1TileShape{});
    static constexpr uint32_t kElementsPerBlock = Catlass::BYTE_PER_BLK / sizeof(ElementC);
    static constexpr uint32_t kUbTileNAlign =
        ((kTileN + kElementsPerBlock - 1) / kElementsPerBlock) * kElementsPerBlock;
    static constexpr uint32_t kUbTileElements = kTileM * kUbTileNAlign;
    static constexpr uint32_t kUbTileBytes = kUbTileElements * sizeof(ElementC);
    static constexpr uint32_t kUbCapacityBytes = UbCapacityBytes_;
    static constexpr uint32_t kReadinessTileM = ReadinessTileM_;
    static constexpr uint32_t kReadinessTileN = ReadinessTileN_;
    static constexpr uint32_t kReadinessStride = Catccos::Detail::Ascend950MoeReady::kStride;
    static constexpr bool kWorldBarrierAtEnd = WorldBarrierAtEnd_;

    static constexpr uint16_t kUbSyncMode = 0x4;
    static constexpr uint16_t kAicSyncAivFlag = 4;
    static constexpr uint16_t kAivSyncAicFlag = 6;
    static constexpr uint16_t kDispatchUbFreeFlag = 2;
    static constexpr uint16_t kAiv1RouteOffset = 16;

    static_assert(
        std::is_same_v<ArchTag, Catlass::Arch::Ascend950>,
        "Direct GMM2 target-UB path is only implemented for Ascend950");
    static_assert(
        std::is_same_v<LayoutC, Catlass::layout::RowMajor>, "Direct GMM2 combine currently requires row-major output");
    static_assert(
        2 * kUbTileBytes <= kUbCapacityBytes, "GMM2 target-UB ping-pong buffers exceed Ascend950 UB capacity");
    static_assert(kReadinessTileM > 0 && kReadinessTileN > 0, "SwiGLU readiness tile shape must be non-zero");
    static_assert(
        kReadinessStride * sizeof(int32_t) == 64, "SwiGLU readiness counters must occupy separate cache lines");

    struct Params {
        Catlass::GemmCoord problemShape;
        int32_t EP;
        int32_t expertPerRank;
        uint32_t maxOutputSize;
        uint32_t rank;
        uint32_t rankSize;
        __gm__ int32_t* ptrTokenPerExpert;
        __gm__ int32_t* ptrCumsumMM;
        __gm__ ElementA* ptrA;
        LayoutA layoutA;
        __gm__ ElementB* ptrB;
        LayoutB layoutB;
        __gm__ ElementScaleA* ptrScaleA;
        __gm__ ElementScaleB* ptrScaleB;
        GM_ADDR ptrWorkspace;
        __gm__ int32_t* ptrSwigluReady;
        GM_ADDR symmetricPtr;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(
            Catlass::GemmCoord problemShape_, uint32_t EP_, uint32_t expertPerRank_, uint32_t maxOutputSize_,
            uint32_t rank_, uint32_t rankSize_, GM_ADDR ptrTokenPerExpert_, GM_ADDR ptrCumsumMM_, GM_ADDR ptrA_,
            LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrScaleA_, GM_ADDR ptrScaleB_,
            GM_ADDR ptrWorkspace_, GM_ADDR ptrSwigluReady_, GM_ADDR symmetricPtr_)
            : problemShape(problemShape_),
              EP(EP_),
              expertPerRank(expertPerRank_),
              maxOutputSize(maxOutputSize_),
              rank(rank_),
              rankSize(rankSize_),
              ptrTokenPerExpert(reinterpret_cast<__gm__ int32_t*>(ptrTokenPerExpert_)),
              ptrCumsumMM(reinterpret_cast<__gm__ int32_t*>(ptrCumsumMM_)),
              ptrA(reinterpret_cast<__gm__ ElementA*>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB*>(ptrB_)),
              layoutB(layoutB_),
              ptrScaleA(reinterpret_cast<__gm__ ElementScaleA*>(ptrScaleA_)),
              ptrScaleB(reinterpret_cast<__gm__ ElementScaleB*>(ptrScaleB_)),
              ptrWorkspace(ptrWorkspace_),
              ptrSwigluReady(reinterpret_cast<__gm__ int32_t*>(ptrSwigluReady_)),
              symmetricPtr(symmetricPtr_)
        {}
    };

    struct WorkspaceInfo {
        GM_ADDR ptrDstExpertOffset;
        __gm__ int32_t* ptrCumsumMM;

        CATLASS_DEVICE
        explicit WorkspaceInfo(Params const& params)
            : ptrDstExpertOffset(params.ptrWorkspace), ptrCumsumMM(params.ptrCumsumMM)
        {}
    };

    CATLASS_DEVICE
    Ascend950GmmAllToAllKernel() = default;

    template <int32_t CoreType = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params, Catlass::Arch::Resource<ArchTag> resource);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params, Catlass::Arch::Resource<ArchTag> resource)
    {
        WaitForDispatchUbFree();

        BlockScheduler blockScheduler;
        BlockMmadToUb blockMmad(resource);
        WorkspaceInfo workspaceInfo(params);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementScaleA> gmScaleA;
        gmScaleA.SetGlobalBuffer(params.ptrScaleA);
        AscendC::GlobalTensor<ElementScaleB> gmScaleB;
        gmScaleB.SetGlobalBuffer(params.ptrScaleB);
        AscendC::GlobalTensor<int32_t> cumsumMM;
        cumsumMM.SetGlobalBuffer(workspaceInfo.ptrCumsumMM);

        auto ubFirst = resource.ubBuf.template GetBufferByByte<ElementC>(0);
        auto ubSecond = resource.ubBuf.template GetBufferByByte<ElementC>(kUbTileBytes);
        auto layoutUb = tla::MakeLayout<ElementC, Catlass::layout::RowMajor>(kTileM, kUbTileNAlign);
        auto tensorUbFirst = tla::MakeTensor(ubFirst, layoutUb, Catlass::Arch::PositionUB{});
        auto tensorUbSecond = tla::MakeTensor(ubSecond, layoutUb, Catlass::Arch::PositionUB{});

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t startCoreIdx = 0;
        uint32_t producedTiles = 0;
        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetScaleA = 0;
        int64_t gmGroupOffsetScaleB = 0;

        for (uint32_t groupIdx = 0; groupIdx < static_cast<uint32_t>(params.expertPerRank); ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            Catlass::GemmCoord groupProblem{currentM, params.problemShape.n(), params.problemShape.k()};

            LayoutA layoutTagA = params.layoutA.GetTileLayout(groupProblem.GetCoordMK());
            auto tensorA =
                tla::MakeTensor(gmA[gmGroupOffsetA], tla::MakeLayoutFromTag(layoutTagA), Catlass::Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(
                gmB[gmGroupOffsetB], tla::MakeLayoutFromTag(params.layoutB), Catlass::Arch::PositionGM{});
            uint32_t scaleK = RoundUp<2>(CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(groupProblem.k()));
            auto layoutScaleA = tla::MakeMxScaleLayout<ElementScaleA, LayoutA, false>(currentM, scaleK);
            auto layoutScaleB = tla::MakeMxScaleLayout<ElementScaleB, LayoutB, true>(scaleK, groupProblem.n());
            auto tensorScaleA =
                tla::MakeTensor(gmScaleA[gmGroupOffsetScaleA], layoutScaleA, Catlass::Arch::PositionGM{});
            auto tensorScaleB =
                tla::MakeTensor(gmScaleB[gmGroupOffsetScaleB], layoutScaleB, Catlass::Arch::PositionGM{});

            blockScheduler.Update(groupProblem, Catlass::MakeCoord(kTileM, kTileN));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;

            if (startLoopIdx < coreLoops) {
                WaitForSwigluReady(params, groupIdx, currentM);
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                uint32_t pingPong = producedTiles & 1U;
                if (producedTiles >= 2) {
                    WaitForAiv1Free(pingPong);
                }

                Catlass::GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                Catlass::GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                Catlass::MatrixCoord offsetA{blockCoord.m() * kTileM, blockCoord.k() * kTileK};
                Catlass::MatrixCoord offsetB{blockCoord.k() * kTileK, blockCoord.n() * kTileN};

                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(offsetA.row(), offsetA.column()),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB = GetTile(
                    tensorB, tla::MakeCoord(offsetB.row(), offsetB.column()),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockScaleA = GetTile(
                    tensorScaleA, tla::MakeCoord(offsetA.row(), offsetA.column() / Catlass::MX_SCALE_GROUP_NUM),
                    tla::MakeShape(actualBlockShape.m(), CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualBlockShape.k())));
                auto tensorBlockScaleB = GetTile(
                    tensorScaleB, tla::MakeCoord(offsetB.row() / Catlass::MX_SCALE_GROUP_NUM, offsetB.column()),
                    tla::MakeShape(CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));
                auto tensorUb = pingPong == 0 ? tensorUbFirst : tensorUbSecond;
                auto tensorBlockC =
                    GetTile(tensorUb, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                blockMmad(
                    tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockScaleA, tensorBlockScaleB);
                NotifyAiv1Ready(pingPong);
                ++producedTiles;
            }

            gmGroupOffsetA += static_cast<int64_t>(currentM) * params.problemShape.k();
            gmGroupOffsetB += static_cast<int64_t>(params.problemShape.k()) * params.problemShape.n();
            gmGroupOffsetScaleA += static_cast<int64_t>(currentM) * scaleK;
            gmGroupOffsetScaleB += static_cast<int64_t>(scaleK) * params.problemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if (producedTiles == 1) {
            WaitForAiv1Free(0);
        } else if (producedTiles >= 2) {
            WaitForAiv1Free(0);
            WaitForAiv1Free(1);
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params, Catlass::Arch::Resource<ArchTag> resource)
    {
        WorkspaceInfo workspaceInfo(params);
        AscendC::GlobalTensor<int32_t> cumsumMM;
        cumsumMM.SetGlobalBuffer(workspaceInfo.ptrCumsumMM);
        AscendC::GlobalTensor<int32_t> dstExpertOffset;
        dstExpertOffset.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrDstExpertOffset));

        if (AscendC::GetSubBlockIdx() == 1) {
            ConsumeTiles(params, cumsumMM, dstExpertOffset, resource);
        }
        if constexpr (kWorldBarrierAtEnd) {
            aclshmemx_barrier_all_vec();
        } else {
            // The fused caller performs the only required world barrier immediately
            // before unpermute. Here it only needs to align local AIV0/AIV1 cores.
            AscendC::SyncAll<true>();
        }
    }

private:
    CATLASS_DEVICE
    static void WaitForSwigluReady(Params const& params, uint32_t groupIdx, uint32_t currentM)
    {
        uint32_t tileRows = CeilDiv(currentM, kReadinessTileM);
        uint32_t tileColumns = CeilDiv(params.problemShape.k(), kReadinessTileN);
        int32_t readyTarget = static_cast<int32_t>(tileRows * tileColumns);
        __gm__ int32_t* ready = params.ptrSwigluReady + groupIdx * kReadinessStride;
        while (AscendC::ReadGmByPassDCache(ready) != readyTarget) {
            int64_t start = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - start < 100) {
            }
        }
    }

    CATLASS_DEVICE
    static void WaitForDispatchUbFree()
    {
        AscendC::CrossCoreWaitFlag<kUbSyncMode, PIPE_FIX>(kAivSyncAicFlag + kDispatchUbFreeFlag + kAiv1RouteOffset);
    }

    CATLASS_DEVICE
    static void NotifyAiv1Ready(uint32_t pingPong)
    {
        AscendC::CrossCoreSetFlag<kUbSyncMode, PIPE_FIX>(kAicSyncAivFlag + kAiv1RouteOffset + pingPong);
    }

    CATLASS_DEVICE
    static void WaitForAiv1Free(uint32_t pingPong)
    {
        AscendC::CrossCoreWaitFlag<kUbSyncMode, PIPE_FIX>(kAivSyncAicFlag + kAiv1RouteOffset + pingPong);
    }

    CATLASS_DEVICE
    static void WaitForAicReady(uint32_t pingPong)
    {
        AscendC::CrossCoreWaitFlag<kUbSyncMode, PIPE_V>(kAicSyncAivFlag + pingPong);
    }

    CATLASS_DEVICE
    static void NotifyAicFree(uint32_t pingPong)
    {
        AscendC::CrossCoreSetFlag<kUbSyncMode, PIPE_V>(kAivSyncAicFlag + pingPong);
    }

    CATLASS_DEVICE
    static void ConsumeTiles(
        Params const& params, AscendC::GlobalTensor<int32_t>& cumsumMM, AscendC::GlobalTensor<int32_t>& dstExpertOffset,
        Catlass::Arch::Resource<ArchTag> resource)
    {
        BlockScheduler blockScheduler;
        auto ubFirst = resource.ubBuf.template GetBufferByByte<ElementC>(0);
        auto ubSecond = resource.ubBuf.template GetBufferByByte<ElementC>(kUbTileBytes);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t startCoreIdx = 0;
        uint32_t consumedTiles = 0;

        for (uint32_t groupIdx = 0; groupIdx < static_cast<uint32_t>(params.expertPerRank); ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            Catlass::GemmCoord groupProblem{currentM, params.problemShape.n(), params.problemShape.k()};
            blockScheduler.Update(groupProblem, Catlass::MakeCoord(kTileM, kTileN));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                uint32_t pingPong = consumedTiles & 1U;
                Catlass::GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                Catlass::GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                uint32_t mLoc = blockCoord.m() * kTileM;
                uint32_t nLoc = blockCoord.n() * kTileN;
                auto ubTile = pingPong == 0 ? ubFirst : ubSecond;

                WaitForAicReady(pingPong);
                SendTileToPeers(params, groupIdx, mLoc, nLoc, actualBlockShape, ubTile, cumsumMM, dstExpertOffset);
                NotifyAicFree(pingPong);
                ++consumedTiles;
            }
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
    }

    CATLASS_DEVICE
    static void SendTileToPeers(
        Params const& params, uint32_t groupIdx, uint32_t mLoc, uint32_t nLoc,
        Catlass::GemmCoord const& actualBlockShape, AscendC::LocalTensor<ElementC> const& ubTile,
        AscendC::GlobalTensor<int32_t>& cumsumMM, AscendC::GlobalTensor<int32_t>& dstExpertOffset)
    {
        uint32_t tileStart = mLoc;
        uint32_t tileEnd = tileStart + actualBlockShape.m();
        uint32_t actualN = actualBlockShape.n();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        for (uint32_t dstRank = 0; dstRank < params.rankSize; ++dstRank) {
            uint32_t rankStart = dstRank == 0 ? 0 : cumsumMM((dstRank - 1) * params.expertPerRank + groupIdx);
            uint32_t rankEnd = cumsumMM(dstRank * params.expertPerRank + groupIdx);
            uint32_t copyStart = tla::max(tileStart, rankStart);
            uint32_t copyEnd = tla::min(tileEnd, rankEnd);
            if (copyStart >= copyEnd) {
                continue;
            }

            uint32_t copyRows = copyEnd - copyStart;
            uint32_t ubRow = copyStart - tileStart;
            uint32_t dstRow = dstExpertOffset(dstRank * params.expertPerRank + groupIdx) + copyStart - rankStart;

            __gm__ void* remotePtr = aclshmem_ptr(params.symmetricPtr, dstRank);
            AscendC::GlobalTensor<ElementC> remoteOutput;
            remoteOutput.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(remotePtr));
            int64_t remoteOffset = static_cast<int64_t>(dstRow) * params.problemShape.n() + nLoc;

            uint32_t rowBytes = actualN * sizeof(ElementC);
            int64_t srcStride = static_cast<int64_t>(kUbTileNAlign - actualN) * sizeof(ElementC);
            int64_t dstStride = static_cast<int64_t>(params.problemShape.n() - actualN) * sizeof(ElementC);
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(copyRows), rowBytes, srcStride, dstStride, 0};
            AscendC::DataCopyPad(remoteOutput[remoteOffset], ubTile[ubRow * kUbTileNAlign], copyParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_ASCEND950_GMM_ALLTOALL_HPP
