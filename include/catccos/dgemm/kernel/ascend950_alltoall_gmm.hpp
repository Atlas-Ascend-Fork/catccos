
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALL_GMM_HPP
#define CATCCOS_DGEMM_KERNEL_ASCEND950_ALLTOALL_GMM_HPP

#include <type_traits>
#include "catccos/detail/ascend950_moe_ready.hpp"

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/catlass.hpp"
// #include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif

#include "catccos/comm/block/comm_block_mx_quant.hpp"
#include "kernel_operator.h"
#include "shmem.h"

using namespace AscendC;
using namespace Catlass;

namespace Catccos::DGemm::Kernel {

template <class BlockMmad, class = void>
struct Ascend950MxMmadTraits {
    static constexpr bool kEnabled = false;
    using ElementScaleA = uint8_t;
    using ElementScaleB = uint8_t;
};

template <class BlockMmad>
struct Ascend950MxMmadTraits<
    BlockMmad,
    std::void_t<typename BlockMmad::TileCopy::ElementMxScaleA, typename BlockMmad::TileCopy::ElementMxScaleB>> {
    static constexpr bool kEnabled = true;
    using ElementScaleA = typename BlockMmad::TileCopy::ElementMxScaleA;
    using ElementScaleB = typename BlockMmad::TileCopy::ElementMxScaleB;
};

template <
    class BlockMmadToUb_, class BlockScheduler_, class LocalCopyBlockEpilogue_, class RemoteCommBlockEpilogue_,
    class BlockEpilogueScheduler_, class ElementDispatch_ = typename BlockMmadToUb_::ElementA>
class Ascend950AllToAllGmmKernel {
public:
    using BlockMmadToUb = BlockMmadToUb_;
    using ArchTag = typename BlockMmadToUb::ArchTag;
    using UbL1TileShape = typename BlockMmadToUb::L1TileShape;
    using ElementA = typename BlockMmadToUb::ElementA;
    using LayoutA = typename BlockMmadToUb::TileCopy::LayoutTagA;
    using ElementB = typename BlockMmadToUb::ElementB;
    using LayoutB = typename BlockMmadToUb::TileCopy::LayoutTagB;
    using ElementUb = typename BlockMmadToUb::ElementC;
    using ElementDispatch = ElementDispatch_;
    using MxTraits = Ascend950MxMmadTraits<BlockMmadToUb>;
    using ElementScaleA = typename MxTraits::ElementScaleA;
    using ElementScaleB = typename MxTraits::ElementScaleB;
    static_assert(MxTraits::kEnabled, "Ascend950AllToAllGmmKernel requires an MX BlockMmad");
    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t UB_TILE_M = tla::get<0>(UbL1TileShape{});
    static constexpr uint32_t UB_TILE_N = tla::get<1>(UbL1TileShape{});
    static constexpr uint32_t UB_TILE_K = tla::get<2>(UbL1TileShape{});
    static constexpr uint32_t kElementPerBlock = BYTE_PER_BLK / sizeof(ElementUb);
    static constexpr uint32_t kUbTileNAlign =
        ((UB_TILE_N + kElementPerBlock - 1) / kElementPerBlock) * kElementPerBlock;
    static constexpr uint32_t kUbTileElem = UB_TILE_M * kUbTileNAlign;
    static constexpr uint32_t kUbPlaneBytes = kUbTileElem * sizeof(ElementUb);
    static constexpr uint32_t kDispatchReadyStride = Catccos::Detail::Ascend950MoeReady::kStride;
    static constexpr uint32_t kSwigluReadyStride = Catccos::Detail::Ascend950MoeReady::kStride;
    static constexpr uint16_t kUbSyncMode = 0x4;
    static constexpr uint16_t kAivSyncAicFlag = 6;
    static constexpr uint16_t kDispatchUbFreeFlag = 2;

    static_assert(kDispatchReadyStride * sizeof(int32_t) == 64);
    static_assert(kSwigluReadyStride * sizeof(int32_t) == 64);
    static_assert(2 * kUbPlaneBytes <= 256 * 1024, "GMM1 gate/up output planes exceed the physical Ascend950 UB");

    using LocalCopyBlockEpilogue = LocalCopyBlockEpilogue_;
    using RemoteCommBlockEpilogue = RemoteCommBlockEpilogue_;

    using BlockEpilogueScheduler = BlockEpilogueScheduler_;

    using LocalCopyParams = typename LocalCopyBlockEpilogue::Params;
    using RemoteCommParams = typename RemoteCommBlockEpilogue::Params;

    struct Params {
        GemmCoord problemShape;
        __gm__ ElementDispatch* ptrA;
        __gm__ ElementB* ptrB;
        __gm__ ElementA* ptrQuantA;
        __gm__ ElementScaleA* ptrScaleA;
        __gm__ ElementScaleB* ptrScaleB;
        LayoutA layoutA;
        LayoutB layoutB;
        GM_ADDR ptrWorkspace;
        __gm__ int32_t* ptrDispatchReady;
        __gm__ int32_t* ptrSwigluReady;
        int32_t EP;
        int32_t expertPerRank;
        uint32_t maxOutputSize;
        uint32_t rank;
        uint32_t rankSize;
        __gm__ int32_t* ptrTokenPerExpert;
        GM_ADDR symmetricPtr;
        LocalCopyParams localCopyParams;
        RemoteCommParams remoteCommParams;
        Callback waitUbFreeCallback;
        Callback notifyUbReadyCallback;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord problemShape_, uint32_t EP_, uint32_t expertPerRank_, uint32_t maxOutputSize_, uint32_t rank_,
            uint32_t rankSize_, GM_ADDR ptrTokenPerExpert_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_,
            LayoutB layoutB_, GM_ADDR ptrQuantA_, GM_ADDR ptrScaleA_, GM_ADDR ptrScaleB_, GM_ADDR ptrWorkspace_,
            GM_ADDR ptrDispatchReady_, GM_ADDR ptrSwigluReady_, GM_ADDR symmetricPtr_, LocalCopyParams localCopyParams_,
            RemoteCommParams remoteCommParams_, const Callback& waitUbFreeCallback_ = Callback{},
            const Callback& notifyUbReadyCallback_ = Callback{})
            : problemShape(problemShape_),
              EP(EP_),
              expertPerRank(expertPerRank_),
              maxOutputSize(maxOutputSize_),
              rank(rank_),
              rankSize(rankSize_),
              ptrTokenPerExpert(reinterpret_cast<__gm__ int32_t*>(ptrTokenPerExpert_)),
              ptrA(reinterpret_cast<__gm__ ElementDispatch*>(ptrA_)),
              layoutA(layoutA_),
              ptrB(reinterpret_cast<__gm__ ElementB*>(ptrB_)),
              ptrQuantA(reinterpret_cast<__gm__ ElementA*>(ptrQuantA_)),
              ptrScaleA(reinterpret_cast<__gm__ ElementScaleA*>(ptrScaleA_)),
              ptrScaleB(reinterpret_cast<__gm__ ElementScaleB*>(ptrScaleB_)),
              layoutB(layoutB_),
              ptrWorkspace(ptrWorkspace_),
              ptrDispatchReady(reinterpret_cast<__gm__ int32_t*>(ptrDispatchReady_)),
              ptrSwigluReady(reinterpret_cast<__gm__ int32_t*>(ptrSwigluReady_)),
              symmetricPtr(symmetricPtr_),
              localCopyParams(localCopyParams_),
              remoteCommParams(remoteCommParams_),
              waitUbFreeCallback(waitUbFreeCallback_),
              notifyUbReadyCallback(notifyUbReadyCallback_)
        {}
    };

    struct WorkspaceInfo {
        GM_ADDR ptrcumsumMM;

        CATLASS_DEVICE
        WorkspaceInfo(const Params& params) { ptrcumsumMM = params.ptrWorkspace; }
    };

    CATLASS_DEVICE
    Ascend950AllToAllGmmKernel()
    {
#ifdef ENABLE_TIMER
        __gm__ uint8_t* timer_buffer = GetTimerBuffer();
        if (timer_buffer != nullptr) {
            timer.Init(timer_buffer);
            timer.Tik();
        }
#endif
        flagAivFinishCumsum = Catlass::Arch::CrossCoreFlag(0);
    }

    CATLASS_DEVICE
    ~Ascend950AllToAllGmmKernel()
    {
#ifdef ENABLE_TIMER
        timer.Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
#endif
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(
        Params const& params, Catlass::Arch::Resource<ArchTag> resource = Catlass::Arch::Resource<ArchTag>());

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params, Catlass::Arch::Resource<ArchTag> resource)
    {
        BlockScheduler blockScheduler;
        BlockMmadToUb blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmWorkspace;
        gmWorkspace.SetGlobalBuffer(params.ptrQuantA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);
        AscendC::GlobalTensor<ElementScaleA> gmScaleA;
        gmScaleA.SetGlobalBuffer(params.ptrScaleA);
        AscendC::GlobalTensor<ElementScaleB> gmScaleB;
        gmScaleB.SetGlobalBuffer(params.ptrScaleB);

        WorkspaceInfo workspaceInfo(params);

        AscendC::GlobalTensor<int32_t> cumsumMM;
        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrcumsumMM));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetScaleA = 0;
        int64_t gmGroupOffsetScaleB = 0;
        uint32_t startCoreIdx = 0;

        Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCumsum);

        auto ubGate = resource.ubBuf.template GetBufferByByte<ElementUb>(0);
        auto ubUp = resource.ubBuf.template GetBufferByByte<ElementUb>(kUbPlaneBytes);
        auto layoutUb = tla::MakeLayout<ElementUb, Catlass::layout::RowMajor>(UB_TILE_M, kUbTileNAlign);
        auto tensorUbGate = tla::MakeTensor(ubGate, layoutUb, Catlass::Arch::PositionUB{});
        auto tensorUbUp = tla::MakeTensor(ubUp, layoutUb, Catlass::Arch::PositionUB{});

        // The M256 path owns the complete physical UB: gate is [0, 128 KiB)
        // and up is [128, 256 KiB). Carry the final AIV0 consumer
        // acknowledgement across expert boundaries. Dispatch readiness for the
        // next active expert can then hide the preceding SwiGLU/MX tail. The final token is
        // drained after GMM2 by the owning callback object.
        bool ubTileOutstanding = false;

        // ========= 矩阵乘法 start =========
        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            WaitForDispatchReady(params, groupIdx);
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            uint32_t computeN = params.problemShape.n() / 2;
            GemmCoord inGroupProblemShape{currentM, computeN, params.problemShape.k()};

            LayoutA layoutTagA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutTagB = params.layoutB;

            auto tensorA = tla::MakeTensor(
                gmWorkspace[gmGroupOffsetA], tla::MakeLayoutFromTag(layoutTagA), Catlass::Arch::PositionGM{});
            auto tensorB =
                tla::MakeTensor(gmB[gmGroupOffsetB], tla::MakeLayoutFromTag(layoutTagB), Catlass::Arch::PositionGM{});

            uint32_t scaleK = RoundUp<2>(CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(inGroupProblemShape.k()));
            auto layoutScaleA = tla::MakeMxScaleLayout<ElementScaleA, LayoutA, false>(currentM, scaleK);
            auto layoutScaleB = tla::MakeMxScaleLayout<ElementScaleB, LayoutB, true>(scaleK, params.problemShape.n());
            auto tensorScaleA =
                tla::MakeTensor(gmScaleA[gmGroupOffsetScaleA], layoutScaleA, Catlass::Arch::PositionGM{});
            auto tensorScaleB =
                tla::MakeTensor(gmScaleB[gmGroupOffsetScaleB], layoutScaleB, Catlass::Arch::PositionGM{});

            blockScheduler.Update(inGroupProblemShape, MakeCoord(UB_TILE_M, UB_TILE_N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;

            // Delay the previous expert's final UB dependency until this core
            // is actually about to reuse AIV0 UB. PIPE_FIX waits at the store
            // boundary while CUBE can start the next expert's MMAD.
            if (startLoopIdx < coreLoops && ubTileOutstanding) {
                params.waitUbFreeCallback();
                ubTileOutstanding = false;
            }
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                MatrixCoord offsetA{blockCoord.m() * UB_TILE_M, blockCoord.k() * UB_TILE_K};
                MatrixCoord offsetB{blockCoord.k() * UB_TILE_K, blockCoord.n() * UB_TILE_N};

                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(offsetA.row(), offsetA.column()),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockB = GetTile(
                    tensorB, tla::MakeCoord(offsetB.row(), offsetB.column()),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                if (ubTileOutstanding) {
                    params.waitUbFreeCallback();
                    ubTileOutstanding = false;
                }

                auto tensorBlockBUp = GetTile(
                    tensorB, tla::MakeCoord(offsetB.row(), offsetB.column() + computeN),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockScaleA = GetTile(
                    tensorScaleA, tla::MakeCoord(offsetA.row(), offsetA.column() / Catlass::MX_SCALE_GROUP_NUM),
                    tla::MakeShape(actualBlockShape.m(), CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualBlockShape.k())));
                auto tensorBlockScaleB = GetTile(
                    tensorScaleB, tla::MakeCoord(offsetB.row() / Catlass::MX_SCALE_GROUP_NUM, offsetB.column()),
                    tla::MakeShape(CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));
                auto tensorBlockScaleBUp = GetTile(
                    tensorScaleB,
                    tla::MakeCoord(offsetB.row() / Catlass::MX_SCALE_GROUP_NUM, offsetB.column() + computeN),
                    tla::MakeShape(CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));
                auto tensorBlockGate =
                    GetTile(tensorUbGate, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), kUbTileNAlign));
                auto tensorBlockUp =
                    GetTile(tensorUbUp, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), kUbTileNAlign));

                blockMmad(
                    tensorBlockA, tensorBlockB, tensorBlockGate, actualBlockShape, tensorBlockScaleA,
                    tensorBlockScaleB);
                blockMmad(
                    tensorBlockA, tensorBlockBUp, tensorBlockUp, actualBlockShape, tensorBlockScaleA,
                    tensorBlockScaleBUp);
                params.notifyUbReadyCallback();
                ubTileOutstanding = true;
            }

            gmGroupOffsetA += static_cast<int64_t>(inGroupProblemShape.m()) * inGroupProblemShape.k();
            gmGroupOffsetB += static_cast<int64_t>(inGroupProblemShape.k()) * params.problemShape.n();
            gmGroupOffsetScaleA += static_cast<int64_t>(inGroupProblemShape.m()) * scaleK;
            gmGroupOffsetScaleB += static_cast<int64_t>(scaleK) * params.problemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params, Catlass::Arch::Resource<ArchTag> resource)
    {
        uint32_t metadataCoreIdx = get_block_idx() + get_subblockid() * get_block_num();
        uint32_t metadataCoreNum = get_block_num() * get_subblockdim();
        metadataCoreNum = (metadataCoreNum == 0) ? 1 : metadataCoreNum;

        WorkspaceInfo workspaceInfo(params);
        AscendC::GlobalTensor<int32_t> tokenPerExpert;
        tokenPerExpert.SetGlobalBuffer(params.ptrTokenPerExpert);
        AscendC::GlobalTensor<int32_t> cumsumMM;
        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrcumsumMM));
        BlockEpilogueScheduler blockEpilogueScheduler(
            params.rank, params.rankSize, params.expertPerRank, params.EP, params.problemShape, metadataCoreIdx,
            metadataCoreNum, tokenPerExpert, cumsumMM, resource);

        // Metadata preparation above is shared by both AIV subblocks.
        // AIV0 immediately enters the existing SwiGLU consumer. AIV1 owns
        // packed MX dispatch, with all physical AIV1 cores split evenly across
        // source ranks (20 / EP workers per peer on Ascend 950).
        if (AscendC::GetSubBlockIdx() == 0) {
            return;
        }

        uint32_t coreIdx = get_block_idx();
        uint32_t coreNum = get_block_num();
        coreNum = (coreNum == 0) ? 1 : coreNum;
        uint32_t workersPerEP = coreNum / static_cast<uint32_t>(params.EP);
        workersPerEP = workersPerEP == 0 ? 1 : workersPerEP;
        uint32_t activeWorkerNum = static_cast<uint32_t>(params.EP) * workersPerEP;
        bool isDispatchWorker = coreIdx < activeWorkerNum;
        uint32_t peerIdx = isDispatchWorker ? coreIdx / workersPerEP : 0;
        uint32_t workerIdxInPeer = isDispatchWorker ? coreIdx % workersPerEP : 0;

        uint32_t hiddenSize = params.problemShape.k();
        uint32_t quantBytes = hiddenSize * Catlass::SizeOfBits<ElementA>::value / Catlass::SizeOfBits<uint8_t>::value;
        uint32_t scaleK = RoundUp<2>(CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(hiddenSize));
        uint32_t packedRowBytes = RoundUp<Catlass::BYTE_PER_BLK>(quantBytes + scaleK);
        constexpr uint32_t kDispatchRows = 2;
        uint32_t packedBufferBytes = RoundUp<64>(kDispatchRows * packedRowBytes);
        auto packedBuffer0 = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
        auto packedBuffer1 = resource.ubBuf.template GetBufferByByte<uint8_t>(packedBufferBytes);

        AscendC::GlobalTensor<uint8_t> gmQuantBytes;
        gmQuantBytes.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(params.ptrQuantA));
        AscendC::GlobalTensor<uint8_t> gmScaleBytes;
        gmScaleBytes.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(params.ptrScaleA));
        AscendC::GlobalTensor<uint8_t> remotePacked;
        if (isDispatchWorker) {
            auto remotePtr = aclshmem_ptr(reinterpret_cast<__gm__ void*>(params.symmetricPtr), peerIdx);
            remotePacked.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(remotePtr));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }

        int64_t srcExpertRowStart = 0;
        if (isDispatchWorker) {
            int64_t peerTokenBase = static_cast<int64_t>(peerIdx) * params.EP * params.expertPerRank;
            for (int32_t expert = 0; expert < static_cast<int32_t>(params.rank) * params.expertPerRank; ++expert) {
                srcExpertRowStart += tokenPerExpert(peerTokenBase + expert);
            }
        }
        int64_t dstExpertRowStart = 0;
        int32_t pingpongId = -1;

        // ======== all to allv start ========
        for (int32_t localExpertIdx = 0; localExpertIdx < params.expertPerRank; ++localExpertIdx) {
            if (isDispatchWorker) {
                int64_t tokenOffset = static_cast<int64_t>(peerIdx) * params.EP * params.expertPerRank +
                                      static_cast<int64_t>(params.rank) * params.expertPerRank + localExpertIdx;
                uint32_t rowsFromPeer = tokenPerExpert(tokenOffset);
                uint32_t rowsPerWorker = CeilDiv<uint32_t>(rowsFromPeer, workersPerEP);
                uint32_t workerRowOffset = workerIdxInPeer * rowsPerWorker;
                uint32_t rowsToCopy = workerRowOffset < rowsFromPeer ? rowsFromPeer - workerRowOffset : 0;
                rowsToCopy = rowsToCopy < rowsPerWorker ? rowsToCopy : rowsPerWorker;

                int64_t peerDstPrefix =
                    peerIdx == 0 ? 0 : cumsumMM((peerIdx - 1) * params.expertPerRank + localExpertIdx);
                int64_t srcRow = srcExpertRowStart + workerRowOffset;
                int64_t dstRow = dstExpertRowStart + peerDstPrefix + workerRowOffset;

                uint32_t copiedRows = 0;
                while (copiedRows < rowsToCopy) {
                    uint32_t currentRows = rowsToCopy - copiedRows;
                    currentRows = currentRows < kDispatchRows ? currentRows : kDispatchRows;
                    pingpongId = (pingpongId + 1) % 2;
                    TEventID eventId = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
                    auto packedBuffer = pingpongId == 0 ? packedBuffer0 : packedBuffer1;

                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                    int64_t remoteOffset = (srcRow + copiedRows) * packedRowBytes;
                    AscendC::DataCopy(packedBuffer, remotePacked[remoteOffset], currentRows * packedRowBytes);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

                    int64_t quantOffset = (dstRow + copiedRows) * quantBytes;
                    int64_t scaleOffset = (dstRow + copiedRows) * scaleK;
                    AscendC::DataCopyPad(
                        gmQuantBytes[quantOffset], packedBuffer,
                        {static_cast<uint16_t>(currentRows), static_cast<uint16_t>(quantBytes),
                         static_cast<uint16_t>((packedRowBytes - quantBytes) / Catlass::BYTE_PER_BLK), 0, 0});
                    AscendC::DataCopyPad(
                        gmScaleBytes[scaleOffset], packedBuffer[quantBytes],
                        {static_cast<uint16_t>(currentRows), static_cast<uint16_t>(scaleK),
                         static_cast<uint16_t>(
                             (packedRowBytes - RoundUp<Catlass::BYTE_PER_BLK>(scaleK)) / Catlass::BYTE_PER_BLK),
                         0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                    copiedRows += currentRows;
                }
                srcExpertRowStart += rowsFromPeer;
                // Publish readiness only after all Dispatch MTE3 writes for this
                // peer/worker/expert slice are globally visible.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
                AscendC::AtomicAdd(
                    params.ptrDispatchReady + localExpertIdx * kDispatchReadyStride, static_cast<int32_t>(1));
            }
            dstExpertRowStart += cumsumMM((params.EP - 1) * params.expertPerRank + localExpertIdx);
        }
        // ======== all to allv end ========

        if (isDispatchWorker) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::CrossCoreSetFlag<kUbSyncMode, PIPE_MTE3>(kAivSyncAicFlag + kDispatchUbFreeFlag);
    }

private:
    CATLASS_DEVICE
    static void WaitForDispatchReady(Params const& params, uint32_t expertIdx)
    {
        __gm__ int32_t* ready = params.ptrDispatchReady + expertIdx * kDispatchReadyStride;
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t workersPerEP = coreNum / static_cast<uint32_t>(params.EP);
        workersPerEP = workersPerEP == 0 ? 1 : workersPerEP;
        int32_t readyTarget = params.EP * static_cast<int32_t>(workersPerEP);
        while (AscendC::ReadGmByPassDCache(ready) != readyTarget) {
            int64_t start = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - start < 100) {
            }
        }
    }

    Catlass::Arch::CrossCoreFlag flagAivFinishCumsum;
#ifdef ENABLE_TIMER
    AscendTimerDevice timer;
#endif
};
} // namespace Catccos::DGemm::Kernel

#endif
