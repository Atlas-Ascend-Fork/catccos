
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"

#include "catccos/epilogue/dispatch_policy.hpp"

namespace Catlass::Epilogue::Block {

template <uint32_t UB_STAGES_, class CType_, class DType_, class TileCopy_, class EpilogueTileSwizzle_>
class BlockEpilogue<
    Catccos::Epilogue::EpilogueAtlasA2PerTokenDequantSwiglu<UB_STAGES_>, CType_, DType_, TileCopy_,
    EpilogueTileSwizzle_> {
public:
    using DispatchPolicy = Catccos::Epilogue::EpilogueAtlasA2PerTokenDequantSwiglu<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    // Tile copy
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    using EpilogueTileSwizzle = EpilogueTileSwizzle_;

    static_assert(UB_STAGES <= 2, "UB stages too large, event id is not enough.");

    struct Params {
        MatrixCoord tileShape;

        CATLASS_DEVICE
        Params() {};

        CATLASS_DEVICE
        Params(MatrixCoord const& tileShape_) : tileShape(tileShape_) {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const& resource, Params const& params) : params(params)
    {
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += params.tileShape.row() * params.tileShape.column() * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += params.tileShape.row() * params.tileShape.column() * sizeof(ElementD);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
        ubTmpMxN = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += params.tileShape.row() * params.tileShape.column() * sizeof(float);
        ubTmpMxChunkN = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE
    void UpdateParams(Params const& params_) { params = params_; }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementC> const& gmBlockC, LayoutC const& layoutBlockC,
        AscendC::GlobalTensor<ElementD> const& gmBlockD, LayoutC const& layoutBlockD,
        MatrixCoord const& actualBlockShape, Callback&& callback = Callback{})
    {
        callback();

        auto tileShape = params.tileShape;
        auto ubTileStride = MakeCoord(static_cast<int64_t>(tileShape.column()), 1L);
        auto ubChunkTileStride = MakeCoord(static_cast<int64_t>(tileShape.column() >> 1), 1L);
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();

        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();

        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);

            MatrixCoord tileOffset = tileCoord * tileShape;

            auto actualChunkTileShape = MakeCoord(actualTileShape.row(), actualTileShape.column() >> 1);
            auto chunkTileOffset = MakeCoord(tileOffset.row(), tileOffset.column() >> 1);

            auto actualChunkTileCount = actualChunkTileShape.At(0) * actualChunkTileShape.At(1);

            auto gmTileC = gmBlockC[layoutBlockC.GetOffset(tileOffset)];
            auto layoutGmTileC = layoutBlockC.GetTileLayout(actualTileShape);

            auto& ubC = ubCList[ubListId];
            LayoutC layoutUbC{actualTileShape, ubTileStride}; // {{1, 256}, {256, 1}}

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutGmTileC);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            auto& ubD = ubDList[ubListId];

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(
                ubTmpMxN, ubC, AscendC::RoundMode::CAST_NONE, actualTileShape.row() * actualTileShape.column());
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubTmpMxChunkN, ubTmpMxN, -1.0f, actualChunkTileCount); // float -> 512B 256B 1024 -> 8

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(ubTmpMxChunkN, ubTmpMxChunkN, actualChunkTileCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(ubTmpMxChunkN, ubTmpMxChunkN, 1.0f, actualChunkTileCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(ubTmpMxChunkN, ubTmpMxN, ubTmpMxChunkN, actualChunkTileCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(ubTmpMxChunkN, ubTmpMxChunkN, ubTmpMxN[actualChunkTileCount], actualChunkTileCount);
            AscendC::PipeBarrier<PIPE_V>();

            LayoutD layoutUbD{actualChunkTileShape, ubChunkTileStride};

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            AscendC::Cast(ubD, ubTmpMxChunkN, AscendC::RoundMode::CAST_ROUND, actualChunkTileCount);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            auto gmTileD = gmBlockD[layoutBlockD.GetOffset(chunkTileOffset)];
            auto layoutGmTileD = layoutBlockD.GetTileLayout(actualChunkTileShape);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(gmTileD, ubD, layoutGmTileD, layoutUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
        }
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

    AscendC::LocalTensor<float> ubTmpMxN;
    AscendC::LocalTensor<float> ubTmpMxChunkN;

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
};

template <
    uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_,
    class EpilogueTileSwizzle_>
class BlockEpilogue<
    Catccos::Epilogue::EpilogueAtlasA2PerTokenDequantSwiglu<UB_STAGES_>, CType_,
    Gemm::GemmType<float, LayoutPerTokenScale_>, DType_, TileCopy_, EpilogueTileSwizzle_> {
public:
    using DispatchPolicy = Catccos::Epilogue::EpilogueAtlasA2PerTokenDequantSwiglu<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    // Tile copy
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    using EpilogueTileSwizzle = EpilogueTileSwizzle_;

    static_assert(UB_STAGES <= 2, "UB stages too large, event id is not enough.");

    struct Params {
        MatrixCoord tileShape;

        CATLASS_DEVICE
        Params() {};

        CATLASS_DEVICE
        Params(MatrixCoord const& tileShape_) : tileShape(tileShape_) {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const& resource, Params const& params) : params(params)
    {
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        uint32_t tileN = params.tileShape.column();
        uint32_t chunkTileN = tileN / 2;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += tileN * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += tileN * sizeof(ElementD);
            ubCFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += tileN * sizeof(float);
            ubCFp32ChunkNList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += chunkTileN * sizeof(float);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE
    void UpdateParams(Params const& params_) { params = params_; }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementC> const& gmBlockC, LayoutC const& layoutBlockC,
        AscendC::GlobalTensor<ElementPerTokenScale> const& gmBlockPerTokenScale,
        LayoutPerTokenScale const& layoutBlockPerTokenScale, AscendC::GlobalTensor<ElementD> const& gmBlockD,
        LayoutC const& layoutBlockD, MatrixCoord const& actualBlockShape, Callback&& callback = Callback{})
    {
        callback();

        auto tileShape = params.tileShape;
        auto ubTileStride = MakeCoord(static_cast<int64_t>(tileShape.column()), 1L);
        auto ubChunkTileStride = MakeCoord(static_cast<int64_t>(tileShape.column() >> 1), 1L);
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();

        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();

        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto actualTileCount = actualTileShape.At(0) * actualTileShape.At(1);
            MatrixCoord tileOffset = tileCoord * tileShape;

            auto actualChunkTileShape = MakeCoord(actualTileShape.row(), actualTileShape.column() >> 1);
            auto chunkTileOffset = MakeCoord(tileOffset.row(), tileOffset.column() >> 1);
            auto actualChunkTileCount = actualChunkTileShape.At(0) * actualChunkTileShape.At(1);

            auto perTokenScaleTileOffset = tileOffset[0];

            auto gmTileC = gmBlockC[layoutBlockC.GetOffset(tileOffset)];
            auto layoutGmTileC = layoutBlockC.GetTileLayout(actualTileShape);
            auto gmTileD = gmBlockD[layoutBlockD.GetOffset(chunkTileOffset)];
            auto layoutGmTileD = layoutBlockD.GetTileLayout(actualChunkTileShape);

            auto& ubC = ubCList[ubListId];
            LayoutC layoutUbC{actualTileShape, ubTileStride}; // {{1, 256}, {256, 1}}
            auto& ubD = ubDList[ubListId];
            LayoutD layoutUbD{actualChunkTileShape, ubChunkTileStride};
            auto& ubCFp32 = ubCFp32List[ubListId];
            auto& ubCFp32ChunkN = ubCFp32ChunkNList[ubListId];

            // Move C from GM workspace to UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutGmTileC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            // Cast C from FP16 to FP32 in UB
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE, actualTileCount);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            ElementPerTokenScale perTokenScale = gmBlockPerTokenScale.GetValue(perTokenScaleTileOffset);

            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubCFp32, ubCFp32, perTokenScale, actualTileCount);
            AscendC::PipeBarrier<PIPE_V>();

            // Swiglu computation process
            AscendC::Muls(ubCFp32ChunkN, ubCFp32, -1.0f, actualChunkTileCount); // float -> 512B 256B 1024 -> 8
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(ubCFp32ChunkN, ubCFp32ChunkN, actualChunkTileCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(ubCFp32ChunkN, ubCFp32ChunkN, 1.0f, actualChunkTileCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(ubCFp32ChunkN, ubCFp32, ubCFp32ChunkN, actualChunkTileCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(ubCFp32ChunkN, ubCFp32ChunkN, ubCFp32[actualChunkTileCount], actualChunkTileCount);
            AscendC::PipeBarrier<PIPE_V>();

            // cast to bf16
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            AscendC::Cast(ubD, ubCFp32ChunkN, AscendC::RoundMode::CAST_ROUND, actualChunkTileCount);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            // copy ubD to gmD
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(gmTileD, ubD, layoutGmTileD, layoutUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;
    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

    AscendC::LocalTensor<float> ubCFp32List[UB_STAGES];
    AscendC::LocalTensor<float> ubCFp32ChunkNList[UB_STAGES];

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
};

} // namespace Catlass::Epilogue::Block
