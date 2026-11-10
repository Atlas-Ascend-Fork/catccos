
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_BLOCK_COMM_BLOCK_MX_QUANT_HPP
#define CATCCOS_COMM_BLOCK_COMM_BLOCK_MX_QUANT_HPP

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/epilogue/block/block_epilogue_dynamic_mx_quant.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catccos::Comm::Block {

using Catlass::MakeCoord;
using Catlass::MatrixCoord;

// ============================================================================
// CommBlockMxQuant: comm-block adapter for dynamic MX quantization
//
// Wraps BlockEpilogue<EpilogueAscend950DynamicMxQuant> into the flat-streaming
// interface required by MxQuantAllGather kernel.
//
// Data flow per tile:
//   GM(input) --MTE2--> UB(C) --V(MxQuant)--> UB(D) + UB(scale) --MTE3--> GM(ipc)
//
// The operator processes `actualRows` rows of N columns each, outputting:
//   - Quantized data: actualRows × (N / PACK_RATIO) bytes
//   - MX scale: actualRows × (N / BLOCK_SIZE) bytes (E8M0)
// ============================================================================
template <uint32_t UB_STAGES_, uint32_t BLOCK_SIZE_, int64_t ROUND_MODE_, class CType_, class DType_>
class CommBlockMxQuant {
public:
    using DispatchPolicy = EpilogueAscend950DynamicMxQuant<UB_STAGES_, BLOCK_SIZE_, ROUND_MODE_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr uint32_t BLOCK_SIZE = BLOCK_SIZE_;

    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;
    using ElementScale = float8_e8m0_t; // E8M0

    // Type traits for MX quant
    using Traits = Epilogue::Block::mx_quant_detail::MxQuantTraits<ElementC>;
    using CalcType = typename Traits::CalcType;
    using IntCalcType = typename Traits::IntCalcType;
    static constexpr uint32_t VF_LEN = Traits::VF_LEN;
    static constexpr bool IsFp4 = Epilogue::Block::mx_quant_detail::IsFp4Type<ElementD>();
    static constexpr uint32_t PACK_RATIO = IsFp4 ? 2 : 1;

    static constexpr AscendC::RoundMode ROUND_MODE = (ROUND_MODE_ == 1) ? AscendC::RoundMode::CAST_FLOOR :
                                                     (ROUND_MODE_ == 0) ? AscendC::RoundMode::CAST_ROUND :
                                                                          AscendC::RoundMode::CAST_RINT;

    static constexpr IntCalcType TARGET_EMAX_FIELD = []() constexpr {
        if constexpr (std::is_same_v<ElementD, float4_e2m1x2_t>)
            return Traits::FP4_E2M1_EMAX;
        else if constexpr (std::is_same_v<ElementD, float4_e1m2x2_t>)
            return Traits::FP4_E1M2_EMAX;
        else if constexpr (std::is_same_v<ElementD, float8_e4m3_t>)
            return Traits::FP8_E4M3_EMAX;
        else if constexpr (std::is_same_v<ElementD, float8_e5m2_t>)
            return Traits::FP8_E5M2_EMAX;
        else
            return IntCalcType{0};
    }();

    static constexpr uint32_t UB_ALIGN = 64;

    struct Params {
        uint32_t N{0}; // columns per row

        CATLASS_DEVICE Params() {}
        CATLASS_DEVICE Params(uint32_t N_) : N(N_) {}
    };

    CATLASS_DEVICE
    CommBlockMxQuant(Catlass::Arch::Resource<ArchTag>& resource, Params const& params = Params{})
        : params(params), resourceRef(&resource)
    {
        uint32_t N = params.N;
        if (N == 0)
            return;

        uint32_t bytesPerRowInput = N * sizeof(ElementC);
        uint32_t bytesPerRowOutput = N * Catlass::SizeOfBits<ElementD>::value / Catlass::SizeOfBits<uint8_t>::value;
        uint32_t numScalesPerRowRaw = N / BLOCK_SIZE;
        uint32_t numScalesPerRowAligned = ((numScalesPerRowRaw + 1) / 2) * 2;
        uint32_t bytesPerRowScale = numScalesPerRowAligned;
        uint32_t bytesPerRowMaxExp = numScalesPerRowAligned * sizeof(uint16_t);
        uint32_t bytesPerRowRecipScale = numScalesPerRowAligned * sizeof(uint16_t);
        uint32_t bytesPerRow =
            bytesPerRowInput + bytesPerRowOutput + bytesPerRowScale + bytesPerRowMaxExp + bytesPerRowRecipScale;
        tileRows = AscendC::GetUBSizeInBytes() / (UB_STAGES * bytesPerRow);
        tileRows = (tileRows / UB_ALIGN) * UB_ALIGN;
        if (tileRows == 0)
            tileRows = 1;

        size_t ubOffset = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubInputList[i] = resourceRef->ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += tileRows * bytesPerRowInput;
            ubOutputList[i] = resourceRef->ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += tileRows * bytesPerRowOutput;
            ubScaleList[i] = resourceRef->ubBuf.template GetBufferByByte<ElementScale>(ubOffset);
            ubOffset += tileRows * bytesPerRowScale;
            ubMaxExpList[i] = resourceRef->ubBuf.template GetBufferByByte<uint16_t>(ubOffset);
            ubOffset += tileRows * bytesPerRowMaxExp;
            ubRecipScaleList[i] = resourceRef->ubBuf.template GetBufferByByte<uint16_t>(ubOffset);
            ubOffset += tileRows * bytesPerRowRecipScale;
        }
    }

    CATLASS_DEVICE
    ~CommBlockMxQuant() {}

    CATLASS_DEVICE
    void InitBlockLoop()
    {
        int32_t eventVMTE2 = 0, eventMTE2V = 0, eventMTE3V = 0, eventVMTE3 = 0;

        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            eventInputVtoMTE2[i] = eventVMTE2++;
            eventInputMTE2toV[i] = eventMTE2V++;
            eventOutputMTE3toV[i] = eventMTE3V++;
            eventOutputVtoMTE3[i] = eventVMTE3++;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventInputVtoMTE2[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[i]);
        }
    }

    CATLASS_DEVICE
    void FinalizeBlockLoop()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventInputVtoMTE2[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[i]);
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementC> const& gmInput, LayoutC const& layoutInput,
        AscendC::GlobalTensor<ElementD> const& gmQuantOut, LayoutD const& layoutQuantOut,
        AscendC::GlobalTensor<ElementScale> const& gmScaleOut, LayoutD const& layoutScaleOut,
        Catlass::MatrixCoord const& actualBlockShape)
    {
        uint32_t actualRows = actualBlockShape.row();
        uint32_t N = actualBlockShape.column();
        if (actualRows == 0 || N == 0)
            return;

        uint32_t numScalesPerRow = N / BLOCK_SIZE;
        uint32_t totalLoops = (actualRows + tileRows - 1) / tileRows;

        for (uint32_t loopIdx = 0; loopIdx < totalLoops; ++loopIdx) {
            uint32_t ubListId = loopIdx % UB_STAGES;
            uint32_t rowOffset = loopIdx * tileRows;
            uint32_t curRows = (loopIdx == totalLoops - 1) ? (actualRows - rowOffset) : tileRows;

            auto& ubC = ubInputList[ubListId];
            auto& ubD = ubOutputList[ubListId];
            auto& ubScale = ubScaleList[ubListId];
            auto& ubMaxExp = ubMaxExpList[ubListId];
            auto& ubRecipScale = ubRecipScaleList[ubListId];

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventInputVtoMTE2[ubListId]);
            uint32_t inputCount = curRows * N;
            AscendC::DataCopyExtParams srcParams(1, inputCount * sizeof(ElementC), 0, 0, 0);
            AscendC::DataCopyPadExtParams<ElementC> padParams(false, 0, 0, 0);
            AscendC::DataCopyPad(ubC, gmInput[layoutInput.GetOffset({rowOffset, 0})], srcParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventInputMTE2toV[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventInputMTE2toV[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[ubListId]);

            ComputeMxQuant(ubC, ubD, ubScale, ubMaxExp, ubRecipScale, curRows, N);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventInputVtoMTE2[ubListId]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventOutputVtoMTE3[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventOutputVtoMTE3[ubListId]);
            uint32_t quantCount = curRows * N;
            AscendC::DataCopyExtParams dstParams(
                1, quantCount * Catlass::SizeOfBits<ElementD>::value / Catlass::SizeOfBits<uint8_t>::value, 0, 0, 0);
            AscendC::DataCopyPad(gmQuantOut[layoutQuantOut.GetOffset({rowOffset, 0})], ubD, dstParams);

            uint32_t scaleCount = curRows * numScalesPerRow;
            AscendC::DataCopyExtParams scaleParams(1, scaleCount, 0, 0, 0);
            AscendC::DataCopyPad(gmScaleOut[layoutScaleOut.GetOffset({rowOffset, 0})], ubScale, scaleParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[ubListId]);
        }
    }

    CATLASS_DEVICE
    void QuantFromUb(
        AscendC::LocalTensor<ElementC> const& ubInput, AscendC::GlobalTensor<ElementD> const& gmQuantOut,
        LayoutD const& layoutQuantOut, AscendC::GlobalTensor<ElementScale> const& gmScaleOut,
        LayoutD const& layoutScaleOut, Catlass::MatrixCoord const& actualBlockShape)
    {
        uint32_t actualRows = actualBlockShape.row();
        uint32_t N = actualBlockShape.column();
        if (actualRows == 0 || N == 0)
            return;

        uint32_t numScalesPerRow = N / BLOCK_SIZE;
        uint32_t quantBytes = N * Catlass::SizeOfBits<ElementD>::value / Catlass::SizeOfBits<uint8_t>::value;

        // Keep the same one-row execution and scratch-buffer contract as the
        // established GM-input path. The vector program is row-scoped; merging
        // rows here corrupted both quantized data and scales in device tests.
        for (uint32_t row = 0; row < actualRows; ++row) {
            uint32_t ubListId = row % UB_STAGES;
            auto ubC = ubInput[row * N];
            auto& ubD = ubOutputList[ubListId];
            auto& ubScale = ubScaleList[ubListId];
            auto& ubMaxExp = ubMaxExpList[ubListId];
            auto& ubRecipScale = ubRecipScaleList[ubListId];

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[ubListId]);
            ComputeMxQuant(ubC, ubD, ubScale, ubMaxExp, ubRecipScale, 1, N);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventOutputVtoMTE3[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventOutputVtoMTE3[ubListId]);
            AscendC::DataCopyExtParams dstParams{1, quantBytes, 0, 0, 0};
            AscendC::DataCopyPad(gmQuantOut[layoutQuantOut.GetOffset({row, 0})], ubD, dstParams);

            AscendC::DataCopyExtParams scaleParams{1, numScalesPerRow, 0, 0, 0};
            AscendC::DataCopyPad(gmScaleOut[layoutScaleOut.GetOffset({row, 0})], ubScale, scaleParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[ubListId]);
        }
    }

    // Quantize one contiguous UB tile as a single vector program and store the
    // result with two 2D MTE3 copies.  This is intentionally separate from the
    // established row-wise QuantFromUb path so existing users keep identical
    // scheduling and event behavior.
    CATLASS_DEVICE
    void QuantFromUb2D(
        AscendC::LocalTensor<ElementC> const& ubInput, AscendC::GlobalTensor<ElementD> const& gmQuantOut,
        LayoutD const& layoutQuantOut, AscendC::GlobalTensor<ElementScale> const& gmScaleOut,
        LayoutD const& layoutScaleOut, Catlass::MatrixCoord const& actualBlockShape)
    {
        uint32_t actualRows = actualBlockShape.row();
        uint32_t N = actualBlockShape.column();
        if (actualRows == 0 || N == 0)
            return;

        uint32_t numScalesPerRow = N / BLOCK_SIZE;
        if (actualRows > tileRows || numScalesPerRow > Catlass::BYTE_PER_BLK) {
            QuantFromUb(ubInput, gmQuantOut, layoutQuantOut, gmScaleOut, layoutScaleOut, actualBlockShape);
            FinalizeBlockLoop();
            InitBlockLoop();
            return;
        }

        auto ubC = ubInput;
        auto& ubD = ubOutputList[0];
        auto& ubScale = ubScaleList[0];
        auto& ubMaxExp = ubMaxExpList[0];
        auto& ubRecipScale = ubRecipScaleList[0];

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[0]);
        ComputeMxQuant(ubC, ubD, ubScale, ubMaxExp, ubRecipScale, actualRows, N);

        // MTE3 advances an unaligned source row by one 32-byte block.  Expand
        // the compact Mx8 scale rows to Mx32-byte rows in the already-consumed
        // input plane before issuing one 2D copy.
        auto ubScalePadded = ubInputList[0].template ReinterpretCast<ElementScale>();
        PadMxScaleRows(ubScale, ubScalePadded, actualRows, numScalesPerRow);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventOutputVtoMTE3[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventOutputVtoMTE3[0]);

        uint32_t quantBytes = N * Catlass::SizeOfBits<ElementD>::value / Catlass::SizeOfBits<uint8_t>::value;
        Catlass::MatrixCoord rowZero{uint32_t{0}, uint32_t{0}};
        Catlass::MatrixCoord rowOne{uint32_t{1}, uint32_t{0}};
        int64_t quantRowStrideElements = layoutQuantOut.GetOffset(rowOne) - layoutQuantOut.GetOffset(rowZero);
        uint32_t quantRowStrideBytes = static_cast<uint32_t>(
            quantRowStrideElements * Catlass::SizeOfBits<ElementD>::value / Catlass::SizeOfBits<uint8_t>::value);
        AscendC::DataCopyExtParams quantParams{
            static_cast<uint16_t>(actualRows), quantBytes, 0, quantRowStrideBytes - quantBytes, 0};
        AscendC::DataCopyPad(gmQuantOut[layoutQuantOut.GetOffset(rowZero)], ubD, quantParams);

        int64_t scaleRowStrideElements = layoutScaleOut.GetOffset(rowOne) - layoutScaleOut.GetOffset(rowZero);
        uint32_t scaleRowStrideBytes = static_cast<uint32_t>(scaleRowStrideElements * sizeof(ElementScale));
        AscendC::DataCopyExtParams scaleParams{
            static_cast<uint16_t>(actualRows), numScalesPerRow, 0, scaleRowStrideBytes - numScalesPerRow, 0};
        AscendC::DataCopyPad(gmScaleOut[layoutScaleOut.GetOffset(rowZero)], ubScalePadded, scaleParams);

        // The producer may reuse both input planes immediately after this
        // method returns, so complete MTE3 here and re-prime the stage event.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[0]);
    }

    // Quantize each original token once, then scatter the same packed MX row
    // to all of its routed destinations.  A packed row is laid out as
    // [FP8 data, E8M0 scales, padding to a 32-byte row boundary].
    // This is the routing-side producer consumed directly by the packed
    // AllToAllV dispatch path; no expanded BF16 tensor is materialized.
    CATLASS_DEVICE
    void QuantScatterPacked(
        AscendC::GlobalTensor<ElementC> const& gmInput, AscendC::GlobalTensor<int32_t> const& gmExpandedRowIdx,
        AscendC::GlobalTensor<uint8_t> const& gmPackedOut, uint32_t tokenBegin, uint32_t tokenCount, uint32_t topK)
    {
        uint32_t N = params.N;
        if (tokenCount == 0 || N == 0 || topK == 0)
            return;

        uint32_t quantBytes = N * Catlass::SizeOfBits<ElementD>::value / Catlass::SizeOfBits<uint8_t>::value;
        uint32_t scaleBytesRaw = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint32_t scaleBytes = ((scaleBytesRaw + 1) / 2) * 2;
        uint32_t packedRowBytes =
            ((quantBytes + scaleBytes + Catlass::BYTE_PER_BLK - 1) / Catlass::BYTE_PER_BLK) * Catlass::BYTE_PER_BLK;

        for (uint32_t localToken = 0; localToken < tokenCount; ++localToken) {
            uint32_t tokenIdx = tokenBegin + localToken;
            uint32_t ubListId = localToken % UB_STAGES;
            auto& ubC = ubInputList[ubListId];
            auto& ubD = ubOutputList[ubListId];
            auto& ubScale = ubScaleList[ubListId];
            auto& ubMaxExp = ubMaxExpList[ubListId];
            auto& ubRecipScale = ubRecipScaleList[ubListId];

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventInputVtoMTE2[ubListId]);
            AscendC::DataCopyExtParams inputParams{1, static_cast<uint32_t>(N * sizeof(ElementC)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<ElementC> inputPad{false, 0, 0, 0};
            AscendC::DataCopyPad(ubC, gmInput[static_cast<int64_t>(tokenIdx) * N], inputParams, inputPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventInputMTE2toV[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventInputMTE2toV[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[ubListId]);
            ComputeMxQuant(ubC, ubD, ubScale, ubMaxExp, ubRecipScale, 1, N);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventInputVtoMTE2[ubListId]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventOutputVtoMTE3[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventOutputVtoMTE3[ubListId]);
            auto ubQuantBytes = ubD.template ReinterpretCast<uint8_t>();
            auto ubScaleBytes = ubScale.template ReinterpretCast<uint8_t>();
            AscendC::DataCopyExtParams quantParams{1, quantBytes, 0, 0, 0};
            AscendC::DataCopyExtParams scaleParams{1, scaleBytes, 0, 0, 0};
            for (uint32_t route = 0; route < topK; ++route) {
                int32_t dstRow = gmExpandedRowIdx(static_cast<int64_t>(tokenIdx) * topK + route);
                if (dstRow < 0)
                    continue;
                int64_t packedOffset = static_cast<int64_t>(dstRow) * packedRowBytes;
                AscendC::DataCopyPad(gmPackedOut[packedOffset], ubQuantBytes, quantParams);
                AscendC::DataCopyPad(gmPackedOut[packedOffset + quantBytes], ubScaleBytes, scaleParams);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventOutputMTE3toV[ubListId]);
        }
    }

private:
    CATLASS_DEVICE
    void PadMxScaleRows(
        AscendC::LocalTensor<ElementScale>& ubScale, AscendC::LocalTensor<ElementScale>& ubScalePadded,
        uint32_t actualRows, uint32_t numScalesPerRow)
    {
        auto srcAddr = reinterpret_cast<__ubuf__ int8_t*>(ubScale.GetPhyAddr());
        auto dstAddr = reinterpret_cast<__ubuf__ int8_t*>(ubScalePadded.GetPhyAddr());
        uint16_t rows = static_cast<uint16_t>(actualRows);
        uint16_t scales = static_cast<uint16_t>(numScalesPerRow);

        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < rows; ++row) {
                uint32_t elementNum = scales;
                AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<int8_t>(elementNum);
                AscendC::MicroAPI::RegTensor<int8_t> scaleReg;
                AscendC::MicroAPI::UnalignReg unalign;
                auto srcRow = srcAddr + row * scales;
                AscendC::MicroAPI::DataCopyUnAlignPre(unalign, srcRow);
                AscendC::MicroAPI::DataCopyUnAlign(scaleReg, unalign, srcRow);
                auto dstRow = dstAddr + row * Catlass::BYTE_PER_BLK;
                AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(dstRow, scaleReg, mask);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // === tail_axis constants ===
    static constexpr uint32_t VF_LEN_16 = 128;
    static constexpr uint32_t VF_LEN_16_DBL = 256;
    static constexpr uint32_t VF_LEN_32 = 64;
    static constexpr uint16_t ELEM_AFTER_REDUCE = 8;
    static constexpr uint16_t MX_MAX_EXP_BF16 = 0x7F80;
    static constexpr uint16_t MX_MAX_EXP_FP8 = 0x00FF;
    static constexpr uint16_t MX_BF16_EXP_BIAS = 0x7F00;
    static constexpr int16_t MX_SHR_NUM = 7;
    static constexpr uint16_t MX_NAN_CUSTOM = 0x7F81;
    static constexpr uint16_t MX_SPECIAL_EXP = 0x0040;
    static constexpr uint16_t MX_E4M3_EMAX = 0x0400;
    static constexpr uint16_t MX_E5M2_EMAX = 0x0780;

    CATLASS_DEVICE
    void ComputeMxQuant(
        AscendC::LocalTensor<ElementC>& ubC, AscendC::LocalTensor<ElementD>& ubD,
        AscendC::LocalTensor<ElementScale>& ubScale, AscendC::LocalTensor<uint16_t>& ubMaxExp,
        AscendC::LocalTensor<uint16_t>& ubRecipScale, uint32_t curRows, uint32_t N)
    {
        uint32_t numGrp = N / BLOCK_SIZE;
        uint32_t numGrpA = ((numGrp + 1) / 2) * 2;
        uint32_t totalBlk = curRows * numGrpA;
        uint16_t loop2VF = (totalBlk + ELEM_AFTER_REDUCE - 1) / ELEM_AFTER_REDUCE;
        uint16_t loop1VF = (totalBlk + VF_LEN_16 - 1) / VF_LEN_16;
        auto xAddr = reinterpret_cast<__ubuf__ ElementC*>(ubC.GetPhyAddr());
        auto yAddr = reinterpret_cast<__ubuf__ int8_t*>(ubD.GetPhyAddr());
        auto sAddr = reinterpret_cast<__ubuf__ uint16_t*>(ubScale.GetPhyAddr());
        auto meAddr = reinterpret_cast<__ubuf__ uint16_t*>(ubMaxExp.GetPhyAddr());
        auto rsAddr = reinterpret_cast<__ubuf__ uint16_t*>(ubRecipScale.GetPhyAddr());
        uint16_t fpEmax = TARGET_EMAX_FIELD;

        // Phase 1: ComputeMaxExp
        {
            auto xLocalAddr = xAddr;
            auto maxExpAddr = meAddr;
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<ElementC> vdExp0;
                AscendC::MicroAPI::RegTensor<ElementC> vdExp1;
                AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
                AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0;
                AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract1;
                AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
                AscendC::MicroAPI::Duplicate(expMaskBF16, MX_MAX_EXP_BF16);
                AscendC::MicroAPI::MaskReg Mask =
                    AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::UnalignReg ureg;
                for (uint16_t i = 0; i < loop2VF; i++) {
                    AscendC::MicroAPI::LoadAlign<
                        ElementC, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, xLocalAddr, VF_LEN_16_DBL);
                    AscendC::MicroAPI::And(
                        vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, expMaskBF16, Mask);
                    AscendC::MicroAPI::And(
                        vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, expMaskBF16, Mask);
                    AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, Mask);
                    AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);
                    AscendC::MicroAPI::StoreUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        maxExpAddr, vdMaxExp, ureg, ELEM_AFTER_REDUCE);
                }
                AscendC::MicroAPI::StoreUnAlignPost(maxExpAddr, ureg, 0);
            }
        }

        AscendC::PipeBarrier<PIPE_V>();

        // Phase 2: ComputeScale
        {
            auto maxExpAddr = meAddr;
            auto mxScaleLocalAddr = sAddr;
            auto recipScaleLocalAddr = rsAddr;
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
                AscendC::MicroAPI::RegTensor<uint16_t> sharedExp;
                AscendC::MicroAPI::RegTensor<uint16_t> scaleValue;
                AscendC::MicroAPI::RegTensor<uint16_t> halfScale;

                AscendC::MicroAPI::RegTensor<uint16_t> expMask;
                AscendC::MicroAPI::Duplicate(expMask, MX_MAX_EXP_BF16);
                AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue;
                AscendC::MicroAPI::Duplicate(maxExpValue, fpEmax);
                AscendC::MicroAPI::RegTensor<uint16_t> scaleBias;
                AscendC::MicroAPI::Duplicate(scaleBias, MX_BF16_EXP_BIAS);
                AscendC::MicroAPI::RegTensor<uint16_t> fpNanRegTensor;
                AscendC::MicroAPI::Duplicate(fpNanRegTensor, MX_MAX_EXP_FP8);
                AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
                AscendC::MicroAPI::Duplicate(zeroRegTensor, static_cast<uint16_t>(0));
                AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
                AscendC::MicroAPI::Duplicate(nanRegTensor, MX_NAN_CUSTOM);
                AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
                AscendC::MicroAPI::Duplicate(specialExpRegTensor, MX_SPECIAL_EXP);
                AscendC::MicroAPI::MaskReg cmpResult, zeroMask, invalidDataMask, specialDataMask, preMaskScale;
                for (uint16_t i = 0; i < loop1VF; i++) {
                    preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalBlk);
                    AscendC::MicroAPI::LoadAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        vdMaxExp, maxExpAddr, VF_LEN_16);
                    AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(
                        cmpResult, vdMaxExp, expMask, preMaskScale);
                    AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(
                        invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
                    AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
                    AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
                    AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, MX_SHR_NUM, preMaskScale);
                    AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fpNanRegTensor, cmpResult);
                    AscendC::MicroAPI::StoreAlign<
                        uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                        AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                        mxScaleLocalAddr, scaleValue, VF_LEN_32, preMaskScale);
                    AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(
                        zeroMask, sharedExp, zeroRegTensor, preMaskScale);
                    AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(
                        specialDataMask, sharedExp, scaleBias, preMaskScale);
                    AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
                    AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
                    AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
                    AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);
                    AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        recipScaleLocalAddr, halfScale, VF_LEN_16, preMaskScale);
                }
            }
        }

        AscendC::PipeBarrier<PIPE_V>();

        // Phase 3: ComputeData
        if constexpr (IsFp4) {
            // FP4 path (bf16 input): bf16 * recipScale → interleave → Cast<fp4x2> + DIST_PACK4_B32
            static constexpr uint32_t OUT_ELE_NUM_ONE_BLK = 64;
            {
                auto xLocalAddr = xAddr;
                auto recipScaleLocalAddr = rsAddr;
                auto yLocalAddr = yAddr;
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::MaskReg dataMaskB16 = AscendC::MicroAPI::CreateMask<half>();
                    AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
                    AscendC::MicroAPI::RegTensor<ElementC> vdExp0, vdExp1;
                    AscendC::MicroAPI::RegTensor<ElementD> vdExp0FP4, vdExp1FP4;
                    static constexpr AscendC::MicroAPI::CastTrait cFP4 = {
                        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                        AscendC::MicroAPI::MaskMergeMode::ZEROING, ROUND_MODE};
                    for (uint16_t i = 0; i < loop2VF; i++) {
                        AscendC::MicroAPI::LoadAlign<
                            ElementC, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                            AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, xLocalAddr, VF_LEN_16_DBL);
                        AscendC::MicroAPI::LoadAlign<
                            uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                            AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                            halfScaleForMul, recipScaleLocalAddr, ELEM_AFTER_REDUCE);
                        AscendC::MicroAPI::Mul(
                            vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<ElementC>&)halfScaleForMul, dataMaskB16);
                        AscendC::MicroAPI::Mul(
                            vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<ElementC>&)halfScaleForMul, dataMaskB16);
                        AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                        AscendC::MicroAPI::Cast<ElementD, ElementC, cFP4>(vdExp0FP4, vdExp0, dataMaskB16);
                        AscendC::MicroAPI::Cast<ElementD, ElementC, cFP4>(vdExp1FP4, vdExp1, dataMaskB16);
                        AscendC::MicroAPI::StoreAlign<
                            int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                            AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            yLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK,
                            dataMaskB16);
                        AscendC::MicroAPI::StoreAlign<
                            int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                            AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            yLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK,
                            dataMaskB16);
                    }
                }
            }
        } else {
            // FP8 path: bf16 * recipScale → float → Cast<fp8> + DIST_NORM_B8
            {
                auto xLocalAddr = xAddr;
                auto recipScaleLocalAddr = rsAddr;
                auto yLocalAddr = yAddr;
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::MaskReg dataMask1 = AscendC::MicroAPI::CreateMask<ElementC>();
                    AscendC::MicroAPI::MaskReg dataMask2 = AscendC::MicroAPI::CreateMask<ElementC>();
                    AscendC::MicroAPI::MaskReg dataMask3 = AscendC::MicroAPI::CreateMask<float>();
                    AscendC::MicroAPI::MaskReg dataMask4 = AscendC::MicroAPI::CreateMask<float>();
                    AscendC::MicroAPI::MaskReg dataMask5 = AscendC::MicroAPI::CreateMask<ElementD>();
                    AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
                    AscendC::MicroAPI::RegTensor<ElementC> vdExp0, vdExp1;
                    AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
                    AscendC::MicroAPI::RegTensor<ElementD> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;
                    static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
                        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
                    static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
                        AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
                        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
                    static constexpr AscendC::MicroAPI::CastTrait castTrait32to80 = {
                        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
                    static constexpr AscendC::MicroAPI::CastTrait castTrait32to81 = {
                        AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::SAT,
                        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
                    static constexpr AscendC::MicroAPI::CastTrait castTrait32to82 = {
                        AscendC::MicroAPI::RegLayout::TWO, AscendC::MicroAPI::SatMode::SAT,
                        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
                    static constexpr AscendC::MicroAPI::CastTrait castTrait32to83 = {
                        AscendC::MicroAPI::RegLayout::THREE, AscendC::MicroAPI::SatMode::SAT,
                        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
                    constexpr float fp8Max = std::is_same_v<ElementD, float8_e4m3_t> ? 448.0f : 57344.0f;
                    for (uint16_t i = 0; i < loop2VF; i++) {
                        AscendC::MicroAPI::LoadAlign<
                            ElementC, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                            AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, xLocalAddr, VF_LEN_16_DBL);
                        AscendC::MicroAPI::LoadAlign<
                            uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                            AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                            halfScaleForMul, recipScaleLocalAddr, ELEM_AFTER_REDUCE);
                        AscendC::MicroAPI::Mul(
                            vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<ElementC>&)halfScaleForMul, dataMask1);
                        AscendC::MicroAPI::Mul(
                            vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<ElementC>&)halfScaleForMul, dataMask1);
                        AscendC::MicroAPI::Cast<float, ElementC, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                        AscendC::MicroAPI::Cast<float, ElementC, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                        AscendC::MicroAPI::Cast<float, ElementC, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
                        AscendC::MicroAPI::Cast<float, ElementC, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);

                        // Ascend950 converts finite E4M3 overflow to the NaN encoding even with SAT mode.
                        // MX reference quantization clips to the finite FP8 range before RNE conversion.
                        AscendC::MicroAPI::Mins(vdExp0FP32Zero, vdExp0FP32Zero, fp8Max, dataMask3);
                        AscendC::MicroAPI::Maxs(vdExp0FP32Zero, vdExp0FP32Zero, -fp8Max, dataMask3);
                        AscendC::MicroAPI::Mins(vdExp0FP32One, vdExp0FP32One, fp8Max, dataMask3);
                        AscendC::MicroAPI::Maxs(vdExp0FP32One, vdExp0FP32One, -fp8Max, dataMask3);
                        AscendC::MicroAPI::Mins(vdExp1FP32Zero, vdExp1FP32Zero, fp8Max, dataMask4);
                        AscendC::MicroAPI::Maxs(vdExp1FP32Zero, vdExp1FP32Zero, -fp8Max, dataMask4);
                        AscendC::MicroAPI::Mins(vdExp1FP32One, vdExp1FP32One, fp8Max, dataMask4);
                        AscendC::MicroAPI::Maxs(vdExp1FP32One, vdExp1FP32One, -fp8Max, dataMask4);

                        AscendC::MicroAPI::Cast<ElementD, float, castTrait32to80>(
                            vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
                        AscendC::MicroAPI::Cast<ElementD, float, castTrait32to82>(
                            vdExp0FP8One, vdExp0FP32One, dataMask3);
                        AscendC::MicroAPI::Cast<ElementD, float, castTrait32to81>(
                            vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
                        AscendC::MicroAPI::Cast<ElementD, float, castTrait32to83>(
                            vdExp1FP8One, vdExp1FP32One, dataMask4);

                        AscendC::MicroAPI::Add(
                            (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                            (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                            (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8One, dataMask5);
                        AscendC::MicroAPI::Add(
                            (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8Zero,
                            (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8Zero,
                            (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8One, dataMask5);
                        AscendC::MicroAPI::Add(
                            (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                            (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                            (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8Zero, dataMask5);

                        AscendC::MicroAPI::StoreAlign<
                            int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                            AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                            yLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP8Zero, VF_LEN_16_DBL, dataMask5);
                    }
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    Params params;
    Catlass::Arch::Resource<ArchTag>* resourceRef{nullptr};
    AscendC::LocalTensor<ElementC> ubInputList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubOutputList[UB_STAGES];
    AscendC::LocalTensor<ElementScale> ubScaleList[UB_STAGES];
    AscendC::LocalTensor<uint16_t> ubMaxExpList[UB_STAGES];
    AscendC::LocalTensor<uint16_t> ubRecipScaleList[UB_STAGES];
    int32_t eventInputVtoMTE2[UB_STAGES];
    int32_t eventInputMTE2toV[UB_STAGES];
    int32_t eventOutputMTE3toV[UB_STAGES];
    int32_t eventOutputVtoMTE3[UB_STAGES];
    uint32_t tileRows{0};
};

} // namespace Catccos::Comm::Block

#endif // CATCCOS_COMM_BLOCK_COMM_BLOCK_MX_QUANT_HPP
