/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef CATCCOS_COMM_BLOCK_BLOCK_QUANTIZE_HPP
#define CATCCOS_COMM_BLOCK_BLOCK_QUANTIZE_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/tile/tile_perblock_quant.hpp"
#include "shmem.h"

namespace Catccos::Comm::Block {

// Independent GM-to-GM per-token quantizer. Finite BF16 input with finite
// 448/amax (or an all-zero row), row-major FP8
// E4M3FN output and FP32 dequantization scale: x ~= q * scale. Each caller owns
// rows aicoreIdx + n * aicoreNum. No inter-core/ inter-rank barrier is issued;
// callers must synchronize before other workers consume these GM outputs.
class BlockQuantizeBf16Fp8 {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using TileQuant = Catlass::Epilogue::Tile::TilePerBlockQuant<ArchTag, bfloat16_t, float8_e4m3_t, float>;
    static constexpr uint32_t FLAG_BYTES = 2048;
    static constexpr uint32_t VECTOR_ELEMENTS = 128;

    CATLASS_HOST_DEVICE
    static uint32_t PaddedHidden(uint32_t hidden)
    {
        return (hidden + VECTOR_ELEMENTS - 1) / VECTOR_ELEMENTS * VECTOR_ELEMENTS;
    }

    CATLASS_HOST_DEVICE
    static bool CanImplement(uint32_t hidden)
    {
        return hidden > 0 && hidden <= (ArchTag::UB_SIZE - FLAG_BYTES - 32) / 3 &&
               FLAG_BYTES + 3 * PaddedHidden(hidden) + 32 <= ArchTag::UB_SIZE;
    }

    CATLASS_DEVICE
    void operator()(
        Catlass::Arch::Resource<ArchTag>& resource, GM_ADDR input, GM_ADDR output, GM_ADDR scales, uint32_t rows,
        uint32_t hidden, uint32_t aicoreIdx, uint32_t aicoreNum) const
    {
        if (!CanImplement(hidden) || aicoreNum == 0) {
            return;
        }
        const uint32_t padded = PaddedHidden(hidden);
        auto ubInput = resource.ubBuf.template GetBufferByByte<uint8_t>(FLAG_BYTES);
        auto ubOutput = resource.ubBuf.template GetBufferByByte<uint8_t>(FLAG_BYTES + padded * 2);
        auto ubScale = resource.ubBuf.template GetBufferByByte<float>(FLAG_BYTES + padded * 3);
        AscendC::GlobalTensor<uint8_t> gmInput, gmOutput, gmScales;
        gmInput.SetGlobalBuffer(input);
        gmOutput.SetGlobalBuffer(output);
        gmScales.SetGlobalBuffer(scales);
        for (uint32_t row = aicoreIdx; row < rows; row += aicoreNum) {
            aclshmemi_copy_gm2ub(ubInput, gmInput[uint64_t(row) * hidden * 2], hidden * 2);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            if (hidden != padded) {
                // SHMEM's DataCopyPad does not promise zero-valued alignment
                // padding. Clear the reduction tail AFTER the DMA completes.
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                auto ubBits = ubInput.ReinterpretCast<uint16_t>();
                for (uint32_t col = hidden; col < padded; ++col) {
                    ubBits.SetValue(col, uint16_t(0));
                }
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
            TileQuant quant;
            quant.perBlockScaleQuant(
                (__ubuf__ bfloat16_t*)ubInput.GetPhyAddr(), (__ubuf__ float8_e4m3_t*)ubOutput.GetPhyAddr(),
                (__ubuf__ float*)ubScale.GetPhyAddr(), padded / VECTOR_ELEMENTS);
#endif
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            // CATLASS divides by amax without a zero guard. Replace the NaN
            // payload produced for a zero row; the dequantization scale is 0.
            const bool zeroRow = ubScale.GetValue(0) == 0.0F;
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            if (zeroRow) {
                AscendC::Duplicate(ubOutput.ReinterpretCast<uint16_t>(), uint16_t(0), padded / 2);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            aclshmemi_copy_ub2gm(gmOutput[uint64_t(row) * hidden], ubOutput, hidden);
            aclshmemi_copy_ub2gm(
                gmScales[uint64_t(row) * sizeof(float)], ubScale.ReinterpretCast<uint8_t>(), sizeof(float));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }
};
// Per-token symmetric INT8 quantization, matching ops-transformer's cast
// sequence and reciprocal scale. The caller publishes GM outputs before repack.
class BlockQuantizeBf16Int8 {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    static constexpr uint32_t FLAG_BYTES = 2048;
    static constexpr uint32_t VECTOR_ELEMENTS = 128;
    static constexpr uint32_t FP32_ELEMENTS_PER_REPEAT = Catlass::BYTE_PER_VECTOR_FRACTAL / sizeof(float);
    static constexpr uint32_t MAX_VECTOR_REPEAT_COUNT = 255;

    CATLASS_HOST_DEVICE
    static uint32_t PaddedHidden(uint32_t hidden)
    {
        return (hidden + VECTOR_ELEMENTS - 1) / VECTOR_ELEMENTS * VECTOR_ELEMENTS;
    }

    CATLASS_HOST_DEVICE
    static bool CanImplement(uint32_t hidden)
    {
        // BF16 input + INT8 output + two FP32 temporaries + one scale block.
        return hidden > 0 && hidden <= (ArchTag::UB_SIZE - FLAG_BYTES - 32) / 11 &&
               FLAG_BYTES + 11 * PaddedHidden(hidden) + 32 <= ArchTag::UB_SIZE;
    }

    CATLASS_DEVICE
    void operator()(
        Catlass::Arch::Resource<ArchTag>& resource, GM_ADDR input, GM_ADDR output, GM_ADDR scales, uint32_t rows,
        uint32_t hidden, uint32_t aivIdx, uint32_t aivNum) const
    {
        if (!CanImplement(hidden) || aivNum == 0) {
            return;
        }
        const uint32_t padded = PaddedHidden(hidden);
        const uint32_t fullRepeats = hidden / FP32_ELEMENTS_PER_REPEAT;
        const uint32_t tailElements = hidden % FP32_ELEMENTS_PER_REPEAT;
        auto ubInput = resource.ubBuf.template GetBufferByByte<bfloat16_t>(FLAG_BYTES);
        auto ubOutput = resource.ubBuf.template GetBufferByByte<int8_t>(FLAG_BYTES + 2 * padded);
        auto ubFloat = resource.ubBuf.template GetBufferByByte<float>(FLAG_BYTES + 3 * padded);
        auto ubAbs = resource.ubBuf.template GetBufferByByte<float>(FLAG_BYTES + 7 * padded);
        auto ubScale = resource.ubBuf.template GetBufferByByte<float>(FLAG_BYTES + 11 * padded);
        AscendC::GlobalTensor<uint8_t> gmInput, gmOutput, gmScales;
        gmInput.SetGlobalBuffer(input);
        gmOutput.SetGlobalBuffer(output);
        gmScales.SetGlobalBuffer(scales);
        for (uint32_t row = aivIdx; row < rows; row += aivNum) {
            aclshmemi_copy_gm2ub(
                ubInput.ReinterpretCast<uint8_t>(), gmInput[uint64_t(row) * hidden * sizeof(bfloat16_t)],
                hidden * sizeof(bfloat16_t));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(ubFloat, ubInput, AscendC::RoundMode::CAST_NONE, hidden);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs(ubAbs, ubFloat, hidden);
            AscendC::PipeBarrier<PIPE_V>();
            // Accumulate full vectors into the first FP32 vector, as in
            // ops-transformer's ReduceMaxInplace. Split at the repeat-count limit.
            for (uint32_t repeatOffset = 1; repeatOffset < fullRepeats;) {
                const uint32_t repeatCount = fullRepeats - repeatOffset < MAX_VECTOR_REPEAT_COUNT ?
                                                 fullRepeats - repeatOffset :
                                                 MAX_VECTOR_REPEAT_COUNT;
                AscendC::Max(
                    ubAbs, ubAbs[repeatOffset * FP32_ELEMENTS_PER_REPEAT], ubAbs, FP32_ELEMENTS_PER_REPEAT,
                    static_cast<uint8_t>(repeatCount), {1, 1, 1, 0, 8, 0});
                AscendC::PipeBarrier<PIPE_V>();
                repeatOffset += repeatCount;
            }
            if (tailElements > 0 && fullRepeats > 0) {
                AscendC::Max(
                    ubAbs, ubAbs[fullRepeats * FP32_ELEMENTS_PER_REPEAT], ubAbs, tailElements, 1, {1, 1, 1, 0, 8, 0});
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::WholeReduceMax(
                ubAbs, ubAbs, hidden < FP32_ELEMENTS_PER_REPEAT ? hidden : FP32_ELEMENTS_PER_REPEAT, 1, 8, 1, 8);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            const float amax = ubAbs.GetValue(0);
            const float multiplier = amax == 0.0F ? 0.0F : 127.0F / amax;
            ubScale.SetValue(0, amax == 0.0F ? 0.0F : 1.0F / multiplier);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::Muls(ubFloat, ubFloat, multiplier, hidden);
            AscendC::PipeBarrier<PIPE_V>();
            auto ubInt32 = ubFloat.ReinterpretCast<int32_t>();
            auto ubHalf = ubFloat.ReinterpretCast<half>();
            AscendC::Cast(ubInt32, ubFloat, AscendC::RoundMode::CAST_RINT, hidden);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetDeqScale(static_cast<half>(1.0F));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(ubHalf, ubInt32, AscendC::RoundMode::CAST_ROUND, hidden);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(ubOutput, ubHalf, AscendC::RoundMode::CAST_TRUNC, hidden);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            aclshmemi_copy_ub2gm(gmOutput[uint64_t(row) * hidden], ubOutput.ReinterpretCast<uint8_t>(), hidden);
            aclshmemi_copy_ub2gm(
                gmScales[uint64_t(row) * sizeof(float)], ubScale.ReinterpretCast<uint8_t>(), sizeof(float));
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
};
} // namespace Catccos::Comm::Block
#endif
