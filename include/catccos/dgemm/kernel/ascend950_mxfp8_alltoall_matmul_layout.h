/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_LAYOUT_H
#define CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_LAYOUT_H

#include <cstdint>

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
// Separate A, AScale and ready; each source publishes a monotonic window index.
constexpr uint64_t CONTROL_STRIDE = 512;
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
    uint64_t readyFrontier = 0;
    uint64_t readyValues = 0; // Immutable network window sequence values.
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
        return ready + static_cast<uint64_t>(source) * CONTROL_STRIDE;
    }

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
    out.readyFrontier = AlignUp(out.ready + static_cast<uint64_t>(rankSize) * CONTROL_STRIDE, ALIGNMENT);
    out.readyValues = AlignUp(out.readyFrontier + CONTROL_STRIDE, ALIGNMENT);
    out.symmetricBytes = AlignUp(out.readyValues + windows * sizeof(uint64_t), ALIGNMENT);
    return true;
}

} // namespace Catccos::DGemm::Kernel::MxFp8AllToAllMatmul

#endif // CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_LAYOUT_H
