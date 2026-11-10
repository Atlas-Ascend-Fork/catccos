/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DETAIL_ASCEND950_MOE_UB_H
#define CATCCOS_DETAIL_ASCEND950_MOE_UB_H

#include <algorithm>
#include <cstdint>

namespace Catccos::Detail {

// Shape-dependent UB allocations in the BF16/MXFP8 example. GMM/SwiGLU tile
// planes have separate compile-time checks; these stages reuse the same UB.
struct Ascend950MoeUb {
    static constexpr uint64_t kCapacityBytes = 256 * 1024;

    static constexpr uint64_t Align(uint64_t bytes, uint64_t alignment)
    {
        return (bytes + alignment - 1) / alignment * alignment;
    }

    static constexpr uint64_t InputQuantBytes(uint32_t hidden)
    {
        uint64_t scale = Align((static_cast<uint64_t>(hidden) + 31) / 32, 2);
        // CommBlockMxQuant: two stages, each with BF16 input, FP8 output,
        // E8M0 scales and two uint16 scratch arrays; minimum tileRows is one.
        return 2 * (3 * static_cast<uint64_t>(hidden) + 5 * scale);
    }

    static constexpr uint64_t DispatchBytes(uint32_t hidden)
    {
        uint64_t scale = Align((static_cast<uint64_t>(hidden) + 31) / 32, 2);
        uint64_t row = Align(static_cast<uint64_t>(hidden) + scale, 32);
        // Ascend950AllToAllGmmKernel: two packed rows per ping-pong buffer.
        return 2 * Align(2 * row, 64);
    }

    static constexpr uint64_t UnpermuteBytes(uint32_t m, uint32_t hidden, uint32_t topK, uint32_t aivNum)
    {
        if (aivNum == 0) {
            return kCapacityBytes + 1;
        }
        uint64_t rows = std::max<uint64_t>(1, std::min<uint64_t>(m / aivNum, 600));
        // KernelMoeTokenUnpermute<BF16,int32,float,true>: four input
        // buffers, one output, two FP32 temporaries (+256 bytes), and
        // separately 32-byte-aligned index/probability queues.
        return 18 * Align(hidden, 512) + 256 + 2 * Align(rows * topK * sizeof(uint32_t), 32);
    }

    static constexpr bool Fits(uint32_t m, uint32_t hidden, uint32_t topK, uint32_t aivNum)
    {
        return InputQuantBytes(hidden) <= kCapacityBytes && DispatchBytes(hidden) <= kCapacityBytes &&
               UnpermuteBytes(m, hidden, topK, aivNum) <= kCapacityBytes;
    }
};

} // namespace Catccos::Detail
#endif
