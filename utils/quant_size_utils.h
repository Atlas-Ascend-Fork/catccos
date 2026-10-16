/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_SIZE_UTILS_H
#define QUANT_SIZE_UTILS_H

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

inline size_t CheckedBufferSize(std::initializer_list<size_t> factors)
{
    // Buffer sizes must also be representable by signed GM offsets and file sizes.
    constexpr size_t MAX_SIZE = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
    size_t result = 1;
    for (size_t factor : factors) {
        if (factor != 0 && result > MAX_SIZE / factor) {
            throw std::overflow_error("Quant buffer size exceeds the addressable range");
        }
        result *= factor;
    }
    return result;
}

inline void CheckQuantBlockCount(size_t elementCount, uint32_t rankSize)
{
    if (rankSize == 0) {
        throw std::invalid_argument("Quant rank size must be positive");
    }
    // Match TILE_M and FLAG_MAGIC_STRIDE in QuantAllGather/QuantAllToAll.
    constexpr size_t TILE_ELEMENTS = 1U << 20;
    constexpr size_t MAX_BLOCKS = (1U << 20) - 1;
    size_t elementsPerCore = elementCount / rankSize + (elementCount % rankSize != 0);
    if (elementsPerCore > MAX_BLOCKS * TILE_ELEMENTS) {
        throw std::overflow_error("Quant block count exceeds the synchronization flag range");
    }
}

#endif // QUANT_SIZE_UTILS_H
