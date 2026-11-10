/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DETAIL_ASCEND950_MOE_READY_HPP
#define CATCCOS_DETAIL_ASCEND950_MOE_READY_HPP

#include <cstdint>
#include "catlass/detail/macros.hpp"

namespace Catccos::Detail {

// These counters reuse otherwise idle BF16 staging regions. Their capacity
// grows with expert count even when a decode launch has only one token.
struct Ascend950MoeReady {
    static constexpr uint32_t kCacheLineBytes = 64;
    static constexpr uint32_t kStride = kCacheLineBytes / sizeof(int32_t);

    CATLASS_HOST_DEVICE
    static constexpr uint64_t Align(uint64_t bytes, uint64_t alignment = 512)
    {
        return (bytes + alignment - 1) / alignment * alignment;
    }

    CATLASS_HOST_DEVICE
    static constexpr uint64_t CounterBytes(uint32_t localExperts)
    {
        return static_cast<uint64_t>(localExperts) * kCacheLineBytes;
    }

    CATLASS_HOST_DEVICE
    static constexpr uint64_t DispatchOffset(uint32_t ranks, uint32_t localExperts)
    {
        return Align(static_cast<uint64_t>(ranks) * localExperts * sizeof(int32_t), kCacheLineBytes);
    }

    CATLASS_HOST_DEVICE
    static constexpr uint64_t SwigluRegionBytes(uint64_t stagingBytes, uint32_t localExperts)
    {
        uint64_t readyBytes = CounterBytes(localExperts);
        return Align(stagingBytes > readyBytes ? stagingBytes : readyBytes);
    }

    CATLASS_HOST_DEVICE
    static constexpr uint64_t DispatchRegionBytes(uint64_t stagingBytes, uint32_t ranks, uint32_t localExperts)
    {
        uint64_t readyBytes = DispatchOffset(ranks, localExperts) + CounterBytes(localExperts);
        return Align(stagingBytes > readyBytes ? stagingBytes : readyBytes);
    }
};

} // namespace Catccos::Detail
#endif
