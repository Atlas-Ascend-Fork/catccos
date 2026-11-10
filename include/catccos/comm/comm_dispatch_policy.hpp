
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DISPATCH_POLICY_HPP
#define CATCCOS_DISPATCH_POLICY_HPP

// from catlass
#include "catccos/detail/remote_copy_type.hpp"
#include "catlass/arch/arch.hpp"

namespace Catccos::Comm {

// For AtlasA2, an remote copy epilogue of the form D(share mem) = C(share mem)
template <uint32_t UB_STAGES_, bool IsDynamic_ = false>
struct AtlasA2CommToShareMem {
    using ArchTag = Catlass::Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

// For AtlasA2, an remote copy epilogue of the form D(local mem) = C(share mem)
template <uint32_t UB_STAGES_, bool IsDynamic_ = false>
struct AtlasA2CommToLocalMem {
    using ArchTag = Catlass::Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <class ArchTag_, uint32_t UB_STAGES_, bool IsDynamic_ = false>
struct AtlasCommRemoteCopy {
    using ArchTag = ArchTag_;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <class ArchTag_, uint32_t UB_STAGES_>
struct AtlasCommUdmaRemoteCopy {
    using ArchTag = ArchTag_;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <class ArchTag_, uint32_t UB_STAGES_, bool IsDynamic_ = false>
struct AtlasCommHierarchicalCopy {
    using ArchTag = ArchTag_;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <class ArchTag_, uint32_t UB_STAGES_, bool IsDynamic_ = false>
struct AtlasCommRemoteChunkCopy {
    using ArchTag = ArchTag_;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <class ArchTag_, uint32_t UB_STAGES_, bool IsDynamic_ = false>
struct AtlasCommLocalCopy {
    using ArchTag = ArchTag_;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr bool IsDynamic = IsDynamic_;
};

template <uint32_t UB_STAGES_, bool IsDynamic_ = false>
using AtlasA2CommLocalCopy = AtlasCommLocalCopy<Catlass::Arch::AtlasA2, UB_STAGES_, IsDynamic_>;

// For AtlasA2, per tensor dequant
template <uint32_t UB_STAGES_, bool IsDynamic_ = false>
struct AtlasA2PerTensorDequant {
    using ArchTag = Catlass::Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr bool IsDynamic = IsDynamic_;
};

template <uint32_t UB_STAGES_>
struct AtlasA2CommRdmaCopy {
    using ArchTag = Catlass::Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_, bool IsDynamic_ = false>
struct AtlasA5CommLocalCast {
    using ArchTag = Catlass::Arch::Ascend950;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_, uint32_t BLOCK_SIZE_ = 32, int64_t ROUND_MODE_ = 4>
struct EpilogueAscend950DynamicMxQuant {
    using ArchTag = Catlass::Arch::Ascend950;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr uint32_t BLOCK_SIZE = BLOCK_SIZE_;
    static constexpr int64_t ROUND_MODE = ROUND_MODE_; // 4=rint, 1=floor, 0=round
};

} // namespace Catccos::Comm

#endif // CATCCOS_DISPATCH_POLICY_HPP
