// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
#if defined(__NPU_ARCH__)
#define TLA_DIST_SYNC_POLICY_DEVICE __aicore__
#else
#define TLA_DIST_SYNC_POLICY_DEVICE
#endif
namespace tla_dist_sync {
// A translated nonzero slot address is not evidence of MTE reachability: the
// pinned aclshmem_ptr implementation adds offsets even to a zero peer base.
TLA_DIST_SYNC_POLICY_DEVICE constexpr bool signalRouteAllowed(
    bool mteTopology, bool nonzeroMteBase, bool roceTopology, bool rdmaBuilt, bool owner, bool add)
{
    if (mteTopology)
        return nonzeroMteBase;
    return !add && roceTopology && rdmaBuilt && owner;
}
} // namespace tla_dist_sync

#undef TLA_DIST_SYNC_POLICY_DEVICE
