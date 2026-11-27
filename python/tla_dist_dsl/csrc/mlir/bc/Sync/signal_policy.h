// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
namespace tla_dist_sync {
// A translated nonzero slot address is not evidence of MTE reachability: the
// pinned aclshmem_ptr implementation adds offsets even to a zero peer base.
constexpr bool signalRouteAllowed(bool mteTopology, bool nonzeroMteBase, bool roceTopology,
                                  bool rdmaBuilt, bool owner, bool add) {
    if (mteTopology) return nonzeroMteBase;
    return !add && roceTopology && rdmaBuilt && owner;
}
}
