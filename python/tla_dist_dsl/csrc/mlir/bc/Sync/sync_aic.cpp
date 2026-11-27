// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "kernel_operator.h"
#if !defined(__CCE_AICORE_ENABLE_MIX__)
#error "MIX AIC synchronization requires MIX headers"
#endif
extern "C" {
[aicore] __attribute__((always_inline, convergent)) void tla_dist_local_sync_mix_aic() { AscendC::SyncAll<false>(); }
[aicore] __attribute__((always_inline, convergent)) void tla_dist_sync_mix_aic(int32_t team)
{
    if (team == -1)
        return;
    AscendC::SyncAll<false>();
    AscendC::SyncAll<false>();
}
[aicore] __attribute__((always_inline, convergent)) void tla_dist_barrier_mix_aic(int32_t team)
{
    if (team == -1)
        return;
    AscendC::SyncAll<false>();
    AscendC::SyncAll<false>();
}
}
