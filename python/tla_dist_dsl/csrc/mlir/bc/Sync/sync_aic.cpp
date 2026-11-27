// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "kernel_operator.h"
#if !defined(__CCE_AICORE_ENABLE_MIX__)
#error "MIX AIC synchronization requires MIX headers"
#endif
extern "C" {
__aicore__ void tla_dist_local_sync_mix_aic() { AscendC::SyncAll<false>(); }
__aicore__ void tla_dist_sync_mix_aic(int32_t team) {
    if (team == -1) return;
    AscendC::SyncAll<false>(); AscendC::SyncAll<false>();
}
__aicore__ void tla_dist_barrier_mix_aic(int32_t team) {
    if (team == -1) return;
    AscendC::SyncAll<false>(); AscendC::SyncAll<false>();
}
}
