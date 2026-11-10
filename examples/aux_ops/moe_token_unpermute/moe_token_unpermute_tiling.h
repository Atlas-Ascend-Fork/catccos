
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MOE_TOKEN_UNPERMUTE_TILING_H
#define MOE_TOKEN_UNPERMUTE_TILING_H

struct MoeTokenUnpermuteTilingData {
    int64_t hidden_size;
    int64_t top_k;
    int64_t num_out_tokens;
    int64_t hidden_splited_length;
    int64_t hidden_splited_num;
    int64_t hidden_splited_remain;
    int64_t tokens_core_length;
    int64_t tokens_core_remain;
    int64_t tokens_splited_length;
    int64_t tokens_splited_num;
    int64_t tokens_splited_remain;
    int64_t buffer_num;
};

__forceinline__[host, aicore] void MoeTokenUnpermuteTiling(
    int32_t m, int32_t n, int32_t topK, MoeTokenUnpermuteTilingData& tilingData, uint32_t coreNum)
{
    tilingData.hidden_size = static_cast<int64_t>(n);
    tilingData.top_k = static_cast<int64_t>(topK);
    tilingData.num_out_tokens = static_cast<int64_t>(m);
    tilingData.hidden_splited_length = tilingData.hidden_size;
    tilingData.hidden_splited_num = 1;
    tilingData.hidden_splited_remain = 0;
    uint32_t outTokens = m / topK;
    tilingData.tokens_core_length = static_cast<int64_t>(outTokens / coreNum);
    tilingData.tokens_core_remain = static_cast<int64_t>(outTokens % coreNum);
    tilingData.tokens_splited_length = tilingData.tokens_core_length < static_cast<int64_t>(600) ?
                                           tilingData.tokens_core_length :
                                           static_cast<int64_t>(600);
    // Some AIVs own no complete row when outTokens < coreNum.
    if (tilingData.tokens_splited_length == 0) {
        tilingData.tokens_splited_length = 1;
    }
    tilingData.tokens_splited_num =
        static_cast<int64_t>(tilingData.tokens_core_length / tilingData.tokens_splited_length);
    tilingData.tokens_splited_remain =
        static_cast<int64_t>(tilingData.tokens_core_length % tilingData.tokens_splited_length);
    tilingData.buffer_num = 4;
}

#endif
