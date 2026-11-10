/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_TORCH_KERNEL_H
#define CATCCOS_TORCH_KERNEL_H

#include <acl/acl.h>
#include <vector>

namespace CatccosKernel {

// Query the native allocation limit without importing CCEC-only declarations
// or generic info.h include guards into the PyTorch translation unit.
uint64_t catccos_ascend950_symmetric_bytes();

/**
 * @brief AllGather Matmul kernel wrapper
 * @param blockDim number of AICore blocks
 * @param stream ACL stream
 * @param fftsAddr FFTS configuration address
 * @param aPtr pointer to matrix A on device
 * @param bPtr pointer to matrix B on device
 * @param cPtr pointer to matrix C on device (output)
 * @param gmSymmetric pointer to symmetric memory
 * @param m rows of matrix A
 * @param n columns of matrix B
 * @param k columns of matrix A / rows of matrix B
 * @param rankId current rank index
 * @param rankSize number of ranks
 */
void catccos_allgather_matmul_wrapper(
    uint32_t blockDim, aclrtStream stream, uint64_t fftsAddr, uint8_t* aPtr, uint8_t* bPtr, uint8_t* cPtr,
    uint8_t* gmSymmetric, uint32_t m, uint32_t n, uint32_t k, int rankId, int rankSize);

void catccos_ascend950_dispatch_ffn_combine_wrapper(
    uint32_t blockDim, aclrtStream stream, uint64_t fftsAddr, uint8_t* xPtr, uint8_t* expertIdxPtr,
    uint8_t* gateWeightPtr, uint8_t* w1Ptr, uint8_t* w1ScalePtr, uint8_t* w2Ptr, uint8_t* w2ScalePtr,
    uint8_t* outputPtr, uint8_t* workspacePtr, uint8_t* symmetricPtr, uint32_t m, uint32_t k, uint32_t n,
    uint32_t localExperts, uint32_t topK, int rankSize);

} // namespace CatccosKernel

#endif // CATCCOS_TORCH_KERNEL_H
