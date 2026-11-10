/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include <iostream>
#include "catccos/detail/ascend950_moe_ub.h"

int main()
{
    using Ub = Catccos::Detail::Ascend950MoeUb;
    // Independently counted InitBuffer allocations for H=2048, M=512,
    // TopK=8, 48 AIVs: (4+1)*4096 + (8192+256) + 8192 + 2*320.
    if (Ub::UnpermuteBytes(512, 2048, 8, 48) != 37760 || Ub::InputQuantBytes(2048) != 12928 ||
        Ub::DispatchBytes(2048) != 8448) {
        std::cerr << "UB allocation accounting mismatch" << std::endl;
        return EXIT_FAILURE;
    }
    // The 512-element hidden alignment matters even for H divisible by 256.
    // H=14336 fits for tiny decode, whereas H=14592 needs 14848 padded elements.
    if (!Ub::Fits(1, 14336, 8, 48) || Ub::Fits(1, 14592, 8, 48) || Ub::Fits(1, 2048, 8, 0)) {
        std::cerr << "Hidden width or core-count boundary was accepted incorrectly" << std::endl;
        return EXIT_FAILURE;
    }
    // At 600 rows/core the index/probability queues alone can overflow UB.
    // Boundary: TopK=46 fits, TopK=47 exceeds 256 KiB at H=2048.
    if (!Ub::Fits(28800, 2048, 46, 48) || Ub::Fits(28800, 2048, 47, 48) || !Ub::Fits(1, 7168, 6, 48) ||
        !Ub::Fits(512, 2048, 8, 48) || Ub::UnpermuteBytes(1, 256, 1, 48) != 9536) {
        std::cerr << "Token/TopK queue boundary or tiny-row accounting mismatch" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Ascend950 shape-dependent UB capacity checks: PASS" << std::endl;
    return EXIT_SUCCESS;
}
