/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to LICENSE in the root of the software repository for details.
 */
#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <initializer_list>
#include "catccos/detail/ascend950_moe_ready.hpp"

int main()
{
    using Ready = Catccos::Detail::Ascend950MoeReady;
    // A single-token EP2 launch used to reserve only 1024 bytes for 2048 bytes
    // of local32 counters, overlapping the following live FP8 activation.
    static_assert(Ready::SwigluRegionBytes(1024, 32) == 2048);
    static_assert(Ready::DispatchRegionBytes(1024, 2, 32) == 2560);
    for (uint32_t ranks : {2U, 4U, 8U}) {
        for (uint32_t experts : {2U, 16U, 32U, 64U, 128U}) {
            for (uint64_t rows : {1U, 2U, 17U, 138U, 512U}) {
                uint64_t q = rows * ranks;
                uint64_t swiglu = Ready::SwigluRegionBytes(q * 256 * 2, experts);
                uint64_t dispatch = Ready::DispatchRegionBytes(q * 256 * 2, ranks, experts);
                if (!(swiglu % 512 == 0 && dispatch % 512 == 0)) {
                    std::cerr << "Ready regions must be aligned to 512 bytes" << std::endl;
                    return EXIT_FAILURE;
                }
                if (!(swiglu >= q * 256 * 2 && dispatch >= q * 256 * 2)) {
                    std::cerr << "Ready regions must hold the BF16 staging data" << std::endl;
                    return EXIT_FAILURE;
                }
                // Include every cache line, including empty experts reset by
                // the next call. The last counter must not reach FP8 storage.
                for (uint32_t expert = 0; expert < experts; ++expert) {
                    uint64_t end = (static_cast<uint64_t>(expert) + 1) * Ready::kStride * sizeof(int32_t);
                    if (!(end <= swiglu)) {
                        std::cerr << "SwiGLU counter exceeds its region" << std::endl;
                        return EXIT_FAILURE;
                    }
                    if (!(Ready::DispatchOffset(ranks, experts) + end <= dispatch)) {
                        std::cerr << "Dispatch counter exceeds its region" << std::endl;
                        return EXIT_FAILURE;
                    }
                }
            }
        }
    }
    return EXIT_SUCCESS;
}
