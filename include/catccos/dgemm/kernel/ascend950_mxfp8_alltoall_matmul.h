/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_H
#define CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_H

#include <cstddef>
#include <cstdint>

#include "catccos/dgemm/kernel/ascend950_mxfp8_alltoall_matmul_compute.h"
#include "catccos/dgemm/kernel/ascend950_mxfp8_alltoall_matmul_transport.h"

namespace Catccos::DGemm::Kernel {

// AICs own disjoint C tiles; AIVs send each destination's windows and one
// coordinator advances the local receive frontier. No global window barrier.
template <class BlockMmad_, class BlockComm_, class BlockCommScheduler_>
class Ascend950MxFp8AllToAllMatmul {
public:
    using Compute = Ascend950MxFp8AllToAllMatmulCompute<BlockMmad_>;
    using Transport = Ascend950MxFp8AllToAllMatmulTransport<BlockComm_, BlockCommScheduler_>;

    struct Arguments {
        Catlass::GemmCoord problemShape; // M, N, localK; M = rankSize * chunkM
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t groupSize;
        uint32_t windowRows;
        GM_ADDR ptrA; // read-only SHMEM source
        GM_ADDR ptrB;
        GM_ADDR ptrScaleA; // read-only SHMEM source scales
        GM_ADDR ptrScaleB;
        GM_ADDR ptrC;
        GM_ADDR ptrSymmetric;
    };

    struct Params {
        typename Compute::Params compute;
        typename Transport::Params transport;

        CATLASS_HOST_DEVICE Params() = default;
        CATLASS_HOST_DEVICE Params(const Arguments& args, const MxFp8AllToAllMatmul::Layout& layout)
            : compute(typename Compute::Arguments{
                  layout.chunkM, layout.recvSlotRows, args.rankSize, layout.localK, layout.n,
                  args.ptrSymmetric + layout.recvA, args.ptrSymmetric + layout.recvScaleA, args.ptrB, args.ptrScaleB,
                  args.ptrC, args.ptrSymmetric + layout.readyFrontier, layout.windowRows}),
              transport(typename Transport::Arguments{
                  layout.chunkM, layout.recvSlotRows, layout.windowRows, layout.localK, args.rankIdx, args.rankSize,
                  args.groupSize, args.ptrA, args.ptrScaleA, args.ptrSymmetric + layout.recvA,
                  args.ptrSymmetric + layout.recvScaleA, args.ptrSymmetric + layout.ready,
                  args.ptrSymmetric + layout.readyFrontier, args.ptrSymmetric + layout.readyValues})
        {}
    };

    static size_t GetWorkspaceSize(const Arguments&) { return 0; }
    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* = nullptr)
    {
        MxFp8AllToAllMatmul::Layout layout;
        MxFp8AllToAllMatmul::MakeLayout(
            args.problemShape.m(), args.problemShape.n(), args.problemShape.k(), args.rankSize, args.windowRows,
            layout);
        return Params(args, layout);
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params& params)
    {
        Compute compute;
        compute.template operator()<AscendC::AIC>(params.compute);
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params& params)
    {
        Transport transport;
        transport.template operator()<AscendC::AIV>(params.transport);
    }
};

} // namespace Catccos::DGemm::Kernel
#endif // CATCCOS_DGEMM_KERNEL_ASCEND950_MXFP8_ALLTOALL_MATMUL_H
