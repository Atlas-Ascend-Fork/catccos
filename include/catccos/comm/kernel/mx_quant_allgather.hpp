
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_KERNEL_MX_QUANT_ALLGATHER_HPP
#define CATCCOS_COMM_KERNEL_MX_QUANT_ALLGATHER_HPP

#include "catccos/catccos.hpp"
#include "catccos/arch/cross_rank_sync.hpp"
#include "catccos/comm/block/comm_block_mx_quant.hpp"
#include "catccos/comm/block/comm_block_remote_copy.hpp"

#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif

namespace Catccos::Comm::Kernel {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

// ============================================================================
// MxQuantAllGather: fused dynamic MX quantization + AllGather
//
// Architecture (same dual-subcore split as QuantAllGather):
//   subcoreIdx==1 (quant core):
//     Load input → MX quant → Write [quantData | mxScale] to IPC buffer
//   subcoreIdx==0 (comm core):
//     Wait for quant → Copy quantData from remote IPC → output
//     Copy mxScale from remote IPC → mxScale output
//
// IPC buffer layout per block (shared buffer):
//   |<-- quantized data (actualM × N/PACK_RATIO bytes) -->|<-- mxScale (actualM × N/BS bytes) -->|
//
// Final output layout (independent tensors):
//   ptrOutput:  [rank0_data | rank1_data | ... | rankN_data]  (all gathered quantized data)
//   ptrMxScale: [rank0_scale | rank1_scale | ... | rankN_scale]  (all gathered scales)
// ============================================================================
template <
    class BlockMxQuant_, class BlockAllGather_, class BlockScaleGather_, class BlockMxQuantScheduler_,
    class BlockAllGatherScheduler_, uint32_t WORKSPACE_STAGES_>
class MxQuantAllGather {
public:
    using BlockMxQuant = BlockMxQuant_;
    using ArchTag = typename BlockMxQuant::ArchTag;
    using ElementInput = typename BlockMxQuant::ElementC;
    using LayoutInput = typename BlockMxQuant::LayoutC;
    using ElementOutput = typename BlockMxQuant::ElementD;
    using LayoutOutput = typename BlockMxQuant::LayoutD;
    using ElementScale = typename BlockMxQuant::ElementScale; // uint8_t E8M0

    using BlockAllGather = BlockAllGather_;
    using BlockAllGatherParams = typename BlockAllGather::Params;
    using BlockScaleGather = BlockScaleGather_;
    using BlockScaleGatherParams = typename BlockScaleGather::Params;

    using BlockMxQuantScheduler = BlockMxQuantScheduler_;
    using BlockAllGatherScheduler = BlockAllGatherScheduler_;
    using QuantCommParams = typename BlockMxQuantScheduler::Params;
    using AllGatherCommParams = typename BlockAllGatherScheduler::Params;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr uint32_t BLOCK_SIZE = BlockMxQuant::BLOCK_SIZE;
    static constexpr uint32_t PACK_RATIO = BlockMxQuant::PACK_RATIO;
    static constexpr uint32_t TILE_M = 128; // rows per comm block
    static constexpr size_t IPC_BUFF_MAX_SIZE = 512 * 1024 * 1024;

    struct Params {
        GemmCoord problemShape; // M × N (rows × columns)
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t commInterval;

        __gm__ ElementInput* ptrInput;
        LayoutInput layoutInput;

        __gm__ ElementOutput* ptrOutput; // quantized output (all ranks gathered)
        LayoutOutput layoutOutput;
        __gm__ ElementScale* ptrMxScale; // mxscale output (all ranks gathered)

        GM_ADDR ptrSymmetric;

        typename BlockAllGather::Params AllGatherParams;
        typename BlockScaleGather::Params ScaleGatherParams;
        QuantCommParams quantParams;
        AllGatherCommParams commParams;

        CATLASS_DEVICE Params() {}
    };

    CATLASS_DEVICE
    MxQuantAllGather()
    {
#ifdef ENABLE_TIMER
        __gm__ uint8_t* timer_buffer = GetTimerBuffer();
        if (timer_buffer != nullptr) {
            timer.Init(timer_buffer);
            timer.Tik();
        }
#endif
    }

    CATLASS_DEVICE
    ~MxQuantAllGather()
    {
#ifdef ENABLE_TIMER
        timer.Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
#endif
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params& params);

    // AIC: no-op
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params& params)
    {}

    // AIV: main execution
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params& params)
    {
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        uint32_t inputNum = params.problemShape.m();
        uint32_t N = params.problemShape.n();
        uint32_t commSizeM = TILE_M;
        uint32_t blockLoops = static_cast<uint32_t>(CeilDiv(static_cast<uint64_t>(inputNum), commSizeM));

        uint32_t quantBytesPerBlock =
            commSizeM * N * Catlass::SizeOfBits<ElementOutput>::value / Catlass::SizeOfBits<uint8_t>::value;
        uint32_t scaleBytesPerBlock = commSizeM * N / BLOCK_SIZE;
        uint32_t capacity = (quantBytesPerBlock + scaleBytesPerBlock) * aicoreNum;

        BlockMxQuant quant(resource, typename BlockMxQuant::Params{N});

        auto syncQuantFinish = reinterpret_cast<__gm__ int32_t*>(params.ptrSymmetric + IPC_BUFF_MAX_SIZE);
        auto syncCommFinish =
            reinterpret_cast<__gm__ int32_t*>(params.ptrSymmetric + IPC_BUFF_MAX_SIZE + aicoreNum * SYNC_UNIT_SIZE);

        if (subcoreIdx == 0 && aicoreIdx == params.rankIdx) {
            aclshmemx_signal_op(syncQuantFinish + aicoreIdx * SYNC_UNIT_SIZE, 0, ACLSHMEM_SIGNAL_SET, params.rankIdx);
            aclshmemx_signal_op(syncCommFinish + aicoreIdx * SYNC_UNIT_SIZE, 0, ACLSHMEM_SIGNAL_SET, params.rankIdx);
        }

        aclshmemx_barrier_all_vec();
#ifdef ENABLE_TIMER
        timer.Tik(AscendTimer::AIV);
#endif

        for (uint32_t blockIdx = aicoreIdx; blockIdx < blockLoops; blockIdx += aicoreNum) {
            auto actualCommSizeM = Min(commSizeM, inputNum - blockIdx * commSizeM);
            uint32_t actualQuantBytes =
                actualCommSizeM * N * Catlass::SizeOfBits<ElementOutput>::value / Catlass::SizeOfBits<uint8_t>::value;
            uint32_t actualScaleBytes = actualCommSizeM * N / BLOCK_SIZE;
            uint32_t step = blockIdx / aicoreNum;
            uint32_t stageId = step % WORKSPACE_STAGES;

            auto curSymmetric = params.ptrSymmetric + stageId * capacity;

            if (subcoreIdx == 1) {
                // Quant: input → IPC [quantData | mxScale]
                uint64_t offset = static_cast<uint64_t>(blockIdx) * commSizeM;

                if (step >= WORKSPACE_STAGES) {
                    for (uint32_t rankId = 0; rankId < params.rankSize; rankId++) {
                        auto remoteSyncCommFinish = static_cast<__gm__ int32_t*>(shmem_ptr(syncCommFinish, rankId));
                        aclshmem_signal_wait_until(
                            remoteSyncCommFinish + aicoreIdx * SYNC_UNIT_SIZE, ACLSHMEM_CMP_GE,
                            step + 1 - WORKSPACE_STAGES);
                    }
                }

                quant.InitBlockLoop();

                AscendC::GlobalTensor<ElementInput> gmBlockInput;
                gmBlockInput.SetGlobalBuffer(params.ptrInput + offset * N);
                LayoutInput layoutBlockInput{actualCommSizeM, N, N};

                AscendC::GlobalTensor<ElementOutput> gmIpcQuantOut;
                gmIpcQuantOut.SetGlobalBuffer(
                    reinterpret_cast<__gm__ ElementOutput*>(curSymmetric) +
                    aicoreIdx * commSizeM * N * Catlass::SizeOfBits<ElementOutput>::value /
                        Catlass::SizeOfBits<uint8_t>::value);
                uint32_t quantCols = N;
                LayoutOutput layoutIpcQuant{actualCommSizeM, quantCols, quantCols};

                AscendC::GlobalTensor<ElementScale> gmIpcScaleOut;
                gmIpcScaleOut.SetGlobalBuffer(
                    reinterpret_cast<__gm__ ElementScale*>(curSymmetric + quantBytesPerBlock * aicoreNum) +
                    aicoreIdx * commSizeM * (N / BLOCK_SIZE));
                uint32_t numScalesPerRow = N / BLOCK_SIZE;
                LayoutOutput layoutIpcScale{actualCommSizeM, numScalesPerRow, numScalesPerRow};

                MatrixCoord actualBlockShape{actualCommSizeM, N};
                quant(
                    gmBlockInput, layoutBlockInput, gmIpcQuantOut, layoutIpcQuant, gmIpcScaleOut, layoutIpcScale,
                    actualBlockShape);
                quant.FinalizeBlockLoop();
                AscendC::PipeBarrier<PIPE_ALL>();
                aclshmem_fence();
                aclshmemx_signal_op(
                    syncQuantFinish + aicoreIdx * SYNC_UNIT_SIZE, step + 1, ACLSHMEM_SIGNAL_SET, params.rankIdx);

            } else {
                // ====== COMM CORE (AllGather via CommBlock) ======
                uint64_t offset = static_cast<uint64_t>(blockIdx) * commSizeM;
                uint32_t numScalesPerRow = N / BLOCK_SIZE;

                BlockAllGather commBlock(resource, params.AllGatherParams);
                BlockScaleGather scaleBlock(resource, params.ScaleGatherParams);

                for (uint32_t rankId = 0; rankId < params.rankSize; rankId++) {
                    uint32_t remoteRankIdx = (aicoreIdx + rankId) % params.rankSize;
                    uint64_t commDstOffset = static_cast<uint64_t>(remoteRankIdx) * inputNum;

                    auto remoteSyncQuantFinish =
                        static_cast<__gm__ int32_t*>(shmem_ptr(syncQuantFinish, remoteRankIdx));
                    aclshmem_signal_wait_until(
                        remoteSyncQuantFinish + aicoreIdx * SYNC_UNIT_SIZE, ACLSHMEM_CMP_GE, step + 1);

                    // --- Copy quantized data via CommBlock ---
                    {
                        AscendC::GlobalTensor<ElementOutput> gmSymQuant;
                        gmSymQuant.SetGlobalBuffer(reinterpret_cast<__gm__ ElementOutput*>(
                            curSymmetric + aicoreIdx * commSizeM * N * Catlass::SizeOfBits<ElementOutput>::value /
                                               Catlass::SizeOfBits<uint8_t>::value));

                        AscendC::GlobalTensor<ElementOutput> gmOutput;
                        gmOutput.SetGlobalBuffer(params.ptrOutput);
                        uint64_t dstQuantOffset = (commDstOffset + offset) * N;

                        uint32_t quantCols = N;
                        LayoutOutput layoutSrc{actualCommSizeM, quantCols, quantCols};
                        LayoutOutput layoutDst{actualCommSizeM, quantCols, quantCols};
                        MatrixCoord actualShape{actualCommSizeM, quantCols};

                        commBlock.InitBlockLoop();
                        commBlock(
                            gmSymQuant, layoutSrc, gmOutput[dstQuantOffset], layoutDst, actualShape, remoteRankIdx);
                        commBlock.FinalizeBlockLoop();
                    }

                    // --- Copy mxscale data via CommBlock ---
                    {
                        AscendC::GlobalTensor<ElementScale> gmSymScale;
                        gmSymScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale*>(
                            curSymmetric + quantBytesPerBlock * aicoreNum +
                            aicoreIdx * commSizeM * numScalesPerRow * sizeof(ElementScale)));

                        AscendC::GlobalTensor<ElementScale> gmMxScaleOut;
                        gmMxScaleOut.SetGlobalBuffer(params.ptrMxScale);
                        uint64_t dstScaleOffset = (commDstOffset + offset) * numScalesPerRow;

                        LayoutOutput layoutSrc{actualCommSizeM, numScalesPerRow, numScalesPerRow};
                        LayoutOutput layoutDst{actualCommSizeM, numScalesPerRow, numScalesPerRow};
                        MatrixCoord actualShape{actualCommSizeM, numScalesPerRow};

                        scaleBlock.InitBlockLoop();
                        scaleBlock(
                            gmSymScale, layoutSrc, gmMxScaleOut[dstScaleOffset], layoutDst, actualShape, remoteRankIdx);
                        scaleBlock.FinalizeBlockLoop();
                    }
                }

                aclshmem_fence();
                aclshmemx_signal_op(
                    syncCommFinish + aicoreIdx * SYNC_UNIT_SIZE, step + 1, ACLSHMEM_SIGNAL_SET, params.rankIdx);
            }
        }
#ifdef ENABLE_TIMER
        timer.Tok<Overwrite>(AscendTimer::AIV);
#endif
    }

private:
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catccos::Comm::Kernel

#endif // CATCCOS_COMM_KERNEL_MX_QUANT_ALLGATHER_HPP
