
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_KERNEL_QUANT_ALLGATHER_A5_HPP
#define CATCCOS_COMM_KERNEL_QUANT_ALLGATHER_A5_HPP

#include "catccos/catccos.hpp"
#include "catccos/arch/cross_rank_sync.hpp"
#include "catccos/epilogue/block/block_epilogue_per_tensor_quant.hpp"
#include "catccos/comm/block/comm_block_remote_copy.hpp"

#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catccos::Comm::Kernel {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

/**
 * @brief <h1>quantAllgather算子简介</h1>
 *        下发的AICore数目是为rankSize。<br>
 *        <h3>分核策略</h3>
 *        每个AICore中的第一个aiv负责allgather通信搬运，每个AICore中的第二个aiv负责本地的量化操作。
 *        对于负责量化的核而言，按照输入划分成rankSize份，每个核负责其中一份数据的量化以及搬运到共享内存。
 *
 对于负责通信的核而言，类似于量化核，每个通信核负责1/rankSize大小的数据的通信行为，需要遍历所有其他卡，将数据跨卡搬运到本地。<br>
 *        <h3>共享内存上的划分</h3>
 *        共享内存上，首先会划分成两大块，分别表示ping和pong的行为。ping buffer/pong
 buffer又会继续划分IPC_DATA_OFFSET用于标志位区域，
 *        其他用于数据区域。对于标志位区域，按照核数进行划分，每个核分配SYNC_UNIT_SIZE空间记录标志位。
 *        前面的部分用于每个量化核记录自己负责的数据完成；后面的部分用于通信核记录自己负责的数据完成。<br>
 *        <h3>算法步骤</h3>
 *
 量化核和通信核同时并发工作，量化核做完自己负责的那片数据，会将对应的本卡标志位的值置为当前分片的编号；通信核在操作前，首先会查看需要的数据是否已经
          准备完成，即查看相应的标志位，做完自己负责的那片数据，等待所有源卡的数据通信完成后，会将对应的本卡标志位置为当前分片编号。值得注意的是，由于每个
          通信核会轮询所有源卡，会将每个核的链路错开，保证同一时刻不同核访问的是不同链路。<br>
 */
template <
    class BlockQuant_, class BlockAllGather_, class BlockQuantScheduler_, class BlockAllGatherScheduler_,
    uint32_t WORKSPACE_STAGES_>
class QuantAllGather {
public:
    using BlockQuant = BlockQuant_;
    using ArchTag = typename BlockQuant::ArchTag;
    using ElementInput = typename BlockQuant::ElementC;
    using LayoutInput = typename BlockQuant::LayoutC;
    using ElementOutput = typename BlockQuant::ElementD;
    using LayoutOutput = typename BlockQuant::LayoutD;
    using ElementScale = typename BlockQuant::ElementScale;

    using BlockAllGather = BlockAllGather_;
    using BlockAllGatherParams = typename BlockAllGather::Params;

    using BlockQuantScheduler = BlockQuantScheduler_;
    using BlockAllGatherScheduler = BlockAllGatherScheduler_;
    using QuantCommParams = typename BlockQuantScheduler::Params;
    using AllGatherCommParams = typename BlockAllGatherScheduler::Params;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr uint32_t TILE_M = 1024 * 1024; // 单次处理1MB
    static constexpr uint32_t TILE_N = 256;
    static constexpr uint32_t TILE_K = 256;
    static constexpr uint32_t IPC_BUFF_HALF_SIZE = IPC_BUFF_MAX_SIZE / PING_PONG_SIZE;
    static constexpr uint32_t MAX_PEER_RANKS = 128;

    struct Params {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t magic;
        uint32_t commInterval;

        __gm__ ElementInput* ptrInput;
        LayoutInput layoutInput;
        __gm__ ElementScale* ptrScales;
        __gm__ ElementOutput* ptrOutput;
        LayoutOutput layoutOutput;

        GM_ADDR ptrSymmetric;
        __gm__ ElementOutput* buff[MAX_PEER_RANKS];
        __gm__ ElementOutput* anotherBuff[MAX_PEER_RANKS];

        typename BlockAllGather::Params AllGatherParams;
        QuantCommParams quantParams;
        AllGatherCommParams commParams;

        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(
            GemmCoord const& problemShape_, uint32_t rankIdx_, uint32_t rankSize_, uint32_t magic_,
            uint32_t commInterval_, GM_ADDR ptrInput_, LayoutInput const& layoutInput_, GM_ADDR ptrScales_,
            GM_ADDR ptrOutput_, LayoutOutput const& layoutOutput_, GM_ADDR ptrSymmetric_, __gm__ ElementOutput** buff_,
            __gm__ ElementOutput** anotherBuff_, typename BlockAllGather::Params const& AllGatherParams_,
            QuantCommParams const& quantParams_, AllGatherCommParams const& commParams_)
            : problemShape(problemShape_),
              rankIdx(rankIdx_),
              rankSize(rankSize_),
              magic(magic_),
              commInterval(commInterval_),
              ptrInput(reinterpret_cast<__gm__ ElementInput*>(ptrInput_)),
              layoutInput(layoutInput_),
              ptrScales(reinterpret_cast<__gm__ ElementScale*>(ptrScales_)),
              ptrOutput(reinterpret_cast<__gm__ ElementOutput*>(ptrOutput_)),
              layoutOutput(layoutOutput_),
              ptrSymmetric(ptrSymmetric_),
              AllGatherParams(AllGatherParams_),
              quantParams(quantParams_),
              commParams(commParams_)
        {
            for (uint32_t i = 0; i < MAX_PEER_RANKS; ++i) {
                buff[i] = nullptr;
                anotherBuff[i] = nullptr;
            }
            for (uint32_t i = 0; i < rankSize_; ++i) {
                buff[i] = buff_[i];
                anotherBuff[i] = anotherBuff_[i];
            }
        }
    };

    CATLASS_DEVICE
    QuantAllGather() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params& params)
    {}

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params& params)
    {
        constexpr uint32_t FLAG_MAGIC_STRIDE = 1U << 20;
        uint32_t flagBase = params.magic * FLAG_MAGIC_STRIDE;
        AscendC::GlobalTensor<ElementInput> gmInput;
        gmInput.SetGlobalBuffer(params.ptrInput);
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
        // params.problemShape.m() 和 params.problemShape.n()分别表示输入的行数和列数
        uint64_t inputNum =
            static_cast<uint64_t>(params.problemShape.m()) * params.problemShape.n() * params.problemShape.k();
        constexpr uint32_t MIN_COMM_BYTES_PER_CORE = 32;
        constexpr uint32_t MIN_ELMENTS_PER_CORE =
            (MIN_COMM_BYTES_PER_CORE + sizeof(ElementOutput) - 1) / sizeof(ElementOutput);
        uint32_t coreNumPerStep =
            static_cast<uint32_t>(Min(CeilDiv(inputNum, MIN_ELMENTS_PER_CORE), static_cast<uint64_t>(params.rankSize)));
        coreNumPerStep = coreNumPerStep == 0 ? 1 : coreNumPerStep;
        if (aicoreIdx >= coreNumPerStep) {
            return;
        }
        uint64_t elementNumPerCore = CeilDiv(inputNum, coreNumPerStep); // 每个核处理的数据量
        uint64_t processNum = elementNumPerCore;
        if (aicoreIdx == (coreNumPerStep - 1)) { // 如果是最后一个核 需要处理尾块
            processNum = inputNum - aicoreIdx * elementNumPerCore;
        }
        uint32_t commSizeM = TILE_M;
        // 这里是每个核处理的总行数除以每次处理的行数
        uint32_t blockLoops = static_cast<uint32_t>(CeilDiv(processNum, commSizeM)); // Host checks flag range

        BlockQuant quant(resource);
        BlockAllGather allGather(resource, params.AllGatherParams);

        AscendC::GlobalTensor<ElementScale> gmScales;
        gmScales.SetGlobalBuffer(params.ptrScales);
        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM, params.problemShape.n(),
            RoundUp<int64_t>(params.problemShape.n(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementOutput)));
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementOutput*>(params.ptrSymmetric);
        // Each block iterates over one communication slice. Subcore 1 produces quantized tiles into the
        // local symmetric workspace, and subcore 0 waits for the corresponding remote-ready flag before
        // pulling that slice from every rank into the final output buffer.
        for (uint32_t blockIdx = 0; blockIdx < blockLoops; ++blockIdx) {
            uint32_t stageId = blockIdx % WORKSPACE_STAGES;
            uint64_t quantSrcOffset = static_cast<uint64_t>(blockIdx) * commSizeM;
            uint32_t actualCommSizeM =
                static_cast<uint32_t>(Min(static_cast<uint64_t>(commSizeM), processNum - quantSrcOffset));
            uint32_t actualBlockSize = actualCommSizeM;
            uint32_t curFlagValue = flagBase + blockIdx + 1;
            if (subcoreIdx == 1) {                                      // quant
                uint64_t curCoreOffset = aicoreIdx * elementNumPerCore; // 每个核负责1/coreNum的数据块
                // Reuse a workspace stage only after every rank has observed the previous allgather finish.
                if (blockIdx >= WORKSPACE_STAGES) {
                    for (int rankId = 0; rankId < params.rankSize; rankId++) {
                        GM_ADDR quantWaitFlagAddr =
                            (GM_ADDR)params.buff[rankId] + IPC_BUFF_HALF_SIZE +
                            (Catccos::Arch::FLAG_ONE_IDX * coreNumPerStep + aicoreIdx) * SYNC_UNIT_SIZE;
                        aclshmem_signal_wait_until(
                            reinterpret_cast<__gm__ int32_t*>(quantWaitFlagAddr), ACLSHMEM_CMP_GE,
                            flagBase + blockIdx + 1 - WORKSPACE_STAGES);
                    }
                }
                uint32_t blockOffset = (stageId * coreNumPerStep + aicoreIdx) * commSizeM;
                auto srcOffset = curCoreOffset + quantSrcOffset;
                typename BlockQuant::Params quantParamsForBlock{
                    params.ptrInput, params.layoutInput,
                    reinterpret_cast<__gm__ ElementOutput*>(params.buff[params.rankIdx]) + blockOffset,
                    layoutSymmetric};
                quant.UpdateParams(quantParamsForBlock);
                auto gmBlockInput = gmInput[srcOffset];
                ElementScale scaleVal = gmScales.GetValue(0);
                quant(actualBlockSize, gmBlockInput, scaleVal);
                AscendC::PipeBarrier<PIPE_ALL>();
                aclshmem_fence();
                // Publish that this rank's quantized slice for the current stage is ready in symmetric memory.
                auto quantFlagAddr = reinterpret_cast<__gm__ int32_t*>(
                    params.ptrSymmetric + IPC_BUFF_HALF_SIZE +
                    (Catccos::Arch::FLAG_ZERO_IDX * coreNumPerStep + aicoreIdx) * SYNC_UNIT_SIZE);
                aclshmemx_signal_op(quantFlagAddr, curFlagValue, ACLSHMEM_SIGNAL_SET, params.rankIdx);
            } else { // AllGather
                for (int rankId = 0; rankId < params.rankSize; rankId++) {
                    uint32_t remoteRankIdx = (aicoreIdx + rankId) % params.rankSize;
                    GM_ADDR commWaitFlagAddr =
                        (GM_ADDR)params.buff[remoteRankIdx] + IPC_BUFF_HALF_SIZE +
                        (Catccos::Arch::FLAG_ZERO_IDX * coreNumPerStep + aicoreIdx) * SYNC_UNIT_SIZE;
                    // Wait until the remote rank has produced the quantized slice for this stage.
                    aclshmem_signal_wait_until(
                        reinterpret_cast<__gm__ int32_t*>(commWaitFlagAddr), ACLSHMEM_CMP_GE, curFlagValue);
                    allGather.InitBlockLoop();
                    uint32_t commSrcOffset = (stageId * coreNumPerStep + aicoreIdx) * commSizeM;
                    uint32_t blockSrcOffsetInRank = 0;
                    uint64_t blockDstOffsetInRank = aicoreIdx * elementNumPerCore + quantSrcOffset;
                    uint64_t commDstOffset = remoteRankIdx * inputNum;
                    AscendC::GlobalTensor<ElementOutput> gmSymmetric;
                    gmSymmetric.SetGlobalBuffer(ptrSymmetric);

                    uint32_t offsetSrc = commSrcOffset + blockSrcOffsetInRank;
                    uint64_t offsetDst = commDstOffset + blockDstOffsetInRank;

                    auto gmBlockSrc = gmSymmetric[offsetSrc];
                    AscendC::GlobalTensor<ElementOutput> gmOutput;
                    gmOutput.SetGlobalBuffer(params.ptrOutput);
                    auto gmBlockDst = gmOutput[offsetDst];
                    allGather(gmBlockSrc, gmBlockDst, actualBlockSize, remoteRankIdx);
                    allGather.FinalizeBlockLoop();
                }
                aclshmem_fence();
                // Publish that this stage has been consumed so the producer can recycle the workspace slot.
                auto commFlagAddr = reinterpret_cast<__gm__ int32_t*>(
                    params.ptrSymmetric + IPC_BUFF_HALF_SIZE +
                    (Catccos::Arch::FLAG_ONE_IDX * coreNumPerStep + aicoreIdx) * SYNC_UNIT_SIZE);
                aclshmemx_signal_op(commFlagAddr, curFlagValue, ACLSHMEM_SIGNAL_SET, params.rankIdx);
            }
        }
        if (subcoreIdx == 0 && aicoreIdx == params.rankIdx) {
            for (uint32_t flag_idx = 0; flag_idx < (2 * coreNumPerStep); flag_idx++) {
                GM_ADDR syncAddr =
                    (GM_ADDR)params.anotherBuff[params.rankIdx] + IPC_BUFF_HALF_SIZE + flag_idx * SYNC_UNIT_SIZE;
                aclshmemx_signal_op(
                    reinterpret_cast<__gm__ int32_t*>(syncAddr), 0, ACLSHMEM_SIGNAL_SET, params.rankIdx);
            }
        }
        aclshmemx_barrier_all_vec();
    }

private:
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catccos::Comm::Kernel

#endif // CATCCOS_COMM_KERNEL_QUANT_ALLGATHER_A5_HPP
