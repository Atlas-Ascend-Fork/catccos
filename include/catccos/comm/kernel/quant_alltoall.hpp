
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_KERNEL_QUANT_ALLTOALL_A5_HPP
#define CATCCOS_COMM_KERNEL_QUANT_ALLTOALL_A5_HPP

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
 * @brief <h1>quantAlltoall算子简介</h1>
 *        下发的AICore数目是为rankSize。<br>
 *        <h3>分核策略</h3>
 *        每个AICore中的第一个aiv负责alltoall通信搬运，每个AICore中的第二个aiv负责本地的量化操作。
 *        对于负责量化的核而言，按照输入划分成rankSize份，每个核负责其中一份数据的量化以及搬运到共享内存。
 *
 对于负责通信的核而言，类似于量化核，每个通信核负责输入的1/rankSize大小的数据的通信行为，将负责的源卡的数据跨卡搬运到本地。<br>
 *        <h3>共享内存上的划分</h3>
 *        共享内存上，首先会划分成两大块，分别表示ping和pong的行为。ping buffer/pong
 buffer又会继续划分IPC_DATA_OFFSET用于标志位区域，
 *        其他用于数据区域。对于标志位区域，按照核数进行划分，每个核分配SYNC_UNIT_SIZE空间记录标志位。
 *        前面的部分用于每个量化核记录自己负责的数据完成；后面的部分用于通信核记录自己负责的数据完成。<br>
 *        <h3>算法步骤</h3>
 *
 量化核和通信核同时并发工作，量化核做完自己负责的那片数据，会将对应的本卡标志位的值置为当前分片的编号；通信核在操作前，首先会查看需要的数据是否已经
          准备完成，即查看相应的标志位，做完自己负责的那片数据，会将对应的本卡标志位置为当前分片编号。<br>
 */
template <
    class BlockQuant_, class BlockAllToAll_, class BlockQuantScheduler_, class BlockAllToAllScheduler_,
    uint32_t WORKSPACE_STAGES_>
class QuantAllToAll {
public:
    using BlockQuant = BlockQuant_;
    using ArchTag = typename BlockQuant::ArchTag;
    using ElementInput = typename BlockQuant::ElementC;
    using LayoutInput = typename BlockQuant::LayoutC;
    using ElementOutput = typename BlockQuant::ElementD;
    using LayoutOutput = typename BlockQuant::LayoutD;
    using ElementScale = typename BlockQuant::ElementScale;

    using BlockAllToAll = BlockAllToAll_;
    using BlockAllToAllParams = typename BlockAllToAll::Params;

    using BlockQuantScheduler = BlockQuantScheduler_;
    using BlockAllToAllScheduler = BlockAllToAllScheduler_;
    using QuantCommParams = typename BlockQuantScheduler::Params;
    using AllToAllCommParams = typename BlockAllToAllScheduler::Params;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr uint32_t TILE_M = 1024 * 1024; // 单次处理1MB
    static constexpr uint32_t TILE_N = 256;
    static constexpr uint32_t TILE_K = 256;
    static constexpr uint32_t IPC_BUFF_HALF_SIZE = IPC_BUFF_MAX_SIZE / PING_PONG_SIZE;

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
        __gm__ ElementOutput** buff;
        __gm__ ElementOutput** anotherBuff;

        typename BlockAllToAll::Params allToAllParams;
        QuantCommParams quantParams;
        AllToAllCommParams commParams;

        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(
            GemmCoord const& problemShape_, uint32_t rankIdx_, uint32_t rankSize_, uint32_t magic_,
            uint32_t commInterval_, GM_ADDR ptrInput_, LayoutInput const& layoutInput_, GM_ADDR ptrScales_,
            GM_ADDR ptrOutput_, LayoutOutput const& layoutOutput_, GM_ADDR ptrSymmetric_, __gm__ ElementOutput** buff_,
            __gm__ ElementOutput** anotherBuff_, typename BlockAllToAll::Params const& allToAllParams_,
            QuantCommParams const& quantParams_, AllToAllCommParams const& commParams_)
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
              buff(buff_),
              anotherBuff(anotherBuff_),
              allToAllParams(allToAllParams_),
              quantParams(quantParams_),
              commParams(commParams_)
        {}
    };

    CATLASS_DEVICE
    QuantAllToAll() {}

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
        uint64_t inputNum =
            static_cast<uint64_t>(params.problemShape.m()) * params.problemShape.n() * params.problemShape.k();
        uint64_t elementNumPerCore = inputNum / params.rankSize;
        uint32_t commSizeM = TILE_M;
        // 这里是每个核处理的总行数除以每次处理的行数
        uint32_t blockLoops = static_cast<uint32_t>(CeilDiv(elementNumPerCore, commSizeM)); // Host checks flag range
        BlockQuant quant(resource);
        BlockAllToAll allToAll(resource, params.allToAllParams);

        AscendC::GlobalTensor<ElementScale> gmScales;
        gmScales.SetGlobalBuffer(params.ptrScales);
        auto layoutSymmetric = Catlass::layout::RowMajor(
            WORKSPACE_STAGES * params.rankSize * commSizeM, params.problemShape.n(),
            RoundUp<int64_t>(params.problemShape.n(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementOutput)));
        auto ptrSymmetric = reinterpret_cast<__gm__ ElementOutput*>(params.ptrSymmetric);
        // Subcore 1 quantizes one peer slice into the local symmetric workspace, and subcore 0 waits for the
        // producer flag before pulling that same slice from the current rank's workspace into the destination
        // region assigned to the target peer.
        for (uint32_t blockIdx = 0; blockIdx < blockLoops; ++blockIdx) {
            uint32_t stageId = blockIdx % WORKSPACE_STAGES;
            uint64_t quantSrcOffset = static_cast<uint64_t>(blockIdx) * commSizeM;
            uint32_t actualCommSizeM =
                static_cast<uint32_t>(Min(static_cast<uint64_t>(commSizeM), elementNumPerCore - quantSrcOffset));
            uint32_t actualBlockSize = actualCommSizeM;
            uint32_t curFlagValue = flagBase + blockIdx + 1;
            if (subcoreIdx == 1) { // quant
                uint32_t processRankIdx = aicoreIdx % params.rankSize;
                // Reuse a workspace stage only after the corresponding alltoall consumer has finished with it.
                if (blockIdx >= WORKSPACE_STAGES) {
                    GM_ADDR quantWaitFlagAddr =
                        (GM_ADDR)params.buff[params.rankIdx] + IPC_BUFF_HALF_SIZE +
                        (Catccos::Arch::FLAG_ONE_IDX * params.rankSize + processRankIdx) * SYNC_UNIT_SIZE;
                    aclshmem_signal_wait_until(
                        reinterpret_cast<__gm__ int32_t*>(quantWaitFlagAddr), ACLSHMEM_CMP_GE,
                        flagBase + blockIdx + 1 - WORKSPACE_STAGES);
                }
                uint32_t blockOffset = (stageId * params.rankSize + processRankIdx) * commSizeM;
                uint64_t curRankOffset = processRankIdx * elementNumPerCore;
                auto srcOffset = curRankOffset + quantSrcOffset;
                typename BlockQuant::Params quantParamsForBlock{
                    params.ptrInput, params.layoutInput,
                    reinterpret_cast<__gm__ ElementOutput*>(params.buff[params.rankIdx]) + blockOffset,
                    layoutSymmetric};
                quant.UpdateParams(quantParamsForBlock);
                auto gmBlockInput = gmInput[srcOffset];
                ElementScale scaleVal = gmScales.GetValue(processRankIdx);
                quant(actualBlockSize, gmBlockInput, scaleVal);
                AscendC::PipeBarrier<PIPE_ALL>();
                aclshmem_fence();
                // Publish that the local workspace slice for this destination peer is ready.
                auto quantFlagAddr = reinterpret_cast<__gm__ int32_t*>(
                    params.ptrSymmetric + IPC_BUFF_HALF_SIZE +
                    (Catccos::Arch::FLAG_ZERO_IDX * params.rankSize + params.rankIdx) * SYNC_UNIT_SIZE);
                aclshmemx_signal_op(quantFlagAddr, curFlagValue, ACLSHMEM_SIGNAL_SET, processRankIdx);
            } else { // alltoall
                uint32_t remoteRankIdx = aicoreIdx % params.rankSize;
                GM_ADDR commWaitFlagAddr =
                    (GM_ADDR)params.buff[params.rankIdx] + IPC_BUFF_HALF_SIZE +
                    (Catccos::Arch::FLAG_ZERO_IDX * params.rankSize + remoteRankIdx) * SYNC_UNIT_SIZE;
                // Wait until the producer rank has materialized this peer-specific slice in symmetric memory.
                aclshmem_signal_wait_until(
                    reinterpret_cast<__gm__ int32_t*>(commWaitFlagAddr), ACLSHMEM_CMP_GE, curFlagValue);
                allToAll.InitBlockLoop();
                uint32_t commSrcOffset = (stageId * params.rankSize + params.rankIdx) * commSizeM;
                uint32_t blockSrcOffsetInRank = 0;
                uint64_t blockDstOffsetInRank = quantSrcOffset;
                uint64_t commDstOffset = remoteRankIdx * elementNumPerCore;
                AscendC::GlobalTensor<ElementOutput> gmSymmetric;
                gmSymmetric.SetGlobalBuffer(ptrSymmetric);

                uint32_t offsetSrc = commSrcOffset + blockSrcOffsetInRank;
                uint64_t offsetDst = commDstOffset + blockDstOffsetInRank;

                auto gmBlockSrc = gmSymmetric[offsetSrc];
                AscendC::GlobalTensor<ElementOutput> gmOutput;
                gmOutput.SetGlobalBuffer(params.ptrOutput);
                auto gmBlockDst = gmOutput[offsetDst];
                allToAll(gmBlockSrc, gmBlockDst, actualBlockSize, remoteRankIdx);
                allToAll.FinalizeBlockLoop();
                aclshmem_fence();
                // Publish that the current stage has been drained so the producer can recycle it later.
                auto commFlagAddr = reinterpret_cast<__gm__ int32_t*>(
                    params.ptrSymmetric + IPC_BUFF_HALF_SIZE +
                    (Catccos::Arch::FLAG_ONE_IDX * params.rankSize + params.rankIdx) * SYNC_UNIT_SIZE);
                aclshmemx_signal_op(commFlagAddr, curFlagValue, ACLSHMEM_SIGNAL_SET, remoteRankIdx);
            }
        }
        if (subcoreIdx == 0 && aicoreIdx == params.rankIdx) {
            for (uint32_t flag_idx = 0; flag_idx < (2 * params.rankSize); flag_idx++) {
                GM_ADDR syncAddr =
                    (GM_ADDR)params.anotherBuff[params.rankIdx] + IPC_BUFF_HALF_SIZE + flag_idx * SYNC_UNIT_SIZE;
                aclshmemx_signal_op(
                    reinterpret_cast<__gm__ int32_t*>(syncAddr), 0, ACLSHMEM_SIGNAL_SET, params.rankIdx);
            }
        }
        aclshmemx_barrier_all_vec();
    }

private:
    Catlass::Arch::CrossCoreFlag flagQuantFinish[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAllToAllFinish[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAllDone;
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catccos::Comm::Kernel

#endif // CATCCOS_COMM_KERNEL_QUANT_ALLTOALL_A5_HPP
