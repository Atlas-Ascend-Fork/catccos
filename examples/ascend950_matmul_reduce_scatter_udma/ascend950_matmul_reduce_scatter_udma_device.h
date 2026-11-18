

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_MATMUL_REDUCE_SCATTER_UDMA_KERNEL_H
#define ASCEND950_MATMUL_REDUCE_SCATTER_UDMA_KERNEL_H

#include "info.h"

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_scheduler_reduce_scatter.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/kernel/matmul_reduce_scatter_udma_tla.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"

using namespace AscendC;
using namespace Catccos;

template <
    class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementD, class LayoutD, uint32_t M0_,
    uint32_t N0_, uint32_t K0_>
struct Ascend950MatmulReduceScatterUdmaConfig {
    using ArchTag = Catlass::Arch::Ascend950;

    static constexpr bool enableUnitFlag = true;
    static constexpr bool useHF32 = false;
    static constexpr bool enableL1Resident = false;
    static constexpr uint32_t l0CStages = 1;
    static constexpr uint32_t l1AStages = 2;
    static constexpr uint32_t l1BStages = 2;
    static constexpr uint32_t l0AStages = 2;
    static constexpr uint32_t l0BStages = 2;
    using MmadDispatchPolicy = Catlass::Gemm::MmadPingpong<
        ArchTag, enableUnitFlag, useHF32, l0CStages, enableL1Resident, l1AStages, l1BStages, l0AStages, l0BStages>;

    using L1TileShape = tla::Shape<tla::Int<M0_>, tla::Int<N0_>, tla::Int<K0_>>;
    using L0TileShape = tla::Shape<tla::Int<M0_>, tla::Int<N0_>, tla::Int<64>>;

    using TileCopy =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementD, void, TileCopy>;

    static constexpr bool IS_DYNAMIC = true;

    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
    using BlockCommScheduler = Catccos::Comm::Block::BlockCommSchedulerReduceScatter<IS_DYNAMIC, void, 0, true, 7, 1>;

    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;
    using RemoteSrcType = DType;
    using RemoteDstType = DType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileLocalCopy = Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Get, CopyTransport::Mte>;
    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Get, CopyTransport::Udma>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using LocalcopyDispatch = Comm::AtlasCommLocalCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using BlockLocalCopy =
        Comm::Block::CommBlock<LocalcopyDispatch, RemoteSrcType, RemoteDstType, void, TileLocalCopy, TileScheduler>;

    using UdmaReduceScatterDispatch = Comm::AtlasCommUdmaRemoteCopy<ArchTag, UB_STAGES>;
    using BlockRemoteComm =
        Comm::Block::CommBlock<UdmaReduceScatterDispatch, RemoteSrcType, RemoteDstType, TileRemoteCopy>;

    using Kernel = DGemm::Kernel::MatmulReduceScatterUdmaTla<
        BlockMmad, BlockLocalCopy, BlockRemoteComm, BlockMmadScheduler, BlockCommScheduler, WORKSPACE_STAGES>;

    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

// Pre-defined tiling configurations
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementD, class LayoutD>
using Ascend950MatmulReduceScatterUdmaConfig_M0_128 =
    Ascend950MatmulReduceScatterUdmaConfig<ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, 128, 256, 256>;

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementD, class LayoutD>
using Ascend950MatmulReduceScatterUdmaConfig_M0_256 =
    Ascend950MatmulReduceScatterUdmaConfig<ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, 256, 128, 256>;

#endif // ASCEND950_MATMUL_REDUCE_SCATTER_UDMA_KERNEL_H
