/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_HIERARCHICAL_ALLTOALL_DEVICE_H
#define ASCEND950_HIERARCHICAL_ALLTOALL_DEVICE_H

// from catccos
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_hierarchical_swizzle.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/kernel/ascend950_hierarchical_alltoall.hpp"
#include "catccos/comm/kernel/ascend950_hierarchical_barrier.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"

// from catlass
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/layout/layout.hpp"

// Uniform AllToAll: input[peer] -> remote output[rank], one AIV per peer.
template <class Element, class Layout, class ElementScale, class LayoutScale>
struct Ascend950HierarchicalAllToAllConfig {
    using ArchTag = Catlass::Arch::Ascend950;

    static constexpr uint32_t CORE_COUNT = 32;
    static constexpr uint32_t COMM_BLOCK_M = 64;
    static constexpr uint32_t GROUP_SIZE = CATCCOS_HIERARCHICAL_GROUP_SIZE;
    static constexpr uint32_t UB_STAGES = 1;

    static constexpr bool IS_DYNAMIC = true;

    using RemoteSrcType = Catlass::Gemm::GemmType<Element, Layout>;
    using RemoteDstType = RemoteSrcType;
    using RemoteScaleType = Catlass::Gemm::GemmType<ElementScale, LayoutScale>;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;

    using BlockCommScheduler = Catccos::Comm::Block::HierarchicalCommSwizzle<CORE_COUNT, GROUP_SIZE, COMM_BLOCK_M>;
    using TileRemoteCopy = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Put, CopyTransport::Mte>;
    using TileSwizzle = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using CommDispatchPolicy = Catccos::Comm::AtlasCommHierarchicalCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using BlockComm = Catccos::Comm::Block::CommBlock<
        CommDispatchPolicy, RemoteSrcType, RemoteDstType, void, TileRemoteCopy, TileSwizzle>;

    using TileRemoteCopyScale = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, RemoteScaleType, RemoteScaleType, void, CopyDirect::Put, CopyTransport::Mte>;
    using BlockCommScale = Catccos::Comm::Block::CommBlock<
        CommDispatchPolicy, RemoteScaleType, RemoteScaleType, void, TileRemoteCopyScale, TileSwizzle>;

    using Kernel = Catccos::Comm::Kernel::HierarchicalAllToAll<BlockComm, BlockCommScale, BlockCommScheduler>;
    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
    using BarrierDevice = Catccos::DGemm::Device::DeviceDGemm<Catccos::Comm::Kernel::HierarchicalBarrier>;
};

#endif // ASCEND950_HIERARCHICAL_ALLTOALL_DEVICE_H
