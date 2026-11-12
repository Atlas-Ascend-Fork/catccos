/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_MXFP8_ALLTOALL_MATMUL_DEVICE_H
#define ASCEND950_MXFP8_ALLTOALL_MATMUL_DEVICE_H

#include "catccos/dgemm/block/block_mmad_mx_alltoall_matmul_k_split.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_hierarchical_swizzle.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/kernel/ascend950_hierarchical_barrier.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
#include "catccos/dgemm/kernel/ascend950_mxfp8_alltoall_matmul_compute.h"
#include "catccos/dgemm/kernel/ascend950_mxfp8_alltoall_matmul.h"
#include "catccos/dgemm/kernel/ascend950_mxfp8_alltoall_matmul_transport.h"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

template <
    class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementMxScaleA, class LayoutMxScaleA,
    class ElementMxScaleB, class LayoutMxScaleB, class ElementC, class LayoutC, uint32_t TileK = 256>
struct Ascend950MxFp8AllToAllMatmulConfig {
    using ArchTag = Catlass::Arch::Ascend950;
    static constexpr uint32_t CORE_COUNT = 32;
    static constexpr uint32_t COMM_BLOCK_M = 64;
    static constexpr uint32_t GROUP_SIZE = CATCCOS_MXFP8_ALLTOALL_GROUP_SIZE;
    using MmadDispatchPolicy = Catlass::Gemm::MmadMx<ArchTag, true, 1>;
    using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<TileK>>;
    using L0TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<(TileK < 128 ? TileK : 128)>>;
    using TileCopy = Catlass::Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementMxScaleA,
        decltype(tla::MakeMxScaleLayout<ElementMxScaleA, LayoutMxScaleA, false>(0U, 0U)), ElementMxScaleB,
        decltype(tla::MakeMxScaleLayout<ElementMxScaleB, LayoutMxScaleB, true>(0U, 0U)), ElementC, LayoutC, void>;
    using BlockMmad = Catccos::DGemm::Block::BlockMmadMxAllToAllMatmulKSplit<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    // Byte transfers also support localK=64/192: scales occupy 2/6 bytes per row.
    using Word = uint8_t;
    using WordType = Catlass::Gemm::GemmType<Word, Catlass::layout::RowMajor>;
    using Tile = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, true, WordType, WordType, void, Catccos::detail::CopyDirect::Put, Catccos::detail::CopyTransport::Mte>;
    using TileSwizzle = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using Dispatch = Catccos::Comm::AtlasCommHierarchicalCopy<ArchTag, 2, true>;
    using BlockComm = Catccos::Comm::Block::CommBlock<Dispatch, WordType, WordType, void, Tile, TileSwizzle>;
    using CommScheduler = Catccos::Comm::Block::HierarchicalCommSwizzle<CORE_COUNT, GROUP_SIZE, COMM_BLOCK_M>;
    using Kernel = Catccos::DGemm::Kernel::Ascend950MxFp8AllToAllMatmul<BlockMmad, BlockComm, CommScheduler>;
    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
    using BarrierDevice = Catccos::DGemm::Device::DeviceDGemm<Catccos::Comm::Kernel::HierarchicalBarrier>;
};

#endif // ASCEND950_MXFP8_ALLTOALL_MATMUL_DEVICE_H
