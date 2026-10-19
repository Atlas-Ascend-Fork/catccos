

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_MXFP8_ALLTOALL_MATMUL_SPLIT_K_URMA_DEVICE_H
#define ASCEND950_MXFP8_ALLTOALL_MATMUL_SPLIT_K_URMA_DEVICE_H

#include <type_traits>

#include "catccos/catccos.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/block/block_mmad_mx_alltoall_matmul_k_split.hpp"
#include "catccos/dgemm/block/block_swizzle_alltoall_matmul_k_split.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
#include "catccos/dgemm/kernel/ascend950_alltoall_matmul_split_k_urma.hpp"
#include "catccos/dgemm/kernel/ascend950_alltoall_matmul_split_k_urma_local_atomic.h"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "info.h"

using namespace AscendC;
using namespace Catccos;

template <
    class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementMxScaleA, class LayoutMxScaleA,
    class ElementMxScaleB, class LayoutMxScaleB, class ElementC, class LayoutC, uint32_t SWIZZLE_OFFSET_,
    bool LOCAL_ATOMIC_ = false>
struct Ascend950MxFp8AllToAllMatmulSplitKUrmaConfig {
    using ArchTag = Catlass::Arch::Ascend950;
    static_assert(
        SWIZZLE_OFFSET_ >= 1 && SWIZZLE_OFFSET_ <= 7, "Ascend950 split-K URMA swizzle offset must be in [1, 7].");

    static constexpr bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadMx<ArchTag, ENABLE_UNIT_FLAG, 1>;

    using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
    using L0TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using ScaleAType = Catlass::Gemm::GemmType<ElementMxScaleA, LayoutMxScaleA>;
    using TileCopy = Catlass::Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementMxScaleA,
        decltype(tla::MakeMxScaleLayout<ElementMxScaleA, LayoutA, false>(0U, 0U)), ElementMxScaleB,
        decltype(tla::MakeMxScaleLayout<ElementMxScaleB, LayoutB, true>(0U, 0U)), ElementC, LayoutC, void>;
    using BlockMmad = Catccos::DGemm::Block::BlockMmadMxAllToAllMatmulKSplit<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;

    static constexpr bool IS_DYNAMIC = true;

    using BlockMmadScheduler =
        Catccos::DGemm::Block::GemmBlockSwizzleAllToAllMatmulKSplit<SWIZZLE_OFFSET_, LOCAL_ATOMIC_ ? 0 : 1>;

    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using RemoteScaleSrcType = ScaleAType;
    using RemoteScaleDstType = ScaleAType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileUdmaCopy = Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Put, CopyTransport::Udma>;
    using TileUdmaCopyScale = Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, RemoteScaleSrcType, RemoteScaleDstType, void, CopyDirect::Put, CopyTransport::Udma>;

    using UdmaDispatch = Comm::AtlasCommUdmaRemoteCopy<ArchTag, UB_STAGES>;
    using BlockUdmaComm = Comm::Block::CommBlock<UdmaDispatch, RemoteSrcType, RemoteDstType, TileUdmaCopy>;
    using BlockUdmaCommScale =
        Comm::Block::CommBlock<UdmaDispatch, RemoteScaleSrcType, RemoteScaleDstType, TileUdmaCopyScale>;

    using DefaultKernel = DGemm::Kernel::Ascend950AllToAllMatmulSplitKUrma<
        BlockMmad, BlockUdmaComm, BlockUdmaCommScale, BlockMmadScheduler, WORKSPACE_STAGES>;
    using LocalAtomicKernel = DGemm::Kernel::Ascend950AllToAllMatmulSplitKUrmaLocalAtomic<
        BlockMmad, BlockUdmaComm, BlockUdmaCommScale, BlockMmadScheduler, WORKSPACE_STAGES>;
    using Kernel = std::conditional_t<LOCAL_ATOMIC_, LocalAtomicKernel, DefaultKernel>;

    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

#endif // ASCEND950_MXFP8_ALLTOALL_MATMUL_SPLIT_K_URMA_DEVICE_H
