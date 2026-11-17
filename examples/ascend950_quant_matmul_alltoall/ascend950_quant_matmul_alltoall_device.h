
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_QUANT_MATMUL_ALLTOALL_DEVICE_H
#define ASCEND950_QUANT_MATMUL_ALLTOALL_DEVICE_H

#include "info.h"

// from catccos
#include "catccos/catccos.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_hierarchical_copy.hpp"
#include "catccos/comm/block/comm_block_hierarchical_swizzle.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
#include "catccos/dgemm/kernel/ascend950_quant_matmul_alltoall.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

using namespace AscendC;
using namespace Catccos;
using namespace Catlass;
using namespace tla;

// ============================================================================
// QuantMatmulAllToAll 配置(评审与增量设计 §4/§5 + 设计文档 v2.2)
//
// MXFP8/MXFP4 沿用量化 MmadMx 路径;BF16 使用普通 MmadPingpong,TileCopy
// 不带 MX scale。三种输入共享同一个融合 Kernel 和传输/调度逻辑。
// 同机和跨机都使用 QuantMatmulAllToAll 融合 Kernel;
// UDMA/RoCE PUT 直接写入对端对称输出 Y,没有 RX 中转复制。
// ============================================================================

template <
    class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementAScale, class LayoutAScale,
    class ElementBScale, class LayoutBScale, class ElementC, class LayoutC, uint32_t M0_, uint32_t N0_, uint32_t K0_>
struct Ascend950QuantMatmulAllToAllConfig {
    using ArchTag = Catlass::Arch::Ascend950;

    static constexpr bool enableUnitFlag = true;
    static constexpr uint32_t l0CStages = 1;
    static constexpr uint32_t CORE_COUNT = 32;
    static constexpr uint32_t GROUP_SIZE = CATCCOS_HIERARCHICAL_GROUP_SIZE;
    static constexpr uint32_t COMM_BLOCK_M = 64;
    using MmadDispatchPolicy = Catlass::Gemm::MmadMx<ArchTag, enableUnitFlag, 16>;

    using L1TileShape = tla::Shape<tla::Int<M0_>, tla::Int<N0_>, tla::Int<K0_>>;
    using L0TileShape = tla::Shape<tla::Int<M0_>, tla::Int<N0_>, tla::Int<128>>;

    using TileCopy = Catlass::Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementAScale,
        decltype(tla::MakeMxScaleLayout<ElementAScale, LayoutA, false>(0U, 0U)), ElementBScale,
        decltype(tla::MakeMxScaleLayout<ElementBScale, LayoutB, true>(0U, 0U)), ElementC, LayoutC, void>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;

    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;

    using ByteType = Catlass::Gemm::GemmType<uint8_t, Catlass::layout::RowMajor>;
    using TileComm = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, true, ByteType, ByteType, void, Catccos::detail::CopyDirect::Put, Catccos::detail::CopyTransport::Mte>;
    using TileSwizzle = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using CommDispatch = Catccos::Comm::AtlasCommHierarchicalCopy<ArchTag, 1, true>;
    using BlockComm = Catccos::Comm::Block::CommBlock<CommDispatch, ByteType, ByteType, void, TileComm, TileSwizzle>;
    using CommScheduler = Catccos::Comm::Block::HierarchicalCommSwizzle<CORE_COUNT, GROUP_SIZE, COMM_BLOCK_M>;
    using Kernel = DGemm::Kernel::QuantMatmulAllToAll<BlockMmad, BlockMmadScheduler, BlockComm, CommScheduler, true>;
    using Device = DGemm::Device::DeviceDGemm<Kernel>;
};

// 原生 BF16 档:A/B 直接参与 Cube 计算,不创建或传输 MX scale。
template <
    class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC, uint32_t M0_,
    uint32_t N0_, uint32_t K0_>
struct Ascend950MatmulAllToAllBf16Config {
    using ArchTag = Catlass::Arch::Ascend950;

    static constexpr bool enableUnitFlag = true;
    static constexpr bool useHF32 = false;
    static constexpr bool enableL1Resident = false;
    static constexpr uint32_t l0CStages = 1;
    static constexpr uint32_t l1AStages = 2;
    static constexpr uint32_t l1BStages = 2;
    static constexpr uint32_t l0AStages = 2;
    static constexpr uint32_t l0BStages = 2;
    static constexpr uint32_t CORE_COUNT = 32;
    static constexpr uint32_t GROUP_SIZE = CATCCOS_HIERARCHICAL_GROUP_SIZE;
    static constexpr uint32_t COMM_BLOCK_M = 64;
    using MmadDispatchPolicy = Catlass::Gemm::MmadPingpong<
        ArchTag, enableUnitFlag, useHF32, l0CStages, enableL1Resident, l1AStages, l1BStages, l0AStages, l0BStages>;

    using L1TileShape = tla::Shape<tla::Int<M0_>, tla::Int<N0_>, tla::Int<K0_>>;
    using L0TileShape = tla::Shape<tla::Int<M0_>, tla::Int<N0_>, tla::Int<64>>;

    using TileCopy =
        Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, void>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;

    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
    using ByteType = Catlass::Gemm::GemmType<uint8_t, Catlass::layout::RowMajor>;
    using TileComm = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, true, ByteType, ByteType, void, Catccos::detail::CopyDirect::Put, Catccos::detail::CopyTransport::Mte>;
    using TileSwizzle = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using CommDispatch = Catccos::Comm::AtlasCommHierarchicalCopy<ArchTag, 1, true>;
    using BlockComm = Catccos::Comm::Block::CommBlock<CommDispatch, ByteType, ByteType, void, TileComm, TileSwizzle>;
    using CommScheduler = Catccos::Comm::Block::HierarchicalCommSwizzle<CORE_COUNT, GROUP_SIZE, COMM_BLOCK_M>;
    using Kernel = DGemm::Kernel::QuantMatmulAllToAll<BlockMmad, BlockMmadScheduler, BlockComm, CommScheduler, false>;
    using Device = DGemm::Device::DeviceDGemm<Kernel>;
};

// MXFP8 档(E4M3):L1 tile 128×256×256(与基线 mxfp8 一致)
template <
    class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementAScale, class LayoutAScale,
    class ElementBScale, class LayoutBScale, class ElementC, class LayoutC>
using Ascend950QuantMatmulAllToAllConfigFp8 = Ascend950QuantMatmulAllToAllConfig<
    ElementA, LayoutA, ElementB, LayoutB, ElementAScale, LayoutAScale, ElementBScale, LayoutBScale, ElementC, LayoutC,
    128, 256, 256>;

// MXFP4 档(E2M1 打包):L1 tile 128×256×512(catccos fp4 示例验证过的档位)
template <
    class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementAScale, class LayoutAScale,
    class ElementBScale, class LayoutBScale, class ElementC, class LayoutC>
using Ascend950QuantMatmulAllToAllConfigFp4 = Ascend950QuantMatmulAllToAllConfig<
    ElementA, LayoutA, ElementB, LayoutB, ElementAScale, LayoutAScale, ElementBScale, LayoutBScale, ElementC, LayoutC,
    128, 256, 512>;

// BF16 档:L1 tile 128x256x256,L0 K=64。
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
using Ascend950QuantMatmulAllToAllConfigBf16 =
    Ascend950MatmulAllToAllBf16Config<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 128, 256, 256>;

#endif // ASCEND950_QUANT_MATMUL_ALLTOALL_DEVICE_H
