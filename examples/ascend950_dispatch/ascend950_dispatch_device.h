/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef ASCEND950_DISPATCH_DEVICE_H
#define ASCEND950_DISPATCH_DEVICE_H

#include "catlass/catlass.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/kernel/ascend950_dispatch.hpp"
#include "catccos/comm/kernel/ascend950_dispatch_direct.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
#include "info.h"

template <class ElementToken, class ElementScale, class ElementTopkIdx, class ElementTopkW, class ElementIdx>
struct Ascend950DispatchConfig {
    using ArchTag = Catlass::Arch::Ascend950;
    constexpr static bool IS_DYNAMIC = true;
    constexpr static Catccos::detail::CopyDirect COPY_DIRECT = Catccos::detail::CopyDirect::Put;
    using CommDispatchPolicy = Catccos::Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using Layout = Catlass::layout::RowMajor;
    template <class Element>
    using RemoteType = Catlass::Gemm::GemmType<Element, Layout>;
    template <class Element>
    using TileRemoteCopy = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, RemoteType<Element>, RemoteType<Element>, void, COPY_DIRECT,
        Catccos::detail::CopyTransport::Mte>;
    using ByteType = Catlass::Gemm::GemmType<uint8_t, Layout>;
    using CopyGmToUb = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, ByteType>;
    using CopyUbToGm = Catlass::Epilogue::Tile::CopyUb2Gm<ArchTag, ByteType>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using BlockDispatch = Catccos::Comm::Block::CommBlockDispatch<
        CommDispatchPolicy, ElementToken, ElementScale, ElementTopkIdx, ElementTopkW, ElementIdx, TileRemoteCopy,
        CopyGmToUb, CopyUbToGm, TileScheduler, MAX_RANK_SIZE>;
    using BlockQuantize = Catccos::Comm::Block::BlockQuantizeBf16Int8;
    using BlockCommScheduler = Catccos::Comm::Block::BlockCommSchedulerDispatch;
    using Kernel = Catccos::Comm::Kernel::Ascend950Dispatch<BlockDispatch, BlockCommScheduler, BlockQuantize>;
    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

template <class ElementToken, class ElementScale, class ElementTopkIdx, class ElementTopkW, class ElementIdx>
struct Ascend950DispatchDirectConfig {
    using ArchTag = Catlass::Arch::Ascend950;
    constexpr static bool IS_DYNAMIC = true;
    constexpr static Catccos::detail::CopyDirect COPY_DIRECT = Catccos::detail::CopyDirect::Put;
    using CommDispatchPolicy = Catccos::Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using Layout = Catlass::layout::RowMajor;
    template <class Element>
    using RemoteType = Catlass::Gemm::GemmType<Element, Layout>;
    template <class Element>
    using TileRemoteCopy = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, RemoteType<Element>, RemoteType<Element>, void, COPY_DIRECT,
        Catccos::detail::CopyTransport::Mte>;
    using ByteType = Catlass::Gemm::GemmType<uint8_t, Layout>;
    using CopyGmToUb = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, ByteType>;
    using CopyUbToGm = Catlass::Epilogue::Tile::CopyUb2Gm<ArchTag, ByteType>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using BlockDispatch = Catccos::Comm::Block::CommBlockDispatch<
        CommDispatchPolicy, ElementToken, ElementScale, ElementTopkIdx, ElementTopkW, ElementIdx, TileRemoteCopy,
        CopyGmToUb, CopyUbToGm, TileScheduler, MAX_RANK_SIZE>;
    using BlockQuantize = Catccos::Comm::Block::BlockQuantizeBf16Int8;
    using Kernel = Catccos::Comm::Kernel::Ascend950DispatchDirect<BlockDispatch, BlockQuantize>;
    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

#endif // ASCEND950_DISPATCH_DEVICE_H
