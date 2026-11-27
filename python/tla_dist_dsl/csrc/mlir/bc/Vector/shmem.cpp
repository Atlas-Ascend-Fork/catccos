// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Reuse the pinned upstream TLA device ABI instead of duplicating it here.
#include "dma_common.h"

#include "catlass/arch/arch.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "shmem.h"

// The mixed helper template selects MIX header implementations without running
// ccec's mixed-kernel transformation on these plain C ABI exports.
// Fail rather than silently compile SHMEM's software core barrier.
#if defined(TLA_DIST_REQUIRE_MIX_SYNC) && !defined(__CCE_AICORE_ENABLE_MIX__)
#error "MIX SHMEM bitcode requires __CCE_AICORE_ENABLE_MIX__ for hardware synchronization"
#endif

template <bool Get, bool Roce, pipe_t Pipe>
CATLASS_DEVICE void submitTile(
    memref_t<__gm__ float, 2>* src, memref_t<__gm__ float, 2>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc, __ubuf__ float* scratch, int32_t peer, int32_t qp_idx, int32_t event)
{
    uint64_t rows = srcDesc.originShape0 < dstDesc.originShape0 ? srcDesc.originShape0 : dstDesc.originShape0;
    uint64_t columns = srcDesc.originShape1 < dstDesc.originShape1 ? srcDesc.originShape1 : dstDesc.originShape1;
    if (!rows || !columns)
        return;
    if (srcDesc.stride0 == columns && dstDesc.stride0 == columns) {
        columns *= rows;
        rows = 1;
    }
    int64_t srcOffset = srcDesc.coord0 * srcDesc.stride0 + srcDesc.coord1 * srcDesc.stride1;
    int64_t dstOffset = dstDesc.coord0 * dstDesc.stride0 + dstDesc.coord1 * dstDesc.stride1;
    auto source = basePtr(src) + srcOffset;
    auto target = basePtr(dst) + dstOffset;
    // XSCALE/HNS_1825 WQEs encode a 32-bit byte length; UDMA limits a request
    // to 256 MiB. SHMEM's typed API takes elements, including for each chunk.
    constexpr uint64_t maxBytes = Roce ? UINT32_MAX : 256ULL * 1024 * 1024;
    constexpr uint64_t maxElements = maxBytes / sizeof(float);
    for (uint64_t row = 0; row < rows; ++row) {
        auto from = source + row * srcDesc.stride0;
        auto to = target + row * dstDesc.stride0;
        for (uint64_t offset = 0; offset < columns;) {
            uint64_t remaining = columns - offset;
            uint32_t count = static_cast<uint32_t>(remaining < maxElements ? remaining : maxElements);
            if constexpr (Roce && Get)
                aclshmemx_roce_qp_get_nbi<float>(to + offset, from + offset, scratch, count, peer, qp_idx, event);
            else if constexpr (Roce)
                aclshmemx_roce_qp_put_nbi<float>(to + offset, from + offset, scratch, count, peer, qp_idx, event);
            else if constexpr (Get)
                aclshmemx_udma_qp_get_nbi<float, Pipe>(to + offset, from + offset, scratch, count, peer, qp_idx, event);
            else
                aclshmemx_udma_qp_put_nbi<float, Pipe>(to + offset, from + offset, scratch, count, peer, qp_idx, event);
            offset += count;
        }
    }
}

template <typename T, Catccos::detail::CopyDirect Direction, Catccos::detail::CopyTransport Transport>
CATLASS_DEVICE void shmemRemoteCopyGmRowMajorToGmRowMajor(
    memref_t<__gm__ T, 2>* src, memref_t<__gm__ T, 2>* dst, memref_t<__ubuf__ T, 1>* ub, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc, const TensorDesc& ubDesc, int32_t peer, int32_t copyEventId)
{
    using ArchTag = Catlass::Arch::Ascend950;
    using Layout = Catlass::layout::RowMajor;
    using RemoteType = Catlass::Gemm::GemmType<T, Layout>;
    using TileRemoteCopy =
        Catccos::Comm::Tile::TileRemoteCopy<ArchTag, true, RemoteType, RemoteType, void, Direction, Transport>;

    int64_t ubOffset = ubDesc.coord0 * ubDesc.stride0 + ubDesc.coord1 * ubDesc.stride1;
    uint32_t ubAddress = localAddr(ub) + static_cast<uint32_t>(ubOffset * sizeof(T));
    int64_t srcOffset = srcDesc.coord0 * srcDesc.stride0 + srcDesc.coord1 * srcDesc.stride1;
    int64_t dstOffset = dstDesc.coord0 * dstDesc.stride0 + dstDesc.coord1 * dstDesc.stride1;

    AscendC::GlobalTensor<T> srcTensor;
    srcTensor.SetGlobalBuffer(basePtr(src) + srcOffset);
    AscendC::GlobalTensor<T> dstTensor;
    dstTensor.SetGlobalBuffer(basePtr(dst) + dstOffset);

    uint32_t rows = srcDesc.originShape0 < dstDesc.originShape0 ? srcDesc.originShape0 : dstDesc.originShape0;
    uint32_t columns = srcDesc.originShape1 < dstDesc.originShape1 ? srcDesc.originShape1 : dstDesc.originShape1;
    if (rows == 0 || columns == 0)
        return;
    Layout srcLayout(rows, columns, srcDesc.stride0);
    Layout dstLayout(rows, columns, dstDesc.stride0);
    AscendC::LocalTensor<T> ubTensor(AscendC::TPosition::VECCALC, ubAddress, elementCount(ub));

    TileRemoteCopy tileRemoteCopy;
    Catlass::MatrixCoord copyShape(rows, columns);

    // The caller supplies one tile and guarantees that ubTensor is large
    // enough. TileRemoteCopy synchronizes MTE2->MTE3 internally. Completion
    // and scratch reuse are explicitly synchronized by the caller.
    tileRemoteCopy(
        dstTensor, dstLayout, srcTensor, srcLayout, copyShape, ubTensor, static_cast<uint32_t>(copyEventId),
        static_cast<uint32_t>(peer));
}

template <typename T>
CATLASS_DEVICE void shmemRemoteGetGmToUb(
    memref_t<__gm__ T, 2>* src, memref_t<__ubuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc,
    int32_t peer)
{
    uint32_t rows = srcDesc.originShape0 < dstDesc.originShape0 ? srcDesc.originShape0 : dstDesc.originShape0;
    uint32_t columns = srcDesc.originShape1 < dstDesc.originShape1 ? srcDesc.originShape1 : dstDesc.originShape1;
    if (rows == 0 || columns == 0)
        return;
    int64_t srcOffset = srcDesc.coord0 * srcDesc.stride0 + srcDesc.coord1 * srcDesc.stride1;
    int64_t dstOffset = dstDesc.coord0 * dstDesc.stride0 + dstDesc.coord1 * dstDesc.stride1;
    AscendC::GlobalTensor<T> remote;
    remote.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(aclshmem_ptr(basePtr(src) + srcOffset, peer)));
    AscendC::LocalTensor<T> local(
        AscendC::TPosition::VECCALC, localAddr(dst) + static_cast<uint32_t>(dstOffset * sizeof(T)), elementCount(dst));
    AscendC::DataCopyExtParams params(
        rows, columns * sizeof(T), (srcDesc.stride0 - columns) * sizeof(T),
        (dstDesc.stride0 - columns) / (32 / sizeof(T)), 0);
    aclshmemi_copy_gm2ub(local, remote, params);
}

template <typename T>
CATLASS_DEVICE void shmemRemotePutUbToGm(
    memref_t<__ubuf__ T, 1>* src, memref_t<__gm__ T, 2>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc,
    int32_t peer)
{
    uint32_t rows = srcDesc.originShape0 < dstDesc.originShape0 ? srcDesc.originShape0 : dstDesc.originShape0;
    uint32_t columns = srcDesc.originShape1 < dstDesc.originShape1 ? srcDesc.originShape1 : dstDesc.originShape1;
    if (rows == 0 || columns == 0)
        return;
    int64_t srcOffset = srcDesc.coord0 * srcDesc.stride0 + srcDesc.coord1 * srcDesc.stride1;
    int64_t dstOffset = dstDesc.coord0 * dstDesc.stride0 + dstDesc.coord1 * dstDesc.stride1;
    AscendC::GlobalTensor<T> remote;
    remote.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(aclshmem_ptr(basePtr(dst) + dstOffset, peer)));
    AscendC::LocalTensor<T> local(
        AscendC::TPosition::VECCALC, localAddr(src) + static_cast<uint32_t>(srcOffset * sizeof(T)), elementCount(src));
    AscendC::DataCopyExtParams params(
        rows, columns * sizeof(T), (srcDesc.stride0 - columns) / (32 / sizeof(T)),
        (dstDesc.stride0 - columns) * sizeof(T), 0);
    aclshmemi_copy_ub2gm(remote, local, params);
}

extern "C" {

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

[aicore] __attribute__((always_inline)) uint64_t shmem_ptr_addr(uint64_t address, int32_t peer)
{
    return reinterpret_cast<uint64_t>(aclshmem_ptr(reinterpret_cast<__gm__ void*>(address), peer));
}

#define REGISTER_SHMEM_REMOTE_COPY(Name, Direction, Transport, DType)                                                \
    [aicore]                                                                                                         \
        __attribute__((always_inline)) void _mlir_ciface_shmem_remote_##Name##_gm_row_major_to_gm_row_major_##DType( \
            memref_t<__gm__ DType, 2>* src, memref_t<__gm__ DType, 2>* dst, memref_t<__ubuf__ DType, 1>* ub,         \
            DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), DESC_ABI_PARAMS(ub), int32_t peer, int32_t copyEventId)      \
    {                                                                                                                \
        shmemRemoteCopyGmRowMajorToGmRowMajor<                                                                       \
            DType, Catccos::detail::CopyDirect::Direction, Catccos::detail::CopyTransport::Transport>(               \
            src, dst, ub, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), TENSOR_DESC_12(ub), peer, copyEventId);          \
    }

REGISTER_SHMEM_REMOTE_COPY(get, Get, Mte, float)
REGISTER_SHMEM_REMOTE_COPY(put, Put, Mte, float)

#undef REGISTER_SHMEM_REMOTE_COPY

#if TLA_DIST_ENABLE_RDMA
[aicore] __attribute__((always_inline)) void _mlir_ciface_shmem_rdma_get(
    memref_t<__gm__ float, 2>* src, memref_t<__gm__ float, 2>* dst, memref_t<__ubuf__ uint8_t, 1>* ub,
    DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), int32_t peer, int32_t qp_idx, int32_t event)
{
    submitTile<true, true, PIPE_MTE3>(
        src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), reinterpret_cast<__ubuf__ float*>(localAddr(ub)), peer,
        qp_idx, event);
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_shmem_rdma_put(
    memref_t<__gm__ float, 2>* src, memref_t<__gm__ float, 2>* dst, memref_t<__ubuf__ uint8_t, 1>* ub,
    DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), int32_t peer, int32_t qp_idx, int32_t event)
{
    submitTile<false, true, PIPE_MTE3>(
        src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), reinterpret_cast<__ubuf__ float*>(localAddr(ub)), peer,
        qp_idx, event);
}

#endif

[aicore] __attribute__((always_inline)) void _mlir_ciface_shmem_udma_mte3_get(
    memref_t<__gm__ float, 2>* src, memref_t<__gm__ float, 2>* dst, memref_t<__ubuf__ uint8_t, 1>* ub,
    DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), int32_t peer, int32_t qp_idx, int32_t event)
{
    submitTile<true, false, PIPE_MTE3>(
        src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), reinterpret_cast<__ubuf__ float*>(localAddr(ub)), peer,
        qp_idx, event);
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_shmem_udma_mte3_put(
    memref_t<__gm__ float, 2>* src, memref_t<__gm__ float, 2>* dst, memref_t<__ubuf__ uint8_t, 1>* ub,
    DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst), int32_t peer, int32_t qp_idx, int32_t event)
{
    submitTile<false, false, PIPE_MTE3>(
        src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), reinterpret_cast<__ubuf__ float*>(localAddr(ub)), peer,
        qp_idx, event);
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_shmem_udma_s_get(
    memref_t<__gm__ float, 2>* src, memref_t<__gm__ float, 2>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst),
    int32_t peer, int32_t qp_idx)
{
    submitTile<true, false, PIPE_S>(src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), nullptr, peer, qp_idx, 0);
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_shmem_udma_s_put(
    memref_t<__gm__ float, 2>* src, memref_t<__gm__ float, 2>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst),
    int32_t peer, int32_t qp_idx)
{
    submitTile<false, false, PIPE_S>(src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), nullptr, peer, qp_idx, 0);
}

#if TLA_DIST_ENABLE_RDMA
[aicore] __attribute__((always_inline)) void _mlir_ciface_shmem_rdma_quiet(
    int32_t peer, int32_t qp_idx, memref_t<__ubuf__ uint8_t, 1>* ub, int32_t event)
{
    aclshmemx_roce_qp_quiet(peer, qp_idx, reinterpret_cast<__ubuf__ uint8_t*>(localAddr(ub)), event);
}

#endif

[aicore] __attribute__((always_inline)) void shmem_udma_quiet(int32_t peer, int32_t qp_idx)
{
    aclshmemx_udma_qp_quiet(peer, qp_idx);
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_shmem_remote_get_gm_row_major_to_ub_row_major_float(
    memref_t<__gm__ float, 2>* src, memref_t<__ubuf__ float, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst),
    int32_t peer)
{
    shmemRemoteGetGmToUb(src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), peer);
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_shmem_remote_put_ub_row_major_to_gm_row_major_float(
    memref_t<__ubuf__ float, 1>* src, memref_t<__gm__ float, 2>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst),
    int32_t peer)
{
    shmemRemotePutUbToGm(src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), peer);
}

[aicore] __attribute__((always_inline)) void shmem_barrier_all_vec() { aclshmemx_barrier_all_vec(); }

#endif
}

// Synchronization ABI v1. New calls use public SHMEM entry points only.
namespace {
CATLASS_DEVICE void tlaDistCompleteLocal() {
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<uint8_t> global;
    __asm__ __volatile__("" ::: "memory");
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
        AscendC::DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("" ::: "memory");
}
CATLASS_DEVICE bool tlaDistDefaultOwner() { return AscendC::GetBlockIdx() == 0; }
CATLASS_DEVICE int tlaDistCompare(int code) {
    switch (code) {
        case 0: return ACLSHMEM_CMP_EQ; case 1: return ACLSHMEM_CMP_NE;
        case 2: return ACLSHMEM_CMP_GT; case 3: return ACLSHMEM_CMP_GE;
        case 4: return ACLSHMEM_CMP_LT; case 5: return ACLSHMEM_CMP_LE;
        default: trap(); return ACLSHMEM_CMP_EQ;
    }
}
}
extern "C" {
CATLASS_DEVICE void tla_dist_local_sync_aiv() {
#ifdef __CCE_AICORE_ENABLE_MIX__
    AscendC::SyncAll<true>();
#else
    // This symbol is deliberately absent from software-barrier execution.
    trap();
#endif
}
CATLASS_DEVICE void tla_dist_sync_aiv(int32_t team) { if (team != -1) aclshmem_sync(team); }
CATLASS_DEVICE void tla_dist_barrier_aiv(int32_t team) {
    if (team == -1) return;
    tlaDistCompleteLocal(); aclshmem_barrier(team);
}
#ifdef __CCE_AICORE_ENABLE_MIX__
CATLASS_DEVICE void tla_dist_local_sync_mix_aiv() { AscendC::SyncAll<false>(); }
CATLASS_DEVICE void tla_dist_sync_mix_aiv(int32_t team) {
    if (team == -1) return;
    AscendC::SyncAll<false>(); aclshmem_sync(team); AscendC::SyncAll<false>();
}
CATLASS_DEVICE void tla_dist_barrier_mix_aiv(int32_t team) {
    if (team == -1) return;
    tlaDistCompleteLocal(); AscendC::SyncAll<false>(); aclshmem_barrier(team); AscendC::SyncAll<false>();
}
#endif
CATLASS_DEVICE void tla_dist_quiet_default() {
    if (!tlaDistDefaultOwner()) { trap(); return; }
    aclshmem_quiet();
}
CATLASS_DEVICE void tla_dist_fence() {
    if (!tlaDistDefaultOwner()) { trap(); return; }
    aclshmem_fence();
}
CATLASS_DEVICE void tla_dist_signal_i32(uint64_t addr, int32_t value, int32_t op, int32_t peer) {
    // ADD requires a directly mapped peer. SET may enter RoCE only on the
    // unique flat AIV0 owner. aclshmem_ptr is the public topology check.
    bool mapped = aclshmem_ptr(reinterpret_cast<__gm__ void*>(addr), peer) != nullptr;
    if ((!mapped && op == 1) || (!mapped && !tlaDistDefaultOwner())) { trap(); return; }
    int backendOp;
    switch (op) {
        case 0: backendOp = ACLSHMEM_SIGNAL_SET; break;
        case 1: backendOp = ACLSHMEM_SIGNAL_ADD; break;
        default: trap(); return;
    }
    aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t*>(addr), value, backendOp, peer);
}
// Small-integer thresholds use a 64-bit C ABI, avoiding target-specific
// signext/zeroext attributes. UInt64 never passes through a signed value.
#define TLA_DIST_WAIT(NAME, BACKEND, TYPE, ABI) \
CATLASS_DEVICE void tla_dist_wait_##NAME(uint64_t addr, ABI value, int32_t cmp) { \
    aclshmem_##BACKEND##_wait_until(reinterpret_cast<__gm__ TYPE*>(addr), tlaDistCompare(cmp), static_cast<TYPE>(value)); \
}
TLA_DIST_WAIT(i8, int8, int8_t, int64_t)
TLA_DIST_WAIT(i16, int16, int16_t, int64_t)
TLA_DIST_WAIT(i32, int32, int32_t, int64_t)
TLA_DIST_WAIT(i64, int64, int64_t, int64_t)
TLA_DIST_WAIT(u8, uint8, uint8_t, uint64_t)
TLA_DIST_WAIT(u16, uint16, uint16_t, uint64_t)
TLA_DIST_WAIT(u32, uint32, uint32_t, uint64_t)
TLA_DIST_WAIT(u64, uint64, uint64_t, uint64_t)
TLA_DIST_WAIT(f32, float, float, float)
#undef TLA_DIST_WAIT
}
