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

    int64_t ubOffset =
        static_cast<int64_t>(ubDesc.coord0) * ubDesc.stride0 + static_cast<int64_t>(ubDesc.coord1) * ubDesc.stride1;
    uint32_t ubAddress = localAddr(ub) + static_cast<uint32_t>(ubOffset * sizeof(T));
    int64_t srcOffset =
        static_cast<int64_t>(srcDesc.coord0) * srcDesc.stride0 + static_cast<int64_t>(srcDesc.coord1) * srcDesc.stride1;
    int64_t dstOffset =
        static_cast<int64_t>(dstDesc.coord0) * dstDesc.stride0 + static_cast<int64_t>(dstDesc.coord1) * dstDesc.stride1;

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
