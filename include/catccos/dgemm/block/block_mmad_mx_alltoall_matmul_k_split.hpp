
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_BLOCK_MMAD_MX_ALLTOALL_MATMUL_K_SPLIT_HPP
#define CATCCOS_DGEMM_BLOCK_MMAD_MX_ALLTOALL_MATMUL_K_SPLIT_HPP

#include <type_traits>

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/numeric_size.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catccos::DGemm::Block {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

/// MXFP8 MMAD block for ops-transformer-style K-split AllToAll MatMul.
///
/// A and scaleA are stored in symmetric memory as rank-contiguous blocks:
///
///     A:      [srcRank][commRows][localK_aligned]
///     scaleA: [srcRank][commRows][localScaleK_aligned]
///
/// The logical GEMM K dimension is rankSize * localK. This block follows the
/// CATLASS MXFP8 L1/L0 copy path, but maps every logical K stripe back to the
/// matching physical source-rank block before GM->L1 copy.
template <
    class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_,
    class ElementBias_ = void,
    class TileCopy_ = Catlass::Gemm::Tile::PackedMxTileCopyTla<
        typename DispatchPolicy_::ArchTag, ElementA_, Catlass::layout::RowMajor, ElementB_, Catlass::layout::RowMajor,
        float8_e8m0_t, decltype(tla::MakeMxScaleLayout<float8_e8m0_t, Catlass::layout::RowMajor, false>(0U, 0U)),
        float8_e8m0_t, decltype(tla::MakeMxScaleLayout<float8_e8m0_t, Catlass::layout::RowMajor, true>(0U, 0U)),
        ElementC_, Catlass::layout::RowMajor, ElementBias_>,
    class TileMmad_ = Catlass::Gemm::Tile::TileMmadTla<
        typename DispatchPolicy_::ArchTag, ElementA_, typename TileCopy_::LayoutTagL1A>>
struct BlockMmadMxAllToAllMatmulKSplit {
public:
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy::LayoutB;
    using ElementMxScaleA = typename TileCopy::ElementMxScaleA;
    using LayoutMxScaleA = typename TileCopy::LayoutMxScaleA;
    using ElementMxScaleB = typename TileCopy::ElementMxScaleB;
    using LayoutMxScaleB = typename TileCopy::LayoutMxScaleB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy::LayoutC;
    using ElementBias = ElementBias_;
    using ElementL0A = typename Catlass::Gemm::helper::GetL0Element<ElementA, true>::Element;
    using ElementL0B = typename Catlass::Gemm::helper::GetL0Element<ElementB, true>::Element;
    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyL1ToBT = typename TileCopy::CopyL1ToBT;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr bool HAS_BIAS = TileCopy::HAS_BIAS;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL1MxScaleA = typename TileCopy::LayoutTagL1MxScaleA;
    using LayoutTagL1MxScaleB = typename TileCopy::LayoutTagL1MxScaleB;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static_assert(
        tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
        "L1TileShape must be tla::tuple and static.");
    static_assert(
        tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
        "L0TileShape must be tla::tuple and static.");

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_L1_RESIDENT = DispatchPolicy::ENABLE_L1_RESIDENT;
    static constexpr uint32_t L1A_STAGES = DispatchPolicy::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr uint32_t L1_SCALE_FACTOR_K = DispatchPolicy::L1_SCALE_FACTOR_K;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * Catlass::SizeOfBits<ElementA>::value / 8;
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * Catlass::SizeOfBits<ElementB>::value / 8;
    static constexpr uint32_t L1SCALEA_TILE_SIZE =
        L1_TILE_M * L1_TILE_K / Catlass::MX_SCALE_GROUP_NUM * sizeof(ElementMxScaleA) * L1_SCALE_FACTOR_K;
    static constexpr uint32_t L1SCALEB_TILE_SIZE =
        L1_TILE_N * L1_TILE_K / Catlass::MX_SCALE_GROUP_NUM * sizeof(ElementMxScaleB) * L1_SCALE_FACTOR_K;
    static constexpr uint32_t L1_USED_SIZE = L1A_TILE_SIZE * L1A_STAGES + L1B_TILE_SIZE * L1B_STAGES +
                                             L1SCALEA_TILE_SIZE * L1A_STAGES + L1SCALEB_TILE_SIZE * L1B_STAGES;

    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * Catlass::SizeOfBits<ElementL0A>::value / 8;
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * Catlass::SizeOfBits<ElementL0B>::value / 8;
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    static_assert(!(ENABLE_UNIT_FLAG && L0C_STAGES != 1), "L0C_STAGES must be 1 when UnitFlag is true.");
    static_assert(
        L1_TILE_K % Catlass::MX_BASEK_FACTOR == 0 && L0_TILE_K % Catlass::MX_BASEK_FACTOR == 0,
        "MX L1/L0 K tile must be multiples of 64.");
    static_assert(
        L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
        "This MX K-split block requires matching L1/L0 M/N tile shapes.");
    static_assert(L0_TILE_K <= L1_TILE_K, "L0TileShape::K cannot exceed L1TileShape::K.");
    static_assert(L1_SCALE_FACTOR_K == 1, "K-split symmetric A layout requires one MX scale fetch per L1 K stripe.");
    static_assert(L1_USED_SIZE <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space.");
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space.");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space.");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space.");
    static_assert(
        (!HAS_BIAS && (L1A_STAGES + L1B_STAGES) <= 8) || (HAS_BIAS && (L1A_STAGES + L1B_STAGES) <= 7),
        "L1 Buffer overflow: exceeds EVENT range.");
    static_assert(
        (!HAS_BIAS && (L0A_STAGES + L0B_STAGES) <= 8) || (HAS_BIAS && (L0A_STAGES + L0B_STAGES) <= 7),
        "L0 Buffer overflow: exceeds EVENT_ID range.");

    static constexpr auto L1A_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{});
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});
    static constexpr auto L1SCALEA_LAYOUT = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagL1MxScaleA, false>(
        tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K / Catlass::MX_SCALE_GROUP_NUM * L1_SCALE_FACTOR_K>{});
    static constexpr auto L1SCALEB_LAYOUT = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagL1MxScaleB, true>(
        tla::Int<L1_TILE_K / Catlass::MX_SCALE_GROUP_NUM * L1_SCALE_FACTOR_K>{}, tla::Int<L1_TILE_N>{});
    static constexpr auto L1BIAS_LAYOUT = tla::MakeLayout(tla::Int<L1_TILE_N>{});
    static constexpr auto L0BIAS_LAYOUT = tla::MakeLayout(tla::Int<L0_TILE_N>{});

    CATLASS_DEVICE
    void RestoreStatus()
    {
        for (int i = 0; i < L1A_STAGES; ++i) {
            lastAddrA[i] = nullptr;
            lastCoordA[i] = MatrixCoord{0U, 0U};
        }
        for (int i = 0; i < L1B_STAGES; ++i) {
            lastAddrB[i] = nullptr;
            lastCoordB[i] = MatrixCoord{0U, 0U};
        }
        l1MxScaleAListId = 0;
        l1MxScaleBListId = 0;
    }

    CATLASS_DEVICE
    BlockMmadMxAllToAllMatmulKSplit(Catlass::Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0)
    {
        if ASCEND_IS_AIC {
            if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
                AscendC::SetMMLayoutTransform(true);
            }

            uint32_t l1Offset = l1BufAddrStart;
            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1Offset);
                l1Offset += L1A_TILE_SIZE;
                l1AEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            }
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1Offset);
                l1Offset += L1B_TILE_SIZE;
                l1BEventList[i] = i + L1A_STAGES;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; i++) {
                l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementL0A>(L0A_TILE_SIZE * i);
                l0AEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; i++) {
                l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementL0B>(L0B_TILE_SIZE * i);
                l0BEventList[i] = i + L0A_STAGES;
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            if constexpr (!ENABLE_UNIT_FLAG) {
                for (uint32_t i = 0; i < L0C_STAGES; i++) {
                    l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * i);
                    l0CEventList[i] = i;
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
                }
            } else {
                l0CTensorList[0] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
                l0CEventList[0] = 0;
            }
            if constexpr (HAS_BIAS) {
                l1BiasTensor = resource.l1Buf.template GetBufferByByte<uint8_t>(l1Offset);
                l1Offset += L1_TILE_N * sizeof(ElementBias);
                l0BiasTensor = resource.btBuf.template GetBufferByByte<ElementAccumulator>(0);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
            }
            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                l1MxScaleATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementMxScaleA>(l1Offset);
                l1Offset += L1SCALEA_TILE_SIZE;
            }
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                l1MxScaleBTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementMxScaleB>(l1Offset);
                l1Offset += L1SCALEB_TILE_SIZE;
            }

            if constexpr (ENABLE_L1_RESIDENT) {
                RestoreStatus();
            }
        }
    }

    CATLASS_DEVICE
    ~BlockMmadMxAllToAllMatmulKSplit()
    {
        if ASCEND_IS_AIC {
            if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
                AscendC::SetMMLayoutTransform(false);
            }
            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            }
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            if constexpr (!ENABLE_UNIT_FLAG) {
                for (uint32_t i = 0; i < L0C_STAGES; i++) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
                }
            }
            if constexpr (HAS_BIAS) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
            }
        }
    }

    template <class TensorB, class TensorC, class TensorMxScaleB, class TensorBias = Catlass::EmptyClass>
    CATLASS_DEVICE void operator()(
        AscendC::GlobalTensor<ElementA>& gmAStage, AscendC::GlobalTensor<ElementMxScaleA>& gmMxScaleAStage,
        TensorB& tensorB, TensorC& tensorC, GemmCoord const& actualShape, uint32_t blockMOffset, uint32_t commRows,
        uint32_t localK, int64_t alignedLocalK, uint32_t localScaleK, uint32_t alignedLocalScaleK,
        TensorMxScaleB const& tensorMxScaleB, TensorBias const& tensorBias = {}, bool initAccumulator = true,
        bool storeResult = true, bool finalMmad = true)
    {
        if constexpr (HAS_BIAS) {
            static constexpr uint32_t BIAS_BUF_SIZE = L0_TILE_N * sizeof(ElementAccumulator);
            static constexpr uint32_t L1BIAS_SIZE = L1_TILE_N * sizeof(ElementBias);
            static_assert(BIAS_BUF_SIZE <= ArchTag::BIAS_SIZE, "BIAS_BUF_SIZE exceeding the BT space.");
            static_assert(L1_USED_SIZE + L1BIAS_SIZE <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space.");
        }

        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();
        uint32_t kL1Actual = Min(kBlockActual, L1_TILE_K);
        uint32_t kL1ScaleActual = Min(kBlockActual, L1_TILE_K * L1_SCALE_FACTOR_K);

        auto tensorTileA =
            GetGmTileA(gmAStage, blockMOffset, 0, mBlockActual, kL1Actual, commRows, localK, alignedLocalK);
        auto tensorTileMxScaleA = GetGmTileMxScaleA(
            gmMxScaleAStage, blockMOffset, 0, mBlockActual, CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(kL1ScaleActual),
            commRows, localScaleK, alignedLocalScaleK);

        using FirstTensorA = decltype(tensorTileA);
        using FirstTensorMxScaleA = decltype(tensorTileMxScaleA);
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<FirstTensorA>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
        using CopyGmToL1MxScaleA = typename TileCopy::template CopyGmToL1MxScaleA<FirstTensorMxScaleA>;
        using CopyGmToL1MxScaleB = typename TileCopy::template CopyGmToL1MxScaleB<TensorMxScaleB>;
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyGmToL1MxScaleA copyGmToL1MxScaleA;
        CopyGmToL1MxScaleB copyGmToL1MxScaleB;
        CopyL0CToDst copyL0CToDst;

        uint32_t mL1Actual = mBlockActual;
        if constexpr (std::is_same_v<ArchTag, Catlass::Arch::AtlasA2>) {
            if (mL1Actual == 1) {
                mL1Actual = 16;
            }
        }
        uint32_t nL1Actual = nBlockActual;

        auto layoutInL0C = tla::MakeLayoutL0C(mL1Actual, nL1Actual);
        auto tensorL0C = tla::MakeTensor(l0CTensorList[l0CListId], layoutInL0C, Catlass::Arch::PositionL0C{});
        auto tensorL0Bias = tla::MakeTensor(l0BiasTensor, L0BIAS_LAYOUT, Catlass::Arch::PositionBias{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
        auto tensorL1A = tla::MakeTensor(l1ATensorList[l1AListId], L1A_LAYOUT, Catlass::Arch::PositionL1{});
        CopyGmTileA(copyGmToL1A, tensorL1A, tensorTileA, l1AListId);
        InitZeroInL1A(tensorL1A, tla::MakeShape(mL1Actual, kL1Actual));

        auto tensorL1MxScaleA =
            tla::MakeTensor(l1MxScaleATensorList[l1MxScaleAListId], L1SCALEA_LAYOUT, Catlass::Arch::PositionL1{});
        copyGmToL1MxScaleA(tensorL1MxScaleA, tensorTileMxScaleA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
        auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], L1B_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorTileB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(kL1Actual, nBlockActual));
        CopyGmTileB(copyGmToL1B, tensorL1B, tensorTileB, l1BListId);
        InitZeroInL1B(tensorL1B, tla::MakeShape(kL1Actual, nL1Actual));

        auto tensorL1MxScaleB =
            tla::MakeTensor(l1MxScaleBTensorList[l1MxScaleBListId], L1SCALEB_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorTileMxScaleB = GetTile(
            tensorMxScaleB, tla::MakeCoord(0, 0),
            tla::MakeShape(CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(kL1ScaleActual), nBlockActual));
        copyGmToL1MxScaleB(tensorL1MxScaleB, tensorTileMxScaleB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);

        if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, Catlass::EmptyClass>) {
            using CopyGmToL1Bias = typename TileCopy::template CopyGmToL1Bias<TensorBias>;
            CopyGmToL1Bias copyGmToL1Bias;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
            auto l1Bias = l1BiasTensor.template ReinterpretCast<ElementBias>();
            auto tensorL1Bias = tla::MakeTensor(l1Bias, L1BIAS_LAYOUT, Catlass::Arch::PositionL1{});
            copyGmToL1Bias(tensorL1Bias, tensorBias);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
        }

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
        }

        uint32_t kL1Loop = CeilDiv<L1_TILE_K>(kBlockActual);
        uint32_t kL1ScaleLoop = CeilDiv<L1_SCALE_FACTOR_K>(kL1Loop);
        uint32_t l1ScaleTileK = L1_TILE_K * L1_SCALE_FACTOR_K;
        for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; kL1Idx++) {
            uint32_t l1AListIdNext = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
            uint32_t l1BListIdNext = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;
            uint32_t l1MxScaleAListIdNext = (l1MxScaleAListId + 1 < L1A_STAGES) ? (l1MxScaleAListId + 1) : 0;
            uint32_t l1MxScaleBListIdNext = (l1MxScaleBListId + 1 < L1B_STAGES) ? (l1MxScaleBListId + 1) : 0;
            uint32_t kL1ActualNext{0};

            if (kL1Idx < kL1Loop - 1) {
                uint32_t kL1IdxNext = kL1Idx + 1;
                kL1ActualNext = (kL1IdxNext < kL1Loop - 1) ? L1_TILE_K : (kBlockActual - kL1IdxNext * L1_TILE_K);

                auto tensorL1ANext =
                    tla::MakeTensor(l1ATensorList[l1AListIdNext], L1A_LAYOUT, Catlass::Arch::PositionL1{});
                auto tensorL1BNext =
                    tla::MakeTensor(l1BTensorList[l1BListIdNext], L1B_LAYOUT, Catlass::Arch::PositionL1{});
                auto tensorTileANext = GetGmTileA(
                    gmAStage, blockMOffset, kL1IdxNext * L1_TILE_K, mBlockActual, kL1ActualNext, commRows, localK,
                    alignedLocalK);
                auto tensorTileBNext = GetTile(
                    tensorB, tla::MakeCoord(kL1IdxNext * L1_TILE_K, 0), tla::MakeShape(kL1ActualNext, nBlockActual));

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
                CopyGmTileA(copyGmToL1A, tensorL1ANext, tensorTileANext, l1AListIdNext);
                InitZeroInL1A(tensorL1ANext, tla::MakeShape(mL1Actual, kL1ActualNext));

                if (kL1IdxNext % L1_SCALE_FACTOR_K == 0) {
                    uint32_t kL1ScaleIdxNext = kL1IdxNext / L1_SCALE_FACTOR_K;
                    uint32_t kL1MxScaleActualNext = (kL1ScaleIdxNext < kL1ScaleLoop - 1) ?
                                                        l1ScaleTileK :
                                                        (kBlockActual - kL1ScaleIdxNext * l1ScaleTileK);
                    auto tensorL1MxScaleANext = tla::MakeTensor(
                        l1MxScaleATensorList[l1MxScaleAListIdNext], L1SCALEA_LAYOUT, Catlass::Arch::PositionL1{});
                    auto tensorTileMxScaleANext = GetGmTileMxScaleA(
                        gmMxScaleAStage, blockMOffset, kL1ScaleIdxNext * l1ScaleTileK / Catlass::MX_SCALE_GROUP_NUM,
                        mBlockActual, CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(kL1MxScaleActualNext), commRows, localScaleK,
                        alignedLocalScaleK);
                    copyGmToL1MxScaleA(tensorL1MxScaleANext, tensorTileMxScaleANext);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListIdNext]);

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
                CopyGmTileB(copyGmToL1B, tensorL1BNext, tensorTileBNext, l1BListIdNext);
                InitZeroInL1B(tensorL1BNext, tla::MakeShape(kL1ActualNext, nL1Actual));

                if (kL1IdxNext % L1_SCALE_FACTOR_K == 0) {
                    uint32_t kL1ScaleIdxNext = kL1IdxNext / L1_SCALE_FACTOR_K;
                    uint32_t kL1MxScaleActualNext = (kL1ScaleIdxNext < kL1ScaleLoop - 1) ?
                                                        l1ScaleTileK :
                                                        (kBlockActual - kL1ScaleIdxNext * l1ScaleTileK);
                    auto tensorL1MxScaleBNext = tla::MakeTensor(
                        l1MxScaleBTensorList[l1MxScaleBListIdNext], L1SCALEB_LAYOUT, Catlass::Arch::PositionL1{});
                    auto tensorTileMxScaleBNext = GetTile(
                        tensorMxScaleB, tla::MakeCoord(kL1ScaleIdxNext * l1ScaleTileK / Catlass::MX_SCALE_GROUP_NUM, 0),
                        tla::MakeShape(CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(kL1MxScaleActualNext), nBlockActual));
                    copyGmToL1MxScaleB(tensorL1MxScaleBNext, tensorTileMxScaleBNext);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListIdNext]);
            }

            tensorL1A = tla::MakeTensor(l1ATensorList[l1AListId], L1A_LAYOUT, Catlass::Arch::PositionL1{});
            tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], L1B_LAYOUT, Catlass::Arch::PositionL1{});
            tensorL1MxScaleA =
                tla::MakeTensor(l1MxScaleATensorList[l1MxScaleAListId], L1SCALEA_LAYOUT, Catlass::Arch::PositionL1{});
            tensorL1MxScaleB =
                tla::MakeTensor(l1MxScaleBTensorList[l1MxScaleBListId], L1SCALEB_LAYOUT, Catlass::Arch::PositionL1{});

            uint32_t l0kOffset = L1_TILE_K * (kL1Idx % L1_SCALE_FACTOR_K);
            uint32_t kL0Loop = CeilDiv<L0_TILE_K>(kL1Actual);
            for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? L0_TILE_K : (kL1Actual - kL0Idx * L0_TILE_K);
                kL0Actual = RoundUp<Catlass::MX_BASEK_FACTOR>(kL0Actual);

                auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mL1Actual, kL0Actual);
                auto tensorL0A = tla::MakeTensor(l0ATensorList[l0AListId], layoutAInL0, Catlass::Arch::PositionL0A{});
                auto tensorTileL1A =
                    GetTile(tensorL1A, tla::MakeCoord(0, kL0Idx * L0_TILE_K), tla::MakeShape(mL1Actual, kL0Actual));
                auto tensorTileL1MxScaleA = GetTile(
                    tensorL1MxScaleA, tla::MakeCoord(0, (l0kOffset + kL0Idx * L0_TILE_K) / Catlass::MX_SCALE_GROUP_NUM),
                    tla::MakeShape(mL1Actual, CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(kL0Actual)));

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                if (kL0Idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
                }
                copyL1ToL0A(tensorL0A, tensorTileL1A, tensorTileL1MxScaleA);
                if (kL0Idx == kL0Loop - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                }

                auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, nL1Actual);
                auto tensorL0B = tla::MakeTensor(l0BTensorList[l0BListId], layoutBInL0, Catlass::Arch::PositionL0B{});
                auto tensorTileL1B =
                    GetTile(tensorL1B, tla::MakeCoord(kL0Idx * L0_TILE_K, 0), tla::MakeShape(kL0Actual, nL1Actual));
                auto tensorTileL1MxScaleB = GetTile(
                    tensorL1MxScaleB, tla::MakeCoord((l0kOffset + kL0Idx * L0_TILE_K) / Catlass::MX_SCALE_GROUP_NUM, 0),
                    tla::MakeShape(CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(kL0Actual), nL1Actual));

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                if (kL0Idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
                }
                copyL1ToL0B(tensorL0B, tensorTileL1B, tensorTileL1MxScaleB);
                if (kL0Idx == kL0Loop - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
                }

                bool initC = initAccumulator && ((kL1Idx == 0) && (kL0Idx == 0));
                if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, Catlass::EmptyClass>) {
                    if (initC) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                        auto l1Bias = l1BiasTensor.template ReinterpretCast<ElementBias>();
                        auto tensorL1Bias = tla::MakeTensor(l1Bias, L1BIAS_LAYOUT, Catlass::Arch::PositionL1{});
                        auto tensorTileL1Bias = GetTile(tensorL1Bias, tla::MakeCoord(0), tla::MakeShape(nL1Actual));
                        copyL1ToBT(tensorL0Bias, tensorTileL1Bias);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                    }
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);

                uint8_t unitFlag = 0b00;
                if constexpr (ENABLE_UNIT_FLAG) {
                    if (finalMmad && (kL1Idx == kL1Loop - 1) && (kL0Idx == kL0Loop - 1)) {
                        unitFlag = 0b11;
                    } else {
                        unitFlag = 0b10;
                    }
                }

                if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, Catlass::EmptyClass>) {
                    if (initC) {
                        tileMmad(
                            tensorL0C, tensorL0A, tensorL0B, tensorL0Bias, mL1Actual, nL1Actual, kL0Actual, initC,
                            unitFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                    } else {
                        tileMmad(tensorL0C, tensorL0A, tensorL0B, mL1Actual, nL1Actual, kL0Actual, initC, unitFlag);
                    }
                } else {
                    tileMmad(tensorL0C, tensorL0A, tensorL0B, mL1Actual, nL1Actual, kL0Actual, initC, unitFlag);
                }

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
            }

            l1AListId = l1AListIdNext;
            l1BListId = l1BListIdNext;
            kL1Actual = kL1ActualNext;
            if (((kL1Idx + 1) % L1_SCALE_FACTOR_K == 0) || kL1Idx == kL1Loop - 1) {
                l1MxScaleAListId = l1MxScaleAListIdNext;
                l1MxScaleBListId = l1MxScaleBListIdNext;
            }
        }

        if constexpr (!ENABLE_UNIT_FLAG) {
            if (storeResult) {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                copyL0CToDst(tensorC, tensorL0C);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
                l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
            }
        } else {
            if (storeResult) {
                copyL0CToDst(tensorC, tensorL0C, 0b11);
            }
        }
    }

private:
    CATLASS_DEVICE auto GetGmTileA(
        AscendC::GlobalTensor<ElementA>& gmAStage, uint32_t blockMOffset, uint32_t kIndex, uint32_t mSize,
        uint32_t kSize, uint32_t commRows, uint32_t localK, int64_t alignedLocalK) const
    {
        uint32_t srcRank = kIndex / localK;
        uint32_t localCol = kIndex - srcRank * localK;
        int64_t rankOffset = static_cast<int64_t>(srcRank) * commRows * alignedLocalK;
        auto layoutRankA = tla::MakeLayout(
            tla::MakeShape(commRows, localK), tla::MakeStride(alignedLocalK, tla::Int<1>{}),
            tla::MakeShape(commRows, localK));
        auto tensorRankA = tla::MakeTensor(gmAStage[rankOffset], layoutRankA, Catlass::Arch::PositionGM{});
        return GetTile(tensorRankA, tla::MakeCoord(blockMOffset, localCol), tla::MakeShape(mSize, kSize));
    }

    CATLASS_DEVICE auto GetGmTileMxScaleA(
        AscendC::GlobalTensor<ElementMxScaleA>& gmMxScaleAStage, uint32_t blockMOffset, uint32_t scaleKIndex,
        uint32_t mSize, uint32_t scaleKSize, uint32_t commRows, uint32_t localScaleK, uint32_t alignedLocalScaleK) const
    {
        uint32_t srcRank = scaleKIndex / localScaleK;
        uint32_t localScaleCol = scaleKIndex - srcRank * localScaleK;
        int64_t rankOffset = static_cast<int64_t>(srcRank) * commRows * alignedLocalScaleK;
        auto layoutRankMxScaleA =
            tla::MakeMxScaleLayout<ElementMxScaleA, Catlass::layout::RowMajor, false>(commRows, localScaleK);
        auto tensorRankMxScaleA =
            tla::MakeTensor(gmMxScaleAStage[rankOffset], layoutRankMxScaleA, Catlass::Arch::PositionGM{});
        return GetTile(
            tensorRankMxScaleA, tla::MakeCoord(blockMOffset, localScaleCol), tla::MakeShape(mSize, scaleKSize));
    }

    template <class CopyGmToL1A, class TensorL1A, class TensorA>
    CATLASS_DEVICE void CopyGmTileA(
        CopyGmToL1A& copyGmToL1A, TensorL1A& tensorL1A, TensorA& tensorTileA, uint32_t listId)
    {
        if constexpr (ENABLE_L1_RESIDENT) {
            if (lastAddrA[listId] != tensorTileA.data().GetPhyAddr() ||
                tla::get<0>(tensorTileA.coord()) != lastCoordA[listId].row() ||
                tla::get<1>(tensorTileA.coord()) != lastCoordA[listId].column()) {
                copyGmToL1A(tensorL1A, tensorTileA);
                lastCoordA[listId] = MatrixCoord{tla::get<0>(tensorTileA.coord()), tla::get<1>(tensorTileA.coord())};
                lastAddrA[listId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementA>::PrimType*>(
                    tensorTileA.data().GetPhyAddr());
            }
        } else {
            copyGmToL1A(tensorL1A, tensorTileA);
        }
    }

    template <class CopyGmToL1B, class TensorL1B, class TensorB>
    CATLASS_DEVICE void CopyGmTileB(
        CopyGmToL1B& copyGmToL1B, TensorL1B& tensorL1B, TensorB& tensorTileB, uint32_t listId)
    {
        if constexpr (ENABLE_L1_RESIDENT) {
            if (lastAddrB[listId] != tensorTileB.data().GetPhyAddr() ||
                tla::get<0>(tensorTileB.coord()) != lastCoordB[listId].row() ||
                tla::get<1>(tensorTileB.coord()) != lastCoordB[listId].column()) {
                copyGmToL1B(tensorL1B, tensorTileB);
                lastCoordB[listId] = MatrixCoord{tla::get<0>(tensorTileB.coord()), tla::get<1>(tensorTileB.coord())};
                lastAddrB[listId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementB>::PrimType*>(
                    tensorTileB.data().GetPhyAddr());
            }
        } else {
            copyGmToL1B(tensorL1B, tensorTileB);
        }
    }

    template <class TensorL1, class Shape>
    CATLASS_DEVICE void InitZeroInL1A(TensorL1& tensorL1, Shape actualShape)
    {
        constexpr uint32_t ELE_NUM_PER_C0 =
            Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<typename TensorL1::Element>::value;
        const uint32_t mL1Actual = tla::get<0>(actualShape);
        const uint32_t kL1Actual = tla::get<1>(actualShape);

        uint32_t alignKL1 = RoundUp<Catlass::MX_BASEK_FACTOR>(kL1Actual);
        uint32_t validKL1 = kL1Actual;
        if constexpr (tla::detail::iszN<typename TensorL1::Element, typename TensorL1::Layout>::value) {
            validKL1 = RoundUp<ELE_NUM_PER_C0>(kL1Actual);
        }
        uint32_t padKL1 = alignKL1 - validKL1;
        if (padKL1 > 0) {
            AscendC::InitConstValueParams<uint16_t> initConstValueParams;
            if constexpr (tla::detail::iszN<typename TensorL1::Element, typename TensorL1::Layout>::value) {
                initConstValueParams.repeatTimes = 1;
                initConstValueParams.blockNum = mL1Actual;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = 0;
            } else {
                initConstValueParams.repeatTimes = CeilDiv<ELE_NUM_PER_C0>(mL1Actual);
                initConstValueParams.blockNum = padKL1;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = tla::get<0, 1>(tensorL1.stride()) / ELE_NUM_PER_C0 - padKL1;
            }

            auto offset = tensorL1.layout()(tla::MakeCoord(0, validKL1));
            auto srcUint16 = tensorL1.data()[offset].template ReinterpretCast<uint16_t>();
            InitConstValue(srcUint16, initConstValueParams);
        }
    }

    template <class TensorL1, class Shape>
    CATLASS_DEVICE void InitZeroInL1B(TensorL1& tensorL1, Shape actualShape)
    {
        constexpr uint32_t ELE_NUM_PER_C0 =
            Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<typename TensorL1::Element>::value;
        const uint32_t kL1Actual = tla::get<0>(actualShape);
        const uint32_t nL1Actual = tla::get<1>(actualShape);

        uint32_t alignKL1 = RoundUp<Catlass::MX_BASEK_FACTOR>(kL1Actual);
        uint32_t validKL1 = kL1Actual;
        if constexpr (tla::detail::isnZ<typename TensorL1::Element, typename TensorL1::Layout>::value) {
            validKL1 = RoundUp<ELE_NUM_PER_C0>(kL1Actual);
        }
        uint32_t padKL1 = alignKL1 - validKL1;
        if (padKL1 > 0) {
            AscendC::InitConstValueParams<uint16_t> initConstValueParams;
            if constexpr (tla::detail::iszN<typename TensorL1::Element, typename TensorL1::Layout>::value) {
                initConstValueParams.repeatTimes = CeilDiv<ELE_NUM_PER_C0>(nL1Actual);
                initConstValueParams.blockNum = padKL1;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = tla::get<1, 1>(tensorL1.stride()) / ELE_NUM_PER_C0 - padKL1;
            } else {
                initConstValueParams.repeatTimes = 1;
                initConstValueParams.blockNum = nL1Actual;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = 0;
            }

            auto offset = tensorL1.layout()(tla::MakeCoord(validKL1, 0));
            auto srcUint16 = tensorL1.data()[offset].template ReinterpretCast<uint16_t>();
            InitConstValue(srcUint16, initConstValueParams);
        }
    }

private:
    AscendC::LocalTensor<ElementA> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementMxScaleA> l1MxScaleATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementMxScaleB> l1MxScaleBTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementL0A> l0ATensorList[L0A_STAGES];
    AscendC::LocalTensor<ElementL0B> l0BTensorList[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];
    AscendC::LocalTensor<uint8_t> l1BiasTensor;
    AscendC::LocalTensor<ElementAccumulator> l0BiasTensor;

    int32_t l1AEventList[L1A_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    int32_t l0CEventList[L0C_STAGES];

    __gm__ typename AscendC::GlobalTensor<ElementA>::PrimType* lastAddrA[L1A_STAGES];
    __gm__ typename AscendC::GlobalTensor<ElementB>::PrimType* lastAddrB[L1B_STAGES];
    MatrixCoord lastCoordA[L1A_STAGES];
    MatrixCoord lastCoordB[L1B_STAGES];

    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
    uint32_t l0CListId{0};
    uint32_t l1MxScaleAListId{0};
    uint32_t l1MxScaleBListId{0};

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL1ToBT copyL1ToBT;
};

// Retain CATLASS buffers/events; map each K segment to its source-major receive slot.
template <class BaseMmad>
class BlockMmadBf16AllToAllMatmulKSplit : public BaseMmad {
public:
    using Base = BaseMmad;
    using ArchTag = typename Base::ArchTag;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using TileCopy = typename Base::TileCopy;
    using Base::L0_TILE_K;
    using Base::L1_TILE_K;
    using Base::L1A_LAYOUT;
    using Base::L1B_LAYOUT;
    static_assert(Base::ENABLE_UNIT_FLAG && !Base::HAS_BIAS && !Base::ENABLE_L1_RESIDENT);
    static_assert(Base::L1A_STAGES == 2 && Base::L1B_STAGES == 2);
    static_assert(std::is_same_v<typename Base::ElementAccumulator, float>);

    CATLASS_DEVICE explicit BlockMmadBf16AllToAllMatmulKSplit(Catlass::Arch::Resource<ArchTag>& resource)
        : Base(resource)
    {}

    template <class TensorB, class TensorC>
    CATLASS_DEVICE void operator()(
        AscendC::GlobalTensor<ElementA>& recvA, TensorB& tensorB, TensorC& tensorC, const Catlass::GemmCoord& shape,
        uint32_t row, uint32_t recvRows, uint32_t localK)
    {
        const uint32_t perSource = (localK + L1_TILE_K - 1) / L1_TILE_K;
        const uint32_t steps = (shape.k() / localK) * perSource;
        auto tensorL0C = tla::MakeTensor(
            this->l0CTensorList[0], tla::MakeLayoutL0C(shape.m(), shape.n()), Catlass::Arch::PositionL0C{});
        Load(recvA, tensorB, shape, row, recvRows, localK, perSource, 0, this->l1AListId);
        for (uint32_t step = 0; step < steps; ++step) {
            const uint32_t stage = this->l1AListId;
            const uint32_t next = stage ^ 1U;
            if (step + 1 < steps) {
                Load(recvA, tensorB, shape, row, recvRows, localK, perSource, step + 1, next);
            }
            const uint32_t offset = (step % perSource) * L1_TILE_K;
            const uint32_t kSize = ::Min(localK - offset, L1_TILE_K);
            const uint32_t innerSteps = (kSize + L0_TILE_K - 1) / L0_TILE_K;
            auto l1A = tla::MakeTensor(this->l1ATensorList[stage], L1A_LAYOUT, Catlass::Arch::PositionL1{});
            auto l1B = tla::MakeTensor(this->l1BTensorList[stage], L1B_LAYOUT, Catlass::Arch::PositionL1{});
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1AEventList[stage]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1BEventList[stage]);
            for (uint32_t inner = 0; inner < innerSteps; ++inner) {
                const uint32_t k = ::Min(kSize - inner * L0_TILE_K, L0_TILE_K);
                const uint32_t aId = this->l0AListId;
                const uint32_t bId = this->l0BListId;
                auto l0A = tla::MakeTensor(
                    this->l0ATensorList[aId], tla::MakeLayout<ElementA, typename Base::LayoutTagL0A>(shape.m(), k),
                    Catlass::Arch::PositionL0A{});
                auto l0B = tla::MakeTensor(
                    this->l0BTensorList[bId], tla::MakeLayout<ElementB, typename Base::LayoutTagL0B>(k, shape.n()),
                    Catlass::Arch::PositionL0B{});
                auto a = GetTile(l1A, tla::MakeCoord(0U, inner * L0_TILE_K), tla::MakeShape(shape.m(), k));
                auto b = GetTile(l1B, tla::MakeCoord(inner * L0_TILE_K, 0U), tla::MakeShape(k, shape.n()));
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(this->l0AEventList[aId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(this->l0BEventList[bId]);
                this->copyL1ToL0A(l0A, a);
                this->copyL1ToL0B(l0B, b);
                if (inner + 1 == innerSteps) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1AEventList[stage]);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1BEventList[stage]);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
                const bool last = step + 1 == steps && inner + 1 == innerSteps;
                // Accumulate all source segments in FP32; round to BF16 only once on store.
                this->tileMmad(
                    tensorL0C, l0A, l0B, shape.m(), shape.n(), k, step == 0 && inner == 0, last ? 0b11 : 0b10);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(this->l0AEventList[aId]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(this->l0BEventList[bId]);
                this->l0AListId = (aId + 1) % Base::L0A_STAGES;
                this->l0BListId = (bId + 1) % Base::L0B_STAGES;
            }
            this->l1AListId = next;
            this->l1BListId = next;
        }
        typename TileCopy::template CopyL0CToDst<TensorC> copyOut;
        copyOut(tensorC, tensorL0C, 0b11);
    }

private:
    template <class TensorB>
    CATLASS_DEVICE void Load(
        AscendC::GlobalTensor<ElementA>& recvA, TensorB& tensorB, const Catlass::GemmCoord& shape, uint32_t row,
        uint32_t recvRows, uint32_t localK, uint32_t perSource, uint32_t step, uint32_t stage)
    {
        const uint32_t source = step / perSource;
        const uint32_t offset = (step % perSource) * L1_TILE_K;
        const uint32_t kSize = ::Min(localK - offset, L1_TILE_K);
        auto layoutA = tla::MakeLayout(tla::MakeShape(recvRows, localK), tla::MakeStride(localK, tla::Int<1>{}));
        auto sourceA = tla::MakeTensor(
            recvA[static_cast<uint64_t>(source) * recvRows * localK], layoutA, Catlass::Arch::PositionGM{});
        auto a = GetTile(sourceA, tla::MakeCoord(row, offset), tla::MakeShape(shape.m(), kSize));
        auto b = GetTile(tensorB, tla::MakeCoord(source * localK + offset, 0U), tla::MakeShape(kSize, shape.n()));
        auto l1A = tla::MakeTensor(this->l1ATensorList[stage], L1A_LAYOUT, Catlass::Arch::PositionL1{});
        auto l1B = tla::MakeTensor(this->l1BTensorList[stage], L1B_LAYOUT, Catlass::Arch::PositionL1{});
        typename TileCopy::template CopyGmToL1A<decltype(a)> copyA;
        typename TileCopy::template CopyGmToL1B<decltype(b)> copyB;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1AEventList[stage]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(this->l1BEventList[stage]);
        // ND2NZ handles the per-source tail without padding the public GM inputs.
        copyA(l1A, a);
        copyB(l1B, b);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1AEventList[stage]);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(this->l1BEventList[stage]);
    }
};

} // namespace Catccos::DGemm::Block

#endif // CATCCOS_DGEMM_BLOCK_MMAD_MX_ALLTOALL_MATMUL_K_SPLIT_HPP
