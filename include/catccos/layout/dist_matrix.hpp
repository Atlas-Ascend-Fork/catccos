
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_LAYOUT_SYMMETRIC_HPP
#define CATCCOS_LAYOUT_SYMMETRIC_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/layout/layout.hpp"

#include "catccos/catccos.hpp"
#include "catccos/dist_coord.hpp"

namespace Catccos::layout {

class DistRowMajor {
public:
    static constexpr int32_t RANK = 3;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = Catlass::Coord<RANK, Index>;
    using Stride = Catlass::Coord<RANK, LongIndex>;

    CATLASS_HOST_DEVICE constexpr DistRowMajor() = default;

    CATLASS_HOST_DEVICE constexpr DistRowMajor(Shape const& shape, Stride const& stride)
        : shape_(shape), stride_(stride)
    {}

    CATLASS_HOST_DEVICE constexpr DistRowMajor(
        Index rows, Index cols, Index ranks, LongIndex strideRows, LongIndex strideCols, LongIndex strideRanks)
        : shape_(Catlass::MakeCoord(rows, cols, ranks)),
          stride_(Catlass::MakeCoord(strideRows, strideCols, strideRanks))
    {}

    // 只传 shape，通过 shape 推断 stride 的构造函数
    CATLASS_HOST_DEVICE constexpr DistRowMajor(Shape const& shape)
        : shape_(shape),
          stride_(Catlass::MakeCoord<LongIndex>(shape[1], 1, static_cast<LongIndex>(shape[0]) * shape[1]))
    {}

    CATLASS_HOST_DEVICE constexpr DistRowMajor(Index rows, Index cols, Index ranks)
        : DistRowMajor(Catlass::MakeCoord(rows, cols, ranks))
    {}

    CATLASS_HOST_DEVICE constexpr DistRowMajor(Catlass::MatrixCoord const& shape, Index ranks)
        : DistRowMajor(shape.row(), shape.column(), ranks)
    {}

    // 只传 shape，通过 shape 推断出 512B 对齐的 stride 的构造函数
    template <class Element>
    CATLASS_HOST_DEVICE constexpr static DistRowMajor MakeAlignedLayout(Shape const& shape)
    {
        constexpr auto ELE_NUM_PER_FRACTAL =
            Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;
        LongIndex alignedCols = RoundUp<ELE_NUM_PER_FRACTAL>(shape[1]);
        return {shape, Catlass::MakeCoord<LongIndex>(alignedCols, 1, shape[0] * alignedCols)};
    }

    template <class Element>
    CATLASS_HOST_DEVICE constexpr static DistRowMajor MakeAlignedLayout(Index rows, Index cols, Index ranks)
    {
        return MakeAlignedLayout<Element>(Catlass::MakeCoord(rows, cols, ranks));
    }

    template <class Element>
    CATLASS_HOST_DEVICE constexpr static DistRowMajor MakeAlignedLayout(Catlass::MatrixCoord const& shape, Index ranks)
    {
        return MakeAlignedLayout<Element>(shape.row(), shape.column(), ranks);
    }

    CATLASS_HOST_DEVICE constexpr LongIndex GetOffset(DistMatrixCoord const& coord) const
    {
        return Dot(coord, stride_);
    }

    CATLASS_HOST_DEVICE constexpr DistRowMajor GetTileLayout(DistMatrixCoord const& shape) const
    {
        return {shape, stride_};
    }

    CATLASS_HOST_DEVICE
    Catlass::layout::RowMajor GetTileLayout(Catlass::MatrixCoord const& shape) const
    {
        return {shape, Catlass::MakeCoord(stride_[0], stride_[1])};
    }

    CATLASS_HOST_DEVICE
    Shape shape() const { return shape_; }

    CATLASS_HOST_DEVICE
    Shape& shape() { return shape_; }

    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const { return shape_[idx]; }

    CATLASS_HOST_DEVICE
    typename Shape::Index& shape(int idx) { return shape_[idx]; }

    CATLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CATLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const { return stride_[idx]; }

    CATLASS_HOST_DEVICE
    typename Stride::Index& stride(int idx) { return stride_[idx]; }

private:
    Shape shape_{0};
    Stride stride_{0};
};

// 这个函数应该作为每个layout的成员函数，后续应该在catlass里修改
template <>
CATLASS_HOST_DEVICE constexpr int64_t Capacity<DistRowMajor>(DistRowMajor const& layout)
{
    return layout.shape(2) * layout.stride(2);
}

} // namespace Catccos::layout

#endif
