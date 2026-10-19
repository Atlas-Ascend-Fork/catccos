
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_BLOCK_RDMA_COPY_HPP
#define CATCCOS_COMM_BLOCK_RDMA_COPY_HPP

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/detail/remote_copy_type.hpp"

// from catlass
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"

// from shmem
#include "shmem.h"

namespace Catccos::Comm::Block {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

template <uint32_t UB_STAGES_, class SrcType_, class DstType_, class TileRemoteCopy_>
class CommBlock<AtlasA2CommRdmaCopy<UB_STAGES_>, SrcType_, DstType_, TileRemoteCopy_> {
public:
    // Type aliases
    using DispatchPolicy = AtlasA2CommRdmaCopy<UB_STAGES_>;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;

    using TileRemoteCopy = TileRemoteCopy_;

    CATLASS_DEVICE
    CommBlock() = default;

    CATLASS_DEVICE
    ~CommBlock() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementSrc> const& gmSrc, LayoutSrc const& layoutSrc,
        AscendC::GlobalTensor<ElementDst> const& gmDst, LayoutDst const& layoutDst,
        MatrixCoord const& actualCommBlockShape, uint32_t rankIdx)
    {
        if (actualCommBlockShape.row() == 0) {
            return;
        }

        tileRemoteCopy(
            gmDst, layoutDst, gmSrc, layoutSrc, actualCommBlockShape, ubSList[ubListId], copyEventIdList[ubListId],
            rankIdx);
    }

private:
    AscendC::LocalTensor<ElementDst> ubSList[UB_STAGES]{};
    uint32_t copyEventIdList[UB_STAGES]{};
    uint32_t ubListId{0};
    TileRemoteCopy tileRemoteCopy;
};

} // namespace Catccos::Comm::Block

#endif // CATCCOS_COMM_BLOCK_RDMA_COPY_HPP
