// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "Dialect/TlaDist/IR/TlaDistOps.h"

#include "Dialect/Tla/IR/TlaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "tla_dist/Ops.cpp.inc"

mlir::LogicalResult tla_dist::ShmemPtrOp::verify()
{
    auto src = getPtr().getType();
    auto dst = getRemote().getType();
    if (src.getAddrspace() != ::AddressSpace::gm || dst.getAddrspace() != ::AddressSpace::gm)
        return emitOpError("requires GM input and result pointers");
    if (src.getPointee() != dst.getPointee())
        return emitOpError("result pointee type must match the input pointee type");
    if (dst.getAlignment() != 1)
        return emitOpError("requires byte alignment for the result pointer");
    if (!getOperation()->getParentOfType<::tla::VectorOp>() || getOperation()->getParentOfType<::tla::VecFuncOp>())
        return emitOpError("requires the scalar body of a tla.vector region");
    return mlir::success();
}
