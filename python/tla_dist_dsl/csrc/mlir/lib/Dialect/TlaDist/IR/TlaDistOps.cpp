// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "Dialect/TlaDist/IR/TlaDistOps.h"

#include "Dialect/Tla/IR/TlaOps.h"
#include "Passes/TlaTensorDescriptor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Matchers.h"

#include "tla_dist/Enums.cpp.inc"

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

namespace {
mlir::LogicalResult verifySlot(mlir::Operation* op, mlir::Value slot, mlir::Value value, bool signal)
{
    auto info = ::tla::decodeTensorTypeInfo(slot.getType());
    if (mlir::failed(info) || info->addressSpace != "gm" || !::tla::isLinearLayout(info->layoutTag))
        return op->emitOpError("requires a linear symmetric GM tensor slot");
    // Tensor shape is decoded before descriptor lowering. Never accept a dynamic extent.
    auto raw = ::tla::parseTensorInfo(slot.getType());
    if (mlir::failed(raw) || raw->shape.empty() || raw->shape.size() > 2 ||
        llvm::any_of(raw->shape, [](int64_t n) { return n != 1; }))
        return op->emitOpError("requires a provably single-slot shape (1,) or (1,1)");
    auto dtype = raw->elementType;
    if (dtype != value.getType())
        return op->emitOpError("threshold dtype must match counter dtype");
    auto integer = mlir::dyn_cast<mlir::IntegerType>(dtype);
    bool valid = dtype.isF32() || (integer && (integer.getWidth() == 8 || integer.getWidth() == 16 ||
        integer.getWidth() == 32 || integer.getWidth() == 64));
    if (!valid || (signal && (!dtype.isSignlessInteger(32))))
        return op->emitOpError(signal ? "signal requires Int32" : "unsupported wait counter dtype");
    mlir::APFloat constant(0.0f);
    if (dtype.isF32() && mlir::matchPattern(value, mlir::m_ConstantFloat(&constant)) && constant.isNaN())
        return op->emitOpError("NaN thresholds are unsupported");
    return mlir::success();
}
}

mlir::LogicalResult tla_dist::SignalOp::verify() { return verifySlot(*this, getFlag(), getValue(), true); }
mlir::LogicalResult tla_dist::WaitUntilOp::verify() { return verifySlot(*this, getCounter(), getValue(), false); }
