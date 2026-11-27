// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "Dialect/TlaDist/IR/TlaDistDialect.h"
#include "Dialect/TlaDist/IR/TlaDistOps.h"
#include "Passes/TlaDistPasses.h"

#include "Passes/TlaTensorDescriptor.h"
#include "Passes/TlaTensorToMemref.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <set>
#include <string>
#include <tuple>
#include <optional>

using namespace mlir;

namespace tla_dist {
namespace {

bool isInsideVectorRegion(Operation* op)
{
    for (Operation* parent = op ? op->getParentOp() : nullptr; parent; parent = parent->getParentOp()) {
        if (parent->getName().getStringRef() == "tla.vector")
            return true;
    }
    return false;
}

void annotateAivRuntimeCall(func::FuncOp function)
{
    MLIRContext* context = function.getContext();
    function->setAttr(hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE), UnitAttr::get(context));
    function->setAttr(hivm::TFuncCoreTypeAttr::name, hivm::TFuncCoreTypeAttr::get(context, hivm::TFuncCoreType::AIV));
    function->setAttr("llvm.emit_c_interface", UnitAttr::get(context));
}

func::FuncOp getOrCreateAivRuntimeCall(
    ModuleOp module, StringRef name, ArrayRef<Type> operandTypes, ArrayRef<Type> resultTypes = {})
{
    func::FuncOp function = ::tla::getOrCreateRuntimeCall(module, name, operandTypes, resultTypes);
    annotateAivRuntimeCall(function);
    return function;
}

SmallVector<Value, 12> buildDescriptorPayload(OpBuilder& builder, Location loc, const ::tla::TensorDescriptor& desc)
{
    return {
        ::tla::castValueToI64(builder, loc, desc.shape[0]),
        ::tla::castValueToI64(builder, loc, desc.shape[1]),
        ::tla::castValueToI64(builder, loc, desc.shape[2]),
        ::tla::castValueToI64(builder, loc, desc.shape[3]),
        ::tla::castValueToI64(builder, loc, desc.stride[0]),
        ::tla::castValueToI64(builder, loc, desc.stride[1]),
        ::tla::castValueToI64(builder, loc, desc.stride[2]),
        ::tla::castValueToI64(builder, loc, desc.stride[3]),
        ::tla::castValueToI64(builder, loc, desc.coord[0]),
        ::tla::castValueToI64(builder, loc, desc.coord[1]),
        ::tla::castValueToI64(builder, loc, desc.originShape[0]),
        ::tla::castValueToI64(builder, loc, desc.originShape[1]),
    };
}

// Dynamic GM kernel arguments use a four-slot memref ABI, whereas linear
// pointer-backed GM views and the copy/SHMEM C-interface templates use rank 2.
// Build a private descriptor for a copy operand. Other users retain the
// original tensor descriptor and its base representation.
FailureOr<Value> normalizeLinearGmCopyOperand(Value tensor, PatternRewriter& rewriter)
{
    auto operation = tensor.getDefiningOp<::tla::TensorDescOp>();
    if (!operation)
        return failure();
    Value base = operation.getBase();
    auto baseType = dyn_cast<MemRefType>(base.getType());
    if (!baseType || baseType.getRank() != 4)
        return failure();
    // Only a logical rank-2 tensor maps directly to the first two ABI dimensions.
    SmallVector<int64_t, 2> coordLeaves;
    if (failed(::tla::getTlaIndexTreeLeaves(operation.getResult().getType().getCoord().getTree(), coordLeaves)) ||
        coordLeaves.size() != 2)
        return failure();
    auto info = ::tla::decodeTensorTypeInfo(operation.getResult().getType());
    if (failed(info) || info->addressSpace != "gm" || info->rank != 2 || !::tla::isLinearLayout(info->layoutTag))
        return failure();

    auto metadata = rewriter.create<memref::ExtractStridedMetadataOp>(operation.getLoc(), base);
    auto layout = StridedLayoutAttr::get(
        rewriter.getContext(), ShapedType::kDynamic, ArrayRef<int64_t>{ShapedType::kDynamic, ShapedType::kDynamic});
    auto runtimeType = MemRefType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, baseType.getElementType(), layout, baseType.getMemorySpace());
    // Preserve the storage offset and root layout. Tile coordinates remain in
    // tensor_desc and must not be applied here (the runtime applies them once).
    Value normalized = rewriter
                           .create<memref::ReinterpretCastOp>(
                               operation.getLoc(), runtimeType, base, metadata.getOffset(),
                               ValueRange{metadata.getSizes()[0], metadata.getSizes()[1]},
                               ValueRange{metadata.getStrides()[0], metadata.getStrides()[1]})
                           .getResult();
    Operation* copyDescriptor = rewriter.clone(*operation.getOperation());
    rewriter.modifyOpInPlace(copyDescriptor, [&] { copyDescriptor->setOperand(0, normalized); });
    return copyDescriptor->getResult(0);
}

template <typename CopyOp>
class NormalizeLinearGmCopyPattern final : public OpRewritePattern<CopyOp> {
public:
    using OpRewritePattern<CopyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CopyOp operation, PatternRewriter& rewriter) const override
    {
        bool changed = false;
        // Local and remote copies both store dst/src in operands 0/1. Leave
        // copy parameters and remote peer operands untouched.
        for (unsigned index = 0; index < 2; ++index) {
            auto tensor = normalizeLinearGmCopyOperand(operation->getOperand(index), rewriter);
            if (failed(tensor))
                continue;
            rewriter.modifyOpInPlace(operation, [&] { operation->setOperand(index, *tensor); });
            changed = true;
        }
        return success(changed);
    }
};

// SHMEM uses the scratch as raw storage. Pass a byte memref whose start is
// the actual tile address, including the offset from its backing allocation.
FailureOr<Value> buildControlBuffer(
    Operation* op, const ::tla::TensorDescriptor& desc, int64_t bytes, PatternRewriter& rewriter)
{
    auto constant = [](Value v) -> int64_t {
        llvm::APInt value;
        return matchPattern(v, m_ConstantInt(&value)) ? value.getSExtValue() : -1;
    };
    auto capacity = ::tla::getStaticAllocationElementCount(desc.base);
    if (failed(capacity) || desc.addrspace != "ub" || desc.layoutTag != ::LayoutTag::RowMajor ||
        !desc.elementType.isIntOrFloat() || cast<::tla::PtrType>(desc.base.getType()).getAlignment() % 32 != 0)
        return op->emitOpError("requires a statically sized, 32-byte aligned UB allocation"), failure();
    int64_t bits = desc.elementType.getIntOrFloatBitWidth();
    int64_t row = constant(desc.coord[0]), col = constant(desc.coord[1]);
    int64_t rows = constant(desc.originShape[0]), cols = constant(desc.originShape[1]);
    int64_t stride = constant(desc.stride[0]);
    if (bits < 8 || bits % 8 || row < 0 || col < 0 || rows <= 0 || cols <= 0 || stride < cols ||
        constant(desc.stride[1]) != 1)
        return op->emitOpError("requires a statically known contiguous UB scratch region"), failure();
    int64_t elementBytes = bits / 8;
    int64_t requiredElements = (bytes + elementBytes - 1) / elementBytes;
    int64_t requiredRows = (requiredElements - 1) / cols + 1;
    if (rows < requiredRows || (requiredRows > 1 && stride != cols))
        return op->emitOpError("UB scratch tile has insufficient contiguous space"), failure();

    // Check the element offset before multiplying row by stride.
    if (col > *capacity || row > (*capacity - col) / stride)
        return op->emitOpError("UB scratch offset exceeds allocation"), failure();
    int64_t offsetElements = row * stride + col;
    if (*capacity - offsetElements < requiredElements)
        return op->emitOpError("UB scratch exceeds remaining allocation capacity"), failure();
    int64_t offsetBytes = offsetElements * elementBytes;
    if (offsetBytes % 32 != 0)
        return op->emitOpError("UB scratch tile start must be 32-byte aligned"), failure();

    Location loc = op->getLoc();
    auto ptr = desc.base.getDefiningOp<::tla::IntToPtrOp>();
    Value address = ::tla::castValueToI64(rewriter, loc, ptr.getAddr());
    if (offsetBytes)
        address =
            rewriter.create<arith::AddIOp>(loc, address, rewriter.create<arith::ConstantIntOp>(loc, offsetBytes, 64));
    auto space = cast<MemRefType>(desc.bridgedBaseMemrefType).getMemorySpace();
    auto type = MemRefType::get({bytes}, rewriter.getI8Type(), AffineMap(), space);
    Value buffer = rewriter.create<hivm::PointerCastOp>(loc, type, address).getResult();
    // Match the descriptor ABI of the device helper. A fully static memref
    // would let HIVMC select its bare-pointer calling convention for quiet.
    return ::tla::castMemrefToType(rewriter, loc, buffer, ::tla::getDynamicStridedMemrefType(type));
}

bool usesAutoSync(Operation* op)
{
    for (auto* parent = op->getParentOp(); parent; parent = parent->getParentOp())
        if (parent->hasAttr("tla.auto_sync"))
            return true;
    return false;
}

class LowerRemoteCopyPattern final : public OpRewritePattern<RemoteCopyOp> {
public:
    LowerRemoteCopyPattern(MLIRContext* context, DenseMap<Value, Value>& baseMemrefCache)
        : OpRewritePattern<RemoteCopyOp>(context, /*benefit=*/3), baseMemrefCache(baseMemrefCache)
    {}

    LogicalResult matchAndRewrite(RemoteCopyOp operation, PatternRewriter& rewriter) const override
    {
        if (!isInsideVectorRegion(operation.getOperation()))
            return operation->emitOpError("must be nested inside a tla.vector region"), failure();
        StringRef mode = operation.getMode();
        StringRef transport = operation.getTransport();
#if !TLA_DIST_ENABLE_RDMA
        if (transport == "rdma")
            return operation.emitOpError("RDMA is not enabled in this build; "
                                         "enable RDMA in SHMEM and rebuild CATCCOS DSL against that SHMEM build"),
                   failure();
#endif
        if (mode != "get" && mode != "put")
            return operation->emitOpError("mode must be 'put' or 'get'"), failure();
        if (transport != "mte" && transport != "rdma" && transport != "udma" && transport != "sdma")
            return operation->emitOpError("transport must be 'mte', 'rdma', 'udma', or 'sdma'"), failure();
        auto wqeAttr = operation->getAttrOfType<StringAttr>("wqe_pipe");
        StringRef wqePipe = wqeAttr ? wqeAttr.getValue() : StringRef("mte3");
        if (wqeAttr && transport != "udma")
            return operation->emitOpError("wqe_pipe is only supported for UDMA"), failure();
        if (transport == "udma" && wqePipe != "mte3" && wqePipe != "s")
            return operation->emitOpError("wqe_pipe must be 'mte3' or 's'"), failure();
        Value dstTile = operation.getDst();
        Value srcTile = operation.getSrc();
        Value peer = operation.getPeer();

        auto dstDescOp = dstTile.getDefiningOp<::tla::TensorDescOp>();
        auto srcDescOp = srcTile.getDefiningOp<::tla::TensorDescOp>();
        if (!dstDescOp || !srcDescOp)
            return rewriter.notifyMatchFailure(
                operation.getOperation(), "expected tensor_desc operands from tla-lower-tensor-desc");
        FailureOr<::tla::TensorDescriptor> dstDescOr = ::tla::descriptorFromTensorDescOp(dstDescOp);
        FailureOr<::tla::TensorDescriptor> srcDescOr = ::tla::descriptorFromTensorDescOp(srcDescOp);
        if (failed(dstDescOr) || failed(srcDescOr))
            return rewriter.notifyMatchFailure(operation.getOperation(), "failed to decode tensor descriptors");
        const ::tla::TensorDescriptor& dstDesc = *dstDescOr;
        const ::tla::TensorDescriptor& srcDesc = *srcDescOr;
        bool gmToGm = srcDesc.addrspace == "gm" && dstDesc.addrspace == "gm";
        bool gmToUb = srcDesc.addrspace == "gm" && dstDesc.addrspace == "ub";
        bool ubToGm = srcDesc.addrspace == "ub" && dstDesc.addrspace == "gm";
        if (!gmToGm && !gmToUb && !ubToGm)
            return operation->emitOpError("requires GM -> GM, GM -> UB, or UB -> GM"), failure();
        if (!gmToGm && transport != "mte")
            return operation->emitOpError("GM <-> UB is only supported for MTE"), failure();
        if (gmToUb && mode != "get")
            return operation->emitOpError("GM -> UB requires mode='get'"), failure();
        if (ubToGm && mode != "put")
            return operation->emitOpError("UB -> GM requires mode='put'"), failure();
        if (srcDesc.layoutTag != ::LayoutTag::RowMajor || dstDesc.layoutTag != ::LayoutTag::RowMajor)
            return operation->emitOpError("requires row-major src and dst"), failure();
        auto hasUnitColumnStride = [](const ::tla::TensorDescriptor& desc) {
            llvm::APInt stride;
            return matchPattern(desc.stride[1], m_ConstantInt(&stride)) && stride == 1;
        };
        if (!hasUnitColumnStride(srcDesc))
            return operation->emitOpError("requires src column stride of 1; dynamic strides must be provably 1"),
                   failure();
        if (!hasUnitColumnStride(dstDesc))
            return operation->emitOpError("requires dst column stride of 1; dynamic strides must be provably 1"),
                   failure();
        if (srcDesc.elementType != dstDesc.elementType)
            return operation->emitOpError("requires identical src/dst element types"), failure();
        if (!srcDesc.elementType.isF32())
            return operation->emitOpError("currently supports only f32 tensors"), failure();
        bool enableAtomicAdd = operation->hasAttr("atomic_add");
        if (enableAtomicAdd && (transport != "mte" || dstDesc.addrspace != "gm"))
            return operation->emitOpError("atomic ADD requires MTE and a GM destination"), failure();

        bool needsScratch = gmToGm && !(transport == "udma" && wqePipe == "s");
        Value ubTile = operation.getUbTensor();
        IntegerAttr eventId = operation.getEventIdAttr();
        std::optional<::tla::TensorDescriptor> ubDesc;
        if (needsScratch) {
            if (!ubTile || !eventId)
                return operation->emitOpError("requires both a UB scratch tensor and event_id"), failure();
            auto ubDescOp = ubTile.getDefiningOp<::tla::TensorDescOp>();
            if (!ubDescOp)
                return rewriter.notifyMatchFailure(operation, "expected UB tensor_desc operand");
            auto decoded = ::tla::descriptorFromTensorDescOp(ubDescOp);
            if (failed(decoded))
                return operation->emitOpError("failed to decode UB descriptor"), failure();
            ubDesc = *decoded;
            if (transport == "mte" && ubDesc->addrspace != "ub")
                return operation->emitOpError("requires its scratch tensor in UB"), failure();
            if (transport == "mte" && ubDesc->elementType != srcDesc.elementType)
                return operation->emitOpError("requires identical src and UB scratch element types"), failure();
        } else if (ubTile || eventId) {
            return operation->emitOpError("this route does not accept ub_tensor or event_id"), failure();
        }
        if (transport == "sdma")
            return operation->emitOpError() << transport << " device lowering is not implemented", failure();

        std::string route = gmToGm ? "gm_row_major_to_gm_row_major" :
                            gmToUb ? "gm_row_major_to_ub_row_major" :
                                     "ub_row_major_to_gm_row_major";
        std::string calleeName = "shmem_remote_" + mode.str() + "_" + route + "_float";

        if (transport != "mte") {
            calleeName =
                "shmem_" + transport.str() + "_" + (transport == "udma" ? wqePipe.str() + "_" : "") + mode.str();
        }

        auto buildRuntimeMemref = [&](const ::tla::TensorDescriptor& descriptor) -> FailureOr<Value> {
            FailureOr<Value> base = ::tla::getOrMaterializeDescriptorBaseMemref(
                rewriter, operation->getLoc(), descriptor, operation.getOperation(), baseMemrefCache);
            if (failed(base))
                return failure();
            auto baseType = dyn_cast<MemRefType>((*base).getType());
            if (!baseType)
                return failure();
            MemRefType runtimeType = ::tla::getDynamicStridedMemrefType(baseType);
            return ::tla::castMemrefToType(rewriter, operation->getLoc(), *base, runtimeType);
        };

        FailureOr<Value> srcMemref = buildRuntimeMemref(srcDesc);
        FailureOr<Value> dstMemref = buildRuntimeMemref(dstDesc);
        if (failed(srcMemref) || failed(dstMemref))
            return operation->emitOpError("failed to materialize runtime memrefs"), failure();

        SmallVector<Value, 40> operands = {*srcMemref, *dstMemref};
        if (ubDesc) {
            auto ubMemref = transport == "mte" ?
                                buildRuntimeMemref(*ubDesc) :
                                buildControlBuffer(operation, *ubDesc, transport == "rdma" ? 128 : 256, rewriter);
            if (failed(ubMemref))
                return operation->emitOpError("failed to materialize the UB scratch runtime memref"), failure();
            operands.push_back(*ubMemref);
        }
        SmallVector<Value, 24> payload = buildDescriptorPayload(rewriter, operation->getLoc(), srcDesc);
        auto dstPayload = buildDescriptorPayload(rewriter, operation->getLoc(), dstDesc);
        payload.append(dstPayload.begin(), dstPayload.end());
        operands.append(payload.begin(), payload.end());
        if (transport == "mte" && ubDesc) {
            auto ubPayload = buildDescriptorPayload(rewriter, operation->getLoc(), *ubDesc);
            operands.append(ubPayload.begin(), ubPayload.end());
        }
        operands.push_back(peer);
        if (transport == "rdma" || transport == "udma")
            operands.push_back(operation.getQpIdx());
        if (needsScratch)
            operands.push_back(rewriter.create<arith::ConstantIntOp>(operation.getLoc(), eventId.getInt(), 32));
        SmallVector<Type, 40> operandTypes;
        for (Value value : operands)
            operandTypes.push_back(value.getType());
        func::FuncOp callee =
            getOrCreateAivRuntimeCall(operation->getParentOfType<ModuleOp>(), calleeName, operandTypes);

        // MTE GET/PUT accumulates at the local/remote destination respectively.
        Type dstType = cast<MemRefType>((*dstMemref).getType()).getElementType();
        if (enableAtomicAdd) {
            auto modeAttr = hivm::AtomicKindAttr::get(rewriter.getContext(), hivm::AtomicKind::ADD);
            rewriter.create<hivm::SetAtomicOp>(operation->getLoc(), modeAttr, TypeAttr::get(dstType));
        }
        rewriter.create<func::CallOp>(operation->getLoc(), callee, operands);
        if (enableAtomicAdd) {
            auto modeAttr = hivm::AtomicKindAttr::get(rewriter.getContext(), hivm::AtomicKind::NONE);
            rewriter.create<hivm::SetAtomicOp>(operation->getLoc(), modeAttr, TypeAttr::get(dstType));
        }
        rewriter.eraseOp(operation);
        return success();
    }

private:
    DenseMap<Value, Value>& baseMemrefCache;
};

class LowerShmemPtrPass final : public PassWrapper<LowerShmemPtrPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerShmemPtrPass)

    StringRef getArgument() const override { return "tla_dist-lower-shmem-ptr"; }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<::tla::TlaDialect, func::FuncDialect, hivm::HIVMDialect, hacc::HACCDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        SmallVector<ShmemPtrOp> translations;
        module.walk([&](ShmemPtrOp op) { translations.push_back(op); });
        if (translations.empty())
            return;

        IRRewriter rewriter(&getContext());
        Type i64 = rewriter.getI64Type();
        auto callee = getOrCreateAivRuntimeCall(module, "shmem_ptr_addr", {i64, rewriter.getI32Type()}, {i64});
        for (ShmemPtrOp op : translations) {
            rewriter.setInsertionPoint(op);
            Location loc = op.getLoc();
            Value address = rewriter.create<::tla::PtrToIntOp>(loc, i64, op.getPtr()).getResult();
            Value remote = rewriter.create<func::CallOp>(loc, callee, ValueRange{address, op.getPeer()}).getResult(0);
            rewriter.replaceOpWithNewOp<::tla::IntToPtrOp>(op, op.getRemote().getType(), remote);
        }
    }
};

class LowerQuietPattern final : public OpRewritePattern<QuietOp> {
public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(QuietOp op, PatternRewriter& rewriter) const override
    {
        if (!isInsideVectorRegion(op))
            return op.emitOpError("requires a tla.vector region with explicit synchronization"), failure();
        StringRef transport = op.getTransport();
#if !TLA_DIST_ENABLE_RDMA
        if (transport == "rdma")
            return op.emitOpError("RDMA is not enabled in this build; "
                                  "enable RDMA in SHMEM and rebuild CATCCOS DSL against that SHMEM build"),
                   failure();
#endif
        SmallVector<Value> args{op.getPeer(), op.getQpIdx()};
        if (transport == "rdma") {
            if (!op.getUbTensor() || !op.getEventIdAttr())
                return op.emitOpError("requires both a UB scratch tensor and event_id"), failure();
            auto tile = op.getUbTensor().getDefiningOp<::tla::TensorDescOp>();
            if (!tile)
                return rewriter.notifyMatchFailure(op, "expected UB tensor_desc operand");
            auto desc = ::tla::descriptorFromTensorDescOp(tile);
            if (failed(desc))
                return failure();
            auto buffer = buildControlBuffer(op, *desc, 128, rewriter);
            if (failed(buffer))
                return failure();
            args.push_back(*buffer);
            args.push_back(rewriter.create<arith::ConstantIntOp>(op.getLoc(), op.getEventIdAttr().getInt(), 32));
        } else if (transport != "udma" || op.getUbTensor() || op.getEventIdAttr()) {
            return op.emitOpError("expected rdma with both ub_tensor and event_id, or udma without either"), failure();
        }
        SmallVector<Type> types;
        for (Value arg : args)
            types.push_back(arg.getType());
        auto callee =
            getOrCreateAivRuntimeCall(op->getParentOfType<ModuleOp>(), "shmem_" + transport.str() + "_quiet", types);
        rewriter.create<func::CallOp>(op.getLoc(), callee, args);
        rewriter.eraseOp(op);
        return success();
    }
};

class LowerBarrierAllPattern final : public OpRewritePattern<BarrierAllOp> {
public:
    explicit LowerBarrierAllPattern(MLIRContext* context) : OpRewritePattern<BarrierAllOp>(context, /*benefit=*/3) {}

    LogicalResult matchAndRewrite(BarrierAllOp operation, PatternRewriter& rewriter) const override
    {
        if (!isInsideVectorRegion(operation.getOperation()))
            return operation->emitOpError("must be nested inside a tla.vector region"), failure();
        func::FuncOp callee =
            getOrCreateAivRuntimeCall(operation->getParentOfType<ModuleOp>(), "shmem_barrier_all_vec", {});
        rewriter.create<func::CallOp>(operation->getLoc(), callee, ValueRange{});
        rewriter.eraseOp(operation);
        return success();
    }
};

class ValidateCommAutoSyncPass final : public PassWrapper<ValidateCommAutoSyncPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ValidateCommAutoSyncPass)

    StringRef getArgument() const override { return "tla_dist-validate-comm-auto-sync"; }

    void runOnOperation() override
    {
        WalkResult result = getOperation().walk([&](Operation* operation) {
            StringRef name = operation->getName().getStringRef();
            auto transport = operation->getAttrOfType<StringAttr>("transport");
            bool needsExplicitSync = (name.starts_with("tla_dist.") && name != "tla_dist.remote_copy" &&
                                      name != "tla_dist.shmem_ptr" && name != "tla_dist.barrier_all") ||
                                     (name == "tla_dist.remote_copy" && transport && transport.getValue() != "mte");
            if (needsExplicitSync && usesAutoSync(operation)) {
                operation->emitOpError("RDMA/UDMA requires explicit synchronization");
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        if (result.wasInterrupted())
            signalPassFailure();
    }
};

class PrepareCommFlagsPass final : public PassWrapper<PrepareCommFlagsPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareCommFlagsPass)

    StringRef getArgument() const override { return "tla_dist-prepare-comm-flags"; }

    void getDependentDialects(DialectRegistry& registry) const override { registry.insert<::tla::TlaDialect>(); }

    void runOnOperation() override
    {
        bool defaults = false;
        getOperation().walk(
            [&](Operation* op) { defaults |= isa<SyncOp, BarrierOp, QuietDefaultOp, FenceOp, SignalOp>(op); });
        auto defaultDirection = [](::Pipe src, ::Pipe dst) {
            return (src == ::Pipe::scalar && dst == ::Pipe::mte3) || (src == ::Pipe::mte3 && dst == ::Pipe::scalar) ||
                   (src == ::Pipe::mte2 && dst == ::Pipe::scalar) || (src == ::Pipe::scalar && dst == ::Pipe::mte2) ||
                   (src == ::Pipe::mte2 && dst == ::Pipe::mte3) || (src == ::Pipe::mte3 && dst == ::Pipe::mte2);
        };
        bool conflict = false;
        if (defaults)
            getOperation().walk([&](::tla::FlagOp flag) {
                auto src = flag->getAttrOfType<::tla::PipeAttr>("src_pipe");
                auto dst = flag->getAttrOfType<::tla::PipeAttr>("dst_pipe");
                if (flag.getId() == 0 && src && dst && defaultDirection(src.getPipe(), dst.getPipe())) {
                    flag.emitOpError("event 0 in this pipe direction is reserved by SHMEM synchronization");
                    conflict = true;
                }
            });
        if (conflict) {
            signalPassFailure();
            return;
        }
        llvm::StringSet<> names;
        getOperation().walk([&](::tla::FlagOp flag) { names.insert(flag.getName()); });
        // Match CATLASS's module-wide allocation scope, deduplicating by pipe pair and id.
        std::set<std::tuple<::Pipe, ::Pipe, int64_t>> reserved;
        unsigned nextName = 0;
        getOperation().walk([&](Operation* op) {
            auto copy = dyn_cast<RemoteCopyOp>(op);
            auto quiet = dyn_cast<QuietOp>(op);
            bool defaultOp = isa<SyncOp, BarrierOp, QuietDefaultOp, FenceOp, SignalOp>(op);
            if (!copy && !quiet && !defaultOp)
                return;
            IntegerAttr eventId = defaultOp ? IntegerAttr::get(IntegerType::get(&getContext(), 32), 0) :
                                              (copy ? copy.getEventIdAttr() : quiet.getEventIdAttr());
            if (!eventId)
                return;
            StringRef transport = defaultOp ? "default" : (copy ? copy.getTransport() : quiet.getTransport());
            auto reserve = [&](::Pipe src, ::Pipe dst) {
                if (!reserved.emplace(src, dst, eventId.getInt()).second)
                    return;
                std::string name;
                do {
                    name = "__tla_dist_comm_flag_" + std::to_string(nextName++);
                } while (!names.insert(name).second);
                OpBuilder builder(op);
                builder.create<::tla::FlagOp>(
                    op->getLoc(), ::tla::FlagType::get(&getContext()), name, ::tla::PipeAttr::get(&getContext(), src),
                    ::tla::PipeAttr::get(&getContext(), dst), eventId.getInt());
            };
            if (defaultOp) {
                reserve(::Pipe::scalar, ::Pipe::mte3);
                reserve(::Pipe::mte3, ::Pipe::scalar);
                reserve(::Pipe::mte2, ::Pipe::scalar);
                reserve(::Pipe::scalar, ::Pipe::mte2);
                reserve(::Pipe::mte2, ::Pipe::mte3);
                reserve(::Pipe::mte3, ::Pipe::mte2);
            } else if (defaults && eventId.getInt() == 0) {
                op->emitOpError(
                    "event 0 for explicit communication conflicts with the SHMEM default synchronization domain");
                conflict = true;
            } else if (copy && transport == "mte") {
                reserve(::Pipe::mte2, ::Pipe::mte3);
            } else if (transport == "rdma") {
                reserve(::Pipe::scalar, ::Pipe::mte3);
                reserve(::Pipe::mte3, ::Pipe::scalar);
#if TLA_DIST_RDMA_BACKEND_HNS_1825
                reserve(::Pipe::mte2, ::Pipe::scalar);
#endif
            } else if (copy && transport == "udma") {
                auto pipe = copy->getAttrOfType<StringAttr>("wqe_pipe");
                if (!pipe || pipe.getValue() == "mte3") {
                    reserve(::Pipe::scalar, ::Pipe::mte3);
                    reserve(::Pipe::mte3, ::Pipe::scalar);
                }
            }
            // CATLASS reserves explicit ids before allocating automatic flags,
            // then erases this unused declaration without emitting set/wait.
        });
        if (conflict)
            signalPassFailure();
    }
};

// The slot ABI is an address plus an exactly typed threshold; it is independent
// of the copy ABI and does not inherit its Float32 restriction.
FailureOr<Value> materializeSlotAddress(Operation* op, Value tensor, OpBuilder& builder, DenseMap<Value, Value>& cache)
{
    auto source = tensor.getDefiningOp<::tla::TensorDescOp>();
    if (!source)
        return op->emitOpError("requires a materialized tensor descriptor"), failure();
    auto decoded = ::tla::descriptorFromTensorDescOp(source);
    if (failed(decoded) || decoded->addrspace != "gm" || !::tla::isLinearLayout(decoded->layoutTag))
        return op->emitOpError("requires a linear symmetric GM slot"), failure();
    auto& desc = *decoded;
    auto base = ::tla::getOrMaterializeDescriptorBaseMemref(builder, op->getLoc(), desc, op, cache);
    if (failed(base))
        return failure();
    auto metadata = builder.create<memref::ExtractStridedMetadataOp>(op->getLoc(), *base);
    auto i64 = [&](Value v) { return ::tla::castValueToI64(builder, op->getLoc(), v); };
    Location loc = op->getLoc();
    Value offset = i64(metadata.getOffset());
    for (unsigned dim = 0; dim != 2; ++dim) {
        Value delta = builder.create<arith::MulIOp>(loc, i64(desc.coord[dim]), i64(desc.stride[dim]));
        offset = builder.create<arith::AddIOp>(loc, offset, delta);
    }
    Value pointer = builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, metadata.getBaseBuffer());
    Value bytes = builder.create<arith::ConstantIntOp>(loc, desc.elementType.getIntOrFloatBitWidth() / 8, 64);
    Value displacement = builder.create<arith::MulIOp>(loc, offset, bytes);
    return builder.create<arith::AddIOp>(loc, i64(pointer), displacement).getResult();
}

bool isScopedSync(Operation* op) { return isa<LocalSyncOp, SyncOp, BarrierOp>(op); }
bool usesDefaultResources(Operation* op) { return isa<SyncOp, BarrierOp, QuietDefaultOp, FenceOp, SignalOp>(op); }

class LowerSyncPattern final : public RewritePattern {
public:
    LowerSyncPattern(MLIRContext* ctx, DenseMap<Value, Value>& cache)
        : RewritePattern(MatchAnyOpTypeTag(), 2, ctx), cache(cache)
    {}
    LogicalResult matchAndRewrite(Operation* op, PatternRewriter& rewriter) const override
    {
        if (!isa<LocalSyncOp, SyncOp, BarrierOp, QuietDefaultOp, FenceOp, SignalOp, WaitUntilOp>(op))
            return failure();
        auto function = op->getParentOfType<func::FuncOp>();
        auto core = function->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
        bool aic = core && core.getFuncCoreType() == hivm::TFuncCoreType::AIC;
        std::string suffix = aic ? "mix_aic" : "aiv";
        SmallVector<Value> args;
        std::string name;
        if (isScopedSync(op)) {
            auto scope = cast<CoreScopeAttr>(op->getAttr("scope")).getValue();
            if (scope == CoreScope::MIX)
                suffix = aic ? "mix_aic" : "mix_aiv";
            if (isa<LocalSyncOp>(op))
                name = "tla_dist_local_sync_" + suffix;
            else {
                name = std::string(isa<SyncOp>(op) ? "tla_dist_sync_" : "tla_dist_barrier_") + suffix;
                args.push_back(op->getOperand(0));
            }
        } else if (auto signal = dyn_cast<SignalOp>(op)) {
            auto address = materializeSlotAddress(op, signal.getFlag(), rewriter, cache);
            if (failed(address))
                return failure();
            int32_t code;
            switch (signal.getOp()) {
                case SignalKind::SET:
                    code = 0;
                    break;
                case SignalKind::ADD:
                    code = 1;
                    break;
            }
            args = {
                *address, signal.getValue(), rewriter.create<arith::ConstantIntOp>(op->getLoc(), code, 32),
                signal.getPeer()};
            name = "tla_dist_signal_i32";
        } else if (auto wait = dyn_cast<WaitUntilOp>(op)) {
            auto address = materializeSlotAddress(op, wait.getCounter(), rewriter, cache);
            if (failed(address))
                return failure();
            auto type = wait.getValue().getType();
            Value threshold = wait.getValue();
            if (type.isF32())
                name = "tla_dist_wait_f32";
            else {
                auto integer = cast<IntegerType>(type);
                name = std::string("tla_dist_wait_") + (integer.isUnsigned() ? "u" : "i") +
                       std::to_string(integer.getWidth());
                if (!integer.isSignless()) {
                    threshold = rewriter
                                    .create<UnrealizedConversionCastOp>(
                                        op->getLoc(), rewriter.getIntegerType(integer.getWidth()), threshold)
                                    .getResult(0);
                }
                if (integer.getWidth() < 64) {
                    if (integer.isUnsigned())
                        threshold = rewriter.create<arith::ExtUIOp>(op->getLoc(), rewriter.getI64Type(), threshold);
                    else
                        threshold = rewriter.create<arith::ExtSIOp>(op->getLoc(), rewriter.getI64Type(), threshold);
                }
            }
            int32_t code;
            switch (wait.getCmp()) {
                case CompareKind::EQ:
                    code = 0;
                    break;
                case CompareKind::NE:
                    code = 1;
                    break;
                case CompareKind::GT:
                    code = 2;
                    break;
                case CompareKind::GE:
                    code = 3;
                    break;
                case CompareKind::LT:
                    code = 4;
                    break;
                case CompareKind::LE:
                    code = 5;
                    break;
            }
            args = {*address, threshold, rewriter.create<arith::ConstantIntOp>(op->getLoc(), code, 32)};
        } else
            name = isa<FenceOp>(op) ? "tla_dist_fence" : "tla_dist_quiet_default";
        SmallVector<Type> types;
        for (Value arg : args)
            types.push_back(arg.getType());
        auto callee = ::tla::getOrCreateRuntimeCall(op->getParentOfType<ModuleOp>(), name, types, {});
        annotateAivRuntimeCall(callee);
        if (aic)
            callee->setAttr(
                hivm::TFuncCoreTypeAttr::name,
                hivm::TFuncCoreTypeAttr::get(rewriter.getContext(), hivm::TFuncCoreType::AIC));
        // Kept through func-to-LLVM; never mark waits willreturn/readnone.
        callee->setAttr("passthrough", rewriter.getArrayAttr({rewriter.getStringAttr("convergent")}));
        auto call = rewriter.create<func::CallOp>(op->getLoc(), callee, args);
        call->setAttr("convergent", rewriter.getUnitAttr());
        rewriter.eraseOp(op);
        return success();
    }

private:
    DenseMap<Value, Value>& cache;
};

class LowerCommPass final : public PassWrapper<LowerCommPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCommPass)

    StringRef getArgument() const override { return "tla_dist-lower-comm"; }
    StringRef getName() const override { return "TlaDistLowerCommPass"; }
    StringRef getDescription() const override
    {
        return "Lower CATCCOS communication operations to SHMEM runtime calls";
    }

    void getDependentDialects(DialectRegistry& registry) const override
    {
        registry.insert<
            TlaDistDialect, arith::ArithDialect, func::FuncDialect, memref::MemRefDialect, ::tla::TlaDialect,
            hivm::HIVMDialect>();
    }

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        RewritePatternSet basePatterns(&getContext());
        basePatterns.add<NormalizeLinearGmCopyPattern<::tla::CopyOp>, NormalizeLinearGmCopyPattern<RemoteCopyOp>>(
            &getContext());
        if (failed(applyPatternsGreedily(module, std::move(basePatterns)))) {
            signalPassFailure();
            return;
        }

        DenseMap<Value, Value> baseMemrefCache;
        RewritePatternSet patterns(&getContext());
        patterns.add<LowerRemoteCopyPattern>(&getContext(), baseMemrefCache);
        patterns.add<LowerSyncPattern>(&getContext(), baseMemrefCache);
        patterns.add<LowerQuietPattern, LowerBarrierAllPattern>(&getContext());
        if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
            signalPassFailure();
            return;
        }
        WalkResult remaining = module.walk([&](Operation* operation) {
            if (operation->getName().getDialectNamespace() == TlaDistDialect::getDialectNamespace()) {
                operation->emitOpError("was not lowered by tla_dist-lower-comm");
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        if (remaining.wasInterrupted())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createValidateCommAutoSyncPass() { return std::make_unique<ValidateCommAutoSyncPass>(); }

std::unique_ptr<Pass> createLowerShmemPtrPass() { return std::make_unique<LowerShmemPtrPass>(); }

std::unique_ptr<Pass> createPrepareCommFlagsPass() { return std::make_unique<PrepareCommFlagsPass>(); }

std::unique_ptr<Pass> createLowerCommPass() { return std::make_unique<LowerCommPass>(); }

void registerTlaDistPasses()
{
    registerSyncContractPasses();
    PassRegistration<ValidateCommAutoSyncPass>();
    PassRegistration<LowerShmemPtrPass>();
    PassRegistration<PrepareCommFlagsPass>();
    PassRegistration<LowerCommPass>();
}

} // namespace tla_dist
