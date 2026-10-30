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

#include <string>
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

func::FuncOp getOrCreateAivRuntimeCall(ModuleOp module, StringRef name, ArrayRef<Type> operandTypes)
{
    func::FuncOp function = ::tla::getOrCreateRuntimeCall(module, name, operandTypes);
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
        Value flagValue = operation.getFlagValue();
        std::optional<::tla::TensorDescriptor> ubDesc;
        int32_t copyEventId = -1;
        if (needsScratch) {
            if (!ubTile || !flagValue)
                return operation->emitOpError("requires both a UB scratch tensor and flag_value"), failure();
            auto ubDescOp = ubTile.getDefiningOp<::tla::TensorDescOp>();
            if (!ubDescOp)
                return rewriter.notifyMatchFailure(operation, "expected UB tensor_desc operand");
            auto decoded = ::tla::descriptorFromTensorDescOp(ubDescOp);
            if (failed(decoded))
                return operation->emitOpError("failed to decode UB descriptor"), failure();
            ubDesc = *decoded;
            if (ubDesc->addrspace != "ub")
                return operation->emitOpError("requires its scratch tensor in UB"), failure();
            if (transport == "mte" && ubDesc->elementType != srcDesc.elementType)
                return operation->emitOpError("requires identical src and UB scratch element types"), failure();
            auto flag = flagValue.getDefiningOp<::tla::FlagOp>();
            if (!flag || !isa<::tla::FlagType>(flagValue.getType()))
                return operation->emitOpError("flag_value must be created by tla.flag"), failure();
            int64_t id = flag.getId();
            if (id < 0 || id >= 8)
                return operation->emitOpError(
                           "flag_value requires a resolved id in [0, 8); run tla-lower-flag-barrier-to-hivm first"),
                       failure();
            copyEventId = static_cast<int32_t>(id);
            ::Pipe srcPipe = ::Pipe::mte2;
            ::Pipe dstPipe = ::Pipe::mte3;
            if (transport == "rdma")
                srcPipe = ::Pipe::scalar;
            else if (transport == "sdma") {
                srcPipe = ::Pipe::mte3;
                dstPipe = ::Pipe::mte2;
            } else if (transport == "udma") {
                srcPipe = ::Pipe::mte3;
                dstPipe = ::Pipe::scalar;
            }
            if (flag.getSrcPipe().getPipe() != srcPipe || flag.getDstPipe().getPipe() != dstPipe)
                return operation->emitOpError("flag_value has the wrong source/destination pipe direction"), failure();
        } else if (ubTile || flagValue) {
            return operation->emitOpError("this route does not accept ub_tensor or flag_value"), failure();
        }
        if (transport != "mte")
            return operation->emitOpError() << transport << " device lowering is not implemented", failure();

        std::string route = gmToGm ? "gm_row_major_to_gm_row_major" :
                            gmToUb ? "gm_row_major_to_ub_row_major" :
                                     "ub_row_major_to_gm_row_major";
        std::string calleeName = "shmem_remote_" + mode.str() + "_" + route + "_float";

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
            auto ubMemref = buildRuntimeMemref(*ubDesc);
            if (failed(ubMemref))
                return operation->emitOpError("failed to materialize the UB scratch runtime memref"), failure();
            operands.push_back(*ubMemref);
        }
        SmallVector<Value, 24> payload = buildDescriptorPayload(rewriter, operation->getLoc(), srcDesc);
        auto dstPayload = buildDescriptorPayload(rewriter, operation->getLoc(), dstDesc);
        payload.append(dstPayload.begin(), dstPayload.end());
        operands.append(payload.begin(), payload.end());
        if (ubDesc) {
            auto ubPayload = buildDescriptorPayload(rewriter, operation->getLoc(), *ubDesc);
            operands.append(ubPayload.begin(), ubPayload.end());
        }
        operands.push_back(peer);
        if (needsScratch)
            operands.push_back(rewriter.create<arith::ConstantIntOp>(operation.getLoc(), copyEventId, 32));
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
        patterns.add<LowerBarrierAllPattern>(&getContext());
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

std::unique_ptr<Pass> createLowerCommPass() { return std::make_unique<LowerCommPass>(); }

void registerTlaDistPasses() { PassRegistration<LowerCommPass>(); }

} // namespace tla_dist
