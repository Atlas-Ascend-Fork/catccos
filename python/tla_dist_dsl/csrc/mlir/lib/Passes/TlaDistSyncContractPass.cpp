// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "Dialect/TlaDist/IR/TlaDistOps.h"
#include "Passes/TlaDistPasses.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
using namespace mlir;
namespace tla_dist {
namespace {
constexpr int64_t userUbLimit = 188 * 1024;
bool scoped(Operation* op) { return isa<LocalSyncOp, SyncOp, BarrierOp>(op); }
bool resources(Operation* op) { return isa<SyncOp, BarrierOp, QuietDefaultOp, FenceOp, SignalOp>(op); }
bool inRegion(Operation* op, StringRef name) {
    for (Operation* p = op->getParentOp(); p; p = p->getParentOp())
        if (p->getName().getStringRef() == name) return true;
    return false;
}
class ValidateSyncContractPass final : public PassWrapper<ValidateSyncContractPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ValidateSyncContractPass)
    StringRef getArgument() const override { return "tla_dist-validate-sync-contract"; }
    void runOnOperation() override {
        int64_t point = 0;
        auto result = getOperation().walk([&](Operation* op) {
            bool collective = scoped(op);
            if (!collective && !isa<SignalOp, WaitUntilOp, QuietDefaultOp, FenceOp>(op)) return WalkResult::advance();
            auto fail = [&](StringRef message) { op->emitOpError(message); return WalkResult::interrupt(); };
            if (inRegion(op, "tla.vec.func")) return fail("synchronization requires SIMD, not SIMT");
            bool vector = inRegion(op, "tla.vector");
            bool mix = collective && cast<CoreScopeAttr>(op->getAttr("scope")).getValue() == CoreScope::MIX;
            if (!mix && !vector) return fail("AIV synchronization requires a tla.vector region");
            if (mix && (vector || inRegion(op, "tla.cube"))) return fail("MIX synchronization requires the common layer outside cube/vector regions");
            auto function = op->getParentOfType<func::FuncOp>();
            auto core = function->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
            if (mix) {
                if (!core || core.getFuncCoreType() != hivm::TFuncCoreType::MIX)
                    return fail("MIX scope requires a real paired cube/vector kernel");
                op->setAttr("tla_dist.sync_point", IntegerAttr::get(IntegerType::get(&getContext(), 64), point++));
            }
            if (collective) {
                for (Operation* parent = op->getParentOp(); parent != function.getOperation(); parent = parent->getParentOp()) {
                    if (isa<scf::IfOp, scf::WhileOp>(parent)) return fail("collective cannot be nested in a dynamic if/while");
                    if (auto loop = dyn_cast<scf::ForOp>(parent)) {
                        APInt v;
                        if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&v)) ||
                            !matchPattern(loop.getUpperBound(), m_ConstantInt(&v)) ||
                            !matchPattern(loop.getStep(), m_ConstantInt(&v)))
                            return fail("collective for bounds and step must be compile-time constants");
                    }
                }
                if (!isa<LocalSyncOp>(op)) {
                    APInt team;
                    if (matchPattern(op->getOperand(0), m_ConstantInt(&team)) && team.getSExtValue() < -1)
                        return fail("team constant must be WORLD, INVALID=-1 or a valid nonnegative handle");
                }
                if (!mix && core && core.getFuncCoreType() == hivm::TFuncCoreType::AIV) {
                    // A standalone vector hard barrier needs mixed hardware metadata,
                    // but remains one AIV function: never manufacture a dummy AIC.
                    function->setAttr("tla_dist.vector_hard_sync", UnitAttr::get(&getContext()));
                }
            }
            return WalkResult::advance();
        });
        if (result.wasInterrupted()) signalPassFailure();
    }
};
class ReserveSyncResourcesPass final : public PassWrapper<ReserveSyncResourcesPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReserveSyncResourcesPass)
    StringRef getArgument() const override { return "tla_dist-reserve-sync-resources"; }
    void runOnOperation() override {
        auto result = getOperation().walk([&](func::FuncOp function) {
            bool needed = false;
            function.walk([&](Operation* op) { needed |= resources(op); });
            if (!needed) return WalkResult::advance();
            for (StringRef key : {"tla.ub_static_bytes", "tla.ub_dynamic_base"}) {
                auto size = function->getAttrOfType<IntegerAttr>(key);
                if (size && size.getInt() > userUbLimit) {
                    function.emitOpError("SHMEM synchronization reserves UB from byte 192512; user UB exceeds 188 KiB");
                    return WalkResult::interrupt();
                }
            }
            auto budget = function->getAttrOfType<IntegerAttr>("tla.ub_programmable_bytes");
            function->setAttr("tla.ub_programmable_bytes", IntegerAttr::get(IntegerType::get(&getContext(), 64),
                budget ? std::min(budget.getInt(), userUbLimit) : userUbLimit));
            function->setAttr("tla_dist.sync_resource_abi", IntegerAttr::get(IntegerType::get(&getContext(), 32), 1));
            return WalkResult::advance();
        });
        if (result.wasInterrupted()) signalPassFailure();
    }
};
// The single-vector hard barrier uses a real MIX_AIV_1_0 ELF. Preserve the
// AIV function (no artificial cube), with explicit zero-cube/one-vector ratio.
class FinalizeVectorSyncMetadataPass final : public PassWrapper<FinalizeVectorSyncMetadataPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeVectorSyncMetadataPass)
    StringRef getArgument() const override { return "tla_dist-finalize-vector-sync-metadata"; }
    void runOnOperation() override {
        bool found = false;
        getOperation().walk([&](func::FuncOp function) {
            if (!function->hasAttr("tla_dist.vector_hard_sync")) return;
            function->setAttr(hivm::TCoreRatioAttr::name, hivm::TCoreRatioAttr::get(&getContext(), 0, 1));
            function->setAttr("mix_mode", StringAttr::get(&getContext(), "mix"));
            found = true;
        });
        if (found) getOperation()->setAttr(hivm::TModuleCoreTypeAttr::name,
            hivm::TModuleCoreTypeAttr::get(&getContext(), hivm::TModuleCoreType::MIX));
    }
};
class ValidateSyncPairsPass final : public PassWrapper<ValidateSyncPairsPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ValidateSyncPairsPass)
    StringRef getArgument() const override { return "tla_dist-validate-sync-pairs"; }
    void runOnOperation() override {
        llvm::DenseMap<int64_t, std::pair<Operation*, Operation*>> pairs;
        bool invalid = false;
        getOperation().walk([&](Operation* op) {
            auto id = op->getAttrOfType<IntegerAttr>("tla_dist.sync_point");
            if (!id) return;
            auto core = op->getParentOfType<func::FuncOp>()->getAttrOfType<hivm::TFuncCoreTypeAttr>(hivm::TFuncCoreTypeAttr::name);
            if (!core || (core.getFuncCoreType() != hivm::TFuncCoreType::AIC && core.getFuncCoreType() != hivm::TFuncCoreType::AIV)) {
                op->emitOpError("MIX synchronization point was not split"); invalid = true; return;
            }
            auto& pair = pairs[id.getInt()];
            auto& side = core.getFuncCoreType() == hivm::TFuncCoreType::AIC ? pair.first : pair.second;
            if (side) { op->emitOpError("duplicate MIX synchronization side"); invalid = true; }
            side = op;
        });
        for (auto& entry : pairs) {
            auto [aic, aiv] = entry.second;
            if (!aic || !aiv) {
                (aic ? aic : aiv)->emitOpError("MIX synchronization point missing AIC or AIV side"); invalid = true;
            } else if (aic->getName() != aiv->getName() || aic->getAttr("scope") != aiv->getAttr("scope")) {
                aic->emitOpError("MIX synchronization sides do not match"); invalid = true;
            }
        }
        if (invalid) signalPassFailure();
    }
};
}
std::unique_ptr<Pass> createValidateSyncContractPass() { return std::make_unique<ValidateSyncContractPass>(); }
std::unique_ptr<Pass> createReserveSyncResourcesPass() { return std::make_unique<ReserveSyncResourcesPass>(); }
std::unique_ptr<Pass> createValidateSyncPairsPass() { return std::make_unique<ValidateSyncPairsPass>(); }
std::unique_ptr<Pass> createFinalizeVectorSyncMetadataPass() { return std::make_unique<FinalizeVectorSyncMetadataPass>(); }
void registerSyncContractPasses() {
    PassRegistration<FinalizeVectorSyncMetadataPass>();
    PassRegistration<ValidateSyncContractPass>(); PassRegistration<ReserveSyncResourcesPass>(); PassRegistration<ValidateSyncPairsPass>();
}
}
