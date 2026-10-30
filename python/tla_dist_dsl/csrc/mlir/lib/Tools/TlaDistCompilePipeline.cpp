#include "Dialect/TlaDist/IR/TlaDistDialect.h"
#include "Passes/TlaDistPasses.h"
#include "Tools/TlaDistCompilePipeline.h"

#include "Passes.h"
#include "Passes/PassesInternal.h"

#include "bishengir/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.h"
#include "bishengir/Conversion/HIVMAVEToStandard/HIVMAVEToStandard.h"
#include "bishengir/Conversion/HIVMToStandard/HIVMToStandard.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/ErrorHandling.h"

namespace tla_dist {

void buildTlaDistPipeline(mlir::OpPassManager& manager)
{
    // Resolve synchronization ids before communication consumes flag values.
    // Keep communication after descriptor materialization and before vector
    // lowering, which removes the enclosing tla.vector regions.
    manager.addPass(::tla::createTlaLowerFuncPass());
    manager.addPass(::tla::createTlaInsertAutoMutexPass());
    manager.addPass(::tla::createTlaLowerExternCallPass());
    manager.addPass(::tla::createTlaLowerPtrPass());
    manager.addPass(::tla::createTlaSplitMixedFuncPass());
    manager.addPass(::tla::createTlaLowerTensorDescPass());
    manager.addPass(::tla::createTlaLowerFlagBarrierToHivmPass());
    manager.addPass(createLowerCommPass());
    manager.addPass(::tla::createTlaLowerScalarAccessPass());
    manager.addPass(::tla::createTlaVectorRegionSimtPass());
    manager.addPass(::tla::createTlaVectorRegionSimdPass());
    manager.addPass(::tla::createTlaCubeRegionPass());
    manager.addPass(::tla::createTlaFinalizeMemrefPass());
    manager.addPass(::tla::createTlaLowerDebugPrintPass());
    manager.addPass(::tla::createTlaLowerScalarOpsPass());
    manager.addPass(::tla::createTlaLowerBlockIdxPass());
    manager.addPass(::tla::createTlaLowerMutexToStdPass());
    manager.addPass(::tla::createTlaPrologueEpiloguePass());
    manager.addPass(mlir::createCSEPass());
    manager.nest<mlir::func::FuncOp>().addPass(mlir::hivmave::createCombineAVEOPsPass());
    manager.nest<mlir::func::FuncOp>().addPass(mlir::hivm::createHIVMDecomposeOpPass());
    mlir::ConvertHIVMToStandardOptions hivmOptions;
    manager.addPass(mlir::createConvertHIVMToStandardPass(hivmOptions));
    manager.addPass(mlir::createConvertHIVMAVEToStandardPass());
    manager.addPass(mlir::memref::createExpandStridedMetadataPass());
    {
        std::unique_ptr<mlir::Pass> hivmaveToIntrin = mlir::createConvertHIVMAVEToAVEIntrinPass();
        if (failed(hivmaveToIntrin->initializeOptions("enable-i16-indvar=true", [](const llvm::Twine& msg) {
                llvm::errs() << "convert-hivmave-to-ave-intrin options: " << msg << "\n";
                return mlir::failure();
            }))) {
            llvm::report_fatal_error("failed to set enable-i16-indvar=true on convert-hivmave-to-ave-intrin");
        }
        manager.addPass(std::move(hivmaveToIntrin));
    }
    manager.addPass(::tla::createTlaLowerAVEToRegbaseIntrinsPass());
    manager.addPass(mlir::createConvertSCFToCFPass());
}

} // namespace tla_dist

namespace tla::tools {

void buildTlaDistCompilePassManagers(mlir::MLIRContext& context, mlir::PassManager& tlaManager)
{
    context.getOrLoadDialect<::tla_dist::TlaDistDialect>();
    ::tla_dist::buildTlaDistPipeline(tlaManager);
}

} // namespace tla::tools
