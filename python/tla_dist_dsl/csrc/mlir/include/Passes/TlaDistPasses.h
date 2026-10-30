#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <memory>

namespace tla_dist {

std::unique_ptr<mlir::Pass> createLowerCommPass();
void registerTlaDistPasses();
void buildTlaDistPipeline(mlir::OpPassManager& manager);

} // namespace tla_dist
