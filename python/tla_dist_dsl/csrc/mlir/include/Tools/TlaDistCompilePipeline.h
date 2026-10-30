#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

namespace tla::tools {

void buildTlaDistCompilePassManagers(mlir::MLIRContext& context, mlir::PassManager& tlaManager);

} // namespace tla::tools
