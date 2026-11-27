// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <memory>

namespace tla_dist {

std::unique_ptr<mlir::Pass> createValidateCommAutoSyncPass();
std::unique_ptr<mlir::Pass> createLowerShmemPtrPass();
std::unique_ptr<mlir::Pass> createPrepareCommFlagsPass();
std::unique_ptr<mlir::Pass> createLowerCommPass();
void registerTlaDistPasses();
void buildTlaDistPipeline(mlir::OpPassManager& manager);

} // namespace tla_dist
