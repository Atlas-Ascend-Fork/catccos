#pragma once

// CATCCOS-owned communication dialect layered on the upstream TLA dialect.

#include "mlir/IR/Dialect.h"

namespace tla_dist {

class TlaDistDialect final : public mlir::Dialect {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TlaDistDialect)

    explicit TlaDistDialect(mlir::MLIRContext* context);
    static llvm::StringRef getDialectNamespace() { return "tla_dist"; }
    void initialize();
};

} // namespace tla_dist
