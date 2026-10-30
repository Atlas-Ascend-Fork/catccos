#include "Dialect/TlaDist/IR/TlaDistDialect.h"
#include "Dialect/TlaDist/IR/TlaDistOps.h"

namespace tla_dist {

TlaDistDialect::TlaDistDialect(mlir::MLIRContext* context)
    : Dialect(getDialectNamespace(), context, mlir::TypeID::get<TlaDistDialect>())
{
    initialize();
}

void TlaDistDialect::initialize()
{
#define GET_OP_LIST
    addOperations<
#include "tla_dist/Ops.cpp.inc"
        >();
#undef GET_OP_LIST
}

} // namespace tla_dist
