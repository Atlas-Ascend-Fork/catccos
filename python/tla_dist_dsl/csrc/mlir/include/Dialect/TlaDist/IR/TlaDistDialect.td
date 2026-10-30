include "mlir/IR/DialectBase.td"

def TlaDistDialect : Dialect {
  let name = "tla_dist";
  let cppNamespace = "::tla_dist";
  let summary = "CATCCOS communication operations";
  let description = [{
    Operations that express device-side communication and synchronization for
    CATCCOS kernels before they are lowered to ACLSHMEM runtime calls.
  }];
}
