add_library(TlaDistPasses STATIC
  TlaDistLowerCommPass.cpp
  ../Dialect/TlaDist/IR/TlaDistDialect.cpp
  ../Dialect/TlaDist/IR/TlaDistOps.cpp
)
set_target_properties(TlaDistPasses PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(TlaDistPasses
  PUBLIC
    "${TLA_DIST_DSL_MLIR_SOURCE_DIR}/include"
    "${TLA_DIST_TABLEGEN_INCLUDE_DIR}"
  PRIVATE
    "${CATLASS_TLA_DSL_ROOT}/csrc/mlir/include"
    "${TLA_DIST_UPSTREAM_MLIR_BINARY_DIR}/tblgen"
)
target_link_libraries(TlaDistPasses PUBLIC TlaPasses)
add_dependencies(TlaDistPasses TlaDistTblgen TlaPasses)
