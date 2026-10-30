add_library(TlaDistCompilePipeline STATIC TlaDistCompilePipeline.cpp)
set_target_properties(TlaDistCompilePipeline PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(TlaDistCompilePipeline
  PUBLIC
    "${TLA_DIST_DSL_MLIR_SOURCE_DIR}/include"
  PRIVATE
    "${CATLASS_TLA_DSL_ROOT}/csrc/mlir/include"
    "${CATLASS_TLA_DSL_ROOT}/csrc/mlir/lib"
    "${TLA_DIST_UPSTREAM_MLIR_BINARY_DIR}/tblgen"
)
target_link_libraries(TlaDistCompilePipeline PUBLIC TlaDistPasses TlaCompilePipeline)
add_dependencies(TlaDistCompilePipeline TlaDistPasses TlaCompilePipeline)
