# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

find_package(Threads REQUIRED)

# Host LLVM builds require the host glibc. Match the pinned upstream tool's
# behavior when CMake was configured with a Conda compiler driver.
if(CMAKE_CXX_COMPILER MATCHES "conda" AND EXISTS "/usr/bin/g++")
  set(CMAKE_CXX_LINK_EXECUTABLE
    "/usr/bin/g++ <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

# Keep the upstream command-line frontend and replace only the pass-manager
# builder selected by the preprocessor. This avoids maintaining a CATCCOS copy
# of main.cpp while still compiling tla_dist operations.
add_executable(TlaDistCompile
  "${CATLASS_TLA_DSL_ROOT}/csrc/mlir/tools/tla-compile/main.cpp"
)
target_compile_definitions(TlaDistCompile PRIVATE
  buildTlaCompilePassManagers=buildTlaDistCompilePassManagers
)
target_include_directories(TlaDistCompile PRIVATE
  "${TLA_DIST_DSL_MLIR_SOURCE_DIR}/include"
  "${CATLASS_TLA_DSL_ROOT}/csrc/mlir/include"
  "${TLA_DIST_UPSTREAM_MLIR_BINARY_DIR}/tblgen"
)
target_link_libraries(TlaDistCompile PRIVATE
  TlaDistCompilePipeline
  TlaDistPasses
  MLIRParser
  Threads::Threads
)
add_dependencies(TlaDistCompile TlaDistCompilePipeline TlaDistPasses)

# Keep the test capability record beside the compiler, including packaged copies.
file(GENERATE OUTPUT "$<TARGET_FILE:TlaDistCompile>.json"
  CONTENT "{\"rdma_backend\": \"${TLA_DIST_RDMA_BACKEND}\"}\n")

# Preserve the normal build-tree tool path for diagnostics and also include a
# copy in the Python package payload used by catccos.tla_dist.compile.
add_custom_command(TARGET TlaDistCompile POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${TLA_DIST_PYTHON_OUTPUT_DIR}/bin"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:TlaDistCompile>"
          "$<TARGET_FILE:TlaDistCompile>.json"
          "${TLA_DIST_PYTHON_OUTPUT_DIR}/bin/"
  VERBATIM
)
