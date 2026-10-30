cmake_minimum_required(VERSION 3.20)
project(catccos_tla_dist LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Match the public headers and generated interfaces of AscendNPU-IR.
add_compile_definitions(BSPUB_DAVINCI_BISHENGIR_A5)

get_filename_component(TLA_DIST_DSL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
get_filename_component(CATCCOS_ROOT "${TLA_DIST_DSL_ROOT}/../.." ABSOLUTE)
set(TLA_DIST_DSL_MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CATLASS_TLA_DSL_ROOT
    "${CATCCOS_ROOT}/3rdparty/catlass/python/tla_dsl"
    CACHE PATH "Root of the pinned Catlass Python TLA DSL")
get_filename_component(CATLASS_TLA_DSL_ROOT "${CATLASS_TLA_DSL_ROOT}" ABSOLUTE)
if(NOT EXISTS "${CATLASS_TLA_DSL_ROOT}/csrc/mlir/CMakeLists.txt" OR
   NOT EXISTS "${CATLASS_TLA_DSL_ROOT}/catlass/__init__.py")
  message(FATAL_ERROR
    "CATLASS_TLA_DSL_ROOT does not contain the Python TLA DSL: "
    "${CATLASS_TLA_DSL_ROOT}. Pin 3rdparty/catlass to a compatible revision "
    "or pass -DCATLASS_TLA_DSL_ROOT=<path>.")
endif()

set(TLA_DIST_DSL_SHMEM_ROOT "${CATCCOS_ROOT}/3rdparty/shmem" CACHE PATH
    "Root of the ACLSHMEM source tree")
get_filename_component(TLA_DIST_DSL_SHMEM_ROOT "${TLA_DIST_DSL_SHMEM_ROOT}" ABSOLUTE)
if(NOT EXISTS "${TLA_DIST_DSL_SHMEM_ROOT}/include/shmem.h" OR
   NOT EXISTS "${TLA_DIST_DSL_SHMEM_ROOT}/src/device")
  message(FATAL_ERROR
    "TLA_DIST_DSL_SHMEM_ROOT must contain include/shmem.h and src/device: "
    "${TLA_DIST_DSL_SHMEM_ROOT}")
endif()

if(NOT DEFINED ASCEND_HOME_PATH OR "${ASCEND_HOME_PATH}" STREQUAL "")
  if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
  else()
    message(FATAL_ERROR "ASCEND_HOME_PATH is not defined")
  endif()
endif()

# Resolve the shared AscendNPU-IR installation before adding the upstream TLA
# project. Dependencies discovered in this parent scope are visible to both
# the upstream `tla` subtree and the sibling TLA_DIST targets.
if(DEFINED ENV{CATLASS_DSL_ASCENDNPU_IR_ROOT} AND
   NOT "$ENV{CATLASS_DSL_ASCENDNPU_IR_ROOT}" STREQUAL "")
  set(TLA_DIST_ASCENDNPU_IR_ROOT "$ENV{CATLASS_DSL_ASCENDNPU_IR_ROOT}")
elseif(DEFINED CATLASS_DSL_PREBUILT_ASCENDNPU_IR AND
   NOT "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}" STREQUAL "")
  set(TLA_DIST_ASCENDNPU_IR_ROOT "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}")
elseif(DEFINED ENV{CATLASS_DSL_PREBUILT_ASCENDNPU_IR} AND
       NOT "$ENV{CATLASS_DSL_PREBUILT_ASCENDNPU_IR}" STREQUAL "")
  set(TLA_DIST_ASCENDNPU_IR_ROOT "$ENV{CATLASS_DSL_PREBUILT_ASCENDNPU_IR}")
elseif(DEFINED TLA_DSL_PREBUILT_ASCENDNPU_IR AND
       NOT "${TLA_DSL_PREBUILT_ASCENDNPU_IR}" STREQUAL "")
  set(TLA_DIST_ASCENDNPU_IR_ROOT "${TLA_DSL_PREBUILT_ASCENDNPU_IR}")
elseif(DEFINED ENV{TLA_DSL_PREBUILT_ASCENDNPU_IR} AND
       NOT "$ENV{TLA_DSL_PREBUILT_ASCENDNPU_IR}" STREQUAL "")
  set(TLA_DIST_ASCENDNPU_IR_ROOT "$ENV{TLA_DSL_PREBUILT_ASCENDNPU_IR}")
else()
  set(TLA_DIST_ASCENDNPU_IR_ROOT
      "${CATLASS_TLA_DSL_ROOT}/3rdparty/AscendNPU-IR")
endif()
get_filename_component(TLA_DIST_ASCENDNPU_IR_ROOT
                       "${TLA_DIST_ASCENDNPU_IR_ROOT}" ABSOLUTE)

# Catlass requires both source and install paths, and embeds its own
# MLIR Python package. Share one source/install pair with the downstream build.
if(DEFINED ENV{CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR} AND
   NOT "$ENV{CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR}" STREQUAL "")
  set(TLA_DIST_ASCENDNPU_IR_INSTALL_DIR "$ENV{CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR}")
else()
  set(TLA_DIST_ASCENDNPU_IR_INSTALL_DIR "${TLA_DIST_ASCENDNPU_IR_ROOT}/build/install")
endif()
set(ENV{CATLASS_DSL_ASCENDNPU_IR_ROOT} "${TLA_DIST_ASCENDNPU_IR_ROOT}")
set(ENV{CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR} "${TLA_DIST_ASCENDNPU_IR_INSTALL_DIR}")
set(MLIR_DIR "${TLA_DIST_ASCENDNPU_IR_INSTALL_DIR}/lib/cmake/mlir")
set(LLVM_DIR "${TLA_DIST_ASCENDNPU_IR_INSTALL_DIR}/lib/cmake/llvm")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG QUIET)
find_package(MLIR REQUIRED CONFIG)
if(NOT TARGET LLVM)
  find_package(LLVM REQUIRED CONFIG HINTS "${MLIR_CMAKE_DIR}/../llvm")
endif()

set(TLA_DIST_CATLASS_INCLUDE_DIR "${CATCCOS_ROOT}/3rdparty/catlass/include")
set(TLA_DSL_SHMEM_ROOT "${TLA_DIST_DSL_SHMEM_ROOT}" CACHE PATH "" FORCE)
set(CATLASS_INCLUDE_DIR "${TLA_DIST_CATLASS_INCLUDE_DIR}" CACHE PATH "" FORCE)
set(TLA_DIST_UPSTREAM_MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/tla")
add_subdirectory("${CATLASS_TLA_DSL_ROOT}/csrc/mlir"
                 "${TLA_DIST_UPSTREAM_MLIR_BINARY_DIR}")

set(TLA_DIST_PYTHON_OUTPUT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/python/catccos/tla_dist")
file(MAKE_DIRECTORY "${TLA_DIST_PYTHON_OUTPUT_DIR}")
# CATCCOS lowering is an executable-only extension. Remove native modules from
# earlier builds so they cannot be packaged or loaded beside Catlass's bridge.
file(GLOB TLA_DIST_STALE_NATIVE_BRIDGES
     "${TLA_DIST_PYTHON_OUTPUT_DIR}/_tla_type_bridge_native*"
     "${TLA_DIST_PYTHON_OUTPUT_DIR}/_tla_dist_type_bridge_native*"
     "${TLA_DIST_PYTHON_OUTPUT_DIR}/_tla_dist_compiler_bridge_native*")
if(TLA_DIST_STALE_NATIVE_BRIDGES)
  file(REMOVE ${TLA_DIST_STALE_NATIVE_BRIDGES})
endif()

# Generate the registered TLA_DIST operation classes from ODS.
find_program(TLA_DIST_MLIR_TABLEGEN_EXE
  NAMES mlir-tblgen
  HINTS
    "${TLA_DIST_ASCENDNPU_IR_INSTALL_DIR}/bin"
    "${TLA_DIST_ASCENDNPU_IR_ROOT}/build/bin"
  REQUIRED
)
set(TLA_DIST_TD_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Dialect/TlaDist/IR/TlaDistOps.td")
set(TLA_DIST_TABLEGEN_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/tblgen")
set(TLA_DIST_TABLEGEN_OUTPUT_DIR "${TLA_DIST_TABLEGEN_INCLUDE_DIR}/tla_dist")
set(TLA_DIST_TABLEGEN_INCLUDE_FLAGS
  -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
  -I "${TLA_DIST_ASCENDNPU_IR_INSTALL_DIR}/include"
  -D BSPUB_DAVINCI_BISHENGIR_A5
)
add_custom_command(
  OUTPUT
    "${TLA_DIST_TABLEGEN_OUTPUT_DIR}/Ops.h.inc"
    "${TLA_DIST_TABLEGEN_OUTPUT_DIR}/Ops.cpp.inc"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${TLA_DIST_TABLEGEN_OUTPUT_DIR}"
  COMMAND "${TLA_DIST_MLIR_TABLEGEN_EXE}" --gen-op-decls "${TLA_DIST_TD_FILE}"
          ${TLA_DIST_TABLEGEN_INCLUDE_FLAGS}
          -o "${TLA_DIST_TABLEGEN_OUTPUT_DIR}/Ops.h.inc"
  COMMAND "${TLA_DIST_MLIR_TABLEGEN_EXE}" --gen-op-defs "${TLA_DIST_TD_FILE}"
          ${TLA_DIST_TABLEGEN_INCLUDE_FLAGS}
          -o "${TLA_DIST_TABLEGEN_OUTPUT_DIR}/Ops.cpp.inc"
  COMMAND_EXPAND_LISTS
  DEPENDS
    "${TLA_DIST_TD_FILE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Dialect/TlaDist/IR/TlaDistDialect.td"
  VERBATIM
)
add_custom_target(TlaDistTblgen DEPENDS
  "${TLA_DIST_TABLEGEN_OUTPUT_DIR}/Ops.h.inc"
  "${TLA_DIST_TABLEGEN_OUTPUT_DIR}/Ops.cpp.inc"
)

# Mirror the upstream TLA layout: device bitcode and compiler libraries are
# configured by their own subdirectories.
option(BISHENGIR_BUILD_TEMPLATE "Build CATCCOS TLA compute and communication bitcode" ON)
add_subdirectory(bc)
add_subdirectory(lib)
add_subdirectory(tools/tla_dist-compile)

add_custom_target(tla_dist-compiler DEPENDS TlaDistCompile)
if(TARGET tla_dist_shmem_bitcode)
  add_dependencies(tla_dist-compiler tla_dist_shmem_bitcode)
endif()

# Lit regression tests cover only the CATCCOS-owned dialect and lowering. The
# upstream Catlass project owns tests for the base TLA operations and passes.
set(TLA_DIST_DSL_COMPILE_TOOL
    "${CMAKE_CURRENT_BINARY_DIR}/tools/tla_dist-compile/TlaDistCompile")
if(EXISTS "${TLA_DIST_DSL_ROOT}/tests/lit/CMakeLists.txt")
  add_subdirectory("${TLA_DIST_DSL_ROOT}/tests/lit"
                   "${CMAKE_CURRENT_BINARY_DIR}/tests/lit")
endif()
