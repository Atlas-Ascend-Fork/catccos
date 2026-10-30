# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.10)
project(CATCCOS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)")
endif()

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT DEFINED ASCEND_HOME_PATH AND NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
elseif(NOT DEFINED ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

if(NOT DEFINED SHMEM_HOME_PATH AND NOT DEFINED ENV{SHMEM_HOME_PATH})
    message(FATAL_ERROR "Cannot find SHMEM_HOME_PATH, please run set_env.sh.")
elseif(NOT DEFINED SHMEM_HOME_PATH)
    set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})
endif()

set(_CATCCOS_BISHENG_CANDIDATES
    "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng"
    "${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin/bisheng"
)

if(DEFINED CCEC AND CCEC)
    if(NOT EXISTS "${CCEC}")
        message(FATAL_ERROR "Configured CCEC does not exist: ${CCEC}")
    endif()
    set(_CATCCOS_BISHENG "${CCEC}")
else()
    foreach(_CATCCOS_BISHENG_CANDIDATE IN LISTS _CATCCOS_BISHENG_CANDIDATES)
        if(EXISTS "${_CATCCOS_BISHENG_CANDIDATE}")
            set(_CATCCOS_BISHENG "${_CATCCOS_BISHENG_CANDIDATE}")
            break()
        endif()
    endforeach()
endif()

if(NOT _CATCCOS_BISHENG)
    string(REPLACE ";" "\n  " _CATCCOS_BISHENG_CANDIDATES_MSG "${_CATCCOS_BISHENG_CANDIDATES}")
    message(FATAL_ERROR
        "Cannot find bisheng compiler. Checked:\n"
        "  ${_CATCCOS_BISHENG_CANDIDATES_MSG}\n"
        "Set CCEC to the full bisheng path or check ASCEND_HOME_PATH."
    )
endif()

set(CCEC "${_CATCCOS_BISHENG}" CACHE FILEPATH "Path to bisheng compiler" FORCE)
message(STATUS "Using bisheng compiler: ${CCEC}")

add_subdirectory(examples)

option(CATCCOS_ENABLE_TLA_DIST_DSL "Build the CATCCOS communication TLA DSL" OFF)
if(CATCCOS_ENABLE_TLA_DIST_DSL)
    add_subdirectory(python/tla_dist_dsl/csrc/mlir)
endif()

option(CATCCOS_BUILD_TESTS "Build tests/dynamic_tiling targets" OFF)
if(CATCCOS_BUILD_TESTS)
    add_subdirectory(tests)
endif()
