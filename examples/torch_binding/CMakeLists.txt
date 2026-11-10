# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(NOT DEFINED Python3_EXECUTABLE)
    message(WARNING "Python3_EXECUTABLE is not defined, using default python3")
else()
    message("Using python: ${Python3_EXECUTABLE}")
endif()

find_package(Python3 COMPONENTS Interpreter Development)
list(APPEND CMAKE_PREFIX_PATH "${Python3_SITELIB}")
find_package(Torch REQUIRED)
message(STATUS "Found Torch: ${Torch_DIR}")
message(STATUS "Torch Libraries: ${TORCH_LIBRARIES}")


# Include directories
set(TORCH_BINDING_INCLUDES
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_SITELIB}/torch_npu/include

    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/3rdparty/catlass/include
    ${CMAKE_SOURCE_DIR}/examples
    ${CMAKE_SOURCE_DIR}/utils

    ${SHMEM_HOME_PATH}/shmem/include
    ${SHMEM_HOME_PATH}/shmem/src

    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof
)

message("Start processing torch_binding")
# Compile torch_binding as shared library
add_library(catccos_torch SHARED
    src/torch_bindings.cpp
    src/torch_bindings_meta.cpp
)

set_target_properties(catccos_torch PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)


if(CATLASS_BISHENG_ARCH STREQUAL "a5")
    target_compile_definitions(catccos_torch PRIVATE CATCCOS_TORCH_A5)
endif()

target_include_directories(catccos_torch PRIVATE
    ${TORCH_BINDING_INCLUDES}
)


target_link_directories(catccos_torch PRIVATE
    ${Python3_SITELIB}/torch_npu/lib
    ${ASCEND_HOME_PATH}/lib64
    ${SHMEM_HOME_PATH}/shmem/lib
)

# Configure kernel library directory variables
set(CATCCOS_KERNEL_LIB_PATHS "")
foreach(op ${CATCCOS_TORCH_SUPPORTED})
    list(APPEND CATCCOS_KERNEL_LIB_PATHS
        ${CMAKE_BINARY_DIR}/lib/lib${op}_kernel.so
    )
endforeach()

# Auto add build dependencies for all kernel targets
foreach(op ${CATCCOS_TORCH_SUPPORTED})
    set(kernel_target ${op}_kernel_build)
    add_dependencies(catccos_torch ${kernel_target})
    message(STATUS "Adding build dependency: ${kernel_target}")
endforeach()


target_link_libraries(catccos_torch PRIVATE
    ${CATCCOS_KERNEL_LIB_PATHS}
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    ${SHMEM_HOME_PATH}/shmem/lib/libshmem.so
    ${SHMEM_HOME_PATH}/shmem/lib/libshmem_utils.so
    torch_npu
    tiling_api
)

install(TARGETS catccos_torch
    LIBRARY DESTINATION lib COMPONENT catccos_torch
)
