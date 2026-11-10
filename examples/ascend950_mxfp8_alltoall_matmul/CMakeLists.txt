# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(NOT CATLASS_BISHENG_ARCH STREQUAL "a5")
    return()
endif()

# build.sh supplies the RDMA backend and matching firmware mode.
set(RDMA_DEFINITIONS)
if(CATCCOS_RDMA_BACKEND)
    list(APPEND RDMA_DEFINITIONS ACLSHMEMI_RDMA_K_BACKEND_${CATCCOS_RDMA_BACKEND}=1)
endif()
if(CATCCOS_RDMA_BACKEND STREQUAL "HNS_1825")
    if(NOT CATCCOS_HNS_1825_PI_ON_CHIP MATCHES "^[01]$")
        message(FATAL_ERROR "Set CATCCOS_HNS_1825_PI_ON_CHIP=0 or 1 to match the SHMEM SDK")
    endif()
    list(APPEND RDMA_DEFINITIONS ACLSHMEMI_HNS_1825_PI_ON_CHIP=${CATCCOS_HNS_1825_PI_ON_CHIP})
    set(SHMEM_TOPOLOGY_INCLUDES
        "${CMAKE_SOURCE_DIR}/3rdparty/shmem/src/host/transport/topo/rootinfo"
        "${CMAKE_SOURCE_DIR}/3rdparty/shmem/src/host/utils/under_api")
endif()

catccos_example_add_executable(
    a2a_matmul
    SOURCES main.cpp
    INCLUDES ${SHMEM_TOPOLOGY_INCLUDES}
    COMPILE_DEFINITIONS CATLASS_ARCH=3510 ${RDMA_DEFINITIONS}
        ASCEND_TIMER_MAX_DYNAMIC_ITER=2048 ASCEND_TIMER_CYCLES_PER_US=1000.0f
)
