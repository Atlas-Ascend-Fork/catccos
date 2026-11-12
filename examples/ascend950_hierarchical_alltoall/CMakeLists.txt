# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(CATCCOS_HIERARCHICAL_GROUP_SIZE 16 CACHE STRING "Consecutive ranks per UDMA domain")

# scripts/build.sh passes the explicitly selected backend/firmware configuration.
set(RDMA_DEFINITIONS)
if(CATCCOS_RDMA_BACKEND)
    list(APPEND RDMA_DEFINITIONS ACLSHMEMI_RDMA_K_BACKEND_${CATCCOS_RDMA_BACKEND}=1)
endif()
if(CATCCOS_RDMA_BACKEND STREQUAL "HNS_1825")
    list(APPEND RDMA_DEFINITIONS
        ACLSHMEMI_HNS_1825_PI_ON_CHIP=${CATCCOS_HNS_1825_PI_ON_CHIP})
endif()

catccos_example_add_executable(
    ascend950_hierarchical_alltoall
    SOURCES main.cpp
    COMPILE_DEFINITIONS
        CATLASS_ARCH=3510
        CATCCOS_HIERARCHICAL_GROUP_SIZE=${CATCCOS_HIERARCHICAL_GROUP_SIZE}
        ${RDMA_DEFINITIONS}
)
