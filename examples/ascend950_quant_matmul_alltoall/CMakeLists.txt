# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# This example only has a multi-QP data path. QP0..N-1 carry data and the
# READY reuses QP0 only after every data QP has been drained.
# Its build marker keeps multi-QP behavior explicit and independent of older caches.
set(QMMA_WITH_QP ON CACHE BOOL "Multi-QP data plane for the target" FORCE)

# HierarchicalCommSwizzle uses this compile-time boundary to select UDMA for
# peers on the same server and RDMA for peers on another server.
set(CATCCOS_HIERARCHICAL_GROUP_SIZE 16 CACHE STRING "Consecutive ranks per UDMA domain")
if(NOT CATCCOS_HIERARCHICAL_GROUP_SIZE MATCHES "^[1-9][0-9]*$"
   OR CATCCOS_HIERARCHICAL_GROUP_SIZE GREATER 32)
    message(FATAL_ERROR "CATCCOS_HIERARCHICAL_GROUP_SIZE must be in [1, 32]")
endif()
message(STATUS "QMMA hierarchical group size: ${CATCCOS_HIERARCHICAL_GROUP_SIZE}")

# Pin the exact SHMEM SDK used at build time. run.sh rejects an SDK swap that
# could otherwise mix a new RDMA ABI with an old kernel. CI configures every
# example against one preinstalled SDK, so skip this RDMA-only example when that
# SDK cannot support it. scripts/build.sh still validates direct builds strictly.
if(NOT EXISTS "${SHMEM_HOME_PATH}/version.info")
    message(STATUS "Skipping ascend950_quant_matmul_alltoall: "
        "SHMEM version.info not found under ${SHMEM_HOME_PATH}")
    return()
endif()
file(READ "${SHMEM_HOME_PATH}/version.info" QMMA_SHMEM_VERSION)

string(REGEX MATCH "RDMA Backend : ([A-Za-z0-9_, ]+)" QMMA_SHMEM_BACKEND_MATCH "${QMMA_SHMEM_VERSION}")
if(NOT QMMA_SHMEM_BACKEND_MATCH)
    message(STATUS "Skipping ascend950_quant_matmul_alltoall: "
        "cannot read 'RDMA Backend' from ${SHMEM_HOME_PATH}/version.info")
    return()
endif()
string(STRIP "${CMAKE_MATCH_1}" QMMA_DETECTED_RDMA_BACKEND)
if(NOT QMMA_DETECTED_RDMA_BACKEND MATCHES "^(XSCALE|HNS_1825)$")
    message(STATUS "Skipping ascend950_quant_matmul_alltoall: "
        "unsupported SHMEM RDMA backend '${QMMA_DETECTED_RDMA_BACKEND}'; "
        "configure SHMEM with -enable_rdma -rdma_backend XSCALE|HNS_1825")
    return()
endif()

# version.info is authoritative.  QMMA_RDMA_BACKEND is only an optional
# assertion; an old build directory may still hold values such as NONE from
# before SHMEM was rebuilt with RDMA, so ignore anything that is not a backend.
set(QMMA_RDMA_BACKEND "$ENV{SHMEM_RDMA_BACKEND}" CACHE STRING
    "Optional assertion of the installed SHMEM RDMA backend: XSCALE or HNS_1825")
if(QMMA_RDMA_BACKEND)
    string(TOUPPER "${QMMA_RDMA_BACKEND}" QMMA_RDMA_BACKEND_UPPER)
    if(QMMA_RDMA_BACKEND_UPPER MATCHES "^(XSCALE|HNS_1825)$")
        if(NOT QMMA_RDMA_BACKEND_UPPER STREQUAL QMMA_DETECTED_RDMA_BACKEND)
            message(FATAL_ERROR
                "QMMA_RDMA_BACKEND=${QMMA_RDMA_BACKEND} differs from the installed SHMEM backend "
                "${QMMA_DETECTED_RDMA_BACKEND} (${SHMEM_HOME_PATH}/version.info). Remove build/ and "
                "reconfigure, or unset QMMA_RDMA_BACKEND to auto-detect.")
        endif()
    else()
        message(STATUS "Ignoring stale QMMA_RDMA_BACKEND=${QMMA_RDMA_BACKEND}; using ${QMMA_DETECTED_RDMA_BACKEND}")
    endif()
endif()
set(QMMA_RDMA_BACKEND "${QMMA_DETECTED_RDMA_BACKEND}")

set(QMMA_HNS_1825_PI_ON_CHIP "$ENV{SHMEM_HNS_1825_PI_ON_CHIP}" CACHE STRING
    "HNS_1825 firmware doorbell mode: empty (SDK default), 0, or 1")
if(QMMA_HNS_1825_PI_ON_CHIP AND NOT QMMA_HNS_1825_PI_ON_CHIP MATCHES "^[01]$")
    message(FATAL_ERROR "QMMA_HNS_1825_PI_ON_CHIP must be empty, 0, or 1")
endif()

# SHMEM's installed headers do not export its backend compile definitions.
# Defining the backend unconditionally is required both by the multi-QP RoCE
# static assertions and by the device-side RDMA backend selection.
set(QMMA_COMPILE_DEFS
    CATLASS_ARCH=3510
    ACLSHMEMI_RDMA_K_BACKEND_${QMMA_RDMA_BACKEND}=1
    ASCEND_TIMER_A2A_MATMUL=1
    CATCCOS_SHMEM_WITH_QP=1
    CATCCOS_HIERARCHICAL_GROUP_SIZE=${CATCCOS_HIERARCHICAL_GROUP_SIZE})

# Installed HNS_1825 headers default to PI-on-chip doorbells. Keep an explicit
# override for build hosts that target legacy firmware; otherwise use the SDK
# default instead of duplicating SHMEM's firmware probe in this example.
if(QMMA_RDMA_BACKEND STREQUAL "HNS_1825")
    if(NOT QMMA_HNS_1825_PI_ON_CHIP STREQUAL "")
        list(APPEND QMMA_COMPILE_DEFS ACLSHMEMI_HNS_1825_PI_ON_CHIP=${QMMA_HNS_1825_PI_ON_CHIP})
        message(STATUS "QMMA HNS_1825 doorbell mode: explicit PI_ON_CHIP=${QMMA_HNS_1825_PI_ON_CHIP}")
    else()
        message(STATUS "QMMA HNS_1825 doorbell mode: installed SDK default (PI_ON_CHIP=1)")
    endif()
elseif(NOT QMMA_HNS_1825_PI_ON_CHIP STREQUAL "")
    message(FATAL_ERROR "QMMA_HNS_1825_PI_ON_CHIP is only valid for the HNS_1825 backend")
endif()

configure_file("${SHMEM_HOME_PATH}/version.info"
    "${CMAKE_BINARY_DIR}/bin/qmma_shmem_version.info" COPYONLY)

catccos_example_add_executable(
    ascend950_quant_matmul_alltoall
    SOURCES main.cpp
    COMPILE_DEFINITIONS ${QMMA_COMPILE_DEFS}
)
