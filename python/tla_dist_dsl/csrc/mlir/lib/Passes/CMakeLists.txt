# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_library(TlaDistPasses STATIC
  TlaDistLowerCommPass.cpp
  TlaDistSyncContractPass.cpp
  ../Dialect/TlaDist/IR/TlaDistDialect.cpp
  ../Dialect/TlaDist/IR/TlaDistOps.cpp
)
set_target_properties(TlaDistPasses PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(TlaDistPasses PRIVATE
  TLA_DIST_SHMEM_MAX_TEAMS=${TLA_DIST_SHMEM_MAX_TEAMS}
  TLA_DIST_ENABLE_RDMA=${TLA_DIST_ENABLE_RDMA}
  TLA_DIST_RDMA_BACKEND_HNS_1825=$<STREQUAL:${TLA_DIST_RDMA_BACKEND},HNS_1825>
)
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
