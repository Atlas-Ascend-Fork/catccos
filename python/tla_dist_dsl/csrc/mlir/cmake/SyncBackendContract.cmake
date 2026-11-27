# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


# The signal capability adapter reads the pinned device-state ABI. Bind its
# field layout and team limit to exact headers, including source-only builds.
set(_sync_contract "${TLA_DIST_DSL_ROOT}/catccos/tla_dist/_shmem_contract.json")
file(READ "${_sync_contract}" _sync_contract_json)
string(JSON _sync_header_count LENGTH "${_sync_contract_json}" headers)
math(EXPR _sync_header_last "${_sync_header_count} - 1")
foreach(_sync_index RANGE 0 ${_sync_header_last})
  string(JSON _sync_header MEMBER "${_sync_contract_json}" headers ${_sync_index})
  string(JSON _sync_expected GET "${_sync_contract_json}" headers "${_sync_header}")
  set(_sync_path "${TLA_DIST_DSL_SHMEM_ROOT}/${_sync_header}")
  if(NOT EXISTS "${_sync_path}")
    message(FATAL_ERROR "SHMEM sync state ABI header is missing: ${_sync_path}")
  endif()
  file(SHA256 "${_sync_path}" _sync_actual)
  if(NOT _sync_actual STREQUAL _sync_expected)
    message(FATAL_ERROR "SHMEM sync state ABI does not match the fixed source contract: ${_sync_path}")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_sync_path}")
endforeach()
file(READ "${TLA_DIST_DSL_SHMEM_ROOT}/include/host_device/shmem_common_types.h" _sync_types)
string(REGEX MATCH "#define[ \t]+ACLSHMEM_MAX_TEAMS[ \t]+([0-9]+)" _sync_match "${_sync_types}")
if(NOT _sync_match)
  message(FATAL_ERROR "Cannot resolve ACLSHMEM_MAX_TEAMS from fixed public header")
endif()
set(TLA_DIST_SHMEM_MAX_TEAMS "${CMAKE_MATCH_1}")
string(JSON _sync_max GET "${_sync_contract_json}" max_teams)
if(NOT "${TLA_DIST_SHMEM_MAX_TEAMS}" STREQUAL "${_sync_max}")
  message(FATAL_ERROR "Python and compiler SHMEM team limits disagree")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_sync_contract}")
