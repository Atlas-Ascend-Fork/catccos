# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

catccos_example_add_executable(
    ascend950_mxfp8_alltoall_matmul_split_k_urma
    SOURCES main.cpp
    COMPILE_DEFINITIONS CATLASS_ARCH=3510
)

catccos_example_add_executable(
    ascend950_mxfp8_alltoall_matmul_split_k_urma_local_atomic
    SOURCES main.cpp
    COMPILE_DEFINITIONS CATLASS_ARCH=3510 CATCCOS_LOCAL_ATOMIC=1
)
