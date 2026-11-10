# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

catccos_example_add_executable(
    ascend950_dispatch_ffn_combine
    SOURCES main.cpp
    INCLUDES ../aux_ops ../aux_ops/moe_init_routing_v2
    INCLUDES ../aux_ops ../aux_ops/moe_token_unpermute
    INCLUDES ${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/moe_init_routing_v2
    COMPILE_DEFINITIONS CATLASS_ARCH=3510
)

if(CATCCOS_TORCH_EXTENSION AND CATLASS_BISHENG_ARCH STREQUAL "a5")
    catccos_example_add_library(
        ascend950_dispatch_ffn_combine
        SOURCES ascend950_dispatch_ffn_combine_wrapper.cpp
        INCLUDES ../aux_ops ../aux_ops/moe_init_routing_v2
        INCLUDES ../aux_ops ../aux_ops/moe_token_unpermute
        INCLUDES ${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/moe_init_routing_v2
        COMPILE_DEFINITIONS CATLASS_ARCH=3510
    )
endif()
