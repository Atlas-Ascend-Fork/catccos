# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

# Disable optional timer flags in this example's directory scope only.
set(ENABLE_TIMER OFF)

catccos_example_add_executable(
    ascend950_dispatch
    SOURCES main.cpp
    COMPILE_DEFINITIONS CATLASS_ARCH=3510
)
