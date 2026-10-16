#!/bin/bash
set -e
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))
CMAKE_EXTRA_ARGS=("$@")

# setup.sh consumes the caller's positional parameters when sourced without
# explicit arguments. Clear them here and use the saved copy for CMake below.
set --

source $PROJECT_ROOT/examples/utils/setup.sh || {
    echo "[ERROR] Running setup.sh in $PROJECT_ROOT/examples/utils failed."
    exit 1
}

SOURCE_DIR=$PROJECT_ROOT
BUILD_DIR=$PROJECT_ROOT/build
mkdir -p $BUILD_DIR
cmake -B "$BUILD_DIR" -S "$SOURCE_DIR" -DCATCCOS_BUILD_TESTS=OFF "${CMAKE_EXTRA_ARGS[@]}"
cmake --build "$BUILD_DIR" --target matmul_reduce_scatter --clean-first -j
