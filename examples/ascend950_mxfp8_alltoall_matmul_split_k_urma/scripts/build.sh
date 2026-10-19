#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")

source "$PROJECT_ROOT/examples/utils/setup.sh" -soc_type Ascend950 || {
    echo "[ERROR] Running setup.sh in $PROJECT_ROOT/examples/utils failed."
    exit 1
}

SOURCE_DIR=$PROJECT_ROOT
BUILD_DIR=$PROJECT_ROOT/build
BUILD_VARIANT=${1:-default}
case "$BUILD_VARIANT" in
    default)
        BUILD_TARGETS=(ascend950_mxfp8_alltoall_matmul_split_k_urma)
        ;;
    local_atomic)
        BUILD_TARGETS=(ascend950_mxfp8_alltoall_matmul_split_k_urma_local_atomic)
        ;;
    all)
        BUILD_TARGETS=(
            ascend950_mxfp8_alltoall_matmul_split_k_urma
            ascend950_mxfp8_alltoall_matmul_split_k_urma_local_atomic
        )
        ;;
    *)
        echo "Usage: $0 [default|local_atomic|all]"
        exit 1
        ;;
esac

mkdir -p "$BUILD_DIR"
cmake -B "$BUILD_DIR" -S "$SOURCE_DIR" -DCATCCOS_BUILD_TESTS=OFF -DCATLASS_BISHENG_ARCH=a5
cmake --build "$BUILD_DIR" --target "${BUILD_TARGETS[@]}" -j
