#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -eo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "$PROJECT_ROOT"
RDMA_BACKEND=${SHMEM_RDMA_BACKEND:-HNS_1825}
HNS_PI_ON_CHIP=${SHMEM_HNS_1825_PI_ON_CHIP:-}
case "$RDMA_BACKEND" in
    HNS_1825)
        if [[ "$HNS_PI_ON_CHIP" != 0 && "$HNS_PI_ON_CHIP" != 1 ]]; then
            echo "[ERROR] Set SHMEM_HNS_1825_PI_ON_CHIP=0 or 1 to match the SHMEM SDK and target firmware." >&2
            exit 1
        fi
        ;;
    XSCALE)
        if [ -n "$HNS_PI_ON_CHIP" ]; then
            echo "[ERROR] SHMEM_HNS_1825_PI_ON_CHIP is only valid for HNS_1825." >&2
            exit 1
        fi
        ;;
    *)
        echo "[ERROR] SHMEM_RDMA_BACKEND must be HNS_1825 or XSCALE." >&2
        exit 1
        ;;
esac
read -r -a shmem_args <<< "${SHMEM_BUILD_ARGS:--enable_rdma -rdma_backend ${RDMA_BACKEND} -multi_link}"
source "${PROJECT_ROOT}/examples/utils/setup.sh" -soc_type Ascend950 "${shmem_args[@]}"
set -u
cmake -B "${PROJECT_ROOT}/build" -S "${PROJECT_ROOT}" -DCATCCOS_BUILD_TESTS=OFF -DCATLASS_BISHENG_ARCH=a5 \
    -DSHMEM_HOME_PATH="${SHMEM_HOME_PATH}" \
    -DCATCCOS_RDMA_BACKEND="${RDMA_BACKEND}" \
    -DCATCCOS_HNS_1825_PI_ON_CHIP="${HNS_PI_ON_CHIP}" \
    -DENABLE_TIMER=OFF "$@"
# Device libraries are generated from main.cpp and do not track header-only changes.
rm -f "${PROJECT_ROOT}/build/bin/liba2a_matmul_impl.so"
cmake --build "${PROJECT_ROOT}/build" --target a2a_matmul -j "${BUILD_JOBS:-2}"
cp "${SHMEM_HOME_PATH}/version.info" "${PROJECT_ROOT}/build/bin/mxfp8_alltoall_shmem_version.info"
