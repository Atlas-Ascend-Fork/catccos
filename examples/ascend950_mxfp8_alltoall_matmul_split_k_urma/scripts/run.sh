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
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")
UTILS_PATH=${PROJECT_ROOT}/examples/utils
CSV_FILE="${SCRIPT_DIR}/test_shapes.csv"
KERNEL_VARIANT=${CATCCOS_SPLIT_K_URMA_VARIANT:-default}

case "$KERNEL_VARIANT" in
    default)
        EXECUTABLE_NAME=ascend950_mxfp8_alltoall_matmul_split_k_urma
        GENERATOR_ARGS=()
        ;;
    local_atomic)
        EXECUTABLE_NAME=ascend950_mxfp8_alltoall_matmul_split_k_urma_local_atomic
        GENERATOR_ARGS=(--local-atomic-golden)
        ;;
    *)
        echo "[ERROR] CATCCOS_SPLIT_K_URMA_VARIANT must be default or local_atomic"
        exit 1
        ;;
esac

source "$PROJECT_ROOT/3rdparty/shmem/install/set_env.sh" || {
    echo "[ERROR] Sourcing set_env.sh in $PROJECT_ROOT/3rdparty/shmem/install failed."
    exit 1
}

IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if [ "$RANK_SIZE" -gt 8 ]; then
    echo "Rank size is illegal"
    exit 1
fi

cd "${PROJECT_ROOT}/examples/ascend950_mxfp8_alltoall_matmul_split_k_urma/" || exit 1

DATA_DIR=$(realpath ./out)
echo "DATA_DIR: $DATA_DIR"
mkdir -p "$DATA_DIR"
EXEC_BIN=${PROJECT_ROOT}/build/bin/${EXECUTABLE_NAME}

# out_dtype: 1 = FP16 (matches utils.DataType.FLOAT16)
OUT_DTYPE=1

tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N; do
    echo "Processing MXFP8 split-K URMA ${KERNEL_VARIANT} test case: M=${M}, localK=${K}, N=${N}"

    if [ $((M % RANK_SIZE)) -ne 0 ]; then
        echo "[ERROR] M=${M} must be divisible by RANK_SIZE=${RANK_SIZE}"
        continue
    fi

    CHUNK_M=$((M / RANK_SIZE))
    FULL_K=$((K * RANK_SIZE))

    rm -rf ./out/*.bin
    python3 "${SCRIPT_DIR}/gen_data.py" "${OUT_DTYPE}" "${RANK_SIZE}" "${M}" "${N}" "${K}" 0 1 "${DATA_DIR}" \
        --device-id-list "$1" "${GENERATOR_ARGS[@]}" || exit 1

    IPPORT="tcp://127.0.0.1:8788"

    for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        "${EXEC_BIN}" "$RANK_SIZE" "$idx" "$IPPORT" "$M" "$N" "$K" "${DATA_DIR}" "$1" &
    done

    wait

    for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        echo "========== Verify rank ${idx} =========="
        if [ "$KERNEL_VARIANT" = "local_atomic" ]; then
            python3 "${UTILS_PATH}/verify_result.py" "${DATA_DIR}/output_${idx}.bin" \
                "${DATA_DIR}/golden_${idx}.bin" "${OUT_DTYPE}" "${CHUNK_M}" "${N}" "${FULL_K}" \
                --golden_low "${DATA_DIR}/golden_local_atomic_${idx}.bin" || exit 1
        else
            python3 "${UTILS_PATH}/verify_result.py" "${DATA_DIR}/output_${idx}.bin" \
                "${DATA_DIR}/golden_${idx}.bin" "${OUT_DTYPE}" "${CHUNK_M}" "${N}" "${FULL_K}" || exit 1
        fi
    done
done || exit 1

cd "${CURRENT_DIR}" || exit 1
