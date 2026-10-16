#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))
UTILS_PATH=${PROJECT_ROOT}/examples/utils
CSV_FILE="${SCRIPT_DIR}/test_shapes.csv"

if [ "$#" -ne 1 ]; then
    echo "Usage: bash scripts/run.sh <device_list>" >&2
    exit 1
fi

source $PROJECT_ROOT/3rdparty/shmem/install/set_env.sh || {
    echo "[ERROR] Running set_env.sh in $PROJECT_ROOT/3rdparty/shmem/install failed."
    exit 1
}

IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if [ $RANK_SIZE -gt 8 ]; then
    echo "Rank size is illegal"
    exit 1
fi

cd ${PROJECT_ROOT}/examples/matmul_reduce_scatter/
DATA_DIR=`realpath ./output`
echo "DATA_DIR: $DATA_DIR"
EXEC_BIN=${PROJECT_ROOT}/build/bin/matmul_reduce_scatter

tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N; do
    echo "Processing test case: M=${M}, K=${K}, N=${N}"

    #Generate golden data
    rm -rf ./output/*.bin
    python3 ${UTILS_PATH}/gen_data.py "mmrs" 1 ${RANK_SIZE} ${M} ${N} ${K} 0 0 ${DATA_DIR}

    # Set necessary parameters
    IPPORT="tcp://127.0.0.1:8788"

    # Start Process
    for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
        ${EXEC_BIN} "$RANK_SIZE" "$idx" "$IPPORT" "$M" "$N" "$K" "${DATA_DIR}" "$1" &
    done

    # Wait until all process exit
    wait

    # Verify output
    python3 ${UTILS_PATH}/verify_result.py ${DATA_DIR}/output.bin ${DATA_DIR}/golden.bin 1 ${M} ${N} ${K}
done

cd ${CURRENT_DIR}
