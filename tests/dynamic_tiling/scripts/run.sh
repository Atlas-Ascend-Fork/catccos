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
set -e
# usage: bash run.sh [kernel_name] [data_type] [test_start_line] [test_collect_rows] [device_list]
# eg. bash run.sh "mmar" 1 0,1      # 在 0/1 卡上运行 all reduce 精度测试, 数据类型 FP16, rank size = 2
# eg. bash run.sh "agmm" 27 0 10 4,5,6,7  # 在 4/5/6/7 卡上运行 allgather matmul 性能测试, 数据类型 BF16, 从test_shapes.csv的第0个shape开始, 每10个shape采集一次msprof数据, rank size = 4

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))
DATA_PATH=${PROJECT_ROOT}/tests/dynamic_tiling/output
TILING_UTILS_PATH=${PROJECT_ROOT}/tests/dynamic_tiling/utils
UTILS_PATH=${PROJECT_ROOT}/examples/utils
PARENT_PATH=${PROJECT_ROOT}/tests/dynamic_tiling/

# eg. 精度测试WARM_UP_TIMES设置成0, PERF_TEST_CYCLE_TIMES成1
# eg. 性能测试WARM_UP_TIMES设置成10, PERF_TEST_CYCLE_TIMES成3
export FUNC_WARM_UP_TIMES=0
export FUNC_TEST_CYCLE_TIMES=1
export PERF_WARM_UP_TIMES=10
export PERF_TEST_CYCLE_TIMES=3
export SEARCH_PARAMS=1

CSV_FILE="${SCRIPT_DIR}/test_shapes.csv"

source $PROJECT_ROOT/3rdparty/shmem/install/set_env.sh -soc_type Ascend950 || {
    echo "[ERROR] Running set_env.sh in $PROJECT_ROOT/3rdparty/shmem/install failed."
    exit 1
}

NUM_ARGS=$#

case "$NUM_ARGS" in
  3)
    KERNEL_NAME="$1"
    DATA_TYPE="$2"
    TEST_START_LINE=0
    TEST_COLLECT_ROWS=1
    DEVICE_ID_STR="$3"
    TEST_TYPE=0
    ;;

  5)
    KERNEL_NAME="$1"
    DATA_TYPE="$2"
    TEST_START_LINE="$3"
    TEST_COLLECT_ROWS="$4"
    DEVICE_ID_STR="$5"
    TEST_TYPE=1
    ;;

  *)
    echo "Error: invalid number of arguments: $NUM_ARGS"
    usage
    return 1
    ;;
esac

IFS=',' read -ra DEVICE_ID_LIST <<< "$DEVICE_ID_STR"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if [ $RANK_SIZE -gt 8 ]; then
    echo "Rank size is illegal"
    exit 1
fi

cd ${PROJECT_ROOT}/tests/dynamic_tiling/
EXEC_BIN=${PROJECT_ROOT}/build/bin/dynamic_tiling

verify_rank_processes() {
    local failed=0 index status
    for index in "${!verify_pids[@]}"; do
        if wait "${verify_pids[$index]}"; then
            continue
        else
            status=$?
            echo "[ERROR] Precision verification failed for rank ${verify_ranks[$index]} (exit code ${status})." >&2
            failed=1
        fi
    done
    return "$failed"
}

if [ "$TEST_START_LINE" = "0" ]; then
    rm -rf output
    mkdir -p output
    mkdir -p output/tiling
fi

IDX=0

if [ "$TEST_TYPE" = "0" ]; then
    tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N TA TB; do
        if [ "$IDX" -lt "$TEST_START_LINE" ]; then
            (( IDX+=1 ))
            continue
        fi

        echo "Processing test case: M=${M}, K=${K}, N=${N}, TransA=${TA}, TransB=${TB}"

        rm -rf output/*.bin
        case "$KERNEL_NAME" in
            "mmar"|"agmm"|"a5agmm"|"mmrs"|"a5mmrs"|"agmmwg"|"agmmrdma"|"agmmrr")
                python3 ${UTILS_PATH}/gen_data.py ${KERNEL_NAME} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} ${DATA_PATH}
                ;;
            "gmmata"|"a5gmmata")
                python3 ${UTILS_PATH}/gen_data_gmm_alltoallv.py ${KERNEL_NAME} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} --ep ${RANK_SIZE} --expert 8
                ;;
            "atavgmm"|"a5atavgmm")
                python3 ${UTILS_PATH}/gen_data_alltoallv_gmm.py ${KERNEL_NAME} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} ${DATA_PATH} --ep ${RANK_SIZE} --expert 8
                ;;
            "atavgmmv2")
                python3 ${UTILS_PATH}/gen_data_alltoallv_gmm.py ${KERNEL_NAME} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} ${DATA_PATH} --ep ${RANK_SIZE} --expert 8
                ;;
            "agmmdqbs")
                python3 ${UTILS_PATH}/gen_allgather_quant_data.py ${KERNEL_NAME} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} 0 0 ${DATA_PATH}
                ;;
            "agmmdq")
                python3 ${UTILS_PATH}/gen_allgather_fixpipe_quant_data.py ${KERNEL_NAME} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} 0 0 ${DATA_PATH}
                ;;
            "mmdqrs")
                python3 ${UTILS_PATH}/gen_double_golden_data_mmdqrs.py ${KERNEL_NAME} 1 ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} ${DATA_PATH}
                ;;
            "a5fp8mxagmm")
                python3 ${UTILS_PATH}/gen_data_fp8_mx_allgather_matmul.py ${KERNEL_NAME} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} ${DATA_PATH}
                ;;
            "a5fp4mxagmm")
                python3 ${UTILS_PATH}/gen_data_fp4_mx_allgather_matmul.py ${KERNEL_NAME} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} ${DATA_PATH}
                ;;
            "a5fp8gmmata")
                python3 ${UTILS_PATH}/gen_data_gmm_alltoallv_fp8_mx.py ${KERNEL_NAME} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} --ep ${RANK_SIZE} --expert 8
                ;;
            "a5fp4gmmata")
                python3 ${UTILS_PATH}/gen_data_gmm_alltoallv_fp4_mx.py ${KERNEL_NAME} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} --ep ${RANK_SIZE} --expert 8
                ;;
            "a5fp8mxatavgmm")
 	            python3 ${UTILS_PATH}/gen_data_fp8_mx_alltoallv_gmm.py ${KERNEL_NAME} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} ${DATA_PATH} --ep ${RANK_SIZE} --expert 8
                ;;
            "a5fp4mxatavgmm")
 	            python3 ${UTILS_PATH}/gen_data_fp4_mx_alltoallv_gmm.py ${KERNEL_NAME} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} ${DATA_PATH} --ep ${RANK_SIZE} --expert 8
                ;;
            "a5fp4mmrs")
                python3 ${UTILS_PATH}/gen_data_fp4_mx.py ${KERNEL_NAME} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} ${DATA_PATH}
                ;;
            "a5agmmudma")
                python3 ${UTILS_PATH}/gen_data_allgather_matmul.py ${KERNEL_NAME} ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} ${DATA_PATH}
                ;;
            "a5mxfp8atammsplitkurma")
                python3 ${PROJECT_ROOT}/examples/ascend950_mxfp8_alltoall_matmul_split_k_urma/scripts/gen_data.py ${DATA_TYPE} ${RANK_SIZE} ${M} ${N} ${K} ${TA} ${TB} ${DATA_PATH} --device-id-list "$DEVICE_ID_STR"
                ;;
        esac

        # Set necessary parameters
        IPPORT="tcp://127.0.0.1:27048"

        # Start Process
        for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
            APP="$EXEC_BIN $KERNEL_NAME $DATA_TYPE $RANK_SIZE $idx $IPPORT $M $N $K $TEST_START_LINE $TEST_COLLECT_ROWS $PARENT_PATH $CSV_FILE $FUNC_WARM_UP_TIMES $FUNC_TEST_CYCLE_TIMES $DEVICE_ID_STR $DATA_PATH"
            ${APP}&
        done

        # Wait until all process exit
        wait

        if [ "$KERNEL_NAME" = "agmm" -o "$KERNEL_NAME" = "a5agmm" -o "$KERNEL_NAME" = "agmmrdma" -o "$KERNEL_NAME" = "agmmrr" ]; then
            python3 ${UTILS_PATH}/verify_result.py ./output/output.bin ./output/golden.bin ${DATA_TYPE} ${M} ${N} ${K}
        elif [ "$KERNEL_NAME" = "agmmwg" ]; then
            python3 ${UTILS_PATH}/verify_result.py ./output/output.bin ./output/golden.bin ${DATA_TYPE} ${M} ${N} ${K}
            # Gathered A has shape [RANK_SIZE * M, K], so verify it with K as the row width.
            python3 ${UTILS_PATH}/verify_result.py ./output/output_gather_a.bin ./output/gather_a.bin ${DATA_TYPE} ${M} ${K} ${K}
        elif [ "$KERNEL_NAME" = "agmmdq" -o "$KERNEL_NAME" = "agmmdqbs" ]; then
            verify_pids=()
            verify_ranks=()
            for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1)); do
                python3 ${UTILS_PATH}/verify_result.py ./output/output_rank${idx}.bin ./output/golden_rank${idx}.bin 1 ${M} ${N} ${K} &
                verify_pids+=("$!")
                verify_ranks+=("${idx}")
            done
            if ! verify_rank_processes; then
                exit 1
            fi
        elif [ "$KERNEL_NAME" = "mmdqrs" ]; then
            python3 ${UTILS_PATH}/verify_result.py ./output/output.bin ./output/golden_fp32.bin 1 ${M} ${N} ${K} --golden_low=./output/golden_aclnn.bin
        elif [ "$KERNEL_NAME" = "atagmm" -o "$KERNEL_NAME" = "gmmata" -o "$KERNEL_NAME" = "atavgmm" -o "$KERNEL_NAME" = "atavgmmv2" -o "$KERNEL_NAME" = "a5atavgmm" \
            -o "$KERNEL_NAME" = "a5fp8mxagmm" -o "$KERNEL_NAME" = "a5fp4mxagmm" -o "$KERNEL_NAME" = "a5fp8mxatavgmm" -o "$KERNEL_NAME" = "a5fp4mxatavgmm" \
            -o "$KERNEL_NAME" = "a5gmmata" -o "$KERNEL_NAME" = "a5fp8gmmata" -o "$KERNEL_NAME" = "a5fp4gmmata" -o "$KERNEL_NAME" = "a5agmmudma" ]; then
            verify_pids=()
            verify_ranks=()
            for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1)); do
                python3 ${UTILS_PATH}/verify_result.py ./output/output_${idx}.bin ./output/golden_${idx}.bin ${DATA_TYPE} ${M} ${N} ${K} &
                verify_pids+=("$!")
                verify_ranks+=("${idx}")
            done
            if ! verify_rank_processes; then
                exit 1
            fi
        elif [ "$KERNEL_NAME" = "a5mxfp8atammsplitkurma" ]; then
            verify_pids=()
            verify_ranks=()
            for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1)); do
                python3 ${UTILS_PATH}/verify_result.py ./output/output_${idx}.bin ./output/golden_${idx}.bin ${DATA_TYPE} $((M / RANK_SIZE)) ${N} $((K * RANK_SIZE)) &
                verify_pids+=("$!")
                verify_ranks+=("${idx}")
            done
            if ! verify_rank_processes; then
                exit 1
            fi
        else
            python3 ${UTILS_PATH}/verify_result.py ./output/output.bin ./output/golden.bin ${DATA_TYPE} ${M} ${N} $((K * RANK_SIZE))
        fi

        (( TEST_START_LINE+=TEST_COLLECT_ROWS ))
        (( IDX+=1 ))
    done
else
    tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N TA TB; do
        if [ "$IDX" -lt "$TEST_START_LINE" ]; then
            (( IDX+=1 ))
            continue
        fi

        echo "Processing test case: M=${M}, K=${K}, N=${N}, TransA=${TA}, TransB=${TB}"

        # Set necessary parameters
        IPPORT="tcp://127.0.0.1:27049"

        OUTPUT_PATH="./output/msprof/start_line${IDX}_run_rows${TEST_COLLECT_ROWS}/"

        # Start Process
        rm -rf output/*.bin
        case "$KERNEL_NAME" in
            "atavgmm")
                python3 ${UTILS_PATH}/gen_uniform_tokens_table.py ${RANK_SIZE} ${M} ${N} ${K} --expert 8 --ep ${RANK_SIZE}
                ;;
        esac

        for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
            APP="$EXEC_BIN $KERNEL_NAME $DATA_TYPE $RANK_SIZE $idx $IPPORT $M $N $K $TEST_START_LINE $TEST_COLLECT_ROWS $PARENT_PATH $CSV_FILE $PERF_WARM_UP_TIMES $PERF_TEST_CYCLE_TIMES $DEVICE_ID_STR"
            msprof --application="${APP}" --output="${OUTPUT_PATH}"&
        done

        # Wait until all process exit
        wait

        python3 ${TILING_UTILS_PATH}/process_data.py "${OUTPUT_PATH}"

        (( TEST_START_LINE+=TEST_COLLECT_ROWS ))
        (( IDX+=1 ))
    done
    python3 ${TILING_UTILS_PATH}/get_best_result.py "${CSV_FILE}"

    python3 ${TILING_UTILS_PATH}/process.py "./output_timer/" "${KERNEL_NAME}"
fi

cd ${CURRENT_DIR}
