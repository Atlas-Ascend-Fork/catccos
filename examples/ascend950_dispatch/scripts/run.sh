#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")
EXEC_BIN=${PROJECT_ROOT}/build/bin/ascend950_dispatch
CSV_FILE=${SCRIPT_DIR}/test_shapes.csv
DATA_DIR=${PROJECT_ROOT}/examples/ascend950_dispatch/out

source "${PROJECT_ROOT}/3rdparty/shmem/install/set_env.sh"

IFS=',' read -ra DEVICE_IDS <<< "${1:-0,1,2,3,4,5,6,7}"
RANK_SIZE=${#DEVICE_IDS[@]}
DEVICE_LIST=$(IFS=,; echo "${DEVICE_IDS[*]}")
IP_PORT=${IP_PORT:-tcp://127.0.0.1:27029}
LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/catccos-dispatch.XXXXXX")
trap 'rm -rf "$LOG_DIR"' EXIT
mkdir -p "$DATA_DIR"

while IFS=',' read -r NUM_TOKENS HIDDEN GROUP_SIZE TOP_K EXPERTS_PER_RANK COMM_BLOCK_M COMM_INTERVAL ROUTE_MODE; do
    case_failed=0
    if [ "$GROUP_SIZE" -gt "$RANK_SIZE" ]; then
        echo "[ERROR] group_size=${GROUP_SIZE} exceeds rank_size=${RANK_SIZE}"
        exit 1
    fi
    echo "Dispatch M=${NUM_TOKENS}, K=${HIDDEN}, ranks=${RANK_SIZE}, "\
"groupSize=${GROUP_SIZE}, topK=${TOP_K}, expertsPerRank=${EXPERTS_PER_RANK}, "\
"commInterval=${COMM_INTERVAL}, "\
"commBlockM=${COMM_BLOCK_M}"
    rm -f "${DATA_DIR}"/*.bin
    python3 "${SCRIPT_DIR}/get_data.py" "$RANK_SIZE" "$NUM_TOKENS" "$HIDDEN" \
        "$GROUP_SIZE" "$TOP_K" "$EXPERTS_PER_RANK" \
        "$DATA_DIR" --route-mode "${ROUTE_MODE:-balanced}"

    pids=()
    rank_logs=()
    for ((rank = 0; rank < RANK_SIZE; ++rank)); do
        rank_log=${LOG_DIR}/rank_${rank}.log
        "$EXEC_BIN" "$RANK_SIZE" "$rank" "$IP_PORT" "$NUM_TOKENS" "$HIDDEN" \
            "$GROUP_SIZE" "$TOP_K" "$EXPERTS_PER_RANK" \
            "$COMM_INTERVAL" "$COMM_BLOCK_M" "$DATA_DIR" "$DEVICE_LIST" \
            >"$rank_log" 2>&1 &
        pids+=("$!")
        rank_logs+=("$rank_log")
    done
    for ((rank = 0; rank < RANK_SIZE; ++rank)); do
        if ! wait "${pids[$rank]}"; then
            case_failed=1
        fi
    done
    for ((rank = 0; rank < RANK_SIZE; ++rank)); do
        cat "${rank_logs[$rank]}"
    done
    if [ "$case_failed" -ne 0 ]; then
        exit "$case_failed"
    fi
    for ((rank = 0; rank < RANK_SIZE; ++rank)); do
        if ! python3 "${SCRIPT_DIR}/verify.py" "$DATA_DIR" "$rank"; then
            case_failed=1
        fi
    done
    if [ "$case_failed" -ne 0 ]; then
        exit "$case_failed"
    fi
done < <(tail -n +2 "$CSV_FILE")
