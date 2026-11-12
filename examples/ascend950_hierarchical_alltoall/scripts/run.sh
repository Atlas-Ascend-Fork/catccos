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
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))

source $PROJECT_ROOT/examples/utils/setup.sh -soc_type Ascend950 || {
    echo "[ERROR] Running setup.sh in $PROJECT_ROOT/examples/utils failed."
    exit 1
}

if [ $# -ne 1 ]; then
    echo "Usage: [NODE_RANK=0 NODE_COUNT=1 GROUP_SIZE=16 IPPORT=tcp://RANK0_HOST:8788] \\"
    echo "       [QP_NUM=1|2|4] bash scripts/run.sh <local_device_list>"
    exit 1
fi

IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
LOCAL_RANKS=${#DEVICE_ID_LIST[@]}

# SHMEM maps PE id to the rootinfo rank, and the rootinfo rank is the local device id here.
# A local rank therefore has to run on device <rank>, otherwise the process would attach to
# another PE's symmetric heap while still addressing its own local buffers.
for ((idx = 0; idx < LOCAL_RANKS; ++idx)); do
    if [ "${DEVICE_ID_LIST[idx]}" != "${idx}" ]; then
        echo "[ERROR] single-node device list must be 0,1,...,$((LOCAL_RANKS - 1)); got $1" >&2
        echo "[ERROR] rank r must run on device r, because SHMEM PE r is bound to rootinfo rank r" >&2
        exit 1
    fi
done
NODE_RANK=${NODE_RANK:-0}
NODE_COUNT=${NODE_COUNT:-1}
GROUP_SIZE=${GROUP_SIZE:-${LOCAL_RANKS}}
export QP_NUM=${QP_NUM-1}
case "${QP_NUM}" in
    1|2|4) ;;
    *) echo "[ERROR] QP_NUM must be 1, 2 or 4" >&2; exit 1 ;;
esac
IPPORT=${IPPORT:-"tcp://127.0.0.1:8788"}
PYTHON=${PYTHON:-python3}
CSV_FILE=${CSV_FILE:-"${SCRIPT_DIR}/test_shapes.csv"}
LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/logs/$(date +%Y%m%d-%H%M%S)-node${NODE_RANK}-qp${QP_NUM}"}

WORLD_SIZE=$((NODE_COUNT * LOCAL_RANKS))
if [ ${LOCAL_RANKS} -lt 1 ] || [ ${LOCAL_RANKS} -gt 16 ] || [ ${NODE_COUNT} -lt 1 ] || \
    [ ${NODE_RANK} -lt 0 ] || [ ${NODE_RANK} -ge ${NODE_COUNT} ]; then
    echo "[ERROR] device list must contain 1..16 distinct local devices and NODE_RANK must be in [0, NODE_COUNT)"
    exit 1
fi
if [ ${WORLD_SIZE} -gt 32 ]; then
    echo "[ERROR] rank_size (${WORLD_SIZE}) must be <= 32 for this example"
    exit 1
fi
if [ ${GROUP_SIZE} -lt 1 ] || [ ${GROUP_SIZE} -gt ${WORLD_SIZE} ]; then
    echo "[ERROR] GROUP_SIZE must be in [1, rank_size=${WORLD_SIZE}]"
    exit 1
fi
if [ ${GROUP_SIZE} -ne ${LOCAL_RANKS} ]; then
    echo "[ERROR] GROUP_SIZE must equal local rank count=${LOCAL_RANKS} for this UBX launcher." >&2
    echo "[ERROR] Splitting a server into RDMA groups may hit unsupported same-port HNS1825 traffic." >&2
    exit 1
fi
if [ ${NODE_COUNT} -gt 1 ] && [[ "${IPPORT}" == tcp://127.* || "${IPPORT}" == tcp://localhost:* ]]; then
    echo "[ERROR] Multi-node run needs the rank0 server address, use tcp://RANK0_HOST:8788." >&2
    exit 1
fi

if ! cmp -s "${SHMEM_HOME_PATH}/version.info" "${PROJECT_ROOT}/build/bin/hierarchical_shmem_version.info"; then
    echo "[ERROR] SHMEM SDK differs from the one used to compile the kernel; rebuild this example." >&2
    exit 1
fi
EXEC_BIN=${PROJECT_ROOT}/build/bin/ascend950_hierarchical_alltoall
if [ ! -x ${EXEC_BIN} ]; then
    echo "[ERROR] ${EXEC_BIN} not found, run scripts/build.sh first."
    exit 1
fi

# A rank-0 that cannot listen makes every other rank report
# "Failed to connect to <ip:port> after tried 20000 times".
if [[ "${IPPORT}" == tcp://127.0.0.1:* || "${IPPORT}" == tcp://localhost:* ]]; then
    PORT="${IPPORT##*:}"
    if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p"$"' | grep -q .; then
        echo "[ERROR] ${IPPORT} is already in use; stop the stale process or set another IPPORT."
        exit 1
    fi
fi
echo "node=${NODE_RANK}/${NODE_COUNT} world=${WORLD_SIZE} group=${GROUP_SIZE} qp_num=${QP_NUM} logs=${LOG_DIR}"
mkdir -p "${LOG_DIR}"

status=0
while IFS=',' read -r M N; do
    if [ -z "${M}" ] || [ -z "${N}" ]; then
        continue
    fi
    if ! [[ "${M}" =~ ^[0-9]+$ && "${N}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] CSV cases must contain exactly two non-negative integers: m,n" >&2
        status=1
        break
    fi

    if [ "${M}" -eq 0 ] || [ "${N}" -eq 0 ] || [ $((N % 32)) -ne 0 ]; then
        echo "[ERROR] m/n must be positive and n must be divisible by 32" >&2
        status=1
        break
    fi

    case_dir="${LOG_DIR}/world${WORLD_SIZE}_m${M}_n${N}"
    mkdir -p "${case_dir}"
    DATA_DIR="$(cd "${case_dir}" && pwd)/data"
    if ! "${PYTHON}" "${SCRIPT_DIR}/gen_data.py" "${WORLD_SIZE}" "${M}" "${N}" "${DATA_DIR}" \
        --rank-start "$((NODE_RANK * LOCAL_RANKS))" --rank-count "${LOCAL_RANKS}"; then
        status=1
        break
    fi
    pids=()
    for (( idx = 0; idx < LOCAL_RANKS; idx = idx + 1 )); do
        RANK=$((NODE_RANK * LOCAL_RANKS + idx))
        "${EXEC_BIN}" "${WORLD_SIZE}" "${RANK}" "${IPPORT}" "${M}" "${N}" \
            "${GROUP_SIZE}" "${DATA_DIR}" "$1" >"${case_dir}/rank${RANK}.log" 2>&1 &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid"
        rank_status=$?
        if [ ${rank_status} -ne 0 ]; then
            failed=1
        fi
    done
    for (( idx = 0; idx < LOCAL_RANKS; idx = idx + 1 )); do
        RANK=$((NODE_RANK * LOCAL_RANKS + idx))
        if ! "${PYTHON}" "${SCRIPT_DIR}/verify_result.py" "${DATA_DIR}" "${RANK}" \
            >>"${case_dir}/rank${RANK}.log" 2>&1; then
            failed=1
        fi
    done
    if [ ${failed} -ne 0 ]; then
        cat "${case_dir}"/rank*.log
        echo "[FAIL] m=${M} n=${N}; logs: ${case_dir}"
        status=1
        # A peer may already have stopped. Do not reuse this bootstrap endpoint for
        # the next CSV case while the other node is still waiting in the failed case.
        break
    else
        # Keep individual results in rank logs; print one summary per local case.
        # Preserve transport information and any warnings from the runtime.
        sed '/^PASS rank=[0-9][0-9]*$/d' "${case_dir}"/rank*.log
        echo "[PASS] m=${M} n=${N}"
    fi
done < <(tail -n +2 "$CSV_FILE")

cd "$CURRENT_DIR"
exit ${status}
