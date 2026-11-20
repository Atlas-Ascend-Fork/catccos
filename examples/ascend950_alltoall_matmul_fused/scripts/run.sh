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

usage() {
    cat <<'EOF'
Usage: bash scripts/run.sh [device_id_list]
  DTYPE=mxfp8       mxfp8 or bf16; both use the same executable.
  BF16 defaults: M=3667 N=3328 K=12288 ITERATIONS=1 SEED=1.
  OVERLAP=1 measures both formats with 5 warmups and 20 samples by default.
  Default topology: two nodes, 16 devices per node, physical devices 0..15.
  LOCAL_RANKS=16      Devices per node: 1, 2, 4, 8, or 16; equal on both nodes.
                     Must match GROUP_SIZE selected when building.
  START_DEVICE=0      First physical device; select LOCAL_RANKS consecutive devices.
  device_id_list     Optional explicit list; cannot be combined with START_DEVICE.
  NODE_COUNT=2       Number of nodes: 1 or 2.
  WORLD_SIZE=<ranks> Optional total; otherwise NODE_COUNT * LOCAL_RANKS.
                     When NODE_COUNT is omitted, WORLD_SIZE determines it.
  NODE_RANK=0         This host's index: 0 or 1.
  IPPORT=tcp://<rank0-host>:12560  Both hosts use the same rank-0 endpoint.
  M=10000           Output rows per rank, in [1,300000]; input A has WORLD_SIZE*M rows.
  K=<columns>       Full K after AllToAll; MXFP8 default WORLD_SIZE*256, local K is a multiple of 64.
                     BF16 requires K to be divisible by WORLD_SIZE.
  N=4096            Output columns, positive multiple of 16.
  CSV_FILE=<path>   Optional DTYPE,M,K,N case list, filtered by DTYPE. Legacy M,K,N also works.
  ITERATIONS=20     Includes the first launch (iter=0) and repeated launches.
  WINDOW_ROWS=512   Communication window rows: 256, 512, or 1024; MatMul tile M remains 256.
  DATA_PATH=<dir>   Input/output files; omitted: generate periodic inputs under LOG_DIR/data.
                    Check the first and last row of every 256-row compute tile, across all columns.
  PYTHON_BIN=python3  Python 3.10+ interpreter for data generation and checks.
  RUN_TIMEOUT=300   Timeout in seconds for each rank.
  LOG_DIR=<path>    Rank logs, preflight, manifest, and timing summary.
  CHECK_RESULT=1    Check sampled output after the final call (0 skips that CPU check).
  SHMEM_HEAP_MB=0   Per-rank heap in MiB; 0 sizes it automatically from the shape.
Only selected NPUs must be idle. Device-query errors stop the run.
EOF
}

fail() { echo "[ERROR] $*" >&2; exit 1; }
if [ "${1:-}" = -h ] || [ "${1:-}" = --help ]; then usage; exit 0; fi
if [ "$#" -gt 1 ]; then usage; exit 1; fi
[ -z "${CHUNK_M+x}" ] || fail "CHUNK_M is no longer supported; use M for output rows per rank"
PYTHON_BIN=${PYTHON_BIN:-python3}
DTYPE=${DTYPE:-mxfp8}
OVERLAP=${OVERLAP:-0}
SEED=${SEED:-1}
WARMUP=${WARMUP:-5}
case "$DTYPE" in
    mxfp8) K_PER_RANK_MULTIPLE=64 ;;
    bf16) K_PER_RANK_MULTIPLE=1 ;;
    *) fail "DTYPE must be mxfp8 or bf16" ;;
esac
[[ $OVERLAP == 0 || $OVERLAP == 1 ]] || fail 'OVERLAP must be 0 or 1'
export DTYPE OVERLAP
if [ "$#" -eq 1 ]; then
    [ -z "${START_DEVICE:-}" ] || fail "use either device_id_list or START_DEVICE"
    DEVICE_LIST=$1
else
    LOCAL_RANKS=${LOCAL_RANKS:-16}
    case "$LOCAL_RANKS" in 1|2|4|8|16) ;; *) fail "LOCAL_RANKS must be 1, 2, 4, 8, or 16" ;; esac
    START_DEVICE=${START_DEVICE:-0}
    [[ "$START_DEVICE" =~ ^([0-9]|1[0-5])$ ]] || fail "START_DEVICE must be in [0,15]"
    (( START_DEVICE + LOCAL_RANKS <= 16 )) || fail "selected devices exceed physical device 15"
    DEVICE_LIST=$START_DEVICE
    for ((device = START_DEVICE + 1; device < START_DEVICE + LOCAL_RANKS; ++device)); do
        DEVICE_LIST+=",$device"
    done
fi
[[ "$DEVICE_LIST" =~ ^([0-9]|1[0-5])(,([0-9]|1[0-5]))*$ ]] || fail "device IDs must be in [0,15]"
IFS=',' read -r -a DEVICE_IDS <<< "$DEVICE_LIST"
[ -z "${LOCAL_RANKS:-}" ] || [ "$LOCAL_RANKS" = "${#DEVICE_IDS[@]}" ] || fail "LOCAL_RANKS disagrees with device list"
LOCAL_RANKS=${#DEVICE_IDS[@]}
case "$LOCAL_RANKS" in 1|2|4|8|16) ;; *) fail "use 1, 2, 4, 8, or 16 local devices" ;; esac
if [ -n "${WORLD_SIZE:-}" ]; then
    case "$WORLD_SIZE" in 1|2|4|8|16|32) ;; *) fail "WORLD_SIZE must be 1, 2, 4, 8, 16, or 32" ;; esac
    (( WORLD_SIZE % LOCAL_RANKS == 0 )) || fail "WORLD_SIZE must be divisible by LOCAL_RANKS"
    [ -z "${NODE_COUNT:-}" ] || [ "$NODE_COUNT" = "$((WORLD_SIZE / LOCAL_RANKS))" ] ||
        fail "NODE_COUNT disagrees with WORLD_SIZE / LOCAL_RANKS"
    NODE_COUNT=$((WORLD_SIZE / LOCAL_RANKS))
else
    NODE_COUNT=${NODE_COUNT:-2}
fi
case "$NODE_COUNT" in 1|2) ;; *) fail "NODE_COUNT must be 1 or 2" ;; esac
RANK_SIZE=$((NODE_COUNT * LOCAL_RANKS))
NODE_RANK=${NODE_RANK:-0}
[[ "$NODE_RANK" =~ ^[01]$ ]] && (( NODE_RANK < NODE_COUNT )) || fail "NODE_RANK must be in [0,$((NODE_COUNT - 1))]"
declare -A SEEN_DEVICES=()
for device in "${DEVICE_IDS[@]}"; do
    [ -z "${SEEN_DEVICES[$device]:-}" ] || fail "duplicate device ID: $device"
    SEEN_DEVICES[$device]=1
done

# M always means output rows per rank, including CSV cases.
if [ -z "${M:-}" ] && [ -n "${CSV_FILE:-}" ]; then
    CASES=$("$PYTHON_BIN" - "$CSV_FILE" "$DTYPE" "$RANK_SIZE" "$K_PER_RANK_MULTIPLE" <<'PY'
import csv
import sys
sys.stdout.reconfigure(newline='\n')
path, dtype, world, k_per_rank_multiple = sys.argv[1:]
k_multiple = int(world) * int(k_per_rank_multiple)
fields = ['M', 'K', 'N']
with open(path, encoding='utf-8-sig', newline='') as source:
    reader = csv.DictReader(source)
    has_dtype = reader.fieldnames == ['DTYPE', *fields]
    if not has_dtype and reader.fieldnames != fields:
        raise SystemExit('CSV header must be DTYPE,M,K,N or M,K,N')
    matched = False
    for row in reader:
        values = [row.get(key) or '' for key in fields]
        if None in row or any(not value.isascii() or not value.isdecimal() for value in values):
            raise SystemExit('CSV requires three unsigned integers per row')
        if has_dtype:
            if row['DTYPE'] not in ('mxfp8', 'bf16'):
                raise SystemExit('CSV DTYPE must be mxfp8 or bf16')
            if row['DTYPE'] != dtype:
                continue
        m, k, n = map(int, values)
        if not 0 < m <= 300000 or not 0 < k <= 0xffffffff or k % k_multiple or not 0 < n <= 0xffffffff or n % 16:
            raise SystemExit(f'CSV requires M in [1,300000], K%{k_multiple}=0, N%16=0')
        print(m, k, n)
        matched = True
    if not matched:
        raise SystemExit(f'CSV contains no shapes for DTYPE={dtype}')
PY
    ) || fail "invalid shape CSV: $CSV_FILE"
    CASE_LOG_ROOT=${LOG_DIR:-"${SCRIPT_DIR}/../out/csv-$(date +%Y%m%d-%H%M%S)-node${NODE_RANK}-$$"}
    while read -r case_m case_k case_n; do
        START_DEVICE= WORLD_SIZE="$RANK_SIZE" NODE_COUNT="$NODE_COUNT" M="$case_m" K="$case_k" N="$case_n" \
            LOG_DIR="$CASE_LOG_ROOT/m${case_m}-k${case_k}-n${case_n}" bash "$SCRIPT_DIR/run.sh" "$DEVICE_LIST"
    done <<< "$CASES"
    exit 0
fi

DEFAULT_ITERATIONS=20
if [[ $DTYPE == bf16 ]]; then
    M=${M:-3667}
    K=${K:-12288}
    N=${N:-3328}
    DEFAULT_ITERATIONS=1
    [[ $SEED =~ ^[0-9]+$ ]] || fail "SEED must be an unsigned integer"
else
    M=${M:-10000}
    K=${K:-$((RANK_SIZE * 256))}
    N=${N:-4096}
fi
if [[ $OVERLAP == 1 ]]; then DEFAULT_ITERATIONS=25; fi
K_MULTIPLE=$((RANK_SIZE * K_PER_RANK_MULTIPLE))
[[ "$K" =~ ^[1-9][0-9]{0,9}$ ]] && (( K <= 4294967295 && K % K_MULTIPLE == 0 )) ||
    fail "K must be a positive uint32 multiple of $K_MULTIPLE"
[[ "$N" =~ ^[1-9][0-9]{0,9}$ ]] && (( N <= 4294967295 && N % 16 == 0 )) ||
    fail "N must be a positive uint32 multiple of 16"
LOCAL_K=$((K / RANK_SIZE))
ITERATIONS=${ITERATIONS:-$DEFAULT_ITERATIONS}
RUN_TIMEOUT=${RUN_TIMEOUT:-300}
CHECK_RESULT=${CHECK_RESULT:-1}
SHMEM_HEAP_MB=${SHMEM_HEAP_MB:-0}
[[ "$SHMEM_HEAP_MB" =~ ^(0|[1-9][0-9]{0,9})$ ]] && (( SHMEM_HEAP_MB <= 4294967295 && SHMEM_HEAP_MB % 2 == 0 )) ||
    fail "SHMEM_HEAP_MB must be 0 (auto) or an even number of MiB within uint32"
[[ "$M" =~ ^[1-9][0-9]{0,5}$ ]] && (( M <= 300000 )) || fail "M must be in [1,300000]"
[[ "$ITERATIONS" =~ ^[1-9][0-9]{0,3}$ ]] && (( ITERATIONS <= 1000 )) || fail "ITERATIONS must be in [1,1000]"
[[ "$RUN_TIMEOUT" =~ ^[1-9][0-9]{0,4}$ ]] || fail "RUN_TIMEOUT must be positive seconds"
[[ "$CHECK_RESULT" = 0 || "$CHECK_RESULT" = 1 ]] || fail "CHECK_RESULT must be 0 or 1"
if [[ $OVERLAP == 1 ]]; then
    [[ $WARMUP =~ ^[1-9][0-9]{0,3}$ ]] && (( ITERATIONS >= WARMUP + 2 )) ||
        fail "OVERLAP requires at least one warmup and two measured iterations"
fi
DATA_PATH=${DATA_PATH:-}
IPPORT=${IPPORT:-tcp://127.0.0.1:12560}
[[ "$IPPORT" =~ ^tcp://([A-Za-z0-9.-]+):([0-9]{1,5})$ ]] || fail "IPPORT must be tcp://<rank0-host>:<port>"
PORT=${BASH_REMATCH[2]}
PORT=$((10#$PORT))
(( PORT >= 1 && PORT <= 65535 )) || fail "invalid bootstrap port"
if [ "$NODE_COUNT" -gt 1 ]; then
    [[ "$IPPORT" != tcp://127.* && "$IPPORT" != tcp://localhost:* ]] || fail "two nodes require rank-0 host's non-loopback IPPORT"
fi

set --
WINDOW_ROWS=${WINDOW_ROWS:-512}
case "$WINDOW_ROWS" in 256|512|1024) ;; *) fail "WINDOW_ROWS must be 256, 512, or 1024" ;; esac
export ITERATIONS SHMEM_HEAP_MB WINDOW_ROWS
# Environment setup during a run only loads already-built dependencies.
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
SHMEM_ENV="${SHMEM_HOME_PATH:-${PROJECT_ROOT}/3rdparty/shmem/install}/set_env.sh"
[ -f "$SHMEM_ENV" ] || fail "SHMEM environment not found; run scripts/build.sh first"
source "$SHMEM_ENV"
set -u
EXEC_BIN="${PROJECT_ROOT}/build/bin/a2a_matmul"
[ -x "$EXEC_BIN" ] || fail "executable not found; run scripts/build.sh first"
[ -f "${SHMEM_HOME_PATH}/version.info" ] || fail "SHMEM version.info not found"
[ -f "${PROJECT_ROOT}/build/bin/alltoall_matmul_fused_shmem_version.info" ] || fail "build snapshot not found; run scripts/build.sh"
cmp -s "${SHMEM_HOME_PATH}/version.info" "${PROJECT_ROOT}/build/bin/alltoall_matmul_fused_shmem_version.info" ||
    fail "SHMEM SDK differs from the one used to compile the kernel; rebuild"
command -v npu-smi >/dev/null || fail "npu-smi is required"
command -v timeout >/dev/null || fail "timeout is required"
command -v "$PYTHON_BIN" >/dev/null || fail "Python 3.10+ is required for validation and timing summaries"
if [[ $DTYPE == bf16 && $NODE_RANK == 0 && $NODE_COUNT == 2 ]]; then
    command -v ss >/dev/null || fail "ss is required to detect when BF16 bootstrap is ready"
fi
if [[ $OVERLAP == 1 ]]; then
    if [[ $DTYPE == bf16 ]]; then
        "$PYTHON_BIN" -c 'import torch; import torch_npu' || fail "BF16 overlap requires torch and torch_npu"
    else
        "$PYTHON_BIN" -c 'import torch.distributed as dist; raise SystemExit(not dist.is_gloo_available())' ||
            fail "MXFP8 overlap reporting requires PyTorch with Gloo"
    fi
fi
LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/../out/m${M}-node${NODE_RANK}-$(date +%Y%m%d-%H%M%S)-$$"}
mkdir -p "$LOG_DIR"
LOG_DIR=$(cd "$LOG_DIR" && pwd)
export TRACE_DIR=${TRACE_DIR:-"$LOG_DIR/trace"}
DATA_SCRIPT="$SCRIPT_DIR/gen_data.py"
RANK_START=$((NODE_RANK * LOCAL_RANKS))
DATA_ARGS=(--world "$RANK_SIZE" --m "$M" --k "$K" --n "$N" --dtype "$DTYPE")
RANK_ARGS=(--rank-start "$RANK_START" --rank-count "$LOCAL_RANKS")
if [[ $DTYPE == bf16 ]]; then
    DATA_ARGS+=(--seed "$SEED" "${RANK_ARGS[@]}")
    export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
fi
if [ -z "$DATA_PATH" ]; then
    DATA_PATH="$LOG_DIR/data"
    "$PYTHON_BIN" "$DATA_SCRIPT" generate "$DATA_PATH" "${DATA_ARGS[@]}" \
        > "$LOG_DIR/generate.log" 2>&1 || fail "input generation failed; see $LOG_DIR/generate.log"
fi
[ -d "$DATA_PATH" ] || fail "DATA_PATH is not a directory"
DATA_PATH=$(cd "$DATA_PATH" && pwd)
"$PYTHON_BIN" "$DATA_SCRIPT" check-inputs "$DATA_PATH" "${DATA_ARGS[@]}"
if [ "$NODE_RANK" -eq 0 ]; then
    "$PYTHON_BIN" - "$PORT" <<'PY'
import socket
import sys
with socket.socket() as listener:
    listener.bind(('0.0.0.0', int(sys.argv[1])))
PY
fi

{
    date -Ins
    printf 'dtype=%s\n' "$DTYPE"
    printf 'world=%s\nnode_rank=%s\nlocal_ranks=%s\ndevices=%s\nM=%s\ninput_rows=%s\niterations=%s\ndata_path=%s\n' \
        "$RANK_SIZE" "$NODE_RANK" "$LOCAL_RANKS" "$DEVICE_LIST" "$M" "$((RANK_SIZE * M))" "$ITERATIONS" "$DATA_PATH"
    printf 'K=%s\nlocalK=%s\nN=%s\n' "$K" "$LOCAL_K" "$N"
    sha256sum "$EXEC_BIN" "${PROJECT_ROOT}/build/bin/liba2a_matmul_impl.so"
    printf 'node_count=%s\nshmem_heap_mb=%s\n' "$NODE_COUNT" "$SHMEM_HEAP_MB"
    printf 'window_rows=%s\nmatmul_tile_rows=256\nrdma_qps=1\nrdma_quiet=final\nready_notify=window\n' "$WINDOW_ROWS"
} > "$LOG_DIR/manifest.txt"
if ! npu-smi info > "$LOG_DIR/preflight.txt" 2>&1; then
    fail "NPU query failed; idle status is unknown. See $LOG_DIR/preflight.txt"
fi
for device in "${DEVICE_IDS[@]}"; do
    grep -Eq "No running processes found in NPU ${device}([[:space:]]|\\|)" "$LOG_DIR/preflight.txt" ||
        fail "selected NPU $device is busy or missing from the query; see $LOG_DIR/preflight.txt"
done

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "[RUN] world=$RANK_SIZE node=$NODE_RANK/$NODE_COUNT local_ranks=$LOCAL_RANKS devices=$DEVICE_LIST M=$M N=$N K=$K window_rows=$WINDOW_ROWS iterations=$ITERATIONS logs=$LOG_DIR"
for ((local = 0; local < LOCAL_RANKS; ++local)); do
    rank=$((NODE_RANK * LOCAL_RANKS + local))
    timeout --signal=TERM --kill-after=10s "${RUN_TIMEOUT}s" \
        "$EXEC_BIN" "$RANK_SIZE" "$rank" "$IPPORT" "$M" "$N" "$K" "$DATA_PATH" "$DEVICE_LIST" \
        > "$LOG_DIR/rank${rank}.log" 2>&1 &
    PIDS+=("$!")
    if [[ $DTYPE == bf16 && $NODE_COUNT == 2 && $rank == 0 ]]; then
        for ((attempt = 0; attempt < 1200; ++attempt)); do
            [[ -n $(ss -H -ltn "sport = :$PORT") ]] && break
            kill -0 "${PIDS[0]}" 2>/dev/null || fail "rank 0 exited during bootstrap"
            sleep 0.1
        done
        (( attempt < 1200 )) || fail "rank 0 bootstrap listener timeout"
        echo "[BOOTSTRAP_READY] Start the other node now."
    fi
done
STATUS=0
for ((local = 0; local < LOCAL_RANKS; ++local)); do
    rank=$((NODE_RANK * LOCAL_RANKS + local))
    rc=0
    wait "${PIDS[$local]}" || rc=$?
    unset "PIDS[$local]"
    echo "rank=$rank exit=$rc" | tee -a "$LOG_DIR/exits.txt"
    if [ "$rc" -ne 0 ]; then STATUS=1; fi
done
npu-smi info > "$LOG_DIR/postflight.txt" 2>&1 || true
if [ "$STATUS" -ne 0 ]; then fail "one or more ranks failed; see $LOG_DIR/rank*.log"; fi
if [[ $CHECK_RESULT == 1 || $OVERLAP == 1 ]]; then
    if [[ $DTYPE == mxfp8 ]]; then DATA_ARGS+=("${RANK_ARGS[@]}"); fi
    "$PYTHON_BIN" "$DATA_SCRIPT" check "$DATA_PATH" "${DATA_ARGS[@]}" \
        > "$LOG_DIR/correctness.log" 2>&1 || fail "CPU reference check failed; see $LOG_DIR/correctness.log"
fi
if [[ $DTYPE == bf16 || $OVERLAP == 1 ]]; then
    grep -h "^TIMING " "$LOG_DIR"/rank*.log | tee "$LOG_DIR/timings.txt"
fi
if [[ $OVERLAP == 1 ]]; then
    if [[ $DTYPE == bf16 ]]; then
        # CPU checking can take time; recheck the selected devices before starting HCCL.
        npu-smi info > "$LOG_DIR/overlap_preflight.txt" 2>&1 || fail "NPU query failed before HCCL baseline"
        for device in "${DEVICE_IDS[@]}"; do
            grep -Eq "No running processes found in NPU ${device}([[:space:]]|\\|)" "$LOG_DIR/overlap_preflight.txt" ||
                fail "NPU $device is occupied before HCCL baseline; results remain in $LOG_DIR"
        done
    fi
    MASTER_ADDR=${IPPORT#tcp://}
    MASTER_ADDR=${MASTER_ADDR%:*}
    timeout --signal=TERM --kill-after=10s "${OVERLAP_TIMEOUT:-600}s" \
        "$PYTHON_BIN" -m torch.distributed.run --nnodes="$NODE_COUNT" --nproc-per-node="$LOCAL_RANKS" \
        --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$PORT" \
        "$SCRIPT_DIR/trace.py" overlap --dtype "$DTYPE" --data-path "$DATA_PATH" --own-logs "$LOG_DIR" \
        --shape "$M,$N,$K" --devices "$DEVICE_LIST" --warmup "$WARMUP" \
        --iterations="$((ITERATIONS - WARMUP))" --output "$LOG_DIR/overlap.json" \
        2>&1 | tee "$LOG_DIR/overlap.log"
fi
if [[ $DTYPE == bf16 || $OVERLAP == 1 ]]; then
    echo "[PASS] Logs: $LOG_DIR"
    exit 0
fi
# Summarize this host's existing TIMING records without another device run.
"$PYTHON_BIN" - "$LOG_DIR" "$((NODE_RANK * LOCAL_RANKS))" "$LOCAL_RANKS" "$ITERATIONS" <<'PY' | tee "$LOG_DIR/summary.txt"
import math
from pathlib import Path
import statistics
import sys

log_dir = Path(sys.argv[1])
first, count, iterations = map(int, sys.argv[2:])
groups = [{} for _ in range(iterations)]
for rank in range(first, first + count):
    with (log_dir / f'rank{rank}.log').open() as source:
        for line in source:
            if not line.startswith('TIMING '):
                continue
            fields = dict(field.split('=', 1) for field in line.split()[1:])
            if fields.get('timing_scope') != 'gm_input_to_output':
                raise SystemExit(f'outdated operator timing scope in rank{rank}.log; rebuild the executable')
            iteration, total = int(fields['iter']), float(fields['total_ms'])
            if (int(fields['rank']) != rank or not 0 <= iteration < iterations
                    or rank in groups[iteration] or not math.isfinite(total) or total < 0):
                raise SystemExit(f'invalid or duplicate timing in rank{rank}.log')
            groups[iteration][rank] = total
if any(len(group) != count for group in groups):
    raise SystemExit('missing rank/iteration timings')
totals = [max(group.values()) for group in groups]
print(f'Operator timing summary: local ranks {first}..{first + count - 1}; '
      'ready reset + GM input staging + preparation barrier + fused execution')
for iteration, total in enumerate(totals):
    print(f'iter={iteration} total_max_ms={total:.6f}')
if iterations > 1:
    print(f'excluding iter=0: median_total_max_ms={statistics.median(totals[1:]):.6f}')
PY
echo "[PASS] Logs: $LOG_DIR"
