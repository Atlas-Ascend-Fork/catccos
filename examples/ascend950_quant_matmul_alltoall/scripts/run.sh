#!/bin/bash
# Usage:
#   run.sh [local] <local_device_list> [fmt] [ignored_slots] [lanes] [rpn] [world_size] [rank_start] [ip_port]
#   run.sh cross <node_id> <node0_ip> [local_device_start]
# fmt: 0=MXFP8, 1=MXFP4, 2=BF16.
# Shapes default to M,K,N rows in scripts/test_shapes.csv. Set QMMA_TEST_SHAPES to
# use another CSV, or set QMMA_M, QMMA_N and QMMA_K together for one shape.
set -e
SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
BUILD_DIR=${QMMA_BUILD_DIR:-${PROJECT_ROOT}/build/ascend950_quant_matmul_alltoall}
EXEC_BIN=${BUILD_DIR}/bin/ascend950_quant_matmul_alltoall
TIMER_ENABLED=OFF
BUILD_QP=OFF
HIERARCHICAL_GROUP_SIZE=16
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    TIMER_VALUE=$(sed -n 's/^ENABLE_TIMER:[^=]*=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1 | tr -d '\r')
    QP_VALUE=$(sed -n 's/^QMMA_WITH_QP:[^=]*=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1 | tr -d '\r')
    GROUP_SIZE_VALUE=$(sed -n 's/^CATCCOS_HIERARCHICAL_GROUP_SIZE:[^=]*=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1 | tr -d '\r')
    if [[ "$TIMER_VALUE" == "ON" ]]; then
        TIMER_ENABLED=ON
    fi
    if [[ "$QP_VALUE" == "ON" ]]; then
        BUILD_QP=ON
    fi
    if [[ "$GROUP_SIZE_VALUE" =~ ^[0-9]+$ ]]; then
        HIERARCHICAL_GROUP_SIZE=$((10#$GROUP_SIZE_VALUE))
    fi
fi
TS() { date +%T; }

fail() {
    echo "[$(TS)] [ERROR] $*" >&2
    exit 1
}

parse_uint() {
    local output_var=$1
    local label=$1
    local raw_value=$2
    local parsed

    if (( $# >= 3 )); then
        label=$2
        raw_value=$3
    fi
    [[ "$raw_value" =~ ^[0-9]{1,10}$ ]] || fail "$label must be an unsigned integer"
    parsed=$((10#$raw_value))
    (( parsed <= 4294967295 )) || fail "$label exceeds uint32"
    printf -v "$output_var" '%u' "$parsed"
}

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run.sh [local] <device_list> [fmt] [ignored_slots] [lanes] [rpn] [world_size] [rank_start] [ip_port]
  bash scripts/run.sh cross <node_id> <node0_ip> [local_device_start]

Local mode uses the explicit global rank arguments. Cross-node mode derives
rank_start from node_id * QMMA_RPN and launches the local ranks after both
nodes pass the data-generation rendezvous.

Cross-node environment variables:
  QMMA_WORLD_SIZE  Global rank count (default: 2 * configured group size)
  QMMA_RPN         Ranks per node; must match the configured group size
  QMMA_M/N/K       Per-rank M, full N, K (default: 256/(P*256)/64)
  QMMA_FMT         0=MXFP8, 1=MXFP4, 2=BF16 (default: 0)
  QMMA_SLOTS       Legacy argument retained for command-line compatibility (default: 2)
  QMMA_LANES       Data QPs per peer: 1..4 (default: 1)
  QMMA_PORT        Node-0 bootstrap TCP port (default: 8788)
  QMMA_DATA_DIR    Node-local data and log root
  QMMA_CROSS_LOG   Optional wrapper and rank output log
  QMMA_CROSS_RENDEZVOUS
                   Wait for both nodes before launching ranks (default: 1)
  QMMA_RENDEZVOUS_PORT
                   Launch-gate TCP port (default: QMMA_PORT + 1000)
EOF
}

parse_bool() {
    local output_var=$1
    local label=$2
    local raw_value=$3

    [[ "$raw_value" == 0 || "$raw_value" == 1 ]] || fail "$label must be 0 or 1"
    printf -v "$output_var" '%s' "$raw_value"
}

parse_csv_uint() {
    local output_var=$1
    local label=$2
    local raw_value=$3
    local parsed

    [[ "$raw_value" =~ ^[0-9]{1,10}$ ]] || fail "$label must be a positive unsigned integer"
    parsed=$((10#$raw_value))
    (( parsed > 0 && parsed <= 4294967295 )) || fail "$label must be in [1, 4294967295]"
    printf -v "$output_var" '%u' "$parsed"
}

trim() {
    local value=$1
    value=${value%$'\r'}
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

# Cross-node mode is part of this entry point so both launchers share one
# implementation. The function translates node-local arguments and re-enters
# local mode, which keeps the rank launch and precision checks identical.
run_cross_node() {
    local node_id node0_ip device_start port rank_start remaining local_ranks
    local fmt slots lanes rpn world_size cross_rendezvous m n k seed data_dir ipport gh gen_script
    local device_id local_device_list local_device_ids ready_port_default i prev

    (( $# >= 2 && $# <= 3 )) || { usage >&2; exit 1; }
    node_id=$1
    node0_ip=$2
    parse_uint node_id "node_id" "$node_id"
    (( node_id <= 1 )) || fail "node_id must be 0 or 1; this example supports two nodes"

    if [[ -n "${QMMA_CROSS_LOG:-}" ]]; then
        mkdir -p -- "$(dirname -- "$QMMA_CROSS_LOG")"
        exec > >(tee -a -- "$QMMA_CROSS_LOG") 2>&1
    fi

    node0_ip=${node0_ip#tcp://}
    [[ -n "$node0_ip" && "$node0_ip" != *[[:space:]/]* && "$node0_ip" != *:* ]] || \
        fail "node0_ip must be a host name or IPv4 address without tcp:// or a port"
    [[ "$node0_ip" != "localhost" && "$node0_ip" != 127.* ]] || \
        fail "node0_ip must be reachable from both nodes; loopback is not allowed"

    parse_uint rpn "QMMA_RPN" "${QMMA_RPN:-$HIERARCHICAL_GROUP_SIZE}"
    parse_uint world_size "QMMA_WORLD_SIZE" "${QMMA_WORLD_SIZE:-$((2 * HIERARCHICAL_GROUP_SIZE))}"
    parse_uint port "QMMA_PORT" "${QMMA_PORT:-8788}"
    parse_uint device_start "local_device_start" "${3:-${QMMA_DEVICE_START:-0}}"
    parse_uint fmt "QMMA_FMT" "${QMMA_FMT:-0}"
    parse_uint slots "QMMA_SLOTS" "${QMMA_SLOTS:-2}"
    parse_uint lanes "QMMA_LANES" "${QMMA_LANES:-1}"
    parse_bool cross_rendezvous "QMMA_CROSS_RENDEZVOUS" "${QMMA_CROSS_RENDEZVOUS:-1}"

    (( device_start == 0 )) || fail "local_device_start must be 0; SHMEM binds PE 0 to local device 0"
    (( rpn >= 1 && rpn <= 16 )) || fail "QMMA_RPN must be in [1, 16] for cross-node runs"
    (( rpn == HIERARCHICAL_GROUP_SIZE )) || \
        fail "QMMA_RPN=$rpn must match CATCCOS_HIERARCHICAL_GROUP_SIZE=$HIERARCHICAL_GROUP_SIZE"
    (( world_size > rpn && world_size <= 2 * rpn && world_size <= 32 )) || \
        fail "QMMA_WORLD_SIZE must be in (QMMA_RPN, min(2*QMMA_RPN, 32)] for cross-node runs"
    (( port >= 1 && port <= 65535 )) || fail "QMMA_PORT must be in [1, 65535]"
    if (( port <= 64535 )); then
        ready_port_default=$((port + 1000))
    else
        ready_port_default=$((port - 1000))
    fi
    (( fmt <= 2 && lanes >= 1 && lanes <= 4 )) || \
        fail "QMMA_FMT must be 0/1/2, QMMA_LANES in [1, 4]"
    rank_start=$((node_id * rpn))
    remaining=$((world_size - rank_start))
    (( remaining > 0 )) || fail "node_id=$node_id has no global ranks in QMMA_WORLD_SIZE=$world_size"
    local_ranks=$remaining
    (( local_ranks <= rpn )) || local_ranks=$rpn

    if [[ -n "${QMMA_LOCAL_DEVICES:-}" ]]; then
        local_device_list=$QMMA_LOCAL_DEVICES
    else
        local_device_list=
        for (( i = 0; i < local_ranks; ++i )); do
            parse_uint device_id "local device" "$((device_start + i))"
            (( device_id <= 2147483647 )) || fail "local device id exceeds int32"
            [[ -z "$local_device_list" ]] || local_device_list+=,
            local_device_list+=$device_id
        done
    fi

    IFS=',' read -ra local_device_ids <<< "$local_device_list"
    (( ${#local_device_ids[@]} == local_ranks )) || \
        fail "QMMA_LOCAL_DEVICES must contain exactly $local_ranks device id(s), got: $local_device_list"
    for (( i = 0; i < local_ranks; ++i )); do
        parse_uint device_id "QMMA_LOCAL_DEVICES[$i]" "${local_device_ids[i]}"
        (( device_id <= 2147483647 )) || fail "local device id exceeds int32"
        local_device_ids[i]=$device_id
        for (( prev = 0; prev < i; ++prev )); do
            (( device_id != local_device_ids[prev] )) || fail "duplicate local device id: $device_id"
        done
        (( device_id == i )) || fail \
            "cross-node local devices must be 0,1,...,$((local_ranks - 1)); SHMEM binds each PE to its local device id"
    done
    local_device_list=$(IFS=,; echo "${local_device_ids[*]}")

    gen_script="${SCRIPT_DIR}/gen_data_slice_n.py"
    [[ -f "$gen_script" ]] || fail "data generator not found: $gen_script"
    grep -q "torch.manual_seed" "$gen_script" || \
        fail "data generator does not support a deterministic torch.manual_seed path"

    m=${QMMA_M:-256}
    n=${QMMA_N:-$((world_size * 256))}
    k=${QMMA_K:-64}
    seed=${QMMA_SEED:-20261024}
    [[ "$seed" =~ ^[0-9]{1,20}$ ]] || fail "QMMA_SEED must be a non-negative integer"
    [[ "$m" =~ ^[0-9]{1,10}$ && "$n" =~ ^[0-9]{1,10}$ && "$k" =~ ^[0-9]{1,10}$ ]] || \
        fail "QMMA_M, QMMA_N and QMMA_K must be unsigned integers"
    m=$((10#$m))
    n=$((10#$n))
    k=$((10#$k))
    (( m > 0 && n > 0 && k > 0 )) || fail "QMMA_M, QMMA_N and QMMA_K must be positive"
    (( n % world_size == 0 && (n / world_size) % 256 == 0 )) || \
        fail "N must be divisible by P and N/P must be a multiple of 256: N=$n P=$world_size"
    (( k % 32 == 0 )) || fail "K must be a multiple of 32: K=$k"

    data_dir=${QMMA_DATA_DIR:-${PROJECT_ROOT}/out/qmma/cross_node_P${world_size}/node${node_id}}
    mkdir -p -- "$data_dir"
    data_dir=$(cd -- "$data_dir" && pwd)
    ipport="tcp://${node0_ip}:${port}"

    echo "[$(TS)] [cross-node] node=$node_id P=$world_size rpn=$rpn"
    echo "[$(TS)] [cross-node] PE range=[$rank_start,$((rank_start + local_ranks - 1))] local_devices=$local_device_list"
    echo "[$(TS)] [cross-node] bootstrap=$ipport data_dir=$data_dir"
    echo "[$(TS)] [cross-node] shape=M${m},N${n},K=${k} fmt=$fmt slot=$slots(ignored) lane=$lanes comm_interval=${QMMA_OPT_COMM_INTERVAL:-auto} heap=${QMMA_SHMEM_HEAP_MB:-auto}"
    echo "[$(TS)] [cross-node] transport=fused_udma_roce data_qp=0..$((lanes - 1)) completion_qp=0-after-data-quiet"
    echo "[$(TS)] [cross-node] deterministic seed=$seed"
    echo "[$(TS)] [cross-node] launch gate=$cross_rendezvous port=${QMMA_RENDEZVOUS_PORT:-$ready_port_default}"
    if (( cross_rendezvous == 1 )); then
        echo "[$(TS)] [cross-node] both nodes will rendezvous after data generation; start order does not matter"
    else
        echo "[$(TS)] [WARN] launch gate disabled: start node 0 first, then node 1 promptly (SHMEM bootstrap retries are time-bounded)" >&2
    fi

    if QMMA_M="$m" QMMA_N="$n" QMMA_K="$k" QMMA_DATA_DIR="$data_dir" \
        QMMA_DETERMINISTIC_DATA=1 QMMA_SEED="$seed" QMMA_CROSS_RENDEZVOUS="$cross_rendezvous" \
        bash "$0" local "$local_device_list" "$fmt" "$slots" "$lanes" "$rpn" \
            "$world_size" "$rank_start" "$ipport"; then
        echo "[$(TS)] [cross-node] node=$node_id local ranks completed and were verified"
    else
        local rc=$?
        echo "[$(TS)] [cross-node] node=$node_id failed with status=$rc" >&2
        return "$rc"
    fi
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    cross)
        shift
        run_cross_node "$@"
        exit $?
        ;;
    local)
        shift
        ;;
esac

[[ "${1:-}" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail "provide a comma-separated local device list"
LOCAL_DEVICE_LIST=$1
IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
LOCAL_RANKS=${#DEVICE_ID_LIST[@]}
parse_uint FMT "${2:-0}"
parse_uint SLOT "${3:-2}"
parse_uint LANE "${4:-1}"
parse_uint RPN "${5:-$HIERARCHICAL_GROUP_SIZE}"
parse_uint RANK_SIZE "${6:-$LOCAL_RANKS}"
parse_uint RANK_START "${7:-0}"
IPPORT=${8:-tcp://127.0.0.1:8788}
BOOTSTRAP_PORT=${IPPORT##*:}
[[ "$BOOTSTRAP_PORT" =~ ^[0-9]{1,5}$ ]] || fail "cannot parse SHMEM bootstrap port from ip_port: $IPPORT"
BOOTSTRAP_PORT=$((10#$BOOTSTRAP_PORT))
(( BOOTSTRAP_PORT >= 1 && BOOTSTRAP_PORT <= 65535 )) || \
    fail "SHMEM bootstrap port must be in [1, 65535]"
BOOTSTRAP_READY_TIMEOUT=${QMMA_BOOTSTRAP_READY_TIMEOUT:-120}
[[ "$BOOTSTRAP_READY_TIMEOUT" =~ ^[0-9]{1,5}$ ]] || \
    fail "QMMA_BOOTSTRAP_READY_TIMEOUT must be an unsigned integer"
parse_uint COMM_INTERVAL "${QMMA_OPT_COMM_INTERVAL:-1}"
COMM_INTERVAL_DISPLAY=${QMMA_OPT_COMM_INTERVAL:-auto}
DATA_SEED=${QMMA_SEED:-20261024}
[[ "$DATA_SEED" =~ ^[0-9]{1,20}$ ]] || fail "QMMA_SEED must be a non-negative integer"

(( RANK_SIZE >= 2 && RANK_SIZE <= 32 && RPN > 0 && RPN <= 16 )) || \
    fail "world_size in [2, 32] and rpn in [1, 16] required"
(( RPN == HIERARCHICAL_GROUP_SIZE )) || \
    fail "rpn=$RPN must match CATCCOS_HIERARCHICAL_GROUP_SIZE=$HIERARCHICAL_GROUP_SIZE"
(( FMT <= 2 && LANE >= 1 && LANE <= 4 )) || fail "fmt in [0, 2], lanes in [1, 4] required"
(( COMM_INTERVAL >= 1 && COMM_INTERVAL <= 64 )) || fail "QMMA_OPT_COMM_INTERVAL must be in [1, 64]"
(( RANK_SIZE <= 2 * RPN )) || fail "at most two nodes are supported"
(( LOCAL_RANKS <= RPN )) || fail "local rank count must not exceed rpn=$RPN"
(( RANK_START + LOCAL_RANKS <= RANK_SIZE )) || \
    fail "rank_start=$RANK_START + local ranks=$LOCAL_RANKS exceeds world_size=$RANK_SIZE"
if (( RANK_SIZE > RPN )); then
    [[ $# -ge 8 ]] || fail "cross-node runs require a shared ip_port on node 0"
    [[ "$IPPORT" != *127.0.0.1* && "$IPPORT" != *localhost* ]] || fail "cross-node ip_port must be reachable from both nodes"
fi
(( RANK_START < RANK_SIZE && RANK_START % RPN == 0 )) || fail "rank_start must be a node boundary below world_size"
EXPECTED_LOCAL=$((RANK_SIZE - RANK_START))
(( EXPECTED_LOCAL <= RPN )) || EXPECTED_LOCAL=$RPN
(( LOCAL_RANKS == EXPECTED_LOCAL )) || fail "local device count must equal $EXPECTED_LOCAL on this node"
for (( idx = 0; idx < LOCAL_RANKS; ++idx )); do
    parse_uint DEVICE_ID "${DEVICE_ID_LIST[idx]}"
    (( DEVICE_ID <= 2147483647 )) || fail "device id exceeds int32"
    DEVICE_ID_LIST[idx]=$DEVICE_ID
    for (( prev = 0; prev < idx; ++prev )); do
        (( DEVICE_ID != DEVICE_ID_LIST[prev] )) || fail "duplicate local device id: $DEVICE_ID"
    done
done

if (( RANK_SIZE > RPN )); then
    for (( idx = 0; idx < LOCAL_RANKS; ++idx )); do
        (( DEVICE_ID_LIST[idx] == idx )) || fail \
            "cross-node local devices must be 0,1,...,$((LOCAL_RANKS - 1)); SHMEM binds each PE to its local device id"
    done
fi

wait_for_local_bootstrap_listener() {
    local rank_pid=$1
    local rank_log=$2
    local deadline=$((SECONDS + BOOTSTRAP_READY_TIMEOUT))

    command -v ss >/dev/null 2>&1 || fail "ss is required to verify the SHMEM bootstrap listener"
    while (( SECONDS < deadline )); do
        if ! kill -0 "$rank_pid" 2>/dev/null; then
            echo "[$(TS)] [ERROR] global rank 0 exited before binding bootstrap port $BOOTSTRAP_PORT" >&2
            tail -n 40 -- "$rank_log" >&2 || true
            return 1
        fi
        if ss -H -ltn "sport = :$BOOTSTRAP_PORT" 2>/dev/null | grep -q .; then
            echo "[$(TS)] [bootstrap] rank=0 listening on port $BOOTSTRAP_PORT"
            return 0
        fi
        sleep 0.1
    done
    echo "[$(TS)] [ERROR] timed out after ${BOOTSTRAP_READY_TIMEOUT}s waiting for rank 0 " \
         "to bind bootstrap port $BOOTSTRAP_PORT" >&2
    tail -n 40 -- "$rank_log" >&2 || true
    return 1
}

# Cross-node launch gate.
#
# SHMEM's TCP bootstrap client has a bounded retry budget
# (CONNECT_RETRY_MAX_TIMES=20000, i.e. about 20 s of sleeps when connect()
# fails immediately; a filtered port can block even longer).  Worse, once the
# first node enters a WORLD barrier without its peer, the AICore watchdog traps
# the kernel after 60 s with error 507015.  Data generation happens before the
# rank launch below and can take very different times on the two hosts, so a
# plain "start node 0 first" rule is not enough.  This gate rendezvouses after
# data generation and immediately before launching the ranks.  Node 1 must
# receive node 0's confirmation before either node starts its local ranks.
cross_node_rendezvous() {
    local m=$1
    local n=$2
    local k=$3
    local case_index=$4
    local role
    local host
    local gate_host
    local bootstrap_port
    local ready_port
    local timeout=${QMMA_RENDEZVOUS_TIMEOUT:-1800}

    [[ "$timeout" =~ ^[0-9]{1,6}$ ]] || fail "QMMA_RENDEZVOUS_TIMEOUT must be an unsigned integer"

    if (( RANK_START == 0 )); then
        role=node0
    else
        role=node1
    fi

    host=${IPPORT#*://}
    host=${host%%:*}
    gate_host=${QMMA_RENDEZVOUS_HOST:-$host}
    bootstrap_port=${IPPORT##*:}
    [[ -n "$host" && -n "$gate_host" && "$bootstrap_port" =~ ^[0-9]{1,5}$ ]] || \
        fail "cannot parse host and port from ip_port: $IPPORT"
    if [[ -n "${QMMA_RENDEZVOUS_PORT:-}" ]]; then
        ready_port=$QMMA_RENDEZVOUS_PORT
    elif (( 10#$bootstrap_port <= 64535 )); then
        ready_port=$((10#$bootstrap_port + 1000))
    else
        ready_port=$((10#$bootstrap_port - 1000))
    fi
    [[ "$ready_port" =~ ^[0-9]{1,5}$ ]] || fail "QMMA_RENDEZVOUS_PORT must be an unsigned integer"
    (( ready_port >= 1 && ready_port <= 65535 )) || fail "QMMA_RENDEZVOUS_PORT must be in [1, 65535]"
    (( ready_port != 10#$bootstrap_port )) || fail "QMMA_RENDEZVOUS_PORT must differ from the SHMEM bootstrap port"

    echo "[$(TS)] [rendezvous] role=$role peer=$gate_host:$ready_port timeout=${timeout}s"
    if ! python3 - "$role" "$gate_host" "$ready_port" "$timeout" \
        "$RANK_SIZE" "$m" "$n" "$k" "$DATA_SEED" "$FMT" "$SLOT" "$LANE" "$RPN" \
        "$case_index" "$SHAPE_COUNT" "$COMM_INTERVAL" "$IPPORT" "$RDMA_BACKEND" "$INSTALLED_COMMIT" <<'PY'
import json
import socket
import sys
import time

role, host, port_text, timeout_text = sys.argv[1:5]
port = int(port_text)
deadline = time.monotonic() + float(timeout_text)
names = ("world_size", "M", "N", "K", "seed", "fmt", "slots", "lanes", "rpn", "case", "case_count",
         "comm_interval")
config = {name: int(value) for name, value in zip(names, sys.argv[5:17])}
config.update(bootstrap=sys.argv[17], rdma_backend=sys.argv[18], shmem_commit=sys.argv[19])
protocol = "qmma-rendezvous-v1"


def log(message):
    print("[rendezvous] " + message, flush=True)


def send_message(conn, message):
    conn.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8"))


def receive_line(conn):
    data = bytearray()
    while len(data) <= 4096 and not data.endswith(b"\n"):
        chunk = conn.recv(4096 - len(data) + 1)
        if not chunk:
            raise ValueError("peer closed connection before replying")
        data.extend(chunk)
    if len(data) > 4096:
        raise ValueError("message exceeds 4096 bytes")
    return data.decode("utf-8")


if role == "node0":
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(("0.0.0.0", port))
    except OSError as exc:
        print(f"[rendezvous] bind 0.0.0.0:{port} failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
    server.listen(16)
    server.settimeout(1.0)
    log(f"node0 listening on 0.0.0.0:{port}; waiting for node1")
    last_report = time.monotonic()
    while time.monotonic() < deadline:
        try:
            conn, addr = server.accept()
        except socket.timeout:
            now = time.monotonic()
            if now - last_report >= 30.0:
                log("still waiting for node1")
                last_report = now
            continue
        with conn:
            conn.settimeout(3.0)
            try:
                raw = receive_line(conn)
                if raw.startswith("QMMA_READY|"):
                    print("[rendezvous] incompatible launch gate protocol; update run.sh on both nodes",
                          file=sys.stderr, flush=True)
                    sys.exit(3)
                message = json.loads(raw)
            except (OSError, UnicodeError, ValueError) as exc:
                log(f"ignored invalid connection from {addr[0]}:{addr[1]}: {exc}")
                continue
            if not isinstance(message, dict) or message.get("protocol") != protocol:
                log(f"ignored incompatible request from {addr[0]}:{addr[1]}")
                continue
            other = message.get("config")
            if not isinstance(other, dict):
                differences = ["missing node1 config"]
            else:
                differences = [f"{key}: node0={config.get(key)!r}, node1={other.get(key)!r}"
                               for key in sorted(set(config) | set(other)) if config.get(key) != other.get(key)]
            if differences:
                detail = "configuration mismatch: " + "; ".join(differences)
                try:
                    send_message(conn, {"protocol": protocol, "status": "error", "message": detail})
                except OSError as exc:
                    log(f"failed to report mismatch to node1: {exc}")
                print(f"[rendezvous] {detail}", file=sys.stderr, flush=True)
                sys.exit(3)
            try:
                send_message(conn, {"protocol": protocol, "status": "ok"})
            except OSError as exc:
                log(f"node1 disconnected before confirmation: {exc}")
                continue
            log(f"node1 confirmed from {addr[0]}:{addr[1]}; launching local ranks")
            sys.exit(0)
    print("[rendezvous] timed out waiting for node1", file=sys.stderr, flush=True)
    sys.exit(3)

last_report = time.monotonic()
last_error = "node0 has not accepted a connection yet"
while time.monotonic() < deadline:
    try:
        remaining = max(0.1, deadline - time.monotonic())
        with socket.create_connection((host, port), timeout=min(3.0, remaining)) as conn:
            send_message(conn, {"protocol": protocol, "config": config})
            reply = json.loads(receive_line(conn))
        if not isinstance(reply, dict) or reply.get("protocol") != protocol:
            raise ValueError("unexpected response from node0")
        if reply.get("status") == "error":
            print(f"[rendezvous] {reply.get('message', 'node0 rejected configuration')}",
                  file=sys.stderr, flush=True)
            sys.exit(3)
        if reply.get("status") != "ok":
            raise ValueError("node0 did not confirm the launch")
        log(f"node0 confirmed at {host}:{port}; launching local ranks")
        sys.exit(0)
    except (UnicodeError, ValueError) as exc:
        print(f"[rendezvous] invalid confirmation from node0 {host}:{port}: {exc}; "
              "check that both nodes use the same run.sh and rendezvous port",
              file=sys.stderr, flush=True)
        sys.exit(4)
    except OSError as exc:
        last_error = str(exc)
    now = time.monotonic()
    if now - last_report >= 30.0:
        log(f"still waiting for node0 {host}:{port}: {last_error}")
        last_report = now
    time.sleep(min(1.0, max(0.0, deadline - now)))
print(f"[rendezvous] timed out waiting for node0 {host}:{port}: {last_error}", file=sys.stderr, flush=True)
sys.exit(4)
PY
    then
        fail "cross-node launch gate failed (role=$role peer=$gate_host:$ready_port)"
    fi
}

# The configuration rendezvous above makes both nodes enter the launch path at
# nearly the same time. PE0 still needs time to bind the SHMEM bootstrap port;
# a client can otherwise consume its bounded 20000-attempt budget first. Once
# PE0 is listening, release node 1 through this second gate before launching
# the remaining node-0 ranks and all node-1 ranks.
cross_node_bootstrap_gate() {
    [[ "${QMMA_CROSS_RENDEZVOUS:-1}" != "0" ]] || return 0
    (( RANK_SIZE > RPN )) || return 0

    local role="node0"
    local host=${QMMA_RENDEZVOUS_HOST:-${IPPORT#*://}}
    local ready_port
    local timeout=${QMMA_BOOTSTRAP_READY_TIMEOUT:-120}
    host=${host%%:*}
    if (( RANK_START != 0 )); then
        role=node1
    fi
    if [[ -n "${QMMA_RENDEZVOUS_PORT:-}" ]]; then
        ready_port=$QMMA_RENDEZVOUS_PORT
    elif (( BOOTSTRAP_PORT <= 64535 )); then
        ready_port=$((BOOTSTRAP_PORT + 1000))
    else
        ready_port=$((BOOTSTRAP_PORT - 1000))
    fi
    [[ -n "$host" && "$ready_port" =~ ^[0-9]{1,5}$ ]] || \
        fail "cannot derive bootstrap readiness endpoint from ip_port=$IPPORT"
    (( ready_port >= 1 && ready_port <= 65535 && ready_port != BOOTSTRAP_PORT )) || \
        fail "invalid bootstrap readiness port: $ready_port"

    echo "[$(TS)] [bootstrap] role=$role waiting for PE0 bootstrap readiness on $host:$ready_port"
    if ! python3 - "$role" "$host" "$ready_port" "$timeout" <<'PY'
import json
import socket
import sys
import time

role, host, port_text, timeout_text = sys.argv[1:5]
port = int(port_text)
deadline = time.monotonic() + float(timeout_text)
protocol = "qmma-bootstrap-ready-v1"
message = {"protocol": protocol, "status": "ready"}


if role == "node0":
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(4)
    server.settimeout(max(0.1, deadline - time.monotonic()))
    try:
        conn, _ = server.accept()
    except socket.timeout:
        print("[bootstrap] timed out waiting for node1", file=sys.stderr, flush=True)
        sys.exit(3)
    with conn:
        conn.settimeout(3.0)
        data = bytearray()
        while len(data) <= 4096 and not data.endswith(b"\n"):
            chunk = conn.recv(4097 - len(data))
            if not chunk:
                break
            data.extend(chunk)
        try:
            request = json.loads(data.decode("utf-8"))
        except (UnicodeError, ValueError):
            print("[bootstrap] invalid node1 readiness request", file=sys.stderr, flush=True)
            sys.exit(4)
        if request != message:
            print("[bootstrap] incompatible node1 readiness request", file=sys.stderr, flush=True)
            sys.exit(4)
        conn.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8"))
    print("[bootstrap] PE0 listener confirmed; releasing node1", flush=True)
    sys.exit(0)

last_error = "PE0 is not ready"
while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2.0) as conn:
            conn.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8"))
            data = bytearray()
            while len(data) <= 4096 and not data.endswith(b"\n"):
                chunk = conn.recv(4097 - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            reply = json.loads(data.decode("utf-8"))
        if reply != message:
            raise ValueError("unexpected PE0 readiness response")
        print("[bootstrap] PE0 listener confirmed; launching local ranks", flush=True)
        sys.exit(0)
    except (OSError, UnicodeError, ValueError) as exc:
        last_error = str(exc)
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
print(f"[bootstrap] timed out waiting for PE0 listener: {last_error}", file=sys.stderr, flush=True)
sys.exit(3)
PY
    then
        fail "cross-node bootstrap readiness gate failed (role=$role peer=$host:$ready_port)"
    fi
}

SHAPE_M=()
SHAPE_K=()
SHAPE_N=()
SHAPE_LINE=()

ENV_SHAPE_COUNT=0
for env_name in QMMA_M QMMA_N QMMA_K; do
    if [[ -n "${!env_name:-}" ]]; then
        ENV_SHAPE_COUNT=$((ENV_SHAPE_COUNT + 1))
    fi
done

if (( ENV_SHAPE_COUNT > 0 )); then
    (( ENV_SHAPE_COUNT == 3 )) || fail "QMMA_M, QMMA_N and QMMA_K must be set together"
    parse_csv_uint ENV_M "QMMA_M" "$QMMA_M"
    parse_csv_uint ENV_K "QMMA_K" "$QMMA_K"
    parse_csv_uint ENV_N "QMMA_N" "$QMMA_N"
    SHAPE_M+=("$ENV_M")
    SHAPE_K+=("$ENV_K")
    SHAPE_N+=("$ENV_N")
    SHAPE_LINE+=("environment override")
    SHAPE_SOURCE="environment"
else
    SHAPE_SOURCE=${QMMA_TEST_SHAPES:-${SCRIPT_DIR}/test_shapes.csv}
    [[ -f "$SHAPE_SOURCE" ]] || fail "shape CSV not found: $SHAPE_SOURCE"

    HEADER_SEEN=0
    CSV_LINE_NO=0
    echo "[$(TS)] [shape CSV] reading $SHAPE_SOURCE"
    while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
        CSV_LINE_NO=$((CSV_LINE_NO + 1))
        line=$(trim "$raw_line")
        [[ -z "$line" || "$line" == \#* ]] && continue

        IFS=',' read -r field1 field2 field3 extra <<< "$line"
        field1=$(trim "$field1")
        field2=$(trim "$field2")
        field3=$(trim "$field3")
        extra=$(trim "${extra:-}")

        if (( HEADER_SEEN == 0 )); then
            field1=${field1#$'\xEF\xBB\xBF'}
            if [[ "$field1,$field2,$field3" == "M,K,N" || "$field1,$field2,$field3" == "m,k,n" ]]; then
                [[ -z "$extra" ]] || fail "$SHAPE_SOURCE:$CSV_LINE_NO: expected exactly M,K,N columns"
                HEADER_SEEN=1
                continue
            fi
            fail "$SHAPE_SOURCE:$CSV_LINE_NO: expected header M,K,N"
        fi

        [[ -z "$extra" ]] || fail "$SHAPE_SOURCE:$CSV_LINE_NO: expected exactly three fields (M,K,N)"
        parse_csv_uint CSV_M "$SHAPE_SOURCE:$CSV_LINE_NO M" "$field1"
        parse_csv_uint CSV_K "$SHAPE_SOURCE:$CSV_LINE_NO K" "$field2"
        parse_csv_uint CSV_N "$SHAPE_SOURCE:$CSV_LINE_NO N" "$field3"
        SHAPE_M+=("$CSV_M")
        SHAPE_K+=("$CSV_K")
        SHAPE_N+=("$CSV_N")
        SHAPE_LINE+=("$CSV_LINE_NO")
    done < "$SHAPE_SOURCE"

    (( HEADER_SEEN == 1 )) || fail "$SHAPE_SOURCE: missing M,K,N header"
fi

SHAPE_COUNT=${#SHAPE_M[@]}
(( SHAPE_COUNT > 0 )) || fail "no test shapes found in $SHAPE_SOURCE"
echo "[$(TS)] [shape CSV] loaded $SHAPE_COUNT case(s) from $SHAPE_SOURCE"
for (( shape_idx = 0; shape_idx < SHAPE_COUNT; ++shape_idx )); do
    echo "[$(TS)] [shape $((shape_idx + 1))/$SHAPE_COUNT] source=${SHAPE_LINE[shape_idx]} M=${SHAPE_M[shape_idx]} K=${SHAPE_K[shape_idx]} N=${SHAPE_N[shape_idx]}"
done

echo "[$(TS)] [config] P=$RANK_SIZE rank_start=$RANK_START fmt=$FMT slot=$SLOT(ignored) lane=$LANE rpn=$RPN heap=${QMMA_SHMEM_HEAP_MB:-auto} devices=$LOCAL_DEVICE_LIST"
echo "[$(TS)] [config] comm_interval=$COMM_INTERVAL_DISPLAY build_dir=$BUILD_DIR"
echo "[$(TS)] [config] timer=$TIMER_ENABLED qp=ON"
[[ "$BUILD_QP" == "ON" ]] || fail "this operator requires a multi-QP build; rebuild with scripts/build.sh"

SHMEM_ENV="${SHMEM_HOME_PATH:-${PROJECT_ROOT}/3rdparty/shmem/install}/set_env.sh"
[[ -f "$SHMEM_ENV" ]] || fail "SHMEM environment not found: $SHMEM_ENV"
source "$SHMEM_ENV" || exit 1
SHMEM_HOME_PATH=${SHMEM_HOME_PATH:-${PROJECT_ROOT}/3rdparty/shmem/install}

if (( RANK_SIZE > RPN )); then
    SHMEM_VERSION_FILE="${SHMEM_HOME_PATH}/version.info"
    BUILD_VERSION_FILE="${BUILD_DIR}/bin/qmma_shmem_version.info"
    [[ -f "$SHMEM_VERSION_FILE" ]] || fail "SHMEM SDK metadata not found: $SHMEM_VERSION_FILE"
    [[ -f "$BUILD_VERSION_FILE" ]] || \
        fail "build/SHMEM version snapshot not found: $BUILD_VERSION_FILE; rebuild with scripts/build.sh"
    cmp -s -- "$SHMEM_VERSION_FILE" "$BUILD_VERSION_FILE" || \
        fail "SHMEM SDK changed after this binary was built; rebuild with scripts/build.sh"

    RDMA_BACKEND=$(sed -n 's/^RDMA Backend : //p' "$SHMEM_VERSION_FILE" | head -n 1 | tr -d '\r')
    [[ "$RDMA_BACKEND" == "XSCALE" || "$RDMA_BACKEND" == "HNS_1825" ]] || \
        fail "SHMEM was built without RDMA support (RDMA Backend: ${RDMA_BACKEND:-unknown})"
    SOC_TYPE=$(sed -n 's/^SOC Type : //p' "$SHMEM_VERSION_FILE" | head -n 1 | tr -d '\r')
    [[ "$SOC_TYPE" == "Ascend950" ]] || \
        fail "SHMEM SOC Type must be Ascend950 (got: ${SOC_TYPE:-unknown})"

    INSTALLED_COMMIT=$(sed -n 's/^Commit Id : //p' "$SHMEM_VERSION_FILE" | head -n 1 | tr -d '\r')
    if EXPECTED_COMMIT=$(git -C "${PROJECT_ROOT}" rev-parse HEAD:3rdparty/shmem 2>/dev/null); then
        [[ "$INSTALLED_COMMIT" == "$EXPECTED_COMMIT" ]] || \
            fail "installed SHMEM commit differs from 3rdparty/shmem; rebuild SHMEM and this example"
    fi
fi

[[ -x "$EXEC_BIN" ]] || fail "executable not found: $EXEC_BIN"
DATA_DIR=${QMMA_DATA_DIR:-${PROJECT_ROOT}/out/qmma/functional}
mkdir -p -- "$DATA_DIR"
DATA_DIR=$(cd -- "$DATA_DIR" && pwd)
DATA_CACHE_ROOT="${DATA_DIR}/shapes"
RANK_LOG_DIR=${QMMA_LOG_DIR:-"${DATA_DIR}/logs/$(date +%Y%m%d-%H%M%S)-rank${RANK_START}-$$"}
mkdir -p -- "$RANK_LOG_DIR"
RANK_LOG_DIR=$(cd -- "$RANK_LOG_DIR" && pwd)
echo "[$(TS)] per-rank logs: $RANK_LOG_DIR"
if [[ "$TIMER_ENABLED" == "ON" ]]; then
    echo "[$(TS)] [timer] enabled; files will be written below $RANK_LOG_DIR/shape*/timer"
    if (( RANK_SIZE > RPN )); then
        echo "[$(TS)] [timer] data path=multi-QP cross-node, symmetric output; expected stages=AIC,AIV,AIV_WAIT,RDMA_COPY$([[ "$RPN" -gt 1 ]] && printf ',UDMA_COPY')"
    else
        echo "[$(TS)] [timer] data path=multi-QP single-node; expected stages=AIC,AIV,AIV_WAIT,UDMA_COPY"
    fi
fi

GEN_SCRIPT="${SCRIPT_DIR}/gen_data_slice_n.py"
MXFP4_SCRIPT="${PROJECT_ROOT}/examples/utils/gen_data_fp4_mx.py"
VERIFY_SCRIPT="${PROJECT_ROOT}/examples/utils/verify_result.py"
[[ -f "$GEN_SCRIPT" ]] || fail "gen script not found: $GEN_SCRIPT"
[[ -f "$MXFP4_SCRIPT" ]] || fail "MXFP4 generator dependency not found: $MXFP4_SCRIPT"
[[ -f "$VERIFY_SCRIPT" ]] || fail "verify script not found: $VERIFY_SCRIPT"
command -v python3 >/dev/null || fail "python3 required"
command -v sha256sum >/dev/null || fail "sha256sum required for the data cache"
mkdir -p -- "$DATA_CACHE_ROOT"

# The generator itself invalidates the cache so a source change cannot silently
# reuse data produced by an older layout or quantization path.
GEN_FINGERPRINT=$(
    {
        sha256sum -- "$GEN_SCRIPT"
        sha256sum -- "$MXFP4_SCRIPT"
    } | awk '{print $1}' | sha256sum | awk '{print substr($1, 1, 16)}'
)
[[ "$GEN_FINGERPRINT" =~ ^[0-9a-f]{16}$ ]] || fail "failed to fingerprint the data generator"

CACHE_SCHEMA="qmma-data-v1"
CACHE_TMP_DIR=""
cleanup_cache_tmp() {
    if [[ -n "$CACHE_TMP_DIR" && -d "$CACHE_TMP_DIR" ]]; then
        rm -rf -- "$CACHE_TMP_DIR"
    fi
}
trap cleanup_cache_tmp EXIT

cache_manifest_matches() {
    local case_dir=$1
    local m=$2
    local k=$3
    local n=$4
    local manifest="${case_dir}/.qmma_data_manifest"

    [[ -s "$manifest" ]] || return 1
    grep -Fqx "schema=$CACHE_SCHEMA" "$manifest" || return 1
    grep -Fqx "fmt=$FMT" "$manifest" || return 1
    grep -Fqx "rank_size=$RANK_SIZE" "$manifest" || return 1
    grep -Fqx "m=$m" "$manifest" || return 1
    grep -Fqx "k=$k" "$manifest" || return 1
    grep -Fqx "n=$n" "$manifest" || return 1
    grep -Fqx "seed=$DATA_SEED" "$manifest" || return 1
    grep -Fqx "generator_fingerprint=$GEN_FINGERPRINT" "$manifest" || return 1
}

cache_data_complete() {
    local case_dir=$1
    local m=$2
    local k=$3
    local n=$4
    local rank file
    local pack=1
    local element_bytes=1
    local scale_k=$(((k + 31) / 32))
    local padded_scale_k=$(((scale_k + 1) / 2 * 2))
    local a_size b_size a_scale_size b_scale_size golden_size

    if (( FMT == 1 )); then
        pack=2
    elif (( FMT == 2 )); then
        element_bytes=2
    fi
    a_size=$((m * k / pack * element_bytes))
    a_scale_size=$((m * padded_scale_k))
    b_size=$((k * n / pack * element_bytes))
    b_scale_size=$((padded_scale_k * n))
    golden_size=$((m * n * 4))

    for (( rank = 0; rank < RANK_SIZE; ++rank )); do
        file="${case_dir}/rank_${rank}_a.bin"
        [[ -f "$file" && "$(stat -c '%s' -- "$file")" == "$a_size" ]] || return 1
        file="${case_dir}/rank_${rank}_b.bin"
        [[ -f "$file" && "$(stat -c '%s' -- "$file")" == "$b_size" ]] || return 1
        if (( FMT != 2 )); then
            file="${case_dir}/rank_${rank}_a_scale.bin"
            [[ -f "$file" && "$(stat -c '%s' -- "$file")" == "$a_scale_size" ]] || return 1
            file="${case_dir}/rank_${rank}_b_scale.bin"
            [[ -f "$file" && "$(stat -c '%s' -- "$file")" == "$b_scale_size" ]] || return 1
        fi
        file="${case_dir}/golden_${rank}.bin"
        [[ -f "$file" && "$(stat -c '%s' -- "$file")" == "$golden_size" ]] || return 1
    done
}

case_cache_valid() {
    local case_dir=$1
    local m=$2
    local k=$3
    local n=$4

    cache_manifest_matches "$case_dir" "$m" "$k" "$n" || return 1
    cache_data_complete "$case_dir" "$m" "$k" "$n"
}

write_cache_manifest() {
    local case_dir=$1
    local m=$2
    local k=$3
    local n=$4

    cat > "${case_dir}/.qmma_data_manifest" <<EOF
schema=$CACHE_SCHEMA
fmt=$FMT
rank_size=$RANK_SIZE
m=$m
k=$k
n=$n
seed=$DATA_SEED
generator_fingerprint=$GEN_FINGERPRINT
EOF
}

clean_case_outputs() {
    local case_dir=$1
    local idx rank
    for (( idx = 0; idx < LOCAL_RANKS; ++idx )); do
        rank=$((RANK_START + idx))
        rm -f -- \
            "${case_dir}/output_${rank}.bin"
    done
}

run_shape() {
    local shape_idx=$1
    local m=${SHAPE_M[shape_idx]}
    local k=${SHAPE_K[shape_idx]}
    local n=${SHAPE_N[shape_idx]}
    local source=${SHAPE_LINE[shape_idx]}
    local label="[shape $((shape_idx + 1))/$SHAPE_COUNT]"
    local status=0
    local idx
    local rank
    local rc
    local chunk_n
    local cache_id="qmma_fmt${FMT}_P${RANK_SIZE}_M${m}_K${k}_N${n}_seed${DATA_SEED}"
    local case_data_dir="${DATA_CACHE_ROOT}/${cache_id}"
    local case_log_dir="${RANK_LOG_DIR}/shape$((shape_idx + 1))_M${m}_K${k}_N${n}"
    local case_timer_dir="${case_log_dir}/timer"

    if (( n % RANK_SIZE != 0 )); then
        echo "[$(TS)] $label [SKIP] source=$source M=$m K=$k N=$n: N must be divisible by P=$RANK_SIZE"
        return 2
    fi
    chunk_n=$((n / RANK_SIZE))
    if (( chunk_n % 256 != 0 )); then
        echo "[$(TS)] $label [SKIP] source=$source M=$m K=$k N=$n: N/P must be divisible by 256 (got $chunk_n)"
        return 2
    fi
    if (( k % 32 != 0 )); then
        echo "[$(TS)] $label [SKIP] source=$source M=$m K=$k N=$n: K must be divisible by 32"
        return 2
    fi

    echo "[$(TS)] $label start: M=$m K=$k N=$n chunkN=$chunk_n data_dir=$case_data_dir"
    echo "[$(TS)] $label [1/4] checking shape data cache"
    if case_cache_valid "$case_data_dir" "$m" "$k" "$n"; then
        echo "[$(TS)] $label [cache hit] fmt=$FMT P=$RANK_SIZE seed=$DATA_SEED data_dir=$case_data_dir"
    else
        echo "[$(TS)] $label [cache miss] fmt=$FMT P=$RANK_SIZE seed=$DATA_SEED data_dir=$case_data_dir"
        if [[ -e "$case_data_dir" ]]; then
            echo "[$(TS)] $label stale or incomplete cache will be replaced"
            rm -rf -- "$case_data_dir"
        fi
        echo "[$(TS)] $label [2/4] generating inputs and golden"
        if (( FMT == 2 )); then
            echo "[$(TS)]        Generating BF16 inputs and FP32 golden"
        else
            echo "[$(TS)]        Python quantization may take several minutes for large shapes"
        fi
        CACHE_TMP_DIR=$(mktemp -d "${DATA_CACHE_ROOT}/.${cache_id}.tmp.XXXXXX")
        if ! QMMA_SEED="$DATA_SEED" python3 "$GEN_SCRIPT" "$FMT" "$RANK_SIZE" "$m" "$n" "$k" "$CACHE_TMP_DIR"; then
            echo "[$(TS)] $label [FAIL] golden generation failed" >&2
            return 1
        fi
        if ! cache_data_complete "$CACHE_TMP_DIR" "$m" "$k" "$n"; then
            echo "[$(TS)] $label [FAIL] generated cache is incomplete" >&2
            return 1
        fi
        write_cache_manifest "$CACHE_TMP_DIR" "$m" "$k" "$n"
        if ! mv -- "$CACHE_TMP_DIR" "$case_data_dir"; then
            echo "[$(TS)] $label [FAIL] could not publish generated cache" >&2
            return 1
        fi
        CACHE_TMP_DIR=""
        echo "[$(TS)] $label [cache save] inputs and golden stored at $case_data_dir"
    fi
    if ! case_cache_valid "$case_data_dir" "$m" "$k" "$n"; then
        echo "[$(TS)] $label [FAIL] shape cache validation failed: $case_data_dir" >&2
        return 1
    fi
    clean_case_outputs "$case_data_dir"
    echo "[$(TS)] $label       cached inputs and golden ready"

    if (( RANK_SIZE > RPN )) && [[ "${QMMA_CROSS_RENDEZVOUS:-1}" != "0" ]]; then
        echo "[$(TS)] $label [2.5/4] cross-node launch gate before starting ranks"
        cross_node_rendezvous "$m" "$n" "$k" "$((shape_idx + 1))"
        echo "[$(TS)] $label       launch gate passed"
    fi

    echo "[$(TS)] $label [3/4] launching $LOCAL_RANKS rank process(es)"
    mkdir -p -- "$case_log_dir"
    if [[ "$TIMER_ENABLED" == "ON" ]]; then
        mkdir -p -- "$case_timer_dir"
        echo "[$(TS)] $label timer_dir=$case_timer_dir"
    fi
    local pids=()
    local ranks=()

    launch_local_rank() {
        local idx=$1
        local rank
        rank=$((RANK_START + idx))
        local rank_timer_work_dir="$case_timer_dir/.rank${rank}_work"
        if [[ "$TIMER_ENABLED" == "ON" ]]; then
            mkdir -p -- "$rank_timer_work_dir"
        fi
        echo "[$(TS)] $label launching global rank=$rank on local device=${DEVICE_ID_LIST[idx]} log=${case_log_dir}/rank${rank}.log"
        (
            set -o pipefail
            # DeviceDGemm's generic timer exporter writes to ./output_timer.
            # Isolate each rank so concurrent ranks never share a timer buffer
            # directory, then merge the rank files into case_timer_dir below.
            if [[ "$TIMER_ENABLED" == "ON" ]]; then
                cd -- "$rank_timer_work_dir"
            fi
            "$EXEC_BIN" "$RANK_SIZE" "$rank" "$IPPORT" "$m" "$n" "$k" \
                "$case_data_dir" "$LOCAL_DEVICE_LIST" "$FMT" "$SLOT" "$LANE" "$RPN" 2>&1 | \
                tee "${case_log_dir}/rank${rank}.log"
        ) &
        pids+=($!)
        ranks+=("$rank")
    }

    if (( RANK_START == 0 )); then
        launch_local_rank 0
        # AllToAll requires at least two peers, so rank 0 must open the SHMEM
        # bootstrap listener before the remaining local ranks are started.
        if ! wait_for_local_bootstrap_listener "${pids[0]}" "${case_log_dir}/rank0.log"; then
            kill "${pids[0]}" 2>/dev/null || true
            wait "${pids[0]}" 2>/dev/null || true
            return 1
        fi
        cross_node_bootstrap_gate
        for (( idx = 1; idx < LOCAL_RANKS; ++idx )); do
            launch_local_rank "$idx"
        done
    else
        cross_node_bootstrap_gate
        for (( idx = 0; idx < LOCAL_RANKS; ++idx )); do
            launch_local_rank "$idx"
        done
    fi

    for (( idx = 0; idx < LOCAL_RANKS; ++idx )); do
        rank=${ranks[idx]}
        if wait "${pids[idx]}"; then
            echo "[$(TS)] $label rank=$rank exited successfully"
        else
            rc=$?
            status=1
            echo "[$(TS)] $label rank=$rank exited with status=$rc" >&2
        fi
    done

    if [[ "$TIMER_ENABLED" == "ON" ]]; then
        mkdir -p -- "$case_timer_dir/data"
        for (( idx = 0; idx < LOCAL_RANKS; ++idx )); do
            rank=$((RANK_START + idx))
            local rank_timer_dir="$case_timer_dir/.rank${rank}_work/output_timer"
            if [[ -f "$rank_timer_dir/timer_rank_${rank}.csv" ]]; then
                mv -- "$rank_timer_dir/timer_rank_${rank}.csv" "$case_timer_dir/"
            fi
            if [[ -f "$rank_timer_dir/data/timer_rank_${rank}_timeline.csv" ]]; then
                mv -- "$rank_timer_dir/data/timer_rank_${rank}_timeline.csv" "$case_timer_dir/data/"
            fi
        done
        echo "[$(TS)] $label [3.5/4] checking device timer CSV files"
        local timer_files_found=0
        local raw_csv duration_csv
        local raw_rows duration_rows raw_file_count duration_file_count
        for (( idx = 0; idx < LOCAL_RANKS; ++idx )); do
            rank=$((RANK_START + idx))
            local -a raw_files=()
            local -a duration_files=()
            shopt -s nullglob
            raw_files=(
                "${case_timer_dir}/data/timer_rank_${rank}_timeline.csv"
                "${case_timer_dir}/data/timer_rank_${rank}_"*"_timeline.csv"
            )
            duration_files=(
                "${case_timer_dir}/timer_rank_${rank}.csv"
                "${case_timer_dir}/timer_rank_${rank}_"*.csv
            )
            shopt -u nullglob
            raw_rows=0
            duration_rows=0
            raw_file_count=0
            duration_file_count=0
            for raw_csv in "${raw_files[@]}"; do
                [[ -s "$raw_csv" ]] || continue
                raw_file_count=$((raw_file_count + 1))
                raw_rows=$((raw_rows + $(wc -l < "$raw_csv") - 1))
            done
            for duration_csv in "${duration_files[@]}"; do
                [[ -s "$duration_csv" ]] || continue
                duration_file_count=$((duration_file_count + 1))
                duration_rows=$((duration_rows + $(wc -l < "$duration_csv") - 1))
            done
            if (( raw_file_count == 0 || duration_file_count == 0 )); then
                status=1
                echo "[$(TS)] $label [FAIL] timer CSV missing or empty for rank=$rank" >&2
                continue
            fi
            timer_files_found=1
            echo "[$(TS)] $label timer rank=$rank raw_files=$raw_file_count raw_rows=$raw_rows duration_files=$duration_file_count duration_rows=$duration_rows"
            if (( raw_file_count > 1 )); then
                echo "[$(TS)] $label timer rank=$rank merged sources: $(basename -a "${raw_files[@]}" | paste -sd ',' -)"
            fi
        done
        if (( timer_files_found == 1 )); then
            local trace_dir="${case_log_dir}/trace"
            echo "[$(TS)] $label [3.75/4] summarizing compute/communication overlap"
            if ! python3 "${SCRIPT_DIR}/trace.py" "$case_timer_dir" --output "$trace_dir"; then
                status=1
                echo "[$(TS)] $label [FAIL] timer trace summary generation failed" >&2
            fi
        fi
    fi

    echo "[$(TS)] $label [4/4] verifying $LOCAL_RANKS rank output(s)"
    for (( idx = 0; idx < LOCAL_RANKS; ++idx )); do
        rank=$((RANK_START + idx))
        if [[ ! -f "${case_data_dir}/output_${rank}.bin" ]]; then
            status=1
            echo "[$(TS)] $label [FAIL] output_${rank}.bin missing" >&2
            continue
        fi
        echo "[$(TS)] $label verifying rank=$rank"
        if python3 "$VERIFY_SCRIPT" "${case_data_dir}/output_${rank}.bin" "${case_data_dir}/golden_${rank}.bin" \
            bf16 "$m" "$n" "$k"; then
            echo "[$(TS)] $label rank=$rank precision PASS"
        else
            rc=$?
            status=1
            echo "[$(TS)] $label rank=$rank precision FAIL (status=$rc)" >&2
        fi
    done

    if (( status == 0 )); then
        echo "[$(TS)] $label [PASS] M=$m K=$k N=$n"
    else
        echo "[$(TS)] $label [FAIL] M=$m K=$k N=$n" >&2
    fi
    return "$status"
}

OVERALL_STATUS=0
PASSED=0
SKIPPED=0
for (( shape_idx = 0; shape_idx < SHAPE_COUNT; ++shape_idx )); do
    if run_shape "$shape_idx"; then
        PASSED=$((PASSED + 1))
    else
        rc=$?
        if (( rc == 2 )); then
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        OVERALL_STATUS=1
        break
    fi
done

if (( OVERALL_STATUS == 0 && PASSED == 0 && SKIPPED > 0 )); then
    echo "[$(TS)] [ERROR] no CSV shape is compatible with P=$RANK_SIZE" >&2
    OVERALL_STATUS=1
elif (( OVERALL_STATUS == 0 )); then
    echo "[$(TS)] [PASS] all $PASSED executed shape(s) verified (fmt=$FMT, P=$RANK_SIZE, skipped=$SKIPPED)"
else
    echo "[$(TS)] [FAIL] test stopped after $PASSED passed shape(s) (skipped=$SKIPPED)" >&2
fi
exit "$OVERALL_STATUS"
