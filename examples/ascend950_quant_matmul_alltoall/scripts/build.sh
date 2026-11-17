#!/bin/bash
# 构建自举:先 source examples/utils/setup.sh(CANN 环境、子模块初始化、
# 首次构建 SHMEM 并导出 ASCEND_HOME_PATH / SHMEM_HOME_PATH),再 cmake。
# 跨机场景默认按 HNS_1825 构建 RDMA + multi-link；XSCALE 机器通过
# SHMEM_RDMA_BACKEND=XSCALE 覆盖。附加参数原样透传给 cmake。
set -e
SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
BUILD_DIR=${QMMA_BUILD_DIR:-${PROJECT_ROOT}/build/ascend950_quant_matmul_alltoall}
BUILD_TARGET=ascend950_quant_matmul_alltoall

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

SHMEM_RDMA_BACKEND=${SHMEM_RDMA_BACKEND:-HNS_1825}
SHMEM_RDMA_BACKEND=$(printf '%s' "$SHMEM_RDMA_BACKEND" | tr '[:lower:]' '[:upper:]')
case "$SHMEM_RDMA_BACKEND" in
    XSCALE|HNS_1825) ;;
    *) fail "SHMEM_RDMA_BACKEND must be XSCALE or HNS_1825, got: $SHMEM_RDMA_BACKEND" ;;
esac
export SHMEM_RDMA_BACKEND

if [[ -n "${SHMEM_BUILD_ARGS:-}" ]]; then
    read -ra SHMEM_ARGS <<< "$SHMEM_BUILD_ARGS"
else
    SHMEM_ARGS=(-enable_rdma -rdma_backend "$SHMEM_RDMA_BACKEND" -multi_link)
fi

source "${PROJECT_ROOT}/examples/utils/setup.sh" -soc_type Ascend950 "${SHMEM_ARGS[@]}" || {
    echo "[ERROR] Running setup.sh in ${PROJECT_ROOT}/examples/utils failed."
    exit 1
}

SHMEM_HOME_PATH=${SHMEM_HOME_PATH:-${PROJECT_ROOT}/3rdparty/shmem/install}
SHMEM_VERSION_FILE="${SHMEM_HOME_PATH}/version.info"
[[ -f "$SHMEM_VERSION_FILE" ]] || \
    fail "SHMEM SDK metadata not found: $SHMEM_VERSION_FILE; rebuild 3rdparty/shmem/install"

read_shmem_field() {
    local key=$1
    sed -n "s/^${key} : //p" "$SHMEM_VERSION_FILE" | head -n 1 | tr -d '\r'
}

INSTALLED_BACKEND=$(read_shmem_field 'RDMA Backend')
[[ "$INSTALLED_BACKEND" == "$SHMEM_RDMA_BACKEND" ]] || \
    fail "installed SHMEM RDMA backend '$INSTALLED_BACKEND' does not match requested '$SHMEM_RDMA_BACKEND'; remove and rebuild 3rdparty/shmem/install"

INSTALLED_SOC=$(read_shmem_field 'SOC Type')
[[ "$INSTALLED_SOC" == "Ascend950" ]] || \
    fail "installed SHMEM SOC Type '$INSTALLED_SOC' is not Ascend950; rebuild 3rdparty/shmem/install"

INSTALLED_COMMIT=$(read_shmem_field 'Commit Id')
if EXPECTED_COMMIT=$(git -C "${PROJECT_ROOT}" rev-parse HEAD:3rdparty/shmem 2>/dev/null); then
    [[ "$INSTALLED_COMMIT" == "$EXPECTED_COMMIT" ]] || \
        fail "installed SHMEM commit '$INSTALLED_COMMIT' differs from 3rdparty/shmem '$EXPECTED_COMMIT'; rebuild SHMEM"
fi

cmake -B "${BUILD_DIR}" -S "${PROJECT_ROOT}" \
    -DCATCCOS_BUILD_TESTS=OFF -DCATLASS_BISHENG_ARCH=a5 \
    -DQMMA_RDMA_BACKEND="$SHMEM_RDMA_BACKEND" \
    -DQMMA_HNS_1825_PI_ON_CHIP="${SHMEM_HNS_1825_PI_ON_CHIP:-}" "$@"

# The kernel library's custom command tracks only the entry .cpp. Remove the
# cached device library so header-only transport/backend changes are not skipped.
rm -f "${BUILD_DIR}/bin/lib${BUILD_TARGET}_impl.so"
cmake --build "${BUILD_DIR}" --target "${BUILD_TARGET}" -j
