#!/bin/bash

# Environment setup script
# Used to configure CANN environment, SHMEM, CATLASS dependencies
#
# Usage:
#   source examples/utils/setup.sh [-soc_type Ascend950] [other shmem build options...]
#
# Notes:
#   - Options are forwarded to 3rdparty/shmem/scripts/build.sh when SHMEM is built
#     for the first time (install/ does not exist yet).
#   - If 3rdparty/shmem/install already exists, build options are ignored. Remove
#     that directory to rebuild SHMEM with different options (e.g. switch soc type).
ORIGINAL_DIR=$(pwd)
PROJECT_ROOT="$(git rev-parse --show-toplevel)"

usage() {
    cat <<'EOF'
Usage: source examples/utils/setup.sh [-soc_type <soc>] [shmem build options...]

Configure CANN, initialize submodules, build SHMEM (first run), and source SHMEM env.

Common options (forwarded to 3rdparty/shmem/scripts/build.sh):
  -soc_type Ascend950    Build SHMEM for Ascend950 (A5) targets

Other shmem options (e.g. -debug, -examples) are also accepted; see
3rdparty/shmem/scripts/build.sh for the full list.

If 3rdparty/shmem/install already exists, build options are skipped. Remove that
directory to force a rebuild with new options.
EOF
}

SHMEM_BUILD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -soc_type)
            if [[ -z "${2:-}" ]]; then
                echo "[ERROR] -soc_type requires a value (e.g. Ascend950)."
                cd "$ORIGINAL_DIR"
                return 1 2>/dev/null || exit 1
            fi
            SHMEM_BUILD_ARGS+=(-soc_type "$2")
            shift 2
            ;;
        -h|--help)
            usage
            return 0 2>/dev/null || exit 0
            ;;
        *)
            SHMEM_BUILD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ ${#SHMEM_BUILD_ARGS[@]} -gt 0 ]; then
    echo "[INFO] SHMEM build options: ${SHMEM_BUILD_ARGS[*]}"
fi

# Configure CANN environment variables
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "[WARN] ASCEND_HOME_PATH is not set."
    if [ -d "/usr/local/Ascend/ascend-toolkit/" ]; then
        echo "[INFO] Using default path: /usr/local/Ascend/ascend-toolkit"
        ASCEND_PATH="/usr/local/Ascend/"
    else
        echo "[ERROR] Could not find ascend-toolkit in default path: /usr/local/Ascend/ascend-toolkit."
        cd "$ORIGINAL_DIR"
        return 1 2>/dev/null || exit 1
    fi
    source "${ASCEND_PATH}/ascend-toolkit/set_env.sh" || {
        echo "[ERROR] Running set_env.sh in ${ASCEND_PATH}/ascend-toolkit failed."
        cd "$ORIGINAL_DIR"
        return 1 2>/dev/null || exit 1
    }
else
    echo "[INFO] ASCEND_HOME_PATH has already been set to: $ASCEND_HOME_PATH"
fi

# Configure CATLASS & SHMEM dependencies
submodule_status=$(git submodule status 2>&1)
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to check submodule status."
    cd "$ORIGINAL_DIR"
    return 1 2>/dev/null || exit 1
fi

if echo "$submodule_status" | grep -q '^[-+]'; then
    echo "Initializing and updating submodules..."
    git submodule update --init || {
        echo "[ERROR] Running git submodule update --init failed."
        exit 1
    }
else
    echo "Submodules already initialized and up to date. Skipping."
fi

SHMEM_PATH="${PROJECT_ROOT}/3rdparty/shmem"
SHMEM_INSTALL_PATH="${SHMEM_PATH}/install"
SHMEM_HEADER_PATH="${SHMEM_INSTALL_PATH}/shmem/include/shmem.h"

if [ -n "${SHMEM_HOME_PATH:-}" ] && [ ! -f "${SHMEM_HOME_PATH}/shmem/include/shmem.h" ]; then
    echo "[WARN] SHMEM_HOME_PATH is stale or incomplete: ${SHMEM_HOME_PATH}"
    echo "[INFO] Rebuilding the local SHMEM installation."
    unset SHMEM_HOME_PATH
fi

if [ -z "$SHMEM_HOME_PATH" ]; then
    cd "$SHMEM_PATH"
    if [ ! -f "$SHMEM_INSTALL_PATH/set_env.sh" ] || [ ! -f "$SHMEM_HEADER_PATH" ]; then
        bash scripts/build.sh "${SHMEM_BUILD_ARGS[@]}" || {
            echo "[ERROR] Running build.sh in 3rdparty/shmem failed."
            cd "$ORIGINAL_DIR"
            return 1 2>/dev/null || exit 1
        }
    elif [ ${#SHMEM_BUILD_ARGS[@]} -gt 0 ]; then
        echo "[WARN] SHMEM install already exists at: $SHMEM_PATH/install"
        echo "[WARN] Build options are ignored. Remove that directory to rebuild SHMEM with new options."
    fi
    source "${SHMEM_PATH}/install/set_env.sh" || {
        echo "[ERROR] Running set_env.sh in 3rdparty/shmem/install failed."
        cd "$ORIGINAL_DIR"
        return 1 2>/dev/null || exit 1
    }
else
    echo "[INFO] SHMEM_HOME_PATH has already been set to: $SHMEM_HOME_PATH"
fi

if [ ! -f "${SHMEM_HOME_PATH}/shmem/include/shmem.h" ]; then
    echo "[ERROR] SHMEM installation is incomplete: ${SHMEM_HOME_PATH}/shmem/include/shmem.h not found."
    cd "$ORIGINAL_DIR"
    return 1 2>/dev/null || exit 1
fi

echo "Environment setup completed!"
cd "$ORIGINAL_DIR"
