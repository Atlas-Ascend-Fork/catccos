#!/usr/bin/env bash
set -euo pipefail

tla_dist_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
catccos_root="$(cd "${tla_dist_root}/../.." && pwd)"
export CATLASS_TLA_DSL_ROOT="${catccos_root}/3rdparty/catlass/python/tla_dsl"
catlass_tla_root="${CATLASS_TLA_DSL_ROOT}"
shmem_root="${TLA_DIST_DSL_SHMEM_ROOT:-${catccos_root}/3rdparty/shmem}"
build_dir="${TLA_DIST_DSL_BUILD_DIR:-${tla_dist_root}/csrc/mlir/build}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "error: ASCEND_HOME_PATH is not set" >&2
  exit 1
fi
if [[ ! -f "${catlass_tla_root}/csrc/mlir/CMakeLists.txt" ]]; then
  echo "error: Catlass Python TLA DSL not found at ${catlass_tla_root}" >&2
  echo "initialize the pinned submodule from ${catccos_root}:" >&2
  echo "  git submodule update --init 3rdparty/catlass" >&2
  exit 1
fi
if [[ ! -f "${catlass_tla_root}/build.sh" ]]; then
  echo "error: Catlass Python TLA DSL build script not found at ${catlass_tla_root}/build.sh" >&2
  exit 1
fi
if [[ ! -f "${shmem_root}/include/shmem.h" ]]; then
  echo "error: ACLSHMEM source tree not found at ${shmem_root}" >&2
  exit 1
fi

# Scheme 2 keeps the two native compiler images in separate processes. Build
# Catlass's own Python type bridge first; CATCCOS itself only builds the
# TlaDistCompile executable and communication device bitcode below.
if [[ -z "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR:-}" &&
      -n "${TLA_DSL_PREBUILT_ASCENDNPU_IR:-}" ]]; then
  export CATLASS_DSL_PREBUILT_ASCENDNPU_IR="${TLA_DSL_PREBUILT_ASCENDNPU_IR}"
fi
# Use the active Python package's headers, rather than a different pybind11
# installed under the Conda prefix and found first by CMake.
pybind11_cmake_dir="$(python -m pybind11 --cmakedir)"
pybind11_cmake_arg="$(python -c 'import shlex, sys; print(shlex.quote("-Dpybind11_DIR=" + sys.argv[1]))' "${pybind11_cmake_dir}")"
# Catlass reads these explicit source/install settings. Keep both builds
# on the same AscendNPU-IR tree when the shared-root override is provided.
export CATLASS_DSL_ASCENDNPU_IR_ROOT="${CATLASS_DSL_ASCENDNPU_IR_ROOT:-${CATLASS_DSL_PREBUILT_ASCENDNPU_IR:-${catlass_tla_root}/3rdparty/AscendNPU-IR}}"
export CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR="${CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR:-${CATLASS_DSL_ASCENDNPU_IR_ROOT}/build/install}"
echo "==> Building pinned Catlass native type bridge"
CMAKE_ARGS="${CMAKE_ARGS:-} ${pybind11_cmake_arg}" \
PIP_NO_BUILD_ISOLATION="${PIP_NO_BUILD_ISOLATION:-false}" \
  bash "${catlass_tla_root}/build.sh"
if ! compgen -G "${catlass_tla_root}/catlass/_tla_type_bridge_native*.so" >/dev/null &&
   ! compgen -G "${catlass_tla_root}/csrc/mlir/build/python/catlass/_tla_type_bridge_native*.so" >/dev/null; then
  echo "error: Catlass build completed without producing _tla_type_bridge_native*.so" >&2
  exit 1
fi

cmake_args=(
  -G Ninja
  -S "${tla_dist_root}/csrc/mlir"
  -B "${build_dir}"
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
  -DCMAKE_C_COMPILER="${CC:-$(command -v clang)}"
  -DCMAKE_CXX_COMPILER="${CXX:-$(command -v clang++)}"
  -DCMAKE_SYSROOT=/
  -Dpybind11_DIR="${pybind11_cmake_dir}"
  -DCATLASS_TLA_DSL_ROOT="${catlass_tla_root}"
  -DTLA_DIST_DSL_SHMEM_ROOT="${shmem_root}"
  -DBISHENGIR_BUILD_TEMPLATE="${BISHENGIR_BUILD_TEMPLATE:-ON}"
)
if [[ -n "${BISHENG_COMPILER_PATH:-}" ]]; then
  cmake_args+=(-DBISHENG_COMPILER_PATH="${BISHENG_COMPILER_PATH}")
fi
if [[ -n "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR:-}" ]]; then
  cmake_args+=(-DCATLASS_DSL_PREBUILT_ASCENDNPU_IR="${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}")
fi
if [[ -n "${MLIR_TBLGEN_INCLUDE_DIR:-}" ]]; then
  cmake_args+=(-DMLIR_TBLGEN_INCLUDE_DIR="${MLIR_TBLGEN_INCLUDE_DIR}")
fi

cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --target tla_dist-compiler

echo "CATCCOS TLA_DIST build complete: ${build_dir}"
echo "Built Catlass _tla_type_bridge_native.so, TlaDistCompile and TLA_DIST device bitcode."
echo "Install the CATCCOS Python package:"
echo "  python -m pip install -e ${tla_dist_root}"
