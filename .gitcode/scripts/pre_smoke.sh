#!/usr/bin/env bash
# CATCCOS TLA DSL 预冒烟测试：根据 PR 修改的文件选择是否运行测试。
# 用法：ut_type=ir|lit|all bash .gitcode/scripts/pre_smoke.sh [workspace]

# 不使用 set -e：单个测试失败后继续执行其他测试，最后统一返回失败数量。
set -uo pipefail

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace="${1:-${script_root}}"
if [[ ! -d "${workspace}/python/tla_dist_dsl" ]]; then
  echo "TLA DSL directory not found: ${workspace}/python/tla_dist_dsl; skip pre-smoke tests."
  exit 0
fi
workspace="$(cd "${workspace}" && pwd)"
dsl_root="${workspace}/python/tla_dist_dsl"
build_dir="${TLA_DIST_DSL_BUILD_DIR:-${dsl_root}/csrc/mlir/build}"
mode="${ut_type:-all}"
dry_run="${CI_DRY_RUN:-0}"
failed=0

case "${mode}" in
  ir|lit|all) ;;
  *) echo "ERROR: ut_type must be ir, lit, or all (got: ${mode})" >&2; exit 2 ;;
esac
case "${dry_run}" in
  0|1) ;;
  *) echo "ERROR: CI_DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac

# ============================================================================
# 1. 获取本次提交修改的文件
# ============================================================================

# CI 将变更文件清单下载到工作区根目录，这里直接使用。
difflist_file="${workspace}/pr_filelist.txt"
if [[ ! -s "${difflist_file}" ]]; then
  echo "ERROR: difflist not found or empty: ${difflist_file}" >&2
  exit 1
fi

echo "CATCCOS TLA DSL pre-smoke: ${mode}"
echo "workspace: ${workspace}"
echo "difflist: ${difflist_file}"
echo "changed files:"
cat "${difflist_file}"

# ============================================================================
# 2. 根据修改文件决定是否执行 DSL 测试
# ============================================================================

# 只有文档类文件变化时跳过测试；CMakeLists.txt 属于构建配置，不能跳过。
if ! grep -qE '(^|/)CMakeLists\.txt$' "${difflist_file}" &&
   ! grep -qvE '\.(md|rst|txt)$' "${difflist_file}"; then
  echo "Only doc change, skip all tests."
  exit 0
fi

# 没有修改 python/tla_dist_dsl/ 时，不需要运行本脚本负责的 DSL 测试。
if ! grep -qE '^python/tla_dist_dsl/' "${difflist_file}"; then
  echo "No TLA DSL changes; skip ${mode} tests."
  exit 0
fi

# ============================================================================
# 3. 统一执行测试并记录结果
# ============================================================================

# 执行前打印完整命令；dry-run 只展示命令，实际执行时累计失败项数。
run_test() {
  local name="$1"
  shift
  echo
  echo "=== [${name}] ==="
  printf '    >>> '
  printf '%q ' "$@"
  echo
  if [[ "${dry_run}" == 1 ]]; then
    echo "    [DRY RUN]"
    return 0
  fi
  if "$@"; then
    echo "    [OK] ${name}"
  else
    echo "    [FAIL] ${name}" >&2
    failed=$((failed + 1))
  fi
  return 0
}

# ============================================================================
# 4. 准备 DSL 测试环境
# ============================================================================

# Catlass 构建脚本在子进程中设置此路径；这里也导出，使 pytest 和 lit 启动的
# Python 进程能够导入 mlir.ir。
if [[ -z "${CATLASS_DSL_PREBUILT_ASCENDNPU_IR:-}" ]]; then
  export CATLASS_DSL_PREBUILT_ASCENDNPU_IR="${TLA_DSL_PREBUILT_ASCENDNPU_IR:-${workspace}/3rdparty/catlass/python/tla_dsl/3rdparty/AscendNPU-IR}"
fi
mlir_python="${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}/build/install/python_packages/mlir_core"
if [[ -d "${mlir_python}" ]]; then
  export PYTHONPATH="${mlir_python}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${dsl_root}${PYTHONPATH:+:${PYTHONPATH}}"

# 真实执行需要 CANN 环境；dry-run 不依赖本机 Ascend 工具链。
if [[ "${dry_run}" == 0 ]]; then
  if [[ -z "${ASCEND_HOME_PATH:-}" && -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
    # shellcheck source=/dev/null
    set +u
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
    set -u
  fi
  if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ERROR: ASCEND_HOME_PATH is not set; source the CANN environment first." >&2
    exit 1
  fi
fi

echo "AscendNPU-IR: ${CATLASS_DSL_PREBUILT_ASCENDNPU_IR}"

# ============================================================================
# 5. 按 ut_type 执行对应的 DSL 测试
# ============================================================================

# 两类测试均先构建 DSL；ir 运行前端 pytest，lit 运行编译器 lit 测试。
run_test "dsl_build" bash "${dsl_root}/build.sh"
if [[ "${mode}" == ir || "${mode}" == all ]]; then
  run_test "mlir_import" python -c 'from mlir import ir'
  run_test "dsl_ir" python -m pytest -q "${dsl_root}/tests" --ignore="${dsl_root}/tests/lit"
fi
if [[ "${mode}" == lit || "${mode}" == all ]]; then
  run_test "dsl_lit" cmake --build "${build_dir}" --target check-tla-dist-lit
fi

# ============================================================================
# 6. 汇总结果
# ============================================================================

echo
echo "CATCCOS TLA DSL pre-smoke: ${failed} failed item(s)"
exit "${failed}"
