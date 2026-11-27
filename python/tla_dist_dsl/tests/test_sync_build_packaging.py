# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Exercise the actual packaging CMake rule across an AIC-only source edit."""

import shutil
import subprocess
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("generator", ["Unix Makefiles", "Ninja"])
def test_mix_aic_package_refreshes_after_incremental_relink(tmp_path, generator):
    if shutil.which("cmake") is None:
        pytest.skip("CMake is required for the incremental packaging regression")
    if generator == "Ninja" and shutil.which("ninja") is None:
        pytest.skip("Ninja is required for the selected production build generator")
    source = Path(__file__).resolve().parents[1] / "csrc/mlir/bc/CMakeLists.txt"
    cmake = source.read_text(encoding="utf-8")
    anchor = '  add_custom_command(\n    OUTPUT\n      "${_tla_dist_packaged_base_mix_aic}"'
    begin = cmake.index(anchor)
    end = cmake.index("\n  )", begin) + len("\n  )")
    # Reuse the product rule verbatim. Fake bitcode bytes isolate dependency
    # semantics from SDK compilation; the real device lane separately builds BC.
    packaging_rule = cmake[begin:end]
    lines = ["cmake_minimum_required(VERSION 3.20)", "project(SyncPackageRegression NONE)"]
    for variable in (
        "_tla_dist_base_aic",
        "_tla_dist_linked_aiv",
        "TLA_DIST_AIV_BITCODE",
        "_tla_dist_linked_mix_aiv",
        "TLA_DIST_MIX_AIV_BITCODE",
    ):
        lines += [
            f'set({variable} "${{CMAKE_BINARY_DIR}}/{variable}.bc")',
            f'file(WRITE "${{{variable}}}" "unchanged-{variable}")',
        ]
    lines += ['set(_tla_dist_bc_package_dir "${CMAKE_BINARY_DIR}/package")']
    for variable in (
        "_tla_dist_packaged_base_mix_aic",
        "_tla_dist_packaged_base_aic",
        "_tla_dist_packaged_base_aiv",
        "_tla_dist_packaged_aiv",
        "_tla_dist_packaged_base_mix_aiv",
        "_tla_dist_packaged_mix_aiv",
    ):
        lines += [f'set({variable} "${{_tla_dist_bc_package_dir}}/{variable}.bc")']
    lines += [
        'set(_tla_dist_linked_mix_aic "${CMAKE_BINARY_DIR}/linked.mix_aic.bc")',
        'add_custom_command(OUTPUT "${_tla_dist_linked_mix_aic}" COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/aic-source.txt" "${_tla_dist_linked_mix_aic}" DEPENDS "${CMAKE_SOURCE_DIR}/aic-source.txt")',
        'add_custom_target(tla_dist_linked_mix_aic_bitcode DEPENDS "${_tla_dist_linked_mix_aic}")',
    ]
    for target in (
        "tla_dist_base_bitcode",
        "tla_dist_shmem_vector_bitcode",
        "tla_dist_linked_aiv_bitcode",
        "tla_dist_linked_mix_aiv_bitcode",
    ):
        lines += [f"add_custom_target({target})"]
    lines += [packaging_rule, 'add_custom_target(package ALL DEPENDS "${_tla_dist_packaged_base_mix_aic}")']
    (tmp_path / "CMakeLists.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    input_file = tmp_path / "aic-source.txt"
    input_file.write_bytes(b"aic-helper-v1")
    build = tmp_path / "build"

    def run(*args):
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr

    run("cmake", "-S", str(tmp_path), "-B", str(build), "-G", generator)
    run("cmake", "--build", str(build))
    packaged = build / "package/_tla_dist_packaged_base_mix_aic.bc"
    assert packaged.read_bytes() == b"aic-helper-v1"
    # Make and Ninja must both see a changed input timestamp, including on
    # filesystems/build tools whose dependency timestamps have second precision.
    time.sleep(1.05)
    input_file.write_bytes(b"aic-helper-v2")
    run("cmake", "--build", str(build))
    assert (build / "linked.mix_aic.bc").read_bytes() == b"aic-helper-v2"
    assert packaged.read_bytes() == b"aic-helper-v2"
