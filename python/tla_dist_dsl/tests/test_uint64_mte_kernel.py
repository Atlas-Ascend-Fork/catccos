# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Link the actual UINT64 E2E kernels without initializing SHMEM or an NPU."""

import subprocess
import sys
from pathlib import Path

import catccos  # noqa: F401
import pytest

pytest.importorskip("catlass._mlir.ir")
from catccos.tla_dist import compiler as compiler_module
from catccos.tla_dist.compiler_bridge import resolve_tla_dist_compile


@pytest.mark.parametrize("mode", ["get", "put"])
@pytest.mark.parametrize(
    "geometry", ["aligned-static", "tail-padded-static", "tail-padded-dynamic", "offset-multiround"]
)
def test_uint64_mte_e2e_kernel_links(tmp_path, mode, geometry):
    if resolve_tla_dist_compile() is None or compiler_module._execution._find_hivmc_a5() is None:
        pytest.skip("configure the native DSL compiler and CANN to check device linkage")
    script = Path(__file__).parent / "end_to_end/test_remote_copy_uint64_mte.py"
    case = f"{mode}/{geometry}"
    result = subprocess.run(
        [sys.executable, str(script), "--compile-only", "--case", case, "--cache-dir", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("COMPILE PASS ") == 1
    assert f"COMPILE PASS {case}" in result.stdout
