# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile Matmul-ReduceScatter to frontend IR without NPU execution."""

from pathlib import Path
import runpy

import pytest

import catccos  # noqa: F401 - bootstrap the pinned Catlass package

pytest.importorskip("catlass._mlir.ir", reason="the native MLIR Python package is not installed")


def test_matmul_reduce_scatter_emits_communication_ir():
    source = Path(__file__).resolve().parents[1] / "examples/fusion_ops/matmul_reduce_scatter.py"
    module = runpy.run_path(str(source))
    ir = module["dump_tlair"]()
    copies = [line for line in ir.splitlines() if '"tla_dist.remote_copy"' in line]
    assert copies
    assert '"tla_dist.barrier_all"' in ir
    for copy in copies:
        assert "atomic_add" in copy
        assert 'mode = "get"' in copy
        assert 'transport = "mte"' in copy
    assert "tla.cross_core_set_flag" in ir
    assert "tla.cross_core_wait_flag" in ir
