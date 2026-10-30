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
    assert '"tla_dist.remote_copy"' in ir
    assert '"tla_dist.barrier_all"' in ir
    assert 'atomic_add, mode = "get", transport = "mte"' in ir
    assert "tla.cross_core_set_flag" in ir
    assert "tla.cross_core_wait_flag" in ir
