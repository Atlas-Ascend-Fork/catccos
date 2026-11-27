# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression coverage for translating and consuming a symmetric GM pointer."""

# Native-dependent imports must follow the importorskip check.
# ruff: noqa: E402

import subprocess

import catccos  # noqa: F401
import pytest

pytest.importorskip("catlass._mlir.ir")
from catccos import tla_dist
from catccos.tla_dist.compiler import _execution
from catccos.tla_dist.compiler_bridge import resolve_tla_dist_compile
from catlass import tla
from catlass.execution_lowering import TlaLoweringError
from end_to_end.test_shmem_ptr import load_from_peer


def _args(mixed):
    tensor = tla.make_fake_tensor(tla.Float32, (1, 8), (8, 1), layout_tag=tla.arch.RowMajor)
    return tensor, tensor, tla.Int32(1), mixed


def test_shmem_ptr_frontend_preserves_pointee():
    assert "shmem_ptr" in tla_dist.__all__
    ir = load_from_peer.dump_mlir(type_args=_args(False))
    op = next(line for line in ir.splitlines() if '"tla_dist.shmem_ptr"' in line)
    assert "(!tla.ptr<f32, gm, 4>, i32) -> !tla.ptr<f32, gm, 1>" in op
    assert "tla.ptrtoint" not in ir
    assert "tla.inttoptr" not in ir


def test_shmem_ptr_rejects_ub():
    @tla.kernel
    def kernel():
        ub = tla.allocate(8, tla.Float32, tla.AddressSpace.ub, 32)
        with tla.vector():
            tla_dist.shmem_ptr(ub, 0)

    with pytest.raises(TlaLoweringError, match="requires a symmetric GM pointer"):
        kernel.dump_mlir(type_args=())


@pytest.mark.parametrize(
    ("result_type", "error"),
    [
        ("!tla.ptr<f32, ub, 1>", "requires GM input and result pointers"),
        ("!tla.ptr<i32, gm, 1>", "result pointee type must match the input pointee type"),
        ("!tla.ptr<f32, gm, 4>", "requires byte alignment for the result pointer"),
    ],
)
def test_shmem_ptr_verifies_result_type(tmp_path, result_type, error):
    compiler = resolve_tla_dist_compile()
    if compiler is None:
        pytest.skip("build the DSL compiler to run this test")
    source = tmp_path / "shmem_ptr.mlir"
    source.write_text(
        f"""module {{
  tla.func @invalid_result(%ptr: !tla.ptr<f32, gm, 4>, %peer: i32) {{
    "tla.vector"() ({{
      %remote = tla_dist.shmem_ptr %ptr, %peer : !tla.ptr<f32, gm, 4> -> {result_type}
    }}) : () -> ()
    tla.return
  }}
}}
"""
    )
    result = subprocess.run([str(compiler), str(source), "-o", "-"], check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert error in result.stderr


def test_shmem_ptr_lowers_to_aiv_call(tmp_path):
    compiler = resolve_tla_dist_compile()
    if compiler is None:
        pytest.skip("build the DSL compiler to run this test")
    source = tmp_path / "shmem_ptr.mlir"
    source.write_text(load_from_peer.dump_mlir(type_args=_args(False)))
    result = subprocess.run([str(compiler), str(source), "-o", "-"], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "tla_dist.shmem_ptr" not in result.stdout
    assert "func.func private @shmem_ptr_addr(i64, i32) -> i64" in result.stdout
    assert "call @shmem_ptr_addr(" in result.stdout
    assert "memref.load" in result.stdout


@pytest.mark.parametrize("mixed", [False, True])
def test_shmem_ptr_links_device_bitcode(tmp_path, mixed):
    if resolve_tla_dist_compile() is None or _execution._find_hivmc_a5() is None:
        pytest.skip("build the DSL compiler and configure CANN to check device linkage")
    artifact = tla_dist.compile(
        load_from_peer,
        type_args=_args(mixed),
        arch_scope="aic.c310" if mixed else "aiv.c310",
        cache_dir=tmp_path,
    )
    assert artifact.kernel_binary_path.is_file()
