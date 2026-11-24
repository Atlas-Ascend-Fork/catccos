# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile the actual E2E kernels without initializing SHMEM or an NPU."""

import json
import runpy
import subprocess
from pathlib import Path

import pytest

import catccos  # noqa: F401

pytest.importorskip("catlass._mlir.ir")
from catccos import tla_dist
from catccos.tla_dist.compiler_bridge import resolve_tla_dist_compile

MODULES = {
    transport: runpy.run_path(str(Path(__file__).parent / f"end_to_end/test_remote_copy_{transport}.py"))
    for transport in ("rdma", "udma")
}


@pytest.fixture(scope="module")
def require_rdma():
    compiler = resolve_tla_dist_compile()
    if compiler is None:
        pytest.skip("build the DSL compiler to run this test")
    config = Path(str(compiler) + ".json")
    assert config.is_file(), f"Missing {config}; rebuild the DSL compiler"
    backend = json.loads(config.read_text())["rdma_backend"]
    assert backend in ("NONE", "XSCALE", "HNS_1825"), backend
    if backend == "NONE":
        pytest.skip("RDMA is not enabled in this build")


def kernel_args(module, test):
    def tensor(stride):
        value = tla_dist.make_fake_tensor(
            tla_dist.Float32,
            (test.total_rows, stride),
            (stride, 1),
            addrspace=tla_dist.AddressSpace.gm,
            layout_tag=tla_dist.arch.RowMajor,
        )
        return value.mark_layout_dynamic() if test.dynamic else value

    args = [tensor(test.dst_stride), tensor(test.src_stride)]
    if test.kernel in (module["quiet_observe"], module["get_then_put"]):
        args.append(tensor(test.dst_stride))
    args.append(tla_dist.Int32(1))
    if test.kernel is module["empty_copy"]:
        args.append(tla_dist.Int32(0))
    args.append(test.case)
    return tuple(args)


@pytest.mark.parametrize(
    "transport,test",
    [
        pytest.param(transport, test, id=f"{transport}/{test.name}")
        for transport, module in MODULES.items()
        for test in module["iter_cases"]()
    ],
)
def test_e2e_kernel_compiles(tmp_path, transport, test, request):
    if transport == "rdma":
        request.getfixturevalue("require_rdma")
    compiler = resolve_tla_dist_compile()
    if compiler is None:
        pytest.skip("build the DSL compiler to run this test")
    module = MODULES[transport]
    source = tmp_path / "kernel.mlir"
    source.write_text(test.kernel.dump_mlir(type_args=kernel_args(module, test)))
    result = subprocess.run([str(compiler), str(source), "-o", "-"], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "tla_dist." not in result.stdout
    quiet_count = 2 if test.kernel is module["get_then_put"] else 3 if test.kernel is module["repeated_quiet"] else 1
    assert result.stdout.count(f"call @shmem_{transport}_quiet(") == quiet_count
    prefix = f"udma_{test.case.wqe_pipe}" if transport == "udma" else "rdma"
    assert f"call @shmem_{prefix}_get(" in result.stdout or f"call @shmem_{prefix}_put(" in result.stdout


@pytest.mark.parametrize("test", list(MODULES["rdma"]["iter_cases"]()), ids=lambda test: test.name)
def test_rdma_e2e_kernel_links(tmp_path, test, require_rdma):
    from catccos.tla_dist import compiler as compiler_module

    if compiler_module._execution._find_hivmc_a5() is None:
        pytest.skip("configure CANN to check device linkage")
    artifact = tla_dist.compile(
        test.kernel,
        type_args=kernel_args(MODULES["rdma"], test),
        arch_scope="aic.c310" if test.case.mixed_kernel else "aiv.c310",
        cache_dir=tmp_path,
    )
    assert artifact.kernel_binary_path.is_file()
