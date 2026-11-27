# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Compile the actual UDMA default-domain acceptance fixtures before devices."""

import pytest

import catccos  # noqa: F401

pytest.importorskip("catlass._mlir.ir")
from catlass import tla
from end_to_end import test_sync as common
from end_to_end import test_sync_udma as acceptance
from test_sync_compiler import require_native

from catccos import tla_dist
from catccos.tla_dist.compiler import _execution


def fake(dtype, shape, stride=None):
    return tla.make_fake_tensor(dtype, shape, stride or (shape[-1], 1), layout_tag=tla.arch.RowMajor)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize(
    "action,mode",
    [
        ("quiet", "get"),
        ("quiet", "put"),
        ("barrier", "put"),
        ("fence", "put"),
        ("interlock", "put"),
    ],
)
def test_udma_default_domain_fixture_links(tmp_path, mixed, action, mode):
    require_native()
    if _execution._find_hivmc_a5() is None:
        pytest.skip("configure CANN for device linkage")
    payload = fake(tla.Float32, (14 * 32 - 8,), (1,))
    work = fake(tla.Float32, (8, 16))
    flag = fake(tla.Float32, (1, 16))
    matrix = fake(tla.Float16, (16, 16))
    args = (
        payload,
        payload,
        payload,
        work,
        work,
        work,
        flag,
        flag,
        matrix,
        matrix,
        fake(tla.Float32, (32, 16)),
        tla.Int32(1),
        acceptance.NetworkCase(action, mixed, mode),
    )
    artifact = tla_dist.compile(
        acceptance.network,
        type_args=args,
        cache_dir=tmp_path,
        arch_scope="aic.c310" if mixed else "aiv.c310",
    )
    assert artifact.kernel_binary_path.is_file()
    assert "shmem_udma_s_" in artifact.artifacts.LLVM
    if action == "quiet":
        assert "tla_dist_quiet_default" in artifact.artifacts.LLVM
    if action == "fence":
        assert "tla_dist_fence" in artifact.artifacts.LLVM
    if action == "interlock":
        assert "shmem_udma_quiet" in artifact.artifacts.LLVM


@pytest.mark.parametrize("mixed", [False, True])
def test_udma_publication_fixture_links(tmp_path, mixed):
    require_native()
    if _execution._find_hivmc_a5() is None:
        pytest.skip("configure CANN for device linkage")
    matrix = fake(tla.Float16, (16, 16))
    args = (
        fake(tla.Float32, (14 * 32 - 8,), (1,)),
        fake(tla.Float32, (702 * 32 - 8,), (1,)),
        fake(tla.Int32, (1, 48), (48, 1)),
        fake(tla.Int32, (1, 48), (48, 1)),
        fake(tla.Float32, (702 * 32 - 8,), (1,)),
        matrix,
        matrix,
        fake(tla.Float32, (32, 16)),
        tla.Int32(1),
        tla.Int32(0),
        100,
        mixed,
        "udma",
    )
    artifact = tla_dist.compile(
        common.publication,
        type_args=args,
        cache_dir=tmp_path,
        arch_scope="aic.c310" if mixed else "aiv.c310",
    )
    assert artifact.kernel_binary_path.is_file()
    assert "shmem_udma_s_put" in artifact.artifacts.LLVM
    assert "tla_dist_quiet_default" in artifact.artifacts.LLVM
    assert "tla_dist_signal_i32" in artifact.artifacts.LLVM
