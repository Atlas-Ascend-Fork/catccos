# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Native contracts for the reviewed UInt64 UB tensor-constructor extension."""

# Native-dependent imports must follow the importorskip check.
# ruff: noqa: E402

import catccos  # noqa: F401
import pytest

pytest.importorskip("catlass._mlir.ir")
from catccos import tla_dist
from catlass import core_api, tla
from catlass.execution_lowering import TlaLoweringError


def _construct(*, builder, dtype=tla.UInt64, space=tla.AddressSpace.ub, compute_copy=False):
    @tla.kernel
    def kernel(source: tla.Tensor):
        if tla.const_expr(space == tla.AddressSpace.gm):
            pointer = source.ptr
        else:
            pointer = tla.allocate(8 * 16, dtype, space, 32)
        if tla.const_expr(builder == "make_tensor"):
            scratch = tla.make_tensor(
                pointer,
                tla.make_layout(tla.make_shape(8, 16), tla.make_stride(16, 1), layoutTag=tla.arch.RowMajor),
            )
        else:
            scratch = tla.make_tensor_like(pointer, source, layoutTag=tla.arch.RowMajor)
        with tla.vector():
            if tla.const_expr(compute_copy):
                tla.copy(scratch, source)
            else:
                tla_dist.remote_copy(
                    source,
                    source,
                    tla_dist.params.CopyParams(mode="get"),
                    peer=1,
                    ub_tensor=scratch,
                    event_id=0,
                )

    source = tla.make_fake_tensor(dtype, (8, 16), (16, 1), addrspace=tla.AddressSpace.gm, layout_tag=tla.arch.RowMajor)
    return kernel.dump_mlir(type_args=(source,))


@pytest.mark.parametrize("builder", ["make_tensor", "make_tensor_like"])
def test_uint64_ub_constructor_builds_real_unsigned_communication_scratch(builder):
    text = _construct(builder=builder)
    assert "!tla.ptr<ui64, ub, 32>" in text
    assert "tla_dist.remote_copy" in text
    assert "u64" not in core_api._MAKE_TENSOR_SUPPORTED_ELEMENT_TYPES


@pytest.mark.parametrize("builder", ["make_tensor", "make_tensor_like"])
@pytest.mark.parametrize(
    "dtype,space",
    [
        (tla.Int64, tla.AddressSpace.ub),
        (tla.UInt64, tla.AddressSpace.gm),
        (tla.UInt64, tla.AddressSpace.l1),
    ],
)
def test_uint64_constructor_extension_keeps_other_types_and_spaces_rejected(builder, dtype, space):
    with pytest.raises(TlaLoweringError, match=f"tla.{builder} expects a supported element type"):
        assert _construct(builder=builder, dtype=dtype, space=space)


@pytest.mark.parametrize("builder", ["make_tensor", "make_tensor_like"])
def test_uint64_ub_constructor_does_not_enable_catlass_compute_copy(builder):
    with pytest.raises(TlaLoweringError, match="unsupported copy route"):
        assert _construct(builder=builder, compute_copy=True)
