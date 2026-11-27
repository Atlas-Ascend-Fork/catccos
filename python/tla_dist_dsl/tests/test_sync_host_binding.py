# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Real Torch/DLPack host contracts for the padded synchronization fixtures."""

import catccos  # noqa: F401
import pytest

pytest.importorskip("catlass._mlir.ir")
torch = pytest.importorskip("torch")
from catlass.tla.runtime import RuntimeTensorError  # noqa: E402 - requires the pinned native runtime
from end_to_end import test_sync as acceptance  # noqa: E402 - requires the pinned native runtime


@pytest.mark.parametrize("rows", [14, 702])
def test_padded_binding_uses_the_original_torch_storage(rows):
    root = torch.arange(rows * 32, dtype=torch.float32).reshape(rows, 32)
    bound = acceptance._padded_view(root)
    window = bound._dlpack_source
    assert window.is_contiguous()
    assert window.storage_offset() == 8
    assert window.untyped_storage().data_ptr() == root.untyped_storage().data_ptr()
    assert window.data_ptr() == root.data_ptr() + 8 * root.element_size()
    assert tuple(bound.shape) == (rows * 32 - 8,)
    assert tuple(bound.stride) == (1,)
    assert bound.data_ptr == 0  # Real CPU DLPack is metadata-only, as documented.
    capsule = window.__dlpack__()
    exported = type(bound)._parse_capsule(capsule)
    assert exported["data_ptr"] == window.data_ptr()
    assert exported["shape"] == (rows * 32 - 8,)
    assert exported["strides"] == (1,)
    # Independently evaluate the same matrix offsets with Torch views. No
    # copied buffer can satisfy both pointer identity and these pitched tiles.
    for row in (1, 8) if rows == 14 else (1, 694):
        tile = window[row * 32 :].as_strided((5, 13), (32, 1))
        torch.testing.assert_close(tile, root[row : row + 5, 8:21], rtol=0, atol=0)
        assert tile.untyped_storage().data_ptr() == root.untyped_storage().data_ptr()


def test_legacy_cropped_column_binding_is_rejected_by_actual_runtime():
    root = torch.empty((14, 32), dtype=torch.float32)
    with pytest.raises(RuntimeTensorError, match="requires a buffer prepared"):
        acceptance._view(root[:, 8:])


@pytest.mark.parametrize(
    "shape,dtype,offset",
    [
        ((1, 64), torch.int32, 16),  # publication ready/ack storage offset
        ((1, 16), torch.int32, 0),  # resource flag
        ((1, 4), torch.int32, 0),  # resource output
        ((16, 16), torch.float16, 0),  # shared Cube inputs
    ],
)
def test_other_remaining_host_bindings_are_actual_compact_dlpack(shape, dtype, offset):
    root = torch.zeros(shape, dtype=dtype)
    view = root[:, offset:] if offset else root
    bound = acceptance._view(view)
    assert view.is_contiguous()
    assert tuple(bound.shape) == tuple(view.shape)
    capsule = view.__dlpack__()
    exported = type(bound)._parse_capsule(capsule)
    assert exported["data_ptr"] == root.data_ptr() + offset * root.element_size()
    assert exported["data_ptr"] == view.data_ptr()
    assert bound._dlpack_source is view
