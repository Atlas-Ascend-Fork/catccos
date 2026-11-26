# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Frontend contracts for remote_copy routes, transports and event IDs."""

# Native-dependent imports must follow the importorskip check.
# ruff: noqa: E402

from __future__ import annotations

import inspect

import catccos  # noqa: F401
import pytest

mlir_ir = pytest.importorskip("catlass._mlir.ir")
from catccos import tla_dist
from catccos.tla_dist import _comm
from catlass import _tla_type_bridge, tla
from catlass.execution_lowering import TlaLoweringError
from catlass.runtime import TlaCoreAPIError


def _tensor(dtype=tla.Float32, space=tla.AddressSpace.gm, layout=tla.arch.RowMajor, stride=None):
    return tla.make_fake_tensor(
        dtype,
        (8, 16),
        stride if stride is not None else ((16, 1) if layout == tla.arch.RowMajor else (1, 8)),
        addrspace=space,
        origin_shape=(8, 16),
        coord=(0, 0),
        layout_tag=layout,
    )


def _ir(
    *,
    dst=None,
    src=None,
    scratch=True,
    scratch_dtype=tla.Float32,
    scratch_space=tla.AddressSpace.ub,
    params=None,
    peer=1,
    event_id=2,
    in_vector=True,
):
    @tla.kernel
    def kernel(dst: tla.Tensor, src: tla.Tensor, workspace: tla.Tensor, peer_arg):
        if in_vector:
            with tla.vector():
                tla_dist.remote_copy(
                    dst, src, params, peer=peer_arg, ub_tensor=workspace if scratch else None, event_id=event_id
                )
        else:
            tla_dist.remote_copy(
                dst, src, params, peer=peer_arg, ub_tensor=workspace if scratch else None, event_id=event_id
            )

    return kernel.dump_mlir(
        type_args=(
            _tensor() if dst is None else dst,
            _tensor() if src is None else src,
            _tensor(scratch_dtype, scratch_space),
            tla.Int32(peer) if isinstance(peer, int) else peer,
        )
    )


def _copy_line(text):
    return next(line for line in text.splitlines() if '"tla_dist.remote_copy"' in line)


def test_public_communication_exports():
    for op in (tla_dist.remote_copy, tla_dist.barrier_all):
        assert op.__name__ in tla_dist.__all__
        assert getattr(tla_dist, op.__name__) is op
    assert not hasattr(tla_dist, "comm")
    assert not hasattr(tla_dist, "remote_get_gm_to_gm")
    assert tla_dist.fp32 is tla.Float32
    assert tla_dist.params.CopyUbToGmParams is tla.params.CopyUbToGmParams
    assert tla_dist.params.AtomicMode is tla.params.AtomicMode
    assert set(tla_dist.params.CopyTransport) == {"mte", "rdma", "udma", "sdma"}
    assert set(tla_dist.params.CopyWqePipe) == {"mte3", "s"}
    signature = inspect.signature(tla_dist.remote_copy)
    assert signature.parameters["peer"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["ub_tensor"].default is None
    assert signature.parameters["event_id"].default is None


@pytest.mark.parametrize("mode", ["get", "put"])
@pytest.mark.parametrize("event_id,expected", [(0, 0), (tla.Int32(2), 2), (7, 7)])
def test_mte_gm_to_gm_uses_explicit_event_id(mode, event_id, expected):
    text = _ir(params=tla_dist.params.CopyParams(mode=mode), event_id=event_id)
    line = _copy_line(text)
    assert f'mode = "{mode}"' in line
    assert 'transport = "mte"' in line
    assert f"event_id = {expected} : i32" in line
    assert "operandSegmentSizes" not in text


@pytest.mark.parametrize("operand", ["src", "dst"])
@pytest.mark.parametrize(
    "src_space,dst_space,mode",
    [
        (tla.AddressSpace.gm, tla.AddressSpace.gm, "get"),
        (tla.AddressSpace.gm, tla.AddressSpace.gm, "put"),
        (tla.AddressSpace.gm, tla.AddressSpace.ub, None),
        (tla.AddressSpace.ub, tla.AddressSpace.gm, None),
    ],
)
def test_remote_copy_rejects_nonunit_column_stride(operand, src_space, dst_space, mode):
    src_stride = (32, 2) if operand == "src" else (16, 1)
    dst_stride = (32, 2) if operand == "dst" else (16, 1)
    gm_to_gm = src_space == dst_space == tla.AddressSpace.gm
    with pytest.raises(TlaLoweringError, match=f"requires {operand} column stride of 1"):
        assert _ir(
            src=_tensor(space=src_space, stride=src_stride),
            dst=_tensor(space=dst_space, stride=dst_stride),
            params=tla_dist.params.CopyParams(mode=mode),
            scratch=gm_to_gm,
            event_id=2 if gm_to_gm else None,
        )


@pytest.mark.parametrize(
    "src_space,dst_space,mode",
    [
        (tla.AddressSpace.gm, tla.AddressSpace.gm, "get"),
        (tla.AddressSpace.gm, tla.AddressSpace.gm, "put"),
        (tla.AddressSpace.gm, tla.AddressSpace.ub, None),
        (tla.AddressSpace.ub, tla.AddressSpace.gm, None),
    ],
)
def test_remote_copy_accepts_row_padding(src_space, dst_space, mode):
    gm_to_gm = src_space == dst_space == tla.AddressSpace.gm
    line = _copy_line(
        _ir(
            src=_tensor(space=src_space, stride=(32, 1)),
            dst=_tensor(space=dst_space, stride=(40, 1)),
            params=tla_dist.params.CopyParams(mode=mode),
            scratch=gm_to_gm,
            event_id=2 if gm_to_gm else None,
        )
    )
    assert "!tla.stride<32,1>" in line
    assert "!tla.stride<40,1>" in line


def test_remote_copy_accepts_dynamic_rows_with_unit_column_stride():
    text = _ir(
        src=_tensor(stride=(32, 1)).mark_layout_dynamic(),
        dst=_tensor(stride=(40, 1)).mark_layout_dynamic(),
        params=tla_dist.params.CopyParams(mode="get"),
    )
    assert "!tla.stride<?,1>" in _copy_line(text)


@pytest.mark.parametrize(
    "src_space,dst_space,expected",
    [
        (tla.AddressSpace.gm, tla.AddressSpace.ub, "get"),
        (tla.AddressSpace.ub, tla.AddressSpace.gm, "put"),
    ],
)
@pytest.mark.parametrize("params", [None, tla_dist.params.CopyParams(), tla_dist.params.CopyParams(mode=None)])
def test_mte_routes_select_unspecified_mode(src_space, dst_space, expected, params):
    text = _ir(
        src=_tensor(space=src_space),
        dst=_tensor(space=dst_space),
        scratch=False,
        event_id=None,
        params=params,
    )
    line = _copy_line(text)
    assert f'mode = "{expected}"' in line
    assert "event_id" not in line
    assert line.rstrip().endswith(", i32) -> ()")
    if params is not None:
        assert params.mode is None


@pytest.mark.parametrize(
    "params",
    [
        None,
        tla_dist.params.CopyParams(),
        tla_dist.params.CopyParams(mode=None),
        tla_dist.params.CopyParams(transport="rdma"),
        tla_dist.params.CopyParams(transport="sdma"),
        tla_dist.params.CopyParams(transport="udma"),
        tla_dist.params.CopyParams(transport="udma", wqe_pipe="s"),
    ],
)
def test_gm_to_gm_requires_explicit_mode(params):
    with pytest.raises(TlaLoweringError, match="GM -> GM requires an explicit mode"):
        assert _ir(params=params)


@pytest.mark.parametrize(
    "src_space,dst_space,requested",
    [
        (tla.AddressSpace.gm, tla.AddressSpace.ub, "get"),
        (tla.AddressSpace.ub, tla.AddressSpace.gm, "put"),
        (tla.AddressSpace.gm, tla.AddressSpace.ub, tla_dist.params.CopyMode.GET),
        (tla.AddressSpace.ub, tla.AddressSpace.gm, tla_dist.params.CopyMode.PUT),
    ],
)
def test_mte_routes_accept_matching_explicit_mode(src_space, dst_space, requested):
    text = _ir(
        src=_tensor(space=src_space),
        dst=_tensor(space=dst_space),
        scratch=False,
        event_id=None,
        params=tla_dist.params.CopyParams(mode=requested),
    )
    line = _copy_line(text)
    assert f'mode = "{tla_dist.params.CopyMode(requested).value}"' in line
    assert "event_id" not in line
    assert line.rstrip().endswith(", i32) -> ()")


@pytest.mark.parametrize(
    "src_space,dst_space,requested,message",
    [
        (tla.AddressSpace.gm, tla.AddressSpace.ub, "put", "GM -> UB requires mode='get'"),
        (tla.AddressSpace.ub, tla.AddressSpace.gm, "get", "UB -> GM requires mode='put'"),
    ],
)
def test_mte_routes_reject_conflicting_explicit_mode(src_space, dst_space, requested, message):
    with pytest.raises(TlaLoweringError, match=message):
        assert _ir(
            src=_tensor(space=src_space),
            dst=_tensor(space=dst_space),
            scratch=False,
            event_id=None,
            params=tla_dist.params.CopyParams(mode=requested),
        )


@pytest.mark.parametrize("transport", ["rdma", "sdma", "udma"])
def test_non_mte_frontend_accepts_byte_workspace(transport):
    line = _copy_line(_ir(params=tla_dist.params.CopyParams(mode="get", transport=transport), scratch_dtype=tla.UInt8))
    assert "event_id = 2 : i32" in line
    assert f'transport = "{transport}"' in line
    assert "!tla.ptr<ui8, ub" in line
    if transport == "udma":
        assert 'wqe_pipe = "mte3"' in line


def test_udma_scalar_submission_uses_no_workspace_or_event_id():
    line = _copy_line(
        _ir(scratch=False, event_id=None, params=tla_dist.params.CopyParams(mode="get", transport="udma", wqe_pipe="s"))
    )
    assert 'wqe_pipe = "s"' in line
    assert "event_id" not in line


@pytest.mark.parametrize("scratch,event_id", [(False, None), (False, 2), (True, None)])
def test_gm_to_gm_requires_scratch_and_event_id(scratch, event_id):
    with pytest.raises(TlaLoweringError, match="requires both"):
        assert _ir(params=tla_dist.params.CopyParams(mode="get"), scratch=scratch, event_id=event_id)


@pytest.mark.parametrize("event_id", [-1, 8])
def test_event_id_requires_valid_range(event_id):
    with pytest.raises(TlaLoweringError, match="event_id must be in range 0..7"):
        assert _ir(params=tla_dist.params.CopyParams(mode="get"), event_id=event_id)


@pytest.mark.parametrize("event_id", [2.0, tla.Float32(2), True, "2"])
def test_event_id_rejects_nonintegers(event_id):
    with pytest.raises(TlaLoweringError, match="event_id must be a compile-time integer"):
        assert _ir(params=tla_dist.params.CopyParams(mode="get"), event_id=event_id)


@pytest.mark.parametrize("operation", ["remote_copy", "quiet"])
def test_event_id_rejects_runtime_integer(operation):
    @tla.kernel
    def kernel(dst: tla.Tensor, src: tla.Tensor, workspace: tla.Tensor, event_id: tla.Int32):
        params = tla_dist.params.CopyParams(mode="get", transport="rdma")
        with tla.vector():
            if tla.const_expr(operation == "remote_copy"):
                tla_dist.remote_copy(dst, src, params, peer=1, ub_tensor=workspace, event_id=event_id)
            else:
                tla_dist.quiet(params, peer=1, ub_tensor=workspace, event_id=event_id)

    with pytest.raises(TlaLoweringError, match="event_id must be a compile-time integer"):
        kernel.dump_mlir(type_args=(_tensor(), _tensor(), _tensor(space=tla.AddressSpace.ub), tla.Int32(0)))


@pytest.mark.parametrize("transport", ["mte", "udma"])
@pytest.mark.parametrize("scratch,event_id", [(True, None), (False, 0)])
def test_routes_without_workspace_reject_extra_arguments(transport, scratch, event_id):
    kwargs = {"dst": _tensor(space=tla.AddressSpace.ub)} if transport == "mte" else {}
    params = tla_dist.params.CopyParams(mode="get", transport=transport, wqe_pipe="s" if transport == "udma" else None)
    with pytest.raises(TlaLoweringError, match="does not accept"):
        assert _ir(params=params, scratch=scratch, event_id=event_id, **kwargs)


@pytest.mark.parametrize("transport", ["rdma", "udma", "sdma"])
def test_non_mte_rejects_gm_to_ub(transport):
    with pytest.raises(TlaLoweringError, match="only supported for MTE"):
        assert _ir(dst=_tensor(space=tla.AddressSpace.ub), params=tla_dist.params.CopyParams(transport=transport))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"src": _tensor(tla.Float16)}, "identical src/dst"),
        ({"src": _tensor(layout=tla.arch.ColumnMajor)}, "row-major"),
        ({"scratch_space": tla.AddressSpace.gm}, "scratch tensor in UB"),
        ({"scratch_dtype": tla.Float16}, "identical src/UB"),
        ({"src": _tensor(space=tla.AddressSpace.ub), "dst": _tensor(space=tla.AddressSpace.ub)}, "requires GM"),
        ({"peer": tla.Float32(1.0)}, "integer communication operand"),
    ],
)
def test_invalid_operands_are_rejected(kwargs, message):
    with pytest.raises(TlaLoweringError, match=message):
        assert _ir(params=tla_dist.params.CopyParams(mode="get"), **kwargs)


@pytest.mark.parametrize("dtype", [tla.Float16, tla.BFloat16, tla.Int8, tla.Int16, tla.Int32, tla.Int64, tla.UInt32])
def test_unsupported_payload_dtype_is_rejected(dtype):
    with pytest.raises(TlaLoweringError, match="supports f32 tensors, or ui64"):
        assert _ir(
            params=tla_dist.params.CopyParams(mode="get"), src=_tensor(dtype), dst=_tensor(dtype), scratch_dtype=dtype
        )


@pytest.mark.parametrize("mode", ["get", "put"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_uint64_mte_gm_to_gm(mode, dynamic):
    source = _tensor(tla.UInt64, stride=(24, 1))
    destination = _tensor(tla.UInt64, stride=(32, 1))
    if dynamic:
        source, destination = source.mark_layout_dynamic(), destination.mark_layout_dynamic()
    line = _copy_line(
        _ir(
            src=source,
            dst=destination,
            scratch_dtype=tla.UInt64,
            params=tla_dist.params.CopyParams(mode=mode),
        )
    )
    assert line.count("!tla.ptr<ui64,") == 3
    assert f'mode = "{mode}"' in line
    assert 'transport = "mte"' in line
    assert "event_id = 2 : i32" in line


@pytest.mark.parametrize(
    "params,src_space,dst_space",
    [
        (tla_dist.params.CopyParams(mode="get", transport="udma"), tla.AddressSpace.gm, tla.AddressSpace.gm),
        (tla_dist.params.CopyParams(mode="put", transport="rdma"), tla.AddressSpace.gm, tla.AddressSpace.gm),
        (tla_dist.params.CopyParams(mode="get", transport="sdma"), tla.AddressSpace.gm, tla.AddressSpace.gm),
        (
            tla_dist.params.CopyParams(mode="get", atomic_mode=tla.params.AtomicMode.ADD),
            tla.AddressSpace.gm,
            tla.AddressSpace.gm,
        ),
        (tla_dist.params.CopyParams(mode="get"), tla.AddressSpace.gm, tla.AddressSpace.ub),
        (tla_dist.params.CopyParams(mode="put"), tla.AddressSpace.ub, tla.AddressSpace.gm),
    ],
)
def test_uint64_rejects_unsupported_combinations(params, src_space, dst_space):
    with pytest.raises(TlaLoweringError, match="ui64 supports only non-atomic MTE GM -> GM"):
        assert _ir(
            src=_tensor(tla.UInt64, src_space),
            dst=_tensor(tla.UInt64, dst_space),
            scratch_dtype=tla.UInt64,
            params=params,
        )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"dst": _tensor(tla.Int64)}, "identical src/dst"),
        ({"scratch_dtype": tla.Int64}, "identical src/UB"),
        ({"src": _tensor(tla.UInt64, stride=(32, 2))}, "src column stride of 1"),
        ({"scratch": False, "event_id": None}, "requires both"),
    ],
)
def test_uint64_retains_operand_contracts(kwargs, message):
    operands = {"src": _tensor(tla.UInt64), "dst": _tensor(tla.UInt64), "scratch_dtype": tla.UInt64}
    operands.update(kwargs)
    with pytest.raises(TlaLoweringError, match=message):
        assert _ir(params=tla_dist.params.CopyParams(mode="get"), **operands)


@pytest.mark.parametrize("mode", ["get", "put"])
def test_mte_atomic_add(mode):
    assert "atomic_add" in _copy_line(
        _ir(params=tla_dist.params.CopyParams(mode=mode, atomic_mode=tla.params.AtomicMode.ADD))
    )


def test_atomic_add_requires_gm_destination():
    with pytest.raises(TlaLoweringError, match="GM destination"):
        assert _ir(
            dst=_tensor(space=tla.AddressSpace.ub),
            scratch=False,
            event_id=None,
            params=tla_dist.params.CopyParams(atomic_mode=tla.params.AtomicMode.ADD),
        )


@pytest.mark.parametrize(
    "params,message",
    [
        (tla.params.CopyUbToGmParams(), "requires tla_dist.params.CopyParams"),
        (tla_dist.params.CopyParams(mode="bad"), "mode must be"),
        (tla_dist.params.CopyParams(transport="bad"), "transport must be"),
        (tla_dist.params.CopyParams(atomic_mode="add"), "atomic_mode must be"),
        (tla_dist.params.CopyParams(wqe_pipe="s"), "only supported for UDMA"),
        (tla_dist.params.CopyParams(transport="udma", wqe_pipe="bad"), "wqe_pipe must be"),
        (tla_dist.params.CopyParams(transport="rdma", atomic_mode=tla.params.AtomicMode.ADD), "only supported for MTE"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(TlaLoweringError, match=message):
        assert _ir(params=params)


@pytest.mark.parametrize(
    "peer,cast",
    [
        (tla.Int16(1), "arith.extsi"),
        (tla.UInt16(1), "arith.extui"),
        (tla.Int64(1), "arith.trunci"),
        (tla.UInt64(1), "arith.trunci"),
        (tla.Int32(1), None),
        (tla.UInt32(1), "builtin.unrealized_conversion_cast"),
    ],
)
def test_peer_is_normalized_to_i32(peer, cast):
    text = _ir(params=tla_dist.params.CopyParams(mode="get"), peer=peer)
    if cast:
        assert cast in text
    assert ", i32," in _copy_line(text)


@pytest.mark.parametrize("signedness", ["signless", "signed", "unsigned"])
@pytest.mark.parametrize("source_width", [16, 32, 64])
def test_integer_ssa_is_normalized_to_signless_i32(signedness, source_width):
    with mlir_ir.Context() as context, mlir_ir.Location.unknown():
        _tla_type_bridge.load_tla_dialect(context)
        source_type = getattr(mlir_ir.IntegerType, f"get_{signedness}")(source_width)
        result_type = mlir_ir.IntegerType.get_signless(32)
        module = mlir_ir.Module.create()
        with mlir_ir.InsertionPoint(module.body):
            function = mlir_ir.Operation.create(
                "func.func",
                attributes={
                    "sym_name": mlir_ir.StringAttr.get("normalize_peer"),
                    "function_type": mlir_ir.TypeAttr.get(mlir_ir.FunctionType.get([source_type], [result_type])),
                },
                regions=1,
            )
        block = function.regions[0].blocks.append(source_type)
        source = block.arguments[0]
        with mlir_ir.InsertionPoint(block):
            result = _comm._as_integer_value(source, 32, loc=None)
            mlir_ir.Operation.create("func.return", operands=[result])
        assert result.type == result_type
        assert module.operation.verify()
        if source_width == 32:
            if signedness == "signless":
                assert result == source
            else:
                assert result.owner.name == "builtin.unrealized_conversion_cast"
                assert result.owner.operands[0] == source
        else:
            expected_cast = (
                "arith.trunci" if source_width > 32 else ("arith.extui" if signedness == "unsigned" else "arith.extsi")
            )
            assert result.owner.name == expected_cast
            cast_input = result.owner.operands[0]
            assert cast_input.type == mlir_ir.IntegerType.get_signless(source_width)
            if signedness != "signless":
                assert cast_input.owner.name == "builtin.unrealized_conversion_cast"
                assert cast_input.owner.operands[0] == source


def test_remote_copy_requires_vector():
    with pytest.raises(TlaCoreAPIError, match="tla.vector"):
        assert _ir(in_vector=False)


def test_barrier_all_requires_vector():
    @tla.kernel
    def kernel():
        tla_dist.barrier_all()

    with pytest.raises(TlaCoreAPIError, match="tla.vector"):
        kernel.dump_mlir()


@pytest.mark.parametrize("transport", ["rdma", "udma"])
def test_quiet_frontend(transport):
    @tla.kernel
    def kernel(workspace: tla.Tensor, peer: tla.Int32):
        with tla.vector():
            if tla.const_expr(transport == "rdma"):
                tla_dist.quiet(
                    tla_dist.params.CopyParams(transport=transport), peer=peer, ub_tensor=workspace, event_id=2
                )
            else:
                tla_dist.quiet(tla_dist.params.CopyParams(transport=transport), peer=peer)

    text = kernel.dump_mlir(type_args=(_tensor(space=tla.AddressSpace.ub), tla.Int32(1)))
    line = next(line for line in text.splitlines() if '"tla_dist.quiet"' in line)
    assert f'transport = "{transport}"' in line
    assert ("event_id = 2 : i32" in line) == (transport == "rdma")
    assert tla_dist.quiet.__name__ in tla_dist.__all__


@pytest.mark.parametrize("transport", ["mte", "sdma", "invalid"])
def test_quiet_rejects_other_transports(transport):
    @tla.kernel
    def kernel(peer: tla.Int32):
        with tla.vector():
            tla_dist.quiet(tla_dist.params.CopyParams(transport=transport), peer=peer)

    with pytest.raises(TlaLoweringError, match="rdma or udma"):
        kernel.dump_mlir(type_args=(tla.Int32(1),))


@pytest.mark.parametrize(
    "transport,scratch,event_id,message",
    [
        ("rdma", False, 0, "requires both"),
        ("rdma", True, None, "requires both"),
        ("udma", True, None, "does not accept"),
        ("udma", False, 0, "does not accept"),
        ("rdma", True, -1, "event_id must be in range 0..7"),
        ("rdma", True, 8, "event_id must be in range 0..7"),
        ("rdma", True, True, "event_id must be a compile-time integer"),
    ],
)
def test_quiet_checks_resources(transport, scratch, event_id, message):
    @tla.kernel
    def kernel(workspace: tla.Tensor, peer: tla.Int32):
        with tla.vector():
            tla_dist.quiet(
                tla_dist.params.CopyParams(transport=transport),
                peer=peer,
                ub_tensor=workspace if scratch else None,
                event_id=event_id,
            )

    with pytest.raises(TlaLoweringError, match=message):
        kernel.dump_mlir(type_args=(_tensor(space=tla.AddressSpace.ub), tla.Int32(1)))


@pytest.mark.parametrize("dynamic", [False, True])
def test_mte_ignores_qp_idx(dynamic):
    @tla.kernel
    def kernel(dst: tla.Tensor, src: tla.Tensor, workspace: tla.Tensor, peer: tla.Int32, qp: tla.Int32):
        with tla.vector():
            params = tla_dist.params.CopyParams(mode="get", transport="mte", qp_idx=qp + 1 if dynamic else 1)
            tla_dist.remote_copy(dst, src, params, peer=peer, ub_tensor=workspace, event_id=0)

    text = kernel.dump_mlir(
        type_args=(_tensor(), _tensor(), _tensor(space=tla.AddressSpace.ub), tla.Int32(1), tla.Int32(0))
    )
    copy = _copy_line(text)
    qp_operand = copy.split("(", 1)[1].split(")", 1)[0].split(",")[3].strip()
    definition = next(line for line in text.splitlines() if line.strip().startswith(qp_operand + " ="))
    assert "0 : i32" in definition


@pytest.mark.parametrize("transport", ["rdma", "udma"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_copy_and_quiet_select_the_same_qp(transport, dynamic):
    @tla.kernel
    def kernel(dst: tla.Tensor, src: tla.Tensor, workspace: tla.Tensor, peer: tla.Int32, qp: tla.Int32):
        scratch = workspace if transport == "rdma" else None
        event_id = 0 if transport == "rdma" else None
        with tla.vector():
            params = tla_dist.params.CopyParams(
                mode="get",
                transport=transport,
                qp_idx=qp + 1 if dynamic else 1,
                wqe_pipe="s" if transport == "udma" else None,
            )
            tla_dist.remote_copy(dst, src, params, peer=peer, ub_tensor=scratch, event_id=event_id)
            tla_dist.quiet(params, peer=peer, ub_tensor=scratch, event_id=event_id)

    text = kernel.dump_mlir(
        type_args=(_tensor(), _tensor(), _tensor(space=tla.AddressSpace.ub), tla.Int32(1), tla.Int32(0))
    )
    copy = _copy_line(text)
    qp_operand = copy.split("(", 1)[1].split(")", 1)[0].split(",")[3].strip()
    quiet = next(line for line in text.splitlines() if '"tla_dist.quiet"' in line)
    quiet_qp = quiet.split("(", 1)[1].split(")", 1)[0].split(",")[1].strip()
    if dynamic:
        assert quiet_qp == qp_operand
        assert "arith.addi" in text
    else:
        for value in (qp_operand, quiet_qp):
            definition = next(line for line in text.splitlines() if line.strip().startswith(value + " ="))
            assert "1 : i32" in definition


def test_quiet_uses_only_transport_and_qp():
    @tla.kernel
    def kernel(peer: tla.Int32):
        with tla.vector():
            params = tla_dist.params.CopyParams(
                transport="udma",
                qp_idx=1,
                mode=None,
                atomic_mode=tla_dist.params.AtomicMode.ADD,
                wqe_pipe="unused",
            )
            tla_dist.quiet(params, peer=peer)

    assert '"tla_dist.quiet"' in kernel.dump_mlir(type_args=(tla.Int32(1),))
