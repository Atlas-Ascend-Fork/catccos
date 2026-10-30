"""Frontend contracts for remote_copy routes, transports and submission flags."""

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
    flag_id=2,
    pipes=(tla.arch.MTE2, tla.arch.MTE3),
    in_vector=True,
):
    @tla.kernel
    def kernel(dst: tla.Tensor, src: tla.Tensor, workspace: tla.Tensor, peer_arg):
        flag = tla.flag("submission", pipes[0], pipes[1], id=flag_id) if flag_id is not None else None
        if in_vector:
            with tla.vector():
                tla_dist.remote_copy(
                    dst, src, params, peer=peer_arg, ub_tensor=workspace if scratch else None, flag_value=flag
                )
        else:
            tla_dist.remote_copy(
                dst, src, params, peer=peer_arg, ub_tensor=workspace if scratch else None, flag_value=flag
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
    assert signature.parameters["flag_value"].default is None


@pytest.mark.parametrize("mode", ["get", "put"])
def test_mte_gm_to_gm_uses_explicit_flag(mode):
    text = _ir(params=tla_dist.params.CopyParams(mode=mode))
    line = _copy_line(text)
    assert f'mode = "{mode}"' in line
    assert 'transport = "mte"' in line
    assert "!tla.flag" in line
    assert "id = 2" in text
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
            flag_id=2 if gm_to_gm else None,
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
            flag_id=2 if gm_to_gm else None,
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
        flag_id=None,
        params=params,
    )
    line = _copy_line(text)
    assert f'mode = "{expected}"' in line
    assert "!tla.flag" not in line
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
        flag_id=None,
        params=tla_dist.params.CopyParams(mode=requested),
    )
    line = _copy_line(text)
    assert f'mode = "{tla_dist.params.CopyMode(requested).value}"' in line
    assert "!tla.flag" not in line
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
            flag_id=None,
            params=tla_dist.params.CopyParams(mode=requested),
        )


@pytest.mark.parametrize(
    "transport,pipes",
    [
        ("rdma", (tla.arch.SCALAR, tla.arch.MTE3)),
        ("sdma", (tla.arch.MTE3, tla.arch.MTE2)),
        ("udma", (tla.arch.MTE3, tla.arch.SCALAR)),
    ],
)
def test_non_mte_frontend_accepts_byte_workspace(transport, pipes):
    line = _copy_line(
        _ir(params=tla_dist.params.CopyParams(mode="get", transport=transport), scratch_dtype=tla.UInt8, pipes=pipes)
    )
    assert f'transport = "{transport}"' in line
    assert "!tla.ptr<ui8, ub" in line
    if transport == "udma":
        assert 'wqe_pipe = "mte3"' in line


def test_udma_scalar_submission_uses_no_workspace_or_flag():
    line = _copy_line(
        _ir(scratch=False, flag_id=None, params=tla_dist.params.CopyParams(mode="get", transport="udma", wqe_pipe="s"))
    )
    assert 'wqe_pipe = "s"' in line
    assert "!tla.flag" not in line


@pytest.mark.parametrize("scratch,flag_id", [(False, None), (False, 2), (True, None)])
def test_gm_to_gm_requires_scratch_and_flag(scratch, flag_id):
    with pytest.raises(TlaLoweringError, match="requires both"):
        assert _ir(params=tla_dist.params.CopyParams(mode="get"), scratch=scratch, flag_id=flag_id)


def test_submission_flag_requires_explicit_id():
    with pytest.raises(TlaLoweringError, match="requires an explicit id"):
        assert _ir(params=tla_dist.params.CopyParams(mode="get"), flag_id=-1)


@pytest.mark.parametrize("transport", ["mte", "rdma", "sdma", "udma"])
def test_submission_flag_requires_transport_pipe_direction(transport):
    with pytest.raises(TlaLoweringError, match="flag_value must use"):
        assert _ir(
            params=tla_dist.params.CopyParams(mode="get", transport=transport), pipes=(tla.arch.VECTOR, tla.arch.MTE3)
        )


@pytest.mark.parametrize("transport", ["mte", "udma"])
def test_routes_without_workspace_reject_extra_arguments(transport):
    kwargs = {"dst": _tensor(space=tla.AddressSpace.ub)} if transport == "mte" else {}
    params = tla_dist.params.CopyParams(mode="get", transport=transport, wqe_pipe="s" if transport == "udma" else None)
    with pytest.raises(TlaLoweringError, match="does not accept"):
        assert _ir(params=params, **kwargs)


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


@pytest.mark.parametrize("dtype", [tla.Float16, tla.BFloat16, tla.Int8, tla.Int16, tla.Int32])
def test_only_fp32_data_is_supported(dtype):
    with pytest.raises(TlaLoweringError, match="only f32"):
        assert _ir(
            params=tla_dist.params.CopyParams(mode="get"), src=_tensor(dtype), dst=_tensor(dtype), scratch_dtype=dtype
        )


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
            flag_id=None,
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
