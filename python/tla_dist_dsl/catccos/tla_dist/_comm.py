"""Internal builders for communication operations in the ``tla_dist`` dialect."""

from __future__ import annotations

from typing import Any

from catlass import core_api as _core
from catlass import runtime as _runtime
from catlass._mlir import ir as mlir_ir  # type: ignore[import-not-found]
from catlass.base_dsl.op import dsl_user_op
from catlass.base_dsl.typing import Numeric
from catlass.core_api import FlagLike, IndexLike
from catlass.execution_lowering import TlaLoweringError
from catlass.params import AtomicMode
from catlass.tla.typing import Tensor

from .params import CopyMode, CopyParams, CopyTransport, CopyWqePipe


def _resolve_integer(value: Any) -> tuple[Any, bool]:
    resolved = _core._resolve_bound_value(value)
    signed = True
    if isinstance(resolved, Numeric):
        if not type(resolved).is_integer:
            raise TlaLoweringError(f"expected an integer communication operand, got {type(resolved).__name__}")
        signed = bool(type(resolved).signed)
        if isinstance(resolved.value, (bool, int)):
            return int(resolved.value), signed
        resolved = resolved.ir_value()
    return resolved, signed


def _as_integer_value(value: Any, width: int, *, loc: mlir_ir.Location | None) -> mlir_ir.Value:
    resolved, signed = _resolve_integer(value)
    result_type = mlir_ir.IntegerType.get_signless(width)
    if isinstance(resolved, mlir_ir.Value):
        if isinstance(resolved.type, mlir_ir.IndexType):
            return mlir_ir.Operation.create(
                "arith.index_cast", operands=[resolved], results=[result_type], loc=loc
            ).results[0]
        if mlir_ir.IntegerType.isinstance(resolved.type):
            source_type = mlir_ir.IntegerType(resolved.type)
            if resolved.type == result_type:
                return resolved
            if not source_type.is_signless:
                # Arithmetic integer casts require signless operands. Preserve
                # the bits first, retaining source signedness for widening.
                signed = source_type.is_signed
                resolved = mlir_ir.Operation.create(
                    "builtin.unrealized_conversion_cast",
                    operands=[resolved],
                    results=[mlir_ir.IntegerType.get_signless(source_type.width)],
                    loc=loc,
                ).results[0]
                if resolved.type == result_type:
                    return resolved
            cast_name = ("arith.extsi" if signed else "arith.extui") if source_type.width < width else "arith.trunci"
            return mlir_ir.Operation.create(cast_name, operands=[resolved], results=[result_type], loc=loc).results[0]
    if isinstance(resolved, (bool, int)):
        return mlir_ir.Operation.create(
            "arith.constant",
            results=[result_type],
            attributes={"value": mlir_ir.IntegerAttr.get(result_type, int(resolved))},
            loc=loc,
        ).results[0]
    raise TlaLoweringError(f"expected an i{width}-like communication operand, got {type(value).__name__}")


def _tensor_descriptor(value: mlir_ir.Value) -> Any | None:
    try:
        return _core._tla_tensor_type_for_mlir_value(value)
    except TlaLoweringError:
        return None


def _build_remote_copy(
    dst: Tensor,
    src: Tensor,
    ub_tensor: Tensor | None,
    flag_value: FlagLike | None,
    peer: IndexLike,
    params: CopyParams,
    loc: mlir_ir.Location | None,
) -> None:
    operation_name = "tla_dist.remote_copy"
    if not isinstance(params, CopyParams):
        raise TlaLoweringError(f"{operation_name} requires tla_dist.params.CopyParams")
    try:
        params._validate()
    except ValueError as error:
        raise TlaLoweringError(f"{operation_name}: {error}") from error
    _core._require_category(operation_name, "dst", dst, "tensor", 0)
    _core._require_category(operation_name, "src", src, "tensor", 1)
    _core._require_frontend_state(operation_name)
    _runtime._require_enclosing_region(operation_name, "vector")
    dst_value, src_value = map(_core._as_value, (dst, src))
    dst_desc, src_desc = map(_tensor_descriptor, (dst_value, src_value))
    if dst_desc is None or src_desc is None:
        raise TlaLoweringError(f"{operation_name} requires tensor descriptors")
    route = (src_desc.addrspace.lower(), dst_desc.addrspace.lower())
    transport = CopyTransport(params.transport)
    mode = None if params.mode is None else CopyMode(params.mode)
    wqe_pipe = CopyWqePipe(params.wqe_pipe or CopyWqePipe.MTE3) if transport == CopyTransport.UDMA else None
    if route not in (("gm", "gm"), ("gm", "ub"), ("ub", "gm")):
        raise TlaLoweringError(f"{operation_name} requires GM -> GM, GM -> UB, or UB -> GM")
    if route == ("gm", "gm"):
        if mode is None:
            raise TlaLoweringError(f"{operation_name}: GM -> GM requires an explicit mode ('get' or 'put')")
    else:
        if transport != CopyTransport.MTE:
            raise TlaLoweringError(f"{operation_name}: GM <-> UB is only supported for MTE")
        required_mode = CopyMode.GET if route == ("gm", "ub") else CopyMode.PUT
        if mode is not None and mode != required_mode:
            route_name = "GM -> UB" if route == ("gm", "ub") else "UB -> GM"
            raise TlaLoweringError(f"{operation_name}: {route_name} requires mode='{required_mode.value}'")
        mode = required_mode
    if str(src_desc.element_type) != str(dst_desc.element_type):
        raise TlaLoweringError(f"{operation_name} requires identical src/dst dtypes")
    if (
        src_desc.layout_tag.replace("_", "").lower() != "rowmajor"
        or dst_desc.layout_tag.replace("_", "").lower() != "rowmajor"
    ):
        raise TlaLoweringError(f"{operation_name} currently requires row-major tensors")
    for name, desc in (("src", src_desc), ("dst", dst_desc)):
        column_stride = _core._flatten_tla_tuple(desc.stride)[-1]
        # Dynamic metadata is checked against the materialized SSA descriptor
        # by communication lowering; a RowMajor tag alone is insufficient.
        if column_stride is not None and column_stride != 1:
            raise TlaLoweringError(f"{operation_name} requires {name} column stride of 1")
    if str(src_desc.element_type) != "f32":
        raise TlaLoweringError(f"{operation_name} currently supports only f32 tensors")
    if params.atomic_mode == AtomicMode.ADD and route[1] != "gm":
        raise TlaLoweringError(f"{operation_name}: atomic ADD requires a GM destination")

    required_pair = None
    if route == ("gm", "gm"):
        required_pair = {
            CopyTransport.MTE: ("mte2", "mte3"),
            CopyTransport.RDMA: ("scalar", "mte3"),
            CopyTransport.SDMA: ("mte3", "mte2"),
            CopyTransport.UDMA: ("mte3", "scalar") if wqe_pipe == CopyWqePipe.MTE3 else None,
        }[transport]
    if required_pair is not None and (ub_tensor is None or flag_value is None):
        raise TlaLoweringError(f"{operation_name} requires both a UB scratch tensor and flag_value")
    if required_pair is None and (ub_tensor is not None or flag_value is not None):
        raise TlaLoweringError(f"{operation_name}: this route does not accept ub_tensor or flag_value")

    operands = [dst_value, src_value, _as_integer_value(peer, 32, loc=loc)]
    if ub_tensor is not None:
        _core._require_category(operation_name, "ub_tensor", ub_tensor, "tensor", 3)
        ub_value = _core._as_value(ub_tensor)
        ub_desc = _tensor_descriptor(ub_value)
        if ub_desc is None:
            raise TlaLoweringError(f"{operation_name} requires a UB tensor descriptor")
        if ub_desc.addrspace.lower() != "ub":
            raise TlaLoweringError(f"{operation_name} requires its scratch tensor in UB")
        if transport == CopyTransport.MTE and str(ub_desc.element_type) != str(src_desc.element_type):
            raise TlaLoweringError(f"{operation_name} requires identical src/UB dtypes")
        flag = _core._as_value(flag_value)
        if not isinstance(flag, mlir_ir.Value) or str(flag.type) != "!tla.flag":
            raise TlaLoweringError(f"{operation_name}: flag_value must be a tla.flag value")
        owner = flag.owner
        if not isinstance(owner, mlir_ir.Operation) or owner.name != "tla.flag":
            raise TlaLoweringError(f"{operation_name}: flag_value must be created by tla.flag")
        event_id = int(mlir_ir.IntegerAttr(owner.attributes["id"]).value) if "id" in owner.attributes else -1
        if not 0 <= event_id < 8:
            raise TlaLoweringError(f"{operation_name}: flag_value requires an explicit id in [0, 8)")
        actual_pair = tuple(str(owner.attributes[key]) for key in ("src_pipe", "dst_pipe"))
        if actual_pair != tuple(f"#tla.pipe<{pipe}>" for pipe in required_pair):
            raise TlaLoweringError(f"{operation_name}: flag_value must use {required_pair[0]} -> {required_pair[1]}")
        operands.extend([ub_value, flag])

    attributes: dict[str, mlir_ir.Attribute] = {
        "mode": mlir_ir.StringAttr.get(mode.value),
        "transport": mlir_ir.StringAttr.get(transport.value),
    }
    if params.atomic_mode == AtomicMode.ADD:
        attributes["atomic_add"] = mlir_ir.UnitAttr.get()
    if wqe_pipe is not None:
        attributes["wqe_pipe"] = mlir_ir.StringAttr.get(wqe_pipe.value)
    mlir_ir.Operation.create(
        operation_name,
        operands=operands,
        attributes=attributes,
        loc=loc,
    )


@dsl_user_op
def remote_copy(
    dst: Tensor,
    src: Tensor,
    params: CopyParams | None = None,
    *,
    peer: IndexLike,
    ub_tensor: Tensor | None = None,
    flag_value: FlagLike | None = None,
    loc: mlir_ir.Location | None = None,
) -> None:
    """Asynchronously copy a RowMajor FP32 tile to or from a symmetric peer.

    Like ``tla.copy``, the third argument contains compile-time parameters.
    The defaults are MTE transport and no atomic accumulation. GET resolves
    ``src`` on ``peer``; PUT resolves ``dst`` on ``peer``. GM->GM requires an
    explicit ``CopyParams(mode="get")`` or ``CopyParams(mode="put")``.

    GM->UB requires GET and UB->GM requires PUT, without scratch or a flag.
    For GM<->UB, an unspecified mode (``None``) is inferred from the route;
    an explicit mode that conflicts with the route raises an error.
    GM->GM requires caller-owned UB scratch and an explicit ``tla.flag`` with
    the transport's submission pipe direction. UDMA scalar publication uses
    neither scratch nor a flag. Only MTE currently has device lowering.

    Source and destination column strides must be provably 1. Row padding,
    dynamic shapes and dynamic row strides are supported. A dynamic column
    stride that cannot be proven to be 1 is rejected during device lowering.

    Return does not imply completion. The caller arranges completion waits,
    remote readiness and scratch reuse. MTE staging rows are padded to 32 B;
    control scratch capacity/alignment for other transports is caller-owned.
    """
    _build_remote_copy(
        dst,
        src,
        ub_tensor,
        flag_value,
        peer,
        CopyParams() if params is None else params,
        loc,
    )


@dsl_user_op
def barrier_all(*, loc: mlir_ir.Location | None = None) -> None:
    """Synchronize all ACLSHMEM ranks from an AIV/vector region."""
    _core._require_frontend_state("tla_dist.barrier_all")
    _runtime._require_enclosing_region("tla_dist.barrier_all", "vector")
    mlir_ir.Operation.create("tla_dist.barrier_all", loc=loc)


__all__ = ["barrier_all", "remote_copy"]
