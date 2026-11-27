# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Internal builders for communication operations in the ``tla_dist`` dialect."""

from __future__ import annotations

import math
import struct
from typing import Any

from catlass import core_api as _core
from catlass import runtime as _runtime
from catlass._mlir import ir as mlir_ir  # type: ignore[import-not-found]
from catlass.base_dsl.op import dsl_user_op
from catlass.base_dsl.typing import Int32, Numeric, Pointer
from catlass.core_api import IndexLike
from catlass.execution_lowering import TlaLoweringError
from catlass.params import AtomicMode
from catlass.tla.typing import Tensor
from catlass.types import PtrType

from .params import CompareOp, CoreScope, SignalOp, TEAM_WORLD, CopyMode, CopyParams, CopyTransport, CopyWqePipe


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


@dsl_user_op
def shmem_ptr(ptr: Pointer, peer: IndexLike, *, loc: mlir_ir.Location | None = None) -> Pointer:
    """Translate a symmetric GM pointer into a GM pointer on a world PE.

    Call in the scalar body of ``tla.vector()``. The caller supplies a valid PE
    with a directly accessible mapping and arranges synchronization for later
    accesses. The result retains the input pointee type.
    """
    name = "tla_dist.shmem_ptr"
    _core._require_frontend_state(name)
    _runtime._require_enclosing_region(name, "vector")
    if _runtime._has_enclosing_region("vec.func"):
        raise TlaLoweringError(f"{name} requires the scalar body of tla.vector()")

    _core._require_category(name, "ptr", ptr, "pointer", 0)
    source = _core._coerce_pointer_arg(ptr)
    if source.memspace != _core.AddressSpace.gm:
        raise TlaLoweringError(f"{name} requires a symmetric GM pointer")
    pe = _as_integer_value(peer, 32, loc=loc)
    # Use byte alignment because the peer mapping can have a different base.
    result_type = PtrType.get(source.pointee, "gm", 1)
    op = mlir_ir.Operation.create(name, operands=[source.value, pe], results=[result_type], loc=loc)
    return _core._Pointer(op.results[0])


def _tensor_descriptor(value: mlir_ir.Value) -> Any | None:
    try:
        return _core._tla_tensor_type_for_mlir_value(value)
    except TlaLoweringError:
        return None


def _build_scratch_operands(
    operation_name: str,
    required: bool,
    ub_tensor: Tensor | None,
    event_id: int | None,
    element_type: Any | None = None,
) -> list[mlir_ir.Value]:
    if required and (ub_tensor is None or event_id is None):
        raise TlaLoweringError(f"{operation_name} requires both a UB scratch tensor and event_id")
    if not required and (ub_tensor is not None or event_id is not None):
        raise TlaLoweringError(f"{operation_name}: this route does not accept ub_tensor or event_id")

    operands = []
    if ub_tensor is not None:
        _core._require_category(operation_name, "ub_tensor", ub_tensor, "tensor", 3)
        ub_value = _core._as_value(ub_tensor)
        ub_desc = _tensor_descriptor(ub_value)
        if ub_desc is None:
            raise TlaLoweringError(f"{operation_name} requires a UB tensor descriptor")
        if ub_desc.addrspace.lower() != "ub":
            raise TlaLoweringError(f"{operation_name} requires its scratch tensor in UB")
        if element_type is not None and str(ub_desc.element_type) != str(element_type):
            raise TlaLoweringError(f"{operation_name} requires identical src/UB dtypes")
        operands.append(ub_value)

    return operands


def _event_id_attr(operation_name: str, event_id: int) -> mlir_ir.IntegerAttr:
    value = _core._resolve_bound_value(event_id)
    if isinstance(value, Numeric) and type(value).is_integer:
        value = value.value
    if not isinstance(value, int) or isinstance(value, bool):
        raise TlaLoweringError(f"{operation_name}: event_id must be a compile-time integer")
    if not 0 <= value < 8:
        raise TlaLoweringError(f"{operation_name}: event_id must be in range 0..7")
    return mlir_ir.IntegerAttr.get(mlir_ir.IntegerType.get_signless(32), value)


def _build_remote_copy(
    dst: Tensor,
    src: Tensor,
    ub_tensor: Tensor | None,
    event_id: int | None,
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

    needs_scratch = route == ("gm", "gm") and not (transport == CopyTransport.UDMA and wqe_pipe == CopyWqePipe.S)
    qp_idx = params.qp_idx if transport in (CopyTransport.RDMA, CopyTransport.UDMA) else 0
    operands = [
        dst_value,
        src_value,
        _as_integer_value(peer, 32, loc=loc),
        _as_integer_value(qp_idx, 32, loc=loc),
    ]
    operands.extend(
        _build_scratch_operands(
            operation_name,
            needs_scratch,
            ub_tensor,
            event_id,
            src_desc.element_type if transport == CopyTransport.MTE else None,
        )
    )

    attributes: dict[str, mlir_ir.Attribute] = {
        "mode": mlir_ir.StringAttr.get(mode.value),
        "transport": mlir_ir.StringAttr.get(transport.value),
    }
    if event_id is not None:
        attributes["event_id"] = _event_id_attr(operation_name, event_id)
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
    event_id: int | None = None,
    loc: mlir_ir.Location | None = None,
) -> None:
    """Asynchronously copy a RowMajor FP32 tile to or from a symmetric peer.

    The third argument contains copy parameters. ``qp_idx`` can be a runtime
    DSL integer for RDMA/UDMA and is ignored by MTE.
    The defaults are MTE transport and no atomic accumulation. GET resolves
    ``src`` on ``peer``; PUT resolves ``dst`` on ``peer``. GM->GM requires an
    explicit ``CopyParams(mode="get")`` or ``CopyParams(mode="put")``.

    GM->UB requires GET and UB->GM requires PUT, without scratch or an event
    ID inside. For GM<->UB, an unspecified mode (``None``) is inferred from
    the route; an explicit mode that conflicts with the route raises an error.
    GM->GM requires caller-owned UB scratch and a compile-time ``event_id``
    in 0..7. The compiler reserves this ID in every pipe direction used by
    the operation. UDMA scalar publication uses neither scratch nor an event
    ID. RDMA/UDMA require quiet before consuming results.

    Source and destination column strides must be provably 1. Row padding,
    dynamic shapes and dynamic row strides are supported. A dynamic column
    stride that cannot be proven to be 1 is rejected during device lowering.
    RDMA/UDMA split contiguous segments into NBI requests: at most
    UINT32_MAX // sizeof(float) elements for RDMA and 256 MiB / sizeof(float)
    for UDMA. Contiguous tiles are merged before splitting; padded tiles
    are split within each row. The caller waits for all requests with quiet.

    Return does not imply completion. The caller arranges completion waits,
    remote readiness and scratch reuse. MTE staging rows are padded to 32 B;
    control scratch capacity/alignment for other transports is caller-owned.
    """
    _build_remote_copy(
        dst,
        src,
        ub_tensor,
        event_id,
        peer,
        CopyParams() if params is None else params,
        loc,
    )


@dsl_user_op
def quiet(
    params: CopyParams | None = None,
    *,
    peer: IndexLike | None = None,
    ub_tensor: Tensor | None = None,
    event_id: int | None = None,
    loc: mlir_ir.Location | None = None,
) -> None:
    """Wait for earlier GET/PUT requests to peer on params.transport and params.qp_idx.

    RDMA requires UB scratch and a compile-time event_id in 0..7. The compiler
    reserves the ID in all synchronization directions used by the backend.
    UDMA quiet accepts neither scratch nor event_id.
    """
    name = "tla_dist.quiet"
    _core._require_frontend_state(name)
    _runtime._require_enclosing_region(name, "vector")
    if params is None:
        if peer is not None or ub_tensor is not None or event_id is not None:
            raise TlaLoweringError(f"{name}(): default quiet does not accept peer, ub_tensor or event_id")
        _require_simd(name)
        mlir_ir.Operation.create("tla_dist.quiet_default", loc=loc)
        return
    if peer is None:
        raise TlaLoweringError(f"{name}: explicit QP quiet requires peer")
    if not isinstance(params, CopyParams):
        raise TlaLoweringError(f"{name} requires tla_dist.params.CopyParams")
    transport = params.transport
    if transport not in (CopyTransport.RDMA, CopyTransport.UDMA):
        raise TlaLoweringError(f"{name} requires rdma or udma transport")
    transport = CopyTransport(transport)
    operands = [_as_integer_value(peer, 32, loc=loc), _as_integer_value(params.qp_idx, 32, loc=loc)]
    operands.extend(_build_scratch_operands(name, transport == CopyTransport.RDMA, ub_tensor, event_id))
    attributes = {"transport": mlir_ir.StringAttr.get(transport.value)}
    if event_id is not None:
        attributes["event_id"] = _event_id_attr(name, event_id)
    mlir_ir.Operation.create(
        name,
        operands=operands,
        attributes=attributes,
        loc=loc,
    )


@dsl_user_op
def barrier_all(*, loc: mlir_ir.Location | None = None) -> None:
    """Synchronize all ACLSHMEM ranks from an AIV/vector region."""
    _core._require_frontend_state("tla_dist.barrier_all")
    _runtime._require_enclosing_region("tla_dist.barrier_all", "vector")
    mlir_ir.Operation.create("tla_dist.barrier_all", loc=loc)



# The numeric codes belong to the IR fields, never to Python Enum values.
_SCOPE = {CoreScope.AIV: 0, CoreScope.MIX: 1}
_SIGNAL = {SignalOp.SET: 0, SignalOp.ADD: 1}
_COMPARE = {item: code for code, item in enumerate(CompareOp)}
_WAIT_DTYPES = ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32")


def _enum_attr(name: str, value: Any, enum: type, mapping: dict) -> mlir_ir.IntegerAttr:
    value = _core._resolve_bound_value(value)
    if type(value) is not enum:
        raise TlaLoweringError(f"{name} requires {enum.__name__} member, got {type(value).__name__}")
    return mlir_ir.IntegerAttr.get(mlir_ir.IntegerType.get_signless(32), mapping[value])


def _require_simd(name: str) -> None:
    _runtime._require_enclosing_region(name, "vector")
    if _runtime._has_enclosing_region("vec.func"):
        raise TlaLoweringError(f"{name} requires the SIMD scalar body of tla.vector()")


def _index32(name: str, value: Any, *, minimum: int, loc: Any) -> mlir_ir.Value:
    original = _core._resolve_bound_value(value)
    if isinstance(original, bool):
        raise TlaLoweringError(f"{name} requires an integer, got bool")
    if isinstance(original, Numeric) and (not type(original).is_integer or type(original).width > 32 or type(original).width == 1):
        raise TlaLoweringError(f"{name} requires Index or an integer no wider than 32 bits")
    resolved, _ = _resolve_integer(original)
    if isinstance(resolved, int):
        if not minimum <= resolved <= 2**31 - 1:
            raise TlaLoweringError(f"{name} constant must be in [{minimum}, 2147483647]")
    elif isinstance(resolved, mlir_ir.Value) and mlir_ir.IntegerType.isinstance(resolved.type):
        if mlir_ir.IntegerType(resolved.type).width > 32:
            raise TlaLoweringError(f"{name} runtime integer must be no wider than 32 bits")
    return _as_integer_value(original, 32, loc=loc)


def _collective(name: str, scope: CoreScope, team: Any | None, loc: Any) -> None:
    _core._require_frontend_state(name)
    attr = _enum_attr("scope", scope, CoreScope, _SCOPE)
    if attr.value == 0:
        _require_simd(name)
    elif _runtime._has_enclosing_region("vector") or _runtime._has_enclosing_region("cube"):
        raise TlaLoweringError(f"{name}: MIX scope requires the common layer outside cube/vector regions")
    operands = [] if team is None else [_index32("team", team, minimum=-1, loc=loc)]
    mlir_ir.Operation.create(name, operands=operands, attributes={"scope": attr}, loc=loc)


@dsl_user_op
def local_sync(*, scope: CoreScope = CoreScope.AIV, loc: mlir_ir.Location | None = None) -> None:
    """Arrive at a hard local AIV or paired MIX barrier; no network completion."""
    _collective("tla_dist.local_sync", scope, None, loc)


@dsl_user_op
def sync(*, team: IndexLike = TEAM_WORLD, scope: CoreScope = CoreScope.AIV,
         loc: mlir_ir.Location | None = None) -> None:
    """Team arrival, without completing outstanding NBI; team -1 is a no-op."""
    _collective("tla_dist.sync", scope, team, loc)


@dsl_user_op
def barrier(*, team: IndexLike = TEAM_WORLD, scope: CoreScope = CoreScope.AIV,
            loc: mlir_ir.Location | None = None) -> None:
    """Complete each AIV's local work and the owner's default domain, then arrive."""
    _collective("tla_dist.barrier", scope, team, loc)


@dsl_user_op
def fence(*, loc: mlir_ir.Location | None = None) -> None:
    """Order the default network domain; flat AIV0 is its sole owner."""
    _core._require_frontend_state("tla_dist.fence")
    _require_simd("tla_dist.fence")
    mlir_ir.Operation.create("tla_dist.fence", loc=loc)


def _slot(name: str, tensor: Tensor) -> tuple[Any, Any, str]:
    _core._require_category(name, "counter", tensor, "tensor", 0)
    result = _core._as_value(tensor)
    desc = _tensor_descriptor(result)
    if desc is None or desc.addrspace.lower() != "gm":
        raise TlaLoweringError(f"{name} requires a symmetric GM tensor descriptor")
    shape = _core._flatten_tla_tuple(desc.shape)
    if len(shape) not in (1, 2) or any(dim != 1 for dim in shape):
        raise TlaLoweringError(f"{name} requires a provably single-slot shape (1,) or (1,1)")
    if desc.layout_tag.replace("_", "").lower() not in ("rowmajor", "columnmajor"):
        raise TlaLoweringError(f"{name} requires a linear layout")
    return result, desc, str(desc.element_type)


def _threshold(name: str, value: Any, dtype: str, loc: Any) -> mlir_ir.Value:
    value = _core._resolve_bound_value(value)
    scalar_type = Numeric.from_dtype_token(dtype)
    if isinstance(value, Numeric):
        if type(value) is not scalar_type:
            raise TlaLoweringError(f"{name}: runtime threshold must match counter dtype {dtype}")
        constant = value.value
        if not isinstance(constant, (int, float, bool)):
            return value.ir_value()
        value = constant
    if dtype == "f32":
        if type(value) not in (int, float):
            raise TlaLoweringError(f"{name}: Float32 threshold requires int or float")
        original_finite = math.isfinite(value)
        try:
            value = struct.unpack("f", struct.pack("f", value))[0]
        except (OverflowError, struct.error) as error:
            raise TlaLoweringError(f"{name}: finite threshold outside Float32 range") from error
        if original_finite and not math.isfinite(value):
            raise TlaLoweringError(f"{name}: finite threshold outside Float32 range")
        if math.isnan(value):
            raise TlaLoweringError(f"{name}: NaN thresholds are unsupported")
    else:
        if type(value) is not int:
            raise TlaLoweringError(f"{name}: integer threshold requires int (not bool)")
        bits = int(dtype[1:])
        low, high = (0, 2**bits - 1) if dtype.startswith("u") else (-2**(bits-1), 2**(bits-1)-1)
        if not low <= value <= high:
            raise TlaLoweringError(f"{name}: threshold outside {dtype} range [{low}, {high}]")
    if dtype == "u64" and value >= 2**63:
        # MLIR's Python binding takes a signed int64 argument. Construct the
        # identical 64 bits in ui64 storage; no arithmetic signed conversion.
        ty = scalar_type.mlir_type()
        return mlir_ir.Operation.create("arith.constant", results=[ty],
            attributes={"value": mlir_ir.IntegerAttr.get(ty, value - 2**64)}, loc=loc).results[0]
    return scalar_type(value).ir_value()


@dsl_user_op
def signal(flag: Tensor, value: int | Int32, *, peer: IndexLike, op: SignalOp = SignalOp.SET,
           loc: mlir_ir.Location | None = None) -> None:
    """Publish Int32 SET or MTE ADD to the peer's corresponding symmetric slot."""
    name = "tla_dist.signal"
    _core._require_frontend_state(name)
    _require_simd(name)
    attr = _enum_attr("op", op, SignalOp, _SIGNAL)
    slot, _, dtype = _slot(name, flag)
    if dtype != "i32":
        raise TlaLoweringError(f"{name} requires Int32 flag")
    mlir_ir.Operation.create(name, operands=[slot, _threshold(name, value, dtype, loc),
        _index32("peer", peer, minimum=0, loc=loc)], attributes={"op": attr}, loc=loc)


@dsl_user_op
def wait_until(counter: Tensor, value: Numeric | int | float, *, cmp: CompareOp = CompareOp.EQ,
               loc: mlir_ir.Location | None = None) -> None:
    """Wait on the local typed slot; counter/value must be non-NaN for Float32.

    This operation refreshes the counter, and does not acquire arbitrary payload cachelines.
    """
    name = "tla_dist.wait_until"
    _core._require_frontend_state(name)
    _require_simd(name)
    attr = _enum_attr("cmp", cmp, CompareOp, _COMPARE)
    slot, _, dtype = _slot(name, counter)
    if dtype not in _WAIT_DTYPES:
        raise TlaLoweringError(f"{name}: unsupported counter dtype {dtype}")
    mlir_ir.Operation.create(name, operands=[slot, _threshold(name, value, dtype, loc)],
        attributes={"cmp": attr}, loc=loc)


__all__ = ["barrier_all", "quiet", "remote_copy", "shmem_ptr", "local_sync", "sync", "barrier", "fence", "signal", "wait_until"]
