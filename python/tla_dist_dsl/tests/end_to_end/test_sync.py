"""Standalone synchronization acceptance. Run --list before selecting device cases.

All jobs require the supervisor's external timeout and serialized device lease.
The typed writer is an independent AscendC same-width store/cache fixture; it
uses neither remote_copy nor the cancelled UInt64 copy implementation.
"""

# ruff: noqa: BLE001
import argparse
import fnmatch
import operator
import os
import struct
from dataclasses import dataclass, replace

from catccos import tla_dist  # isort: skip
from catlass import core_api as _core
from catlass import tla
from catlass._mlir import ir as mlir_ir
from catlass.base_dsl.op import dsl_user_op

from catccos.tla_dist.params import CompareOp, CoreScope, SignalOp

DTYPES = {
    name: getattr(tla, name)
    for name in (
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
    )
}
FORMATS = {
    "Int8": "b",
    "Int16": "h",
    "Int32": "i",
    "Int64": "q",
    "UInt8": "B",
    "UInt16": "H",
    "UInt32": "I",
    "UInt64": "Q",
    "Float32": "f",
}
WRITER_SOURCE = r"""
#include "kernel_operator.h"
#include <cstdint>
template<class T> __aicore__ void storeSlot(uint64_t addr, uint64_t bits) {
    union { uint64_t bits; T value; } value;
    value.bits = bits;
    AscendC::GlobalTensor<T> slot;
    slot.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(addr));
    slot.SetValue(0, value.value);
    __asm__ __volatile__("" ::: "memory");
    AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(slot);
    __asm__ __volatile__("" ::: "memory");
}
extern "C" __aicore__ void sync_test_write(uint64_t addr, uint64_t bits, int32_t tag, int32_t delay) {
    for (int i=0; i<delay; ++i) { __asm__ __volatile__("" ::: "memory"); }
    switch(tag) {
        case 0: storeSlot<int8_t>(addr,bits); break; case 1: storeSlot<int16_t>(addr,bits); break;
        case 2: storeSlot<int32_t>(addr,bits); break; case 3: storeSlot<int64_t>(addr,bits); break;
        case 4: storeSlot<uint8_t>(addr,bits); break; case 5: storeSlot<uint16_t>(addr,bits); break;
        case 6: storeSlot<uint32_t>(addr,bits); break; case 7: storeSlot<uint64_t>(addr,bits); break;
        case 8: storeSlot<float>(addr,bits); break;
    }
}
template<class T> __aicore__ void observeSlot(uint64_t addr, uint64_t out) {
    AscendC::GlobalTensor<T> slot;
    slot.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(addr));
    AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(slot);
    union { uint64_t bits; T value; } value;
    value.bits = 0;
    value.value = *reinterpret_cast<__gm__ volatile T*>(addr);
    __asm__ __volatile__("" ::: "memory");
    AscendC::GlobalTensor<uint64_t> result;
    result.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(out));
    result.SetValue(0,value.bits);
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(result);
}
extern "C" __aicore__ void sync_test_observe(uint64_t addr, uint64_t out, int32_t tag) {
    switch(tag) {
        case 0: observeSlot<int8_t>(addr,out); break; case 1: observeSlot<int16_t>(addr,out); break;
        case 2: observeSlot<int32_t>(addr,out); break; case 3: observeSlot<int64_t>(addr,out); break;
        case 4: observeSlot<uint8_t>(addr,out); break; case 5: observeSlot<uint16_t>(addr,out); break;
        case 6: observeSlot<uint32_t>(addr,out); break; case 7: observeSlot<uint64_t>(addr,out); break;
        case 8: observeSlot<float>(addr,out); break;
    }
}
// Independent test fixture: publish entry into the typed wait before the
// producer starts its bounded delay. This prevents a late receiver launch
// from turning a broken early-return wait into an accidental passing trial.
extern "C" __aicore__ void sync_test_delay(int32_t delay) {
    for (int i=0;i<delay;++i) { __asm__ __volatile__("" ::: "memory"); }
}
extern "C" __aicore__ void sync_test_await_started(uint64_t addr) {
    AscendC::GlobalTensor<int32_t> started;
    started.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(addr));
    do {
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(started);
    } while (*reinterpret_cast<__gm__ volatile int32_t*>(addr)!=1);
}

"""


@dsl_user_op
def _address(ptr, *, loc=None):
    """Test-fixture raw-address ABI, independent of descriptor alignment."""
    source = _core._coerce_pointer_arg(ptr)
    result = mlir_ir.Operation.create(
        "tla.ptrtoint",
        operands=[source.value],
        results=[mlir_ir.IntegerType.get_signless(64)],
        loc=loc,
    ).results[0]
    return tla.Int64(result)


@tla.extern(source=WRITER_SOURCE, name="sync_test_write")
def backend_write(
    ptr: tla.Int64,
    bits: tla.Int64,
    tag: tla.Int32,
    delay: tla.Int32,
) -> None: ...


@tla.extern(source=WRITER_SOURCE, name="sync_test_observe")
def backend_observe(
    ptr: tla.Int64,
    out: tla.Int64,
    tag: tla.Int32,
) -> None: ...


@tla.extern(source=WRITER_SOURCE, name="sync_test_await_started")
def backend_await_started(ptr: tla.Int64) -> None: ...


@tla.extern(source=WRITER_SOURCE, name="sync_test_delay")
def backend_delay(delay: tla.Int32) -> None: ...


@tla.jit
def cube_work(lhs: tla.Tensor, rhs: tla.Tensor, result: tla.Tensor):
    l1_ready = tla.flag("sync_cube_l1", tla.arch.MTE2, tla.arch.MTE1, id=2)
    l0_ready = tla.flag("sync_cube_l0", tla.arch.MTE1, tla.arch.CUBE, id=2)
    product_ready = tla.flag("sync_cube_product", tla.arch.CUBE, tla.arch.FIX, id=2)
    stored = tla.flag("sync_cube_stored", tla.arch.FIX, tla.arch.MTE2, id=2)
    l1a = tla.allocate(256, tla.Float16, tla.AddressSpace.l1, 512)
    l1b = tla.allocate(256, tla.Float16, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(256, tla.Float16, tla.AddressSpace.l0a, 512)
    l0b = tla.allocate(256, tla.Float16, tla.AddressSpace.l0b, 512)
    l0c = tla.allocate(256, tla.Float32, tla.AddressSpace.l0c, 512)
    with tla.cube():
        out = tla.tile_view(
            result, tla.make_shape(16, 16), tla.make_coord(tla.arch.block_idx(), 0)
        )
        a = tla.make_tensor_like(l1a, lhs, tla.arch.zN)
        b = tla.make_tensor_like(l1b, rhs, tla.arch.zN)
        tla.copy(a, lhs)
        tla.copy(b, rhs)
        tla.set_flag(l1_ready)
        tla.wait_flag(l1_ready)
        a0 = tla.make_tensor_like(l0a, a, tla.arch.zN)
        b0 = tla.make_tensor_like(l0b, b, tla.arch.nZ)
        c0 = tla.make_tensor_like(l0c, out, tla.arch.L0Clayout)
        tla.copy(a0, a)
        tla.copy(b0, b)
        tla.set_flag(l0_ready)
        tla.wait_flag(l0_ready)
        tla.mmad(c0, a0, b0, init_c=True)
        tla.set_flag(product_ready)
        tla.wait_flag(product_ready)
        tla.copy(out, c0)
        tla.set_flag(stored)
        tla.wait_flag(stored)


@dataclass(frozen=True)
class WaitCase:
    dtype: str
    cmp: CompareOp = CompareOp.EQ
    threshold: int | float = 7
    initial: int | float = 0
    target: int | float = 7
    mixed: bool = False
    offset: bool = False
    rank1: bool = False
    column: bool = False


@tla.kernel
def typed_wait(
    counter: tla.Tensor,
    observed: tla.Tensor,
    snapshot: tla.Tensor,
    started: tla.Tensor,
    counter8: tla.Tensor,
    counterf: tla.Tensor,
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    cube_result: tla.Tensor,
    rank: tla.Int32,
    producer: tla.Int32,
    peer: tla.Int32,
    index: tla.Int32,
    bits: tla.Int64,
    delay: tla.Int32,
    case: tla.Constexpr[WaitCase],
):
    if tla.const_expr(case.mixed):
        cube_work(lhs, rhs, cube_result)
    slot8 = tla.tile_view(counter8, tla.make_shape(1, 1), tla.make_coord(0, 0))
    slotf = tla.tile_view(counterf, tla.make_shape(1, 1), tla.make_coord(0, 0))
    if tla.const_expr(case.rank1):
        counter_slot = tla.tile_view(counter, tla.make_shape(1), tla.make_coord(index))
    elif tla.const_expr(case.column):
        counter_slot = tla.tile_view(
            counter, tla.make_shape(1, 1), tla.make_coord(index, 0)
        )
    else:
        counter_slot = tla.tile_view(
            counter, tla.make_shape(1, 1), tla.make_coord(0, index)
        )
    with tla.vector():
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            if rank == producer:
                if delay >= 0:
                    backend_await_started(_address(started.ptr))
                    pointer = tla_dist.shmem_ptr(counter.ptr + index, peer)
                    backend_write(
                        _address(pointer),
                        bits,
                        tuple(DTYPES).index(case.dtype),
                        delay,
                    )
                    if tla.const_expr(case.mixed):
                        p8 = tla_dist.shmem_ptr(counter8.ptr, peer)
                        pf = tla_dist.shmem_ptr(counterf.ptr, peer)
                        backend_write(_address(p8), tla.Int64(249), 0, 0)
                        backend_write(
                            _address(pf),
                            tla.Int64(1067450368),
                            8,
                            0,
                        )
            else:
                if delay >= 0:
                    entered = tla_dist.shmem_ptr(started.ptr, peer)
                    backend_write(_address(entered), tla.Int64(1), 2, 0)
                tla_dist.wait_until(counter_slot, case.threshold, cmp=case.cmp)
                backend_observe(
                    _address(counter.ptr + index),
                    _address(snapshot.ptr),
                    tuple(DTYPES).index(case.dtype),
                )
                if tla.const_expr(case.mixed):
                    tla_dist.wait_until(slot8, -7)
                    backend_observe(
                        _address(counter8.ptr),
                        _address(snapshot.ptr + 16),
                        0,
                    )
                    tla_dist.wait_until(slotf, 1.25)
                    backend_observe(
                        _address(counterf.ptr),
                        _address(snapshot.ptr + 32),
                        8,
                    )
                observed[0, 0] = 1


@tla.kernel
def collective(
    observed: tla.Tensor,
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    cube_result: tla.Tensor,
    case: tla.Constexpr[str],
    team: tla.Int32,
):
    if tla.const_expr(case.startswith("mix")):
        cube_work(lhs, rhs, cube_result)
    with tla.vector():
        if tla.const_expr(case.startswith("mix")):
            flat = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
        else:
            flat = tla.arch.block_idx()
        backend_write(
            _address(observed.ptr + (flat + 2) * 16),
            tla.Int64(1),
            2,
            0,
        )
        if tla.const_expr(not case.startswith("mix")):
            if tla.const_expr(case.endswith("local")):
                tla_dist.local_sync()
            elif tla.const_expr(case.endswith("barrier")):
                tla_dist.barrier(team=team)
            else:
                tla_dist.sync(team=team)
    if tla.const_expr(case.startswith("mix")):
        if tla.const_expr(case.endswith("local")):
            tla_dist.local_sync(scope=CoreScope.MIX)
        elif tla.const_expr(case.endswith("barrier")):
            tla_dist.barrier(team=team, scope=CoreScope.MIX)
        else:
            tla_dist.sync(team=team, scope=CoreScope.MIX)
    with tla.vector():
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            observed[0, 0] = 3


@tla.kernel
def signal_case(
    flag: tla.Tensor, peer: tla.Int32, value: tla.Int32, add: tla.Constexpr[bool]
):
    slot = tla.tile_view(flag, tla.make_shape(1, 1), tla.make_coord(0, 0))
    with tla.vector():
        if tla.const_expr(add):
            tla_dist.signal(slot, value, peer=peer, op=SignalOp.ADD)
        else:
            if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
                tla_dist.signal(slot, value, peer=peer)
        tla_dist.barrier()


@tla.kernel
def publication(
    payload: tla.Tensor,
    source: tla.Tensor,
    flag: tla.Tensor,
    ack: tla.Tensor,
    output: tla.Tensor,
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    cube_result: tla.Tensor,
    peer: tla.Int32,
    rank: tla.Int32,
    rounds: tla.Constexpr[int],
    mixed: tla.Constexpr[bool],
):
    scratch = tla.make_tensor(
        tla.allocate(5 * 16, tla.Float32, tla.AddressSpace.ub, 32),
        tla.make_layout(
            tla.make_shape(5, 16),
            tla.make_stride(16, 1),
            origin_shape=tla.make_shape(5, 13),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    complete = tla.flag("publication_complete", tla.arch.MTE3, tla.arch.MTE2, id=2)
    loaded = tla.flag("publication_loaded", tla.arch.MTE2, tla.arch.MTE3, id=2)
    layout = tla.make_layout(
        tla.make_shape(5, 13), tla.make_stride(32, 1), layoutTag=tla.arch.RowMajor
    )
    if tla.const_expr(mixed):
        cube_work(lhs, rhs, cube_result)
    for epoch in tla.range(1, rounds + 1):
        slot = (epoch - 1) % 2
        ready = tla.tile_view(flag, tla.make_shape(1, 1), tla.make_coord(0, slot * 16))
        acknowledged = tla.tile_view(
            ack, tla.make_shape(1, 1), tla.make_coord(0, slot * 16)
        )
        destination = tla.make_tensor(payload.ptr + (slot * 7 + 1) * 32, layout)
        src = tla.make_tensor(source.ptr + ((epoch - 1) * 7 + 1) * 32, layout)
        out = tla.make_tensor(output.ptr + ((epoch - 1) * 7 + 1) * 32, layout)
        with tla.vector():
            if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
                backend_delay((rank + 1) * 100000 + (epoch % 3) * 10000)
                tla_dist.remote_copy(
                    destination,
                    src,
                    tla_dist.params.CopyParams(mode="put"),
                    peer=peer,
                    ub_tensor=scratch,
                    event_id=2,
                )
                tla.set_flag(complete)
                tla.wait_flag(complete)
                tla_dist.signal(ready, epoch, peer=peer)
                tla_dist.wait_until(ready, epoch)
                tla.copy(scratch, destination)
                tla.set_flag(loaded)
                tla.wait_flag(loaded)
                tla.copy(out, scratch)
                tla.set_flag(complete)
                tla.wait_flag(complete)
                tla_dist.signal(acknowledged, epoch, peer=peer)
                tla_dist.wait_until(acknowledged, epoch)
            if tla.const_expr(not mixed):
                tla_dist.local_sync()
        if tla.const_expr(mixed):
            tla_dist.local_sync(scope=CoreScope.MIX)


@tla.kernel
def default_ordering(observed: tla.Tensor):
    with tla.vector():
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            tla_dist.fence()
            tla_dist.quiet()
            observed[0, 0] = 9


@tla.kernel
def ub_guard(flag: tla.Tensor, observed: tla.Tensor, peer: tla.Int32):
    ptr = tla.allocate(188 * 1024 // 4, tla.Int32, tla.AddressSpace.ub, 32)
    storage = tla.make_tensor(
        ptr,
        tla.make_layout(
            tla.make_shape(1, 188 * 1024 // 4),
            tla.make_stride(188 * 1024 // 4, 1),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    slot = tla.tile_view(flag, tla.make_shape(1, 1), tla.make_coord(0, 0))
    with tla.vector():
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            storage[0, 0] = tla.Int32(17)
            storage[0, 188 * 1024 // 4 - 1] = tla.Int32(29)
            tla_dist.signal(slot, 1, peer=peer, op=SignalOp.ADD)
            observed[0, 0] = tla.Int32(storage[0, 0])
            observed[0, 1] = tla.Int32(storage[0, 188 * 1024 // 4 - 1])
        tla_dist.barrier()


@tla.kernel
def dynamic_ub_guard(flag: tla.Tensor, observed: tla.Tensor, peer: tla.Int32):
    ptr = tla.arch.get_dyn_ub(tla.Int32, byte_alignment=32)
    storage = tla.make_tensor(
        ptr,
        tla.make_layout(
            tla.make_shape(1, 188 * 1024 // 4),
            tla.make_stride(188 * 1024 // 4, 1),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    slot = tla.tile_view(flag, tla.make_shape(1, 1), tla.make_coord(0, 0))
    with tla.vector():
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            storage[0, 0] = tla.Int32(17)
            storage[0, 188 * 1024 // 4 - 1] = tla.Int32(29)
            tla_dist.signal(slot, 1, peer=peer, op=SignalOp.ADD)
            observed[0, 0] = tla.Int32(storage[0, 0])
            observed[0, 1] = tla.Int32(storage[0, 188 * 1024 // 4 - 1])
        tla_dist.barrier()


def manifest():
    for blocks in (1, 2, 4):
        yield f"p0/vector-local/blocks-{blocks}", ("collective", "aiv-local", 0, blocks)
    yield "p0/vector-local/resident-limit", ("collective", "aiv-local", 0, 0)
    for scope in ("aiv", "mix"):
        for kind in ("local", "sync", "barrier"):
            yield f"collective/{scope}-{kind}", ("collective", f"{scope}-{kind}", 0, 2)
        yield f"collective/{scope}-invalid", ("collective", f"{scope}-barrier", -1, 2)
    for dtype, dtype_class in DTYPES.items():
        for cmp in CompareOp:
            width = dtype_class.width
            threshold = 2 ** (width - 1) if dtype.startswith("UInt") else 0
            target = {
                CompareOp.EQ: threshold,
                CompareOp.NE: threshold + 1,
                CompareOp.GT: threshold + 1,
                CompareOp.GE: threshold,
                CompareOp.LT: threshold - 1,
                CompareOp.LE: threshold,
            }[cmp]
            initial = {
                CompareOp.EQ: threshold + 1,
                CompareOp.NE: threshold,
                CompareOp.GT: threshold - 1,
                CompareOp.GE: threshold - 1,
                CompareOp.LT: threshold + 1,
                CompareOp.LE: threshold + 1,
            }[cmp]
            yield (
                f"typed/{dtype}/{cmp.name}",
                ("wait", WaitCase(dtype, cmp, threshold, initial, target)),
            )
        width = dtype_class.width
        boundary = (
            2**width - 1
            if dtype.startswith("UInt")
            else (-(2 ** (width - 1)) if dtype.startswith("Int") else float("inf"))
        )
        initial = 0 if boundary != 0 else 1
        yield (
            f"boundary/{dtype}",
            (
                "wait",
                WaitCase(dtype, threshold=boundary, initial=initial, target=boundary),
            ),
        )
        yield f"address/{dtype}", ("wait", WaitCase(dtype, offset=True, rank1=True))
    yield (
        "typed/mixed-width",
        (
            "wait",
            WaitCase("UInt64", threshold=2**63, initial=0, target=2**63, mixed=True),
        ),
    )
    yield "signal/set", ("signal", False, 1)
    yield "signal/add", ("signal", True, 1)
    yield "signal/add-multiwriter", ("signal", True, 4)
    yield "ordering/publication-epoch-ack", ("publication", 100)
    yield "ordering/default-fence-quiet", ("default",)
    yield "resource/user-ub-guards", ("ub-guard",)


def _view(tensor, *, column=False):
    return tla_dist.from_dlpack(
        tensor, layout_tag=tla.arch.ColumnMajor if column else tla.arch.RowMajor
    )


def _cube_inputs(runtime, blocks):
    torch = runtime.torch
    device = f"npu:{os.environ['LOCAL_RANK']}"
    lhs = torch.ones((16, 16), dtype=torch.float16, device=device)
    rhs = torch.ones_like(lhs)
    output = torch.full((blocks * 16, 16), -1, dtype=torch.float32, device=device)
    return lhs, rhs, output


def _check_cube(runtime, result):
    runtime.assert_equal(
        result, runtime.torch.full(tuple(result.shape), 16, dtype=runtime.torch.float32)
    )


def _compile(runtime, kernel, args, *, mixed=False):
    artifact, error = None, None
    try:
        artifact = tla_dist.compile(
            kernel,
            *args,
            arch_scope="aic.c310" if mixed else "aiv.c310",
            cache_dir=runtime.cache_dir,
        )
    except Exception as exc:
        error = f"compile: {type(exc).__name__}: {exc}"
    runtime.check_errors(error)
    return artifact


def _launch(runtime, artifact, args, blocks, *, local_only=False, ub=None):
    runtime.torch.npu.synchronize()
    runtime.shmem.barrier_all()
    error = None
    try:
        if not local_only or runtime.rank == 0:
            if ub is None:
                artifact(*args, block_num=blocks)
            else:
                artifact(*args, block_num=blocks, ub=ub)
            runtime.torch.npu.synchronize()
    except Exception as exc:
        error = f"launch: {type(exc).__name__}: {exc}"
    runtime.check_errors(error)


def _wait(runtime, case):
    torch = runtime.torch
    name = case.dtype
    fmt = FORMATS[name]
    size = struct.calcsize(fmt)
    raw = runtime.shmem.symmetric_tensor((1, 256), torch.uint8)
    raw8 = runtime.shmem.symmetric_tensor((1, 64), torch.uint8)
    rawf = runtime.shmem.symmetric_tensor((1, 64), torch.uint8)
    started = runtime.shmem.symmetric_tensor((1, 16), torch.int32)
    output = torch.zeros((1, 32), dtype=torch.int32, device=raw.device)
    snapshot = torch.empty((1, 40), dtype=torch.uint8, device=raw.device)
    typed = raw.view(getattr(torch, {"Float32": "float32"}.get(name, name.lower())))
    counter8 = raw8.view(torch.int8)
    counterf = rawf.view(torch.float32)
    lhs, rhs, cube_result = _cube_inputs(runtime, 1)
    variants = [case]
    if case.cmp in (CompareOp.GE, CompareOp.LE):
        variants.append(
            replace(
                case, target=case.threshold + (1 if case.cmp is CompareOp.GE else -1)
            )
        )
    if case.dtype == "Float32" and case.target == float("inf"):
        variants.append(replace(case, target=-float("inf"), threshold=-float("inf")))
    layouts = (
        [(True, False), (False, True)] if case.offset else [(case.rank1, case.column)]
    )
    launches = 0
    try:
        for variant in variants:
            threshold = (
                struct.unpack("<f", struct.pack("<f", variant.threshold))[0]
                if name == "Float32"
                else variant.threshold
            )
            predicate = {
                CompareOp.EQ: operator.eq,
                CompareOp.NE: operator.ne,
                CompareOp.GT: operator.gt,
                CompareOp.GE: operator.ge,
                CompareOp.LT: operator.lt,
                CompareOp.LE: operator.le,
            }[variant.cmp]
            for rank1, column in layouts:
                selected = replace(variant, rank1=rank1, column=column)
                counter = typed[:, 4:] if case.offset else typed
                if rank1:
                    counter = counter.reshape(-1)
                elif column:
                    counter = counter.T
                artifact = None
                for index in (1, 3) if case.offset else (0,):
                    byte_offset = (4 + index) * size if case.offset else index * size
                    for producer in (0, 1):
                        for delayed in (False, True):
                            initial = selected.initial if delayed else selected.target
                            assert predicate(initial, threshold) is (not delayed)
                            assert predicate(selected.target, threshold)
                            expected = bytearray([0xA5] * 256)
                            expected[byte_offset : byte_offset + size] = struct.pack(
                                "<" + fmt, initial
                            )
                            expected8 = bytearray([0xA5] * 64)
                            expectedf = bytearray([0xA5] * 64)
                            expected8[0:1] = struct.pack("<b", 0 if delayed else -7)
                            expectedf[0:4] = struct.pack("<f", 0 if delayed else 1.25)
                            for target, data in (
                                (raw, expected),
                                (raw8, expected8),
                                (rawf, expectedf),
                            ):
                                target.copy_(
                                    torch.tensor(list(data), dtype=torch.uint8).reshape(
                                        tuple(target.shape)
                                    )
                                )
                            output.zero_()
                            snapshot.fill_(0x5A)
                            started.zero_()
                            cube_result.fill_(-1)
                            bits = int.from_bytes(
                                struct.pack("<" + fmt, selected.target), "little"
                            )
                            args = (
                                _view(counter, column=column),
                                _view(output),
                                _view(snapshot),
                                _view(started),
                                _view(counter8),
                                _view(counterf),
                                _view(lhs),
                                _view(rhs),
                                _view(cube_result),
                                tla.Int32(runtime.rank),
                                tla.Int32(producer),
                                tla.Int32(1 - runtime.rank),
                                tla.Int32(index),
                                tla.Int64(bits if bits < 2**63 else bits - 2**64),
                                tla.Int32(1000000 if delayed else -1),
                            )
                            if artifact is None:
                                artifact = _compile(
                                    runtime,
                                    typed_wait,
                                    (*args, selected),
                                    mixed=selected.mixed,
                                )
                            _launch(runtime, artifact, args, 1)
                            launches += 1
                            receiver = runtime.rank != producer
                            if receiver:
                                expected[byte_offset : byte_offset + size] = (
                                    struct.pack("<" + fmt, selected.target)
                                )
                                if selected.mixed:
                                    expected8[0:1] = struct.pack("<b", -7)
                                    expectedf[0:4] = struct.pack("<f", 1.25)
                            error = None
                            for target, data in (
                                (raw, expected),
                                (raw8, expected8),
                                (rawf, expectedf),
                            ):
                                if bytes(target.cpu().reshape(-1).tolist()) != bytes(
                                    data
                                ):
                                    error = f"typed writer/address/guard mismatch {name}/{selected.cmp.name}"
                            seen = output.cpu()[0].tolist()
                            if seen[0] != (1 if receiver else 0):
                                error = f"wait return marker mismatch {seen}"
                            wanted = bytearray([0x5A] * 40)
                            if receiver:
                                wanted[0:8] = struct.pack(
                                    "<" + fmt, selected.target
                                ).ljust(8, b"\0")
                                if selected.mixed:
                                    wanted[16:24] = struct.pack("<b", -7).ljust(
                                        8, b"\0"
                                    )
                                    wanted[32:40] = struct.pack("<f", 1.25).ljust(
                                        8, b"\0"
                                    )
                            observed = bytes(snapshot.cpu().reshape(-1).tolist())
                            if receiver:
                                captured = struct.unpack("<" + fmt, observed[:size])[0]
                                if not predicate(captured, threshold):
                                    error = f"wait returned before condition: {captured} {selected.cmp.name} {threshold}"
                            if observed != bytes(wanted):
                                error = f"wait returned wrong immediate typed values: {observed.hex()} expected {wanted.hex()}"
                            runtime.check_errors(error)
                            if selected.mixed:
                                _check_cube(runtime, cube_result)
                            if runtime.rank == 0:
                                print(
                                    f"DETAIL producer={producer} delayed={delayed} index={index} column={column} target={selected.target}",
                                    flush=True,
                                )
    finally:
        for tensor in (started, rawf, raw8, raw):
            runtime.shmem.free(tensor)
    return launches


def _run(runtime, item, *, local_only=False):
    torch = runtime.torch
    if item[0] == "wait":
        return _wait(runtime, item[1])
    if item[0] == "collective":
        _, case, team, blocks = item
        if blocks == 0:
            from catccos.tla_dist._sync_runtime import resident_block_limit

            blocks = resident_block_limit(int(os.environ["LOCAL_RANK"]), paired=False)
        observed = torch.zeros(
            (1, 1056), dtype=torch.int32, device=f"npu:{os.environ['LOCAL_RANK']}"
        )
        lhs, rhs, cube_result = _cube_inputs(runtime, blocks)
        args = (
            _view(observed),
            _view(lhs),
            _view(rhs),
            _view(cube_result),
            case,
            tla.Int32(team),
        )
        artifact = _compile(runtime, collective, args, mixed=case.startswith("mix"))
        _launch(runtime, artifact, (*args[:4], args[5]), blocks, local_only=local_only)
        if case.startswith("mix"):
            _check_cube(runtime, cube_result)
        actual = observed.cpu()[0].tolist()
        expected = blocks * (2 if case.startswith("mix") else 1)
        runtime.check_errors(
            None
            if (local_only and runtime.rank != 0)
            or (actual[0] == 3 and sum(actual[32::16]) == expected)
            else f"participant markers {actual}"
        )
    elif item[0] == "signal":
        _, add, blocks = item
        flag = runtime.shmem.symmetric_tensor((1, 16), torch.int32)
        try:
            flag.zero_()
            args = (_view(flag), tla.Int32(1 - runtime.rank), tla.Int32(3), add)
            artifact = _compile(runtime, signal_case, args)
            _launch(runtime, artifact, args[:-1], blocks)
            runtime.check_errors(
                None
                if flag.cpu()[0, 0].item() == 3 * (blocks if add else 1)
                else "signal result mismatch"
            )
        finally:
            runtime.shmem.free(flag)
    elif item[0] == "default":
        output = torch.zeros(
            (1, 4), dtype=torch.int32, device=f"npu:{os.environ['LOCAL_RANK']}"
        )
        args = (_view(output),)
        artifact = _compile(runtime, default_ordering, args)
        _launch(runtime, artifact, args, 1)
        runtime.check_errors(
            None
            if output.cpu()[0, 0].item() == 9
            else "default completion marker missing"
        )
    elif item[0] == "ub-guard":
        flag = runtime.shmem.symmetric_tensor((1, 16), torch.int32)
        try:
            output = torch.zeros((1, 4), dtype=torch.int32, device=flag.device)
            args = (_view(flag), _view(output), tla.Int32(1 - runtime.rank))
            for dynamic in (False, True):
                flag.zero_()
                output.zero_()
                artifact = _compile(
                    runtime, dynamic_ub_guard if dynamic else ub_guard, args
                )
                budget = artifact.max_ub_bytes if dynamic else None
                if dynamic:
                    runtime.check_errors(
                        None
                        if budget == 188 * 1024
                        else f"dynamic UB budget {budget} differs from reserved contract"
                    )
                _launch(runtime, artifact, args, 1, ub=budget)
                seen = output.cpu()[0].tolist()
                runtime.check_errors(
                    None
                    if seen[:2] == [17, 29] and flag.cpu()[0, 0].item() == 1
                    else f"UB guard/signal mismatch {seen}"
                )
                if dynamic:
                    from catlass.execution import TlaExecutionError

                    refused = False
                    try:
                        artifact(*args, block_num=1, ub=budget + 32)
                    except TlaExecutionError as error:
                        refused = "limit" in str(error) or "exceed" in str(error)
                    runtime.check_errors(
                        None
                        if refused
                        else "same artifact accepted out-of-range dynamic UB"
                    )
        finally:
            runtime.shmem.free(flag)
        return 2

    else:
        rounds = item[1]
        payload = runtime.shmem.symmetric_tensor((14, 32), torch.float32)
        flag = runtime.shmem.symmetric_tensor((1, 64), torch.int32)
        ack = runtime.shmem.symmetric_tensor((1, 64), torch.int32)
        sentinel = -1024.0
        try:
            for mixed in (False, True):
                payload.fill_(sentinel)
                flag.fill_(-17)
                ack.fill_(-17)
                flag[:, 16] = 0
                flag[:, 32] = 0
                ack[:, 16] = 0
                ack[:, 32] = 0
                expected_source = torch.full(
                    (rounds * 7 + 2, 32), sentinel, dtype=torch.float32
                )
                expected_output = torch.full_like(expected_source, sentinel)
                expected_payload = torch.full((14, 32), sentinel, dtype=torch.float32)
                for epoch in range(1, rounds + 1):
                    values = (
                        torch.arange(65, dtype=torch.float32).reshape(5, 13)
                        + runtime.rank * 1048576
                        + epoch * 256
                    )
                    remote_values = (
                        torch.arange(65, dtype=torch.float32).reshape(5, 13)
                        + (1 - runtime.rank) * 1048576
                        + epoch * 256
                    )
                    row = (epoch - 1) * 7 + 1
                    expected_source[row : row + 5, 8:21] = values
                    expected_output[row : row + 5, 8:21] = remote_values
                    slot = (epoch - 1) % 2
                    expected_payload[slot * 7 + 1 : slot * 7 + 6, 8:21] = remote_values
                source = expected_source.to(payload.device)
                output = torch.full_like(source, sentinel)
                lhs, rhs, cube_result = _cube_inputs(runtime, 2)
                args = tuple(
                    _view(t)
                    for t in (
                        payload[:, 8:],
                        source[:, 8:],
                        flag[:, 16:],
                        ack[:, 16:],
                        output[:, 8:],
                        lhs,
                        rhs,
                        cube_result,
                    )
                ) + (
                    tla.Int32(1 - runtime.rank),
                    tla.Int32(runtime.rank),
                    rounds,
                    mixed,
                )
                artifact = _compile(runtime, publication, args, mixed=mixed)
                _launch(runtime, artifact, args[:-2], 2)
                runtime.assert_equal(output, expected_output)
                runtime.assert_equal(source, expected_source)
                runtime.assert_equal(payload, expected_payload)
                expected_flag = torch.full((1, 64), -17, dtype=torch.int32)
                for slot in (0, 1):
                    expected_flag[0, 16 + slot * 16] = rounds - (
                        (rounds - 1 - slot) % 2
                    )
                runtime.assert_equal(flag, expected_flag)
                runtime.assert_equal(ack, expected_flag)
                if mixed:
                    _check_cube(runtime, cube_result)
        finally:
            for tensor in (ack, flag, payload):
                runtime.shmem.free(tensor)
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", default="*", help="glob over manifest names")
    args = parser.parse_args()
    cases = [
        (name, item) for name, item in manifest() if fnmatch.fnmatch(name, args.case)
    ]
    if args.list:
        for name, _ in cases:
            print(name)
        return 0
    if not cases:
        raise SystemExit("no matching acceptance cases")
    if os.environ.get("WORLD_SIZE") != "2":
        raise SystemExit("use torchrun --standalone --nproc-per-node=2")
    from test_remote_copy_mte import initialized_runtime

    with initialized_runtime() as runtime:
        for name, item in cases:
            if runtime.rank == 0:
                print(f"RUN  {name}", flush=True)
            local_only = name.startswith("p0/")
            launches = _run(runtime, item, local_only=local_only)
            if not local_only or runtime.rank == 0:
                print(
                    f"DEVICE_PASS {name} rank={runtime.rank} launches={launches if launches is not None else 1}",
                    flush=True,
                )
            runtime.dist.barrier()
            if runtime.rank == 0:
                print(f"PASS {name}", flush=True)
        if runtime.rank == 0:
            print(f"{len(cases)} passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
