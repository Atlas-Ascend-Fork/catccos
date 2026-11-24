# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Standalone UDMA remote_copy/quiet E2E tests; launch with torchrun."""

# Stage failures are gathered and raised on every rank before continuing.
# ruff: noqa: BLE001

import argparse
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

# Bootstrap pinned Catlass before importing it.
from catccos import tla_dist  # isort: skip
from catlass import tla

# (rows, columns, source row stride, destination row stride, dynamic ABI)
GEOMETRIES = {
    "tail-padded-static": (5, 13, 24, 32, False),
    "tail-padded-dynamic": (5, 13, 24, 32, True),
}
SENTINEL = -1024.0


@dataclass(frozen=True)
class RemoteCase:
    rows: int
    columns: int
    mode: str
    wqe_pipe: str = "mte3"
    mixed_kernel: bool = False


@dataclass(frozen=True)
class MultiSubmitRemoteCase(RemoteCase):
    count: int = 2
    rounds: int = 3


@tla.jit
def make_control(offset: tla.Constexpr[bool] = False):
    if tla.const_expr(offset):
        ptr = tla.allocate(128, tla.Float32, tla.AddressSpace.ub, 32)
        root = tla.make_tensor(
            ptr, tla.make_layout(tla.make_shape(2, 64), tla.make_stride(64, 1), layoutTag=tla.arch.RowMajor)
        )
        return tla.tile_view(root, tla.make_shape(1, 64), tla.make_coord(1, 0))
    ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 32)
    return tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(1, 64), tla.make_stride(64, 1), layoutTag=tla.arch.RowMajor)
    )


@tla.jit
def submit_remote(
    dst,
    src,
    peer,
    scratch,
    event_id: tla.Constexpr[int],
    case: tla.Constexpr[RemoteCase],
    mode: tla.Constexpr[str],
):
    params = tla_dist.params.CopyParams(mode=mode, transport="udma", wqe_pipe=case.wqe_pipe)
    if tla.const_expr(case.wqe_pipe == "s"):
        tla_dist.remote_copy(dst, src, params, peer=peer)
    else:
        tla_dist.remote_copy(dst, src, params, peer=peer, ub_tensor=scratch, event_id=event_id)


@tla.jit
def submit_batch(
    dst,
    src,
    peer,
    control,
    event_id: tla.Constexpr[int],
    iteration,
    case: tla.Constexpr[MultiSubmitRemoteCase],
):
    shape = tla.make_shape(case.rows, case.columns)
    for index in tla.range(case.count):
        coord = tla.make_coord(iteration * case.count + index, 0)
        source = tla.tile_view(src, shape, coord)
        target = tla.tile_view(dst, shape, coord)
        submit_remote(target, source, peer, control, event_id, case, case.mode)


@tla.kernel
def remote_copy(dst: tla.Tensor, src: tla.Tensor, peer: tla.Int32, case: tla.Constexpr[RemoteCase]):
    event_id = 0
    control = None
    if tla.const_expr(case.wqe_pipe == "mte3"):
        control = make_control()
    shape = tla.make_shape(case.rows, case.columns)
    source = tla.tile_view(src, shape, tla.make_coord(0, 0))
    target = tla.tile_view(dst, shape, tla.make_coord(0, 0))
    if tla.const_expr(case.mixed_kernel):
        with tla.cube():
            tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla_dist.barrier_all()
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            submit_remote(target, source, peer, control, event_id, case, case.mode)
            tla_dist.quiet(tla_dist.params.CopyParams(transport="udma"), peer=peer)
        tla_dist.barrier_all()


@tla.kernel
def multi_qp(dst: tla.Tensor, src: tla.Tensor, peer: tla.Int32, case: tla.Constexpr[RemoteCase]):
    control = None
    if tla.const_expr(case.wqe_pipe == "mte3"):
        control = make_control()
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        qp = tla.arch.sub_block_idx()
        params = tla_dist.params.CopyParams(
            mode=case.mode,
            transport="udma",
            qp_idx=qp,
            wqe_pipe=case.wqe_pipe,
        )
        shape = tla.make_shape(case.rows, case.columns)
        source = tla.tile_view(src, shape, tla.make_coord(qp, 0))
        target = tla.tile_view(dst, shape, tla.make_coord(qp, 0))
        tla_dist.barrier_all()
        tla_dist.remote_copy(
            target,
            source,
            params,
            peer=peer,
            ub_tensor=control,
            event_id=0 if case.wqe_pipe == "mte3" else None,
        )
        tla_dist.quiet(params, peer=peer)
        tla_dist.barrier_all()


@tla.kernel
def remote_copy_ub_offset(dst: tla.Tensor, src: tla.Tensor, peer: tla.Int32, case: tla.Constexpr[RemoteCase]):
    control = make_control(True)
    event_id = 0
    shape = tla.make_shape(case.rows, case.columns)
    source = tla.tile_view(src, shape, tla.make_coord(0, 0))
    target = tla.tile_view(dst, shape, tla.make_coord(0, 0))
    with tla.vector():
        tla_dist.barrier_all()
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            submit_remote(target, source, peer, control, event_id, case, case.mode)
            tla_dist.quiet(tla_dist.params.CopyParams(transport="udma"), peer=peer)
        tla_dist.barrier_all()


@tla.jit
def observe_batch(dst, observed, local, load, store, iteration, case: tla.Constexpr[MultiSubmitRemoteCase]):
    shape = tla.make_shape(case.rows, case.columns)
    for index in tla.range(case.count):
        coord = tla.make_coord(iteration * case.count + index, 0)
        tile = tla.tile_view(dst, shape, coord)
        output = tla.tile_view(observed, shape, coord)
        tla.copy(local, tile)
        tla.set_flag(load)
        tla.wait_flag(load)
        tla.copy(output, local)
        tla.set_flag(store)
        tla.wait_flag(store)


@tla.kernel
def quiet_observe(
    dst: tla.Tensor, src: tla.Tensor, observed: tla.Tensor, peer: tla.Int32, case: tla.Constexpr[MultiSubmitRemoteCase]
):
    event_id = 0
    control = None
    if tla.const_expr(case.wqe_pipe == "mte3"):
        control = make_control()
    padded = ((case.columns + 7) // 8) * 8
    local_ptr = tla.allocate(case.rows * padded, tla.Float32, tla.AddressSpace.ub, 32)
    local = tla.make_tensor(
        local_ptr,
        tla.make_layout(
            tla.make_shape(case.rows, padded),
            tla.make_stride(padded, 1),
            origin_shape=tla.make_shape(case.rows, case.columns),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    # Observation uses separate scratch and event IDs from UDMA submission.
    load = tla.flag("observe_load", tla.arch.MTE2, tla.arch.MTE3, id=1)
    store = tla.flag("observe_store", tla.arch.MTE3, tla.arch.MTE2, id=1)
    with tla.vector():
        tla_dist.barrier_all()
        for iteration in tla.range(case.rounds):
            if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
                submit_batch(dst, src, peer, control, event_id, iteration, case)
                tla_dist.quiet(tla_dist.params.CopyParams(transport="udma"), peer=peer)
                if tla.const_expr(case.mode == "get"):
                    observe_batch(dst, observed, local, load, store, iteration, case)
            # PUT receivers read only after every sender has completed its quiet.
            tla_dist.barrier_all()
            if tla.const_expr(case.mode == "put"):  # noqa: SIM102 - compile-time branch before device predicate
                if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
                    observe_batch(dst, observed, local, load, store, iteration, case)
            tla_dist.barrier_all()


@tla.kernel
def get_then_put(
    dst: tla.Tensor, src: tla.Tensor, temporary: tla.Tensor, peer: tla.Int32, case: tla.Constexpr[RemoteCase]
):
    event_id = 0
    control = None
    if tla.const_expr(case.wqe_pipe == "mte3"):
        control = make_control()
    shape = tla.make_shape(case.rows, case.columns)
    coord = tla.make_coord(0, 0)
    source = tla.tile_view(src, shape, coord)
    target = tla.tile_view(dst, shape, coord)
    tmp = tla.tile_view(temporary, shape, coord)
    with tla.vector():
        tla_dist.barrier_all()
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            submit_remote(tmp, source, peer, control, event_id, case, "get")
            tla_dist.quiet(tla_dist.params.CopyParams(transport="udma"), peer=peer)
            submit_remote(target, tmp, peer, control, event_id, case, "put")
            tla_dist.quiet(tla_dist.params.CopyParams(transport="udma"), peer=peer)
        tla_dist.barrier_all()


@tla.kernel
def mixed_get_put(dst: tla.Tensor, src: tla.Tensor, peer: tla.Int32, case: tla.Constexpr[RemoteCase]):
    event_id = 0
    control = None
    if tla.const_expr(case.wqe_pipe == "mte3"):
        control = make_control()
    shape = tla.make_shape(case.rows, case.columns)
    get_coord = tla.make_coord(0, 0)
    put_coord = tla.make_coord(1, 0)
    get_src = tla.tile_view(src, shape, get_coord)
    get_dst = tla.tile_view(dst, shape, get_coord)
    put_src = tla.tile_view(src, shape, put_coord)
    put_dst = tla.tile_view(dst, shape, put_coord)
    with tla.vector():
        tla_dist.barrier_all()
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            submit_remote(get_dst, get_src, peer, control, event_id, case, "get")
            submit_remote(put_dst, put_src, peer, control, event_id, case, "put")
            tla_dist.quiet(tla_dist.params.CopyParams(transport="udma"), peer=peer)
        tla_dist.barrier_all()


@tla.kernel
def empty_copy(
    dst: tla.Tensor, src: tla.Tensor, peer: tla.Int32, empty_rows: tla.Int32, case: tla.Constexpr[RemoteCase]
):
    control = make_control()
    event_id = 0
    layout = tla.make_layout(
        tla.make_shape(case.rows, case.columns),
        tla.make_stride(case.columns, 1),
        origin_shape=tla.make_shape(empty_rows, case.columns),
        layoutTag=tla.arch.RowMajor,
    )
    source = tla.make_tensor(src.ptr, layout)
    target = tla.make_tensor(dst.ptr, layout)
    with tla.vector():
        tla_dist.barrier_all()
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            submit_remote(target, source, peer, control, event_id, case, "get")
            tla_dist.quiet(tla_dist.params.CopyParams(transport="udma"), peer=peer)
        tla_dist.barrier_all()


@tla.kernel
def repeated_quiet(dst: tla.Tensor, src: tla.Tensor, peer: tla.Int32, case: tla.Constexpr[RemoteCase]):
    control = make_control()
    event_id = 0
    shape = tla.make_shape(case.rows, case.columns)
    source = tla.tile_view(src, shape, tla.make_coord(0, 0))
    target = tla.tile_view(dst, shape, tla.make_coord(0, 0))
    with tla.vector():
        tla_dist.barrier_all()
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            tla_dist.quiet(tla_dist.params.CopyParams(transport="udma"), peer=peer)
            submit_remote(target, source, peer, control, event_id, case, "get")
            tla_dist.quiet(tla_dist.params.CopyParams(transport="udma"), peer=peer)
            tla_dist.quiet(tla_dist.params.CopyParams(transport="udma"), peer=peer)
        tla_dist.barrier_all()


def _tensor(runtime, tensor, dynamic):
    result = runtime.dsl.from_dlpack(tensor, layout_tag=runtime.dsl.arch.RowMajor)
    return result.mark_layout_dynamic() if dynamic else result


def _source(runtime, rank, rows, columns):
    # Exact integer FP32 values distinguish rank, row and column permutations.
    return runtime.torch.arange(rows * columns, dtype=runtime.torch.float32).reshape(rows, columns) + 256 * rank


def _run(runtime, allocate, test):
    case = test.case
    rows, columns = case.rows, case.columns
    total_rows = test.total_rows
    prefix = 1 if test.offset else 0
    peer = (runtime.rank + 1) % runtime.world_size
    previous = (runtime.rank - 1) % runtime.world_size
    error = None
    observed, temporary = None, None
    try:
        source_root = allocate((total_rows + prefix, test.src_stride))
        destination_root = allocate((total_rows + prefix, test.dst_stride))
        source_root.fill_(SENTINEL)
        destination_root.fill_(SENTINEL)
        source, destination = source_root[prefix:], destination_root[prefix:]
        values = _source(runtime, runtime.rank, total_rows, columns)
        source[:, :columns].copy_(values)
        args = [_tensor(runtime, destination, test.dynamic), _tensor(runtime, source, test.dynamic)]
        if test.kernel is quiet_observe:
            observed = runtime.torch.full_like(destination, SENTINEL)
            args.append(_tensor(runtime, observed, test.dynamic))
        elif test.kernel is get_then_put:
            temporary = allocate((total_rows, test.dst_stride))
            temporary.fill_(SENTINEL)
            args.append(_tensor(runtime, temporary, test.dynamic))
        args.append(runtime.dsl.Int32(peer))
        if test.kernel is empty_copy:
            args.append(runtime.dsl.Int32(0))
        args.append(case)
    except Exception as exc:
        error = f"input preparation failed: {type(exc).__name__}: {exc}"
    runtime.check_errors(error)
    artifact = runtime.compile(test.kernel, *args, arch_scope="aic.c310" if case.mixed_kernel else "aiv.c310")
    runtime.launch(artifact, *args[:-1])

    expected = runtime.torch.full((total_rows, test.dst_stride), SENTINEL)
    if test.kernel is mixed_get_put:
        expected[:rows, :columns] = _source(runtime, peer, total_rows, columns)[:rows]
        expected[rows:, :columns] = _source(runtime, previous, total_rows, columns)[rows:]
    elif test.kernel is get_then_put:
        # The peer sends our own data back after completing its GET.
        expected[:, :columns] = values
    elif test.kernel is not empty_copy:
        source_rank = peer if case.mode == "get" else previous
        expected[:, :columns] = _source(runtime, source_rank, total_rows, columns)
    runtime.assert_equal(destination, expected)
    if observed is not None:
        runtime.assert_equal(observed, expected)
    if temporary is not None:
        expected_tmp = runtime.torch.full((total_rows, test.dst_stride), SENTINEL)
        expected_tmp[:, :columns] = _source(runtime, peer, total_rows, columns)
        runtime.assert_equal(temporary, expected_tmp)
    expected_source = runtime.torch.full((total_rows + prefix, test.src_stride), SENTINEL)
    expected_source[prefix:, :columns] = values
    runtime.assert_equal(source_root, expected_source)
    if prefix:
        runtime.assert_equal(destination_root[:prefix], runtime.torch.full((prefix, test.dst_stride), SENTINEL))


class Runtime:
    def __init__(self, torch, cache_dir):
        self.torch = torch
        self.dist = torch.distributed
        self.dsl = tla_dist
        self.shmem = None
        self.rank = self.dist.get_rank()
        self.world_size = self.dist.get_world_size()
        self.cache_dir = Path(cache_dir)

    def check_errors(self, error):
        errors = [None] * self.world_size
        self.dist.all_gather_object(errors, error)
        failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message is not None]
        if failures:
            raise RuntimeError("\n".join(failures))

    def compile(self, kernel, *args, arch_scope="aic.c310"):
        artifact, error = None, None
        try:
            artifact = self.dsl.compile(kernel, *args, arch_scope=arch_scope, cache_dir=self.cache_dir)
        except Exception as exc:
            error = f"compilation failed: {type(exc).__name__}: {exc}"
        self.check_errors(error)
        return artifact

    def launch(self, artifact, *args):
        error = None
        try:
            self.torch.npu.synchronize()
        except Exception as exc:
            error = f"input synchronization failed: {type(exc).__name__}: {exc}"
        self.check_errors(error)
        self.shmem.barrier_all()
        error = None
        try:
            artifact(*args, block_num=1)
            self.torch.npu.synchronize()
        except Exception as exc:
            error = f"kernel execution failed: {type(exc).__name__}: {exc}"
        self.check_errors(error)

    def assert_equal(self, actual, expected):
        error = None
        try:
            self.torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        except Exception as exc:
            error = f"result comparison failed: {type(exc).__name__}: {exc}"
        self.check_errors(error)

    @contextmanager
    def allocations(self):
        tensors = []

        def allocate(shape):
            tensor = self.shmem.symmetric_tensor(shape, self.torch.float32)
            tensors.append(tensor)
            return tensor

        try:
            yield allocate
        finally:
            # All kernels and comparisons finish before any rank frees memory.
            self.torch.npu.synchronize()
            self.dist.barrier()
            for tensor in reversed(tensors):
                self.shmem.free(tensor)


@contextmanager
def initialized_runtime():
    import torch

    from catccos.tla_dist.compiler_bridge import resolve_tla_dist_compile

    torch.distributed.init_process_group(backend="gloo", timeout=timedelta(seconds=180))
    try:
        with TemporaryDirectory(prefix="udma-e2e-") as cache_dir:
            runtime = Runtime(torch, cache_dir)
            shmem = None
            error = None
            try:
                try:
                    import torch_npu  # noqa: F401 - register the NPU backend

                    if resolve_tla_dist_compile() is None:
                        raise RuntimeError("TlaDistCompile is unavailable; build python/tla_dist_dsl first")
                    local_rank = int(os.environ["LOCAL_RANK"])
                    torch.npu.set_device(local_rank)
                    device_name = torch.npu.get_device_name(local_rank)
                    if "950" not in device_name:
                        raise RuntimeError(f"communication kernels require Ascend 950, got {device_name}")
                    tla_dist.distributed.configure_store_tls(False, "")
                    uid = [tla_dist.distributed.get_unique_id() if runtime.rank == 0 else None]
                except Exception as exc:
                    error = f"setup failed: {type(exc).__name__}: {exc}"
                runtime.check_errors(error)
                runtime.dist.broadcast_object_list(uid, src=0)
                error = None
                try:
                    shmem = tla_dist.distributed.ShmemRuntime(
                        uid=uid[0],
                        rank=runtime.rank,
                        world_size=runtime.world_size,
                        heap_size=64 * 1024 * 1024,
                        device_id=local_rank,
                        transport="udma",
                        qp_num=2,
                    ).initialize()
                except Exception as exc:
                    error = f"SHMEM initialization failed: {type(exc).__name__}: {exc}"
                runtime.check_errors(error)
                runtime.shmem = shmem
                yield runtime
            finally:
                if shmem is not None:
                    try:
                        torch.npu.synchronize()
                    finally:
                        shmem.finalize()
    finally:
        torch.distributed.destroy_process_group()


@dataclass(frozen=True)
class RunCase:
    name: str
    kernel: object
    case: RemoteCase
    src_stride: int
    dst_stride: int
    dynamic: bool = False
    offset: bool = False

    @property
    def total_rows(self):
        if isinstance(self.case, MultiSubmitRemoteCase):
            return self.case.rows * self.case.count * self.case.rounds
        if self.kernel in (mixed_get_put, multi_qp):
            return 2 * self.case.rows
        return self.case.rows


def iter_cases():
    for pipe, mode in product(("mte3", "s"), ("get", "put")):
        yield RunCase(f"multi-qp/{pipe}/{mode}", multi_qp, RemoteCase(8, 16, mode, pipe, mixed_kernel=True), 16, 16)
    # Four submission paths share one contiguous layout and pure AIV kernel.
    for pipe, mode in product(("mte3", "s"), ("get", "put")):
        yield RunCase(f"basic/{pipe}/{mode}", remote_copy, RemoteCase(8, 16, mode, pipe), 16, 16)
    # Layout handling is shared by the submission paths.
    for name in ("tail-padded-static", "tail-padded-dynamic"):
        rows, columns, src_stride, dst_stride, dynamic = GEOMETRIES[name]
        yield RunCase(f"layout/{name}", remote_copy, RemoteCase(rows, columns, "get"), src_stride, dst_stride, dynamic)
    for pipe in ("mte3", "s"):
        yield RunCase(f"mixed-aiv/{pipe}/get", remote_copy, RemoteCase(8, 16, "get", pipe, mixed_kernel=True), 16, 16)
    # Two submissions per quiet, three rounds, with GM storage offsets.
    for pipe, mode in product(("mte3", "s"), ("get", "put")):
        yield RunCase(
            f"batch-one-quiet/{pipe}/{mode}",
            quiet_observe,
            MultiSubmitRemoteCase(8, 13, mode, pipe),
            16,
            20,
            dynamic=True,
            offset=True,
        )
    for pipe in ("mte3", "s"):
        yield RunCase(f"ordering/{pipe}/mixed", mixed_get_put, RemoteCase(5, 13, "get", pipe), 24, 32, dynamic=True)
        yield RunCase(f"ordering/{pipe}/dependency", get_then_put, RemoteCase(5, 13, "get", pipe), 24, 32, dynamic=True)
    yield RunCase("boundary/ub-offset", remote_copy_ub_offset, RemoteCase(8, 16, "get"), 16, 16)
    yield RunCase("boundary/empty", empty_copy, RemoteCase(8, 16, "get"), 16, 16)
    yield RunCase("boundary/repeated-quiet", repeated_quiet, RemoteCase(8, 16, "get"), 16, 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list cases without initializing SHMEM or an NPU")
    parser.add_argument("--case", default="", help="run cases whose names contain this substring")
    args = parser.parse_args()
    cases = [case for case in iter_cases() if args.case in case.name]
    if not cases:
        parser.error(f"no cases match {args.case!r}")
    if args.list:
        for case in cases:
            print(case.name)
        print(f"{len(cases)} cases")
        return 0
    if int(os.environ.get("WORLD_SIZE", "1")) < 2 or "LOCAL_RANK" not in os.environ:
        parser.error("launch with torchrun --standalone --nproc-per-node=2 (or more ranks)")

    with initialized_runtime() as runtime:
        passed = 0
        for test in cases:
            if runtime.rank == 0:
                print(f"RUN  {test.name}", flush=True)
            try:
                with runtime.allocations() as allocate:
                    _run(runtime, allocate, test)
            except Exception as exc:
                if runtime.rank == 0:
                    print(f"FAIL {test.name}: {exc}\n{passed} passed, 1 failed", flush=True)
                raise
            passed += 1
            if runtime.rank == 0:
                print(f"PASS {test.name}", flush=True)
        if runtime.rank == 0:
            print(f"{passed} passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
