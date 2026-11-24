# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Standalone MTE remote_copy E2E tests; launch with torchrun."""

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
    "aligned-static": (8, 16, 16, 16, False),
    "tail-padded-static": (5, 13, 24, 32, False),
    "tail-padded-dynamic": (5, 13, 24, 32, True),
}
SENTINEL = -1024.0
INITIAL = 7.0


@dataclass(frozen=True)
class CopyCase:
    rows: int
    columns: int
    route: str = "gm_to_gm"
    mode: str | None = None
    atomic: bool = False
    reset_atomic: bool = False
    repeats: int = 2


@tla.kernel
def remote_copy(dst: tla.Tensor, src: tla.Tensor, peer: tla.Int32, case: tla.Constexpr[CopyCase]):
    # Bind compact physical buffers on the host and crop logical columns here,
    # retaining each buffer's distinct physical row stride in the tile descriptor.
    tile_shape = tla.make_shape(case.rows, case.columns)
    src = tla.tile_view(src, tile_shape, tla.make_coord(0, 0))
    dst = tla.tile_view(dst, tile_shape, tla.make_coord(0, 0))
    # Each FP32 UB row occupies whole 32-byte blocks, including tail columns.
    padded_columns = ((case.columns + 7) // 8) * 8
    scratch_ptr = tla.allocate(case.rows * padded_columns, tla.Float32, tla.AddressSpace.ub, 32)
    scratch = tla.make_tensor(
        scratch_ptr,
        tla.make_layout(
            tla.make_shape(case.rows, padded_columns),
            tla.make_stride(padded_columns, 1),
            origin_shape=tla.make_shape(case.rows, case.columns),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    if tla.const_expr(case.route != "gm_to_gm"):
        submit = tla.flag("copy_submit", tla.arch.MTE2, tla.arch.MTE3, id=0)
    complete = tla.flag("copy_complete", tla.arch.MTE3, tla.arch.MTE2, id=0)
    params = tla_dist.params.CopyParams(
        mode=case.mode,
        atomic_mode=tla_dist.params.AtomicMode.ADD if case.atomic else tla_dist.params.AtomicMode.NONE,
    )
    # Select the same mixed-kernel hardware barrier path as the fused example.
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla_dist.barrier_all()
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            for _ in tla.range(case.repeats):
                if tla.const_expr(case.route == "gm_to_gm"):
                    tla_dist.remote_copy(
                        dst,
                        src,
                        params,
                        peer=peer,
                        ub_tensor=scratch,
                        event_id=0,
                    )
                elif tla.const_expr(case.route == "gm_to_ub"):
                    tla_dist.remote_copy(scratch, src, params, peer=peer)
                    tla.set_flag(submit)
                    tla.wait_flag(submit)
                    tla.copy(dst, scratch)
                else:
                    tla.copy(scratch, src)
                    tla.set_flag(submit)
                    tla.wait_flag(submit)
                    tla_dist.remote_copy(dst, scratch, params, peer=peer)
                # Wait for the final store before reusing this UB slot.
                tla.set_flag(complete)
                tla.wait_flag(complete)

            if tla.const_expr(case.reset_atomic):
                tla_dist.remote_copy(
                    dst,
                    src,
                    tla_dist.params.CopyParams(mode=case.mode),
                    peer=peer,
                    ub_tensor=scratch,
                    event_id=0,
                )
                tla.set_flag(complete)
                tla.wait_flag(complete)
        # Idle AIVs also participate; completion waits precede rank visibility.
        tla_dist.barrier_all()


@tla.kernel
def barrier_visibility(
    dst: tla.Tensor,
    src: tla.Tensor,
    observed: tla.Tensor,
    peer: tla.Int32,
    case: tla.Constexpr[CopyCase],
):
    padded_columns = ((case.columns + 7) // 8) * 8
    scratch_ptr = tla.allocate(case.rows * padded_columns, tla.Float32, tla.AddressSpace.ub, 32)
    scratch = tla.make_tensor(
        scratch_ptr,
        tla.make_layout(
            tla.make_shape(case.rows, padded_columns),
            tla.make_stride(padded_columns, 1),
            origin_shape=tla.make_shape(case.rows, case.columns),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    complete = tla.flag("barrier_complete", tla.arch.MTE3, tla.arch.MTE2, id=0)
    tile_shape = tla.make_shape(case.rows, case.columns)
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        for iteration in tla.range(case.repeats):
            source_tile = tla.tile_view(src, tile_shape, tla.make_coord(iteration, 0))
            observed_tile = tla.tile_view(observed, tile_shape, tla.make_coord(iteration, 0))
            if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
                tla_dist.remote_copy(
                    dst,
                    source_tile,
                    tla_dist.params.CopyParams(mode="put"),
                    peer=peer,
                    ub_tensor=scratch,
                    event_id=0,
                )
                tla.set_flag(complete)
                tla.wait_flag(complete)
            # Read back the next rank's destination only after all writers finish.
            tla_dist.barrier_all()
            if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
                tla_dist.remote_copy(
                    observed_tile,
                    dst,
                    tla_dist.params.CopyParams(mode="get"),
                    peer=peer,
                    ub_tensor=scratch,
                    event_id=0,
                )
                tla.set_flag(complete)
                tla.wait_flag(complete)
            # Readers must finish before the next iteration overwrites dst.
            tla_dist.barrier_all()


def _tensor(runtime, tensor, dynamic):
    result = runtime.dsl.from_dlpack(tensor, layout_tag=runtime.dsl.arch.RowMajor)
    return result.mark_layout_dynamic() if dynamic else result


def _source(runtime, rank, rows, columns):
    # Exact integer FP32 values distinguish rank, row and column permutations.
    return runtime.torch.arange(rows * columns, dtype=runtime.torch.float32).reshape(rows, columns) + 256 * rank


def _run_copy(runtime, allocate, geometry, case):
    error = None
    try:
        rows, columns, src_stride, dst_stride, dynamic = geometry
        source = allocate((rows, src_stride))
        destination = allocate((rows, dst_stride))
        source.fill_(SENTINEL)
        source[:, :columns].copy_(_source(runtime, runtime.rank, rows, columns))
        destination.fill_(SENTINEL)
        destination[:, :columns].fill_(INITIAL)
        args = (
            _tensor(runtime, destination, dynamic),
            _tensor(runtime, source, dynamic),
            runtime.dsl.Int32((runtime.rank + 1) % runtime.world_size),
            case,
        )
    except Exception as exc:
        error = f"input preparation failed: {type(exc).__name__}: {exc}"
    runtime.check_errors(error)
    artifact = runtime.compile(remote_copy, *args)
    # CopyCase is a compile-time argument and has no launch ABI slot.
    runtime.launch(artifact, *args[:-1])

    # GET pulls from the next rank; PUT receives the previous rank's data.
    is_get = case.mode == "get" or case.route == "gm_to_ub"
    source_rank = (runtime.rank + (1 if is_get else -1)) % runtime.world_size
    expected = runtime.torch.full((rows, dst_stride), SENTINEL)
    values = _source(runtime, source_rank, rows, columns)
    expected[:, :columns] = INITIAL + 2 * values if case.atomic and not case.reset_atomic else values
    runtime.assert_equal(destination, expected)
    # Detect accidental writes into either the source or the destination padding.
    expected_source = runtime.torch.full((rows, src_stride), SENTINEL)
    expected_source[:, :columns] = _source(runtime, runtime.rank, rows, columns)
    runtime.assert_equal(source, expected_source)


def _run_barrier(runtime, allocate):
    rows, columns, repeats = 5, 13, 3
    error = None
    try:
        source = allocate((rows * repeats, columns))
        destination = allocate((rows, columns))
        expected = _source(runtime, runtime.rank, rows * repeats, columns)
        source.copy_(expected)
        destination.fill_(SENTINEL)
        observed = runtime.torch.full_like(source, SENTINEL)
        args = (
            _tensor(runtime, destination, True),
            _tensor(runtime, source, True),
            _tensor(runtime, observed, True),
            runtime.dsl.Int32((runtime.rank + 1) % runtime.world_size),
            CopyCase(rows, columns, repeats=repeats),
        )
    except Exception as exc:
        error = f"input preparation failed: {type(exc).__name__}: {exc}"
    runtime.check_errors(error)
    artifact = runtime.compile(barrier_visibility, *args)
    runtime.launch(artifact, *args[:-1])
    runtime.assert_equal(observed, expected)


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

    def compile(self, kernel, *args):
        artifact, error = None, None
        try:
            artifact = self.dsl.compile(kernel, *args, arch_scope="aic.c310", cache_dir=self.cache_dir)
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
        with TemporaryDirectory(prefix="mte-e2e-") as cache_dir:
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
                        transport="mte",
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
    geometry: tuple | None = None
    case: CopyCase | None = None


def iter_cases():
    for (layout, geometry), mode, atomic in product(GEOMETRIES.items(), ("get", "put"), (False, True)):
        operation = "atomic-add" if atomic else "copy"
        yield RunCase(
            f"gm-to-gm/{operation}/{mode}/{layout}",
            geometry,
            CopyCase(*geometry[:2], mode=mode, atomic=atomic),
        )
    for (layout, geometry), (route, mode), infer in product(
        GEOMETRIES.items(), (("gm_to_ub", "get"), ("ub_to_gm", "put")), (False, True)
    ):
        mode_kind = "inferred-mode" if infer else "explicit-mode"
        yield RunCase(
            f"{route.replace('_', '-')}/{mode}/{mode_kind}/{layout}",
            geometry,
            CopyCase(*geometry[:2], route=route, mode=None if infer else mode),
        )
    geometry = GEOMETRIES["tail-padded-dynamic"]
    for mode in ("get", "put"):
        yield RunCase(
            f"atomic-reset/{mode}",
            geometry,
            CopyCase(*geometry[:2], mode=mode, atomic=True, reset_atomic=True),
        )
    yield RunCase("barrier-visibility")


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
                    if test.case is None:
                        _run_barrier(runtime, allocate)
                    else:
                        _run_copy(runtime, allocate, test.geometry, test.case)
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
