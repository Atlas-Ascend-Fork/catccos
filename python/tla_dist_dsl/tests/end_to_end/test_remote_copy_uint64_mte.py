# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exact-bit UINT64 MTE GM-to-GM E2E tests; launch with torchrun."""

# ruff: noqa: BLE001

import argparse
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

# Bootstrap pinned Catlass before importing it.
from catccos import tla_dist  # isort: skip
from catlass import tla

from test_remote_copy_mte import initialized_runtime

PATTERNS = (
    0,
    1,
    0xFFFFFFFF,
    0x100000000,
    0x7FFFFFFFFFFFFFFF,
    0x8000000000000000,
    0x8000000000000001,
    0xFFFFFFFFFFFFFFFF,
    0x55AA55AA55AA55AA,
    0xAA55AA55AA55AA55,
)
GUARD = 0xDEADBEEFCAFEBABE


@dataclass(frozen=True)
class UInt64Case:
    name: str
    mode: str
    rows: int
    columns: int
    src_stride: int
    dst_stride: int
    dynamic: bool = False
    src_offset: int = 0
    dst_offset: int = 0
    rounds: int = 3

    @property
    def total_rows(self):
        return (self.rounds + max(self.src_offset, self.dst_offset) + 1) * self.rows


def iter_cases():
    for mode in ("get", "put"):
        for name, geometry in (
            ("aligned-static", (8, 16, 16, 16, False, 0, 0)),
            ("tail-padded-static", (5, 13, 24, 32, False, 0, 0)),
            ("tail-padded-dynamic", (5, 13, 24, 32, True, 0, 0)),
            ("offset-multiround", (5, 13, 24, 32, True, 1, 2)),
        ):
            yield UInt64Case(f"{mode}/{name}", mode, *geometry)


@tla.kernel
def uint64_remote_copy(dst: tla.Tensor, src: tla.Tensor, peer: tla.Int32, case: tla.Constexpr[UInt64Case]):
    padded_columns = ((case.columns + 3) // 4) * 4
    scratch_ptr = tla.allocate(case.rows * padded_columns, tla.UInt64, tla.AddressSpace.ub, 32)
    scratch = tla.make_tensor(
        scratch_ptr,
        tla.make_layout(
            tla.make_shape(case.rows, padded_columns),
            tla.make_stride(padded_columns, 1),
            origin_shape=tla.make_shape(case.rows, case.columns),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    complete = tla.flag("uint64_copy_complete", tla.arch.MTE3, tla.arch.MTE2, id=0)
    tile_shape = tla.make_shape(case.rows, case.columns)
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla_dist.barrier_all()
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            for iteration in tla.range(case.rounds):
                source = tla.tile_view(src, tile_shape, tla.make_coord(iteration + case.src_offset, 0))
                target = tla.tile_view(dst, tile_shape, tla.make_coord(iteration + case.dst_offset, 0))
                tla_dist.remote_copy(
                    target,
                    source,
                    tla_dist.params.CopyParams(mode=case.mode, transport="mte"),
                    peer=peer,
                    ub_tensor=scratch,
                    event_id=0,
                )
                tla.set_flag(complete)
                tla.wait_flag(complete)
        tla_dist.barrier_all()


def kernel_args(case):
    """Metadata-only arguments for native lowering/link tests without an NPU."""

    def tensor(stride):
        value = tla.make_fake_tensor(
            tla.UInt64,
            (case.total_rows, stride),
            (stride, 1),
            addrspace=tla.AddressSpace.gm,
            layout_tag=tla.arch.RowMajor,
        )
        return value.mark_layout_dynamic() if case.dynamic else value

    return tensor(case.dst_stride), tensor(case.src_stride), tla.Int32(1), case


def _payload(case, rank):
    result = np.empty((case.rounds, case.rows, case.columns), dtype=np.uint64)
    for iteration in range(case.rounds):
        for row in range(case.rows):
            for col in range(case.columns):
                index = row * case.columns + col
                result[iteration, row, col] = (
                    PATTERNS[(index + rank + iteration) % len(PATTERNS)]
                    ^ (rank << 52)
                    ^ (iteration << 44)
                    ^ (index // len(PATTERNS))
                )
    return result


def _buffers(runtime, case, rank):
    # CPU uint64 construction and int64 storage preserve every bit, including
    # values above INT64_MAX, without NPU unsigned arithmetic or float casts.
    source = np.full((case.total_rows, case.src_stride), GUARD, dtype=np.uint64)
    destination = np.full((case.total_rows, case.dst_stride), GUARD, dtype=np.uint64)
    for iteration, values in enumerate(_payload(case, rank)):
        start = (iteration + case.src_offset) * case.rows
        source[start : start + case.rows, : case.columns] = values
    return runtime.torch.from_numpy(source.view(np.int64)), runtime.torch.from_numpy(destination.view(np.int64))


@contextmanager
def allocations(runtime):
    tensors = []

    def allocate(shape):
        tensor = runtime.shmem.symmetric_tensor(shape, runtime.torch.int64)
        tensors.append(tensor)
        return tensor

    try:
        yield allocate
    finally:
        runtime.torch.npu.synchronize()
        runtime.dist.barrier()
        for tensor in reversed(tensors):
            runtime.shmem.free(tensor)


def _tensor(value, dynamic):
    tensor = tla.from_dlpack(value, element_type=tla.UInt64, layout_tag=tla.arch.RowMajor)
    return tensor.mark_layout_dynamic() if dynamic else tensor


def run_case(runtime, allocate, case):
    error = None
    try:
        source_cpu, initial_cpu = _buffers(runtime, case, runtime.rank)
        source = allocate(tuple(source_cpu.shape))
        destination = allocate(tuple(initial_cpu.shape))
        source.copy_(source_cpu)
        destination.copy_(initial_cpu)
        args = (
            _tensor(destination, case.dynamic),
            _tensor(source, case.dynamic),
            tla.Int32((runtime.rank + 1) % runtime.world_size),
            case,
        )
        frontend_ir = uint64_remote_copy.dump_mlir(type_args=args)
        if "!tla.ptr<ui64," not in frontend_ir:
            raise AssertionError("the device test must compile UInt64 tensors, not signed Int64 storage")
    except Exception as exc:
        error = f"UINT64 preparation failed: {type(exc).__name__}: {exc}"
    runtime.check_errors(error)
    artifact = runtime.compile(uint64_remote_copy, *args)
    runtime.launch(artifact, *args[:-1])

    source_rank = (runtime.rank + (1 if case.mode == "get" else -1)) % runtime.world_size
    expected = initial_cpu.numpy().view(np.uint64).copy()
    for iteration, values in enumerate(_payload(case, source_rank)):
        start = (iteration + case.dst_offset) * case.rows
        expected[start : start + case.rows, : case.columns] = values
    runtime.assert_equal(destination, runtime.torch.from_numpy(expected.view(np.int64)))
    runtime.assert_equal(source, source_cpu)


def compile_cases(cases, cache_dir):
    from catccos.tla_dist.compiler_bridge import resolve_tla_dist_compile

    compiler = resolve_tla_dist_compile()
    if compiler is None:
        raise RuntimeError("build TlaDistCompile before using --compile-only")
    for case in cases:
        args = kernel_args(case)
        ir = uint64_remote_copy.dump_mlir(type_args=args)
        if "!tla.ptr<ui64," not in ir:
            raise AssertionError("expected unsigned UInt64 frontend IR")
        artifact = tla_dist.compile(uint64_remote_copy, type_args=args, arch_scope="aic.c310", cache_dir=cache_dir)
        if not artifact.kernel_binary_path.is_file():
            raise AssertionError("device kernel binary was not emitted")
        print(f"COMPILE PASS {case.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list cases without SHMEM or NPU initialization")
    parser.add_argument("--case", default="", help="run cases whose names contain this substring")
    parser.add_argument("--compile-only", action="store_true", help="compile/link kernels without running them")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    cases = [case for case in iter_cases() if args.case in case.name]
    if not cases:
        parser.error(f"no cases match {args.case!r}")
    if args.list:
        for case in cases:
            print(case.name)
        print(f"{len(cases)} cases")
        return 0
    if args.compile_only:
        with TemporaryDirectory(prefix="uint64-mte-compile-") as temporary:
            compile_cases(cases, args.cache_dir or Path(temporary))
        return 0
    if int(os.environ.get("WORLD_SIZE", "1")) < 2 or "LOCAL_RANK" not in os.environ:
        parser.error("launch with torchrun --standalone --nproc-per-node=2 (or more ranks)")
    with initialized_runtime() as runtime:
        if runtime.rank == 0:
            print("STORAGE int64 backing with explicit UInt64 DLPack view; exact 64-bit comparison", flush=True)
        passed = 0
        for case in cases:
            if runtime.rank == 0:
                print(f"RUN  {case.name}", flush=True)
            try:
                with allocations(runtime) as allocate:
                    run_case(runtime, allocate, case)
            except Exception as exc:
                if runtime.rank == 0:
                    print(f"FAIL {case.name}: {exc}\n{passed} passed, 1 failed", flush=True)
                raise
            passed += 1
            if runtime.rank == 0:
                print(f"PASS {case.name}", flush=True)
        if runtime.rank == 0:
            print(f"{passed} passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
