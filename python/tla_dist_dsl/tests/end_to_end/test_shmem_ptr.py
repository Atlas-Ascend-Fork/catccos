# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Read a peer's symmetric memory through shmem_ptr on two NPUs."""

# Gather per-rank input preparation failures before entering device barriers.
# ruff: noqa: BLE001

import os

from catccos import tla_dist  # isort: skip
from catlass import tla


@tla.kernel
def load_from_peer(dst: tla.Tensor, src: tla.Tensor, peer: tla.Int32, mixed: tla.Constexpr[bool]):
    if tla.const_expr(mixed):
        with tla.cube():
            tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            remote = tla_dist.shmem_ptr(src.ptr + 1, peer)
            layout = tla.make_layout(tla.make_shape(1, 1), tla.make_stride(1, 1), layoutTag=tla.arch.RowMajor)
            view = tla.make_tensor(remote, layout)
            dst[0, 0] = view[0, 0]


def _run_case(runtime, allocate, *, mixed):
    torch = runtime.torch
    peer = 1 - runtime.rank
    error = None
    try:
        src = allocate((1, 8))
        src.copy_((torch.arange(8, dtype=torch.float32) + 100 * runtime.rank).reshape(1, 8))
        dst = torch.full_like(src, -1)
        args = (
            tla_dist.from_dlpack(dst, layout_tag=tla.arch.RowMajor),
            tla_dist.from_dlpack(src, layout_tag=tla.arch.RowMajor),
            tla.Int32(peer),
        )
        expected = torch.full((1, 8), -1, dtype=torch.float32)
        expected[0, 0] = 100 * peer + 1
    except Exception as exc:
        error = f"input preparation failed: {type(exc).__name__}: {exc}"
    runtime.check_errors(error)

    arch_scope = "aic.c310" if mixed else "aiv.c310"
    artifact, error = None, None
    try:
        artifact = tla_dist.compile(load_from_peer, *args, mixed, arch_scope=arch_scope, cache_dir=runtime.cache_dir)
    except Exception as exc:
        error = f"compilation failed: {type(exc).__name__}: {exc}"
    runtime.check_errors(error)

    runtime.launch(artifact, *args)
    runtime.assert_equal(dst, expected)


def main():
    if os.environ.get("WORLD_SIZE") != "2" or "LOCAL_RANK" not in os.environ:
        raise SystemExit("launch with torchrun --standalone --nproc-per-node=2")

    from test_remote_copy_mte import initialized_runtime

    with initialized_runtime() as runtime:
        for name, mixed in (("aiv", False), ("mixed-aic-aiv", True)):
            if runtime.rank == 0:
                print(f"RUN  shmem_ptr/{name}", flush=True)
            with runtime.allocations() as allocate:
                _run_case(runtime, allocate, mixed=mixed)
            if runtime.rank == 0:
                print(f"PASS shmem_ptr/{name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
