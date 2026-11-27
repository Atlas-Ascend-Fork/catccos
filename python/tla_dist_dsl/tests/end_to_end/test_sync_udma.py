# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Local UDMA acceptance for default completion, order and QP ownership.

Requires matching SHMEM UDMA with relay OFF and qp_num=2. Payloads use UDMA;
P3 reuses the completed-UDMA/MTE-notification protocol, without UDMA signal.
Set VQR_SYNC_EVIDENCE_DIR to retain the exact device cache and launch records.
"""

import argparse
import fnmatch
import os
from dataclasses import dataclass

from catccos import tla_dist  # isort: skip
from catlass import tla

from catccos.tla_dist.params import CoreScope

try:
    from . import test_sync as common
except ImportError:
    import test_sync as common


@dataclass(frozen=True)
class NetworkCase:
    action: str
    mixed: bool = False
    mode: str = "put"


@tla.jit
def scope_point(mixed: tla.Constexpr[bool], kind: tla.Constexpr[str]):
    if tla.const_expr(mixed):
        if tla.const_expr(kind == "local"):
            tla_dist.local_sync(scope=CoreScope.MIX)
        elif tla.const_expr(kind == "sync"):
            tla_dist.sync(scope=CoreScope.MIX)
        else:
            tla_dist.barrier(scope=CoreScope.MIX)
    else:
        with tla.vector():
            if tla.const_expr(kind == "local"):
                tla_dist.local_sync()
            elif tla.const_expr(kind == "sync"):
                tla_dist.sync()
            else:
                tla_dist.barrier()


@tla.jit
def observe_tile(dst: tla.Tensor, source: tla.Tensor, scratch: tla.Tensor, loaded, completed):
    tla.copy(scratch, source)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    tla.copy(dst, scratch)
    tla.set_flag(completed)
    tla.wait_flag(completed)


@tla.kernel
def network(
    dst: tla.Tensor,
    source: tla.Tensor,
    observed: tla.Tensor,
    work_source: tla.Tensor,
    work_dst: tla.Tensor,
    work_seen: tla.Tensor,
    control: tla.Tensor,
    control_source: tla.Tensor,
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    cube_result: tla.Tensor,
    peer: tla.Int32,
    case: tla.Constexpr[NetworkCase],
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
    work = tla.make_tensor(
        tla.allocate(16, tla.Float32, tla.AddressSpace.ub, 32),
        tla.make_layout(tla.make_shape(1, 16), tla.make_stride(16, 1), layoutTag=tla.arch.RowMajor),
    )
    loaded = tla.flag("network_read", tla.arch.MTE2, tla.arch.MTE3, id=2)
    completed = tla.flag("network_store", tla.arch.MTE3, tla.arch.MTE2, id=2)
    work_loaded = tla.flag("network_work_read", tla.arch.MTE2, tla.arch.MTE3, id=3)
    work_completed = tla.flag("network_work_store", tla.arch.MTE3, tla.arch.MTE2, id=3)
    layout = tla.make_layout(tla.make_shape(5, 13), tla.make_stride(32, 1), layoutTag=tla.arch.RowMajor)
    target = tla.make_tensor(dst.ptr + 32, layout)
    src = tla.make_tensor(source.ptr + 32, layout)
    out = tla.make_tensor(observed.ptr + 32, layout)
    target1 = tla.make_tensor(dst.ptr + 8 * 32, layout)
    src1 = tla.make_tensor(source.ptr + 8 * 32, layout)
    out1 = tla.make_tensor(observed.ptr + 8 * 32, layout)
    qp0 = tla_dist.params.CopyParams(mode=case.mode, transport="udma", wqe_pipe="s", qp_idx=0)
    qp1 = tla_dist.params.CopyParams(mode=case.mode, transport="udma", wqe_pipe="s", qp_idx=1)
    if tla.const_expr(case.mixed):
        common.cube_work(lhs, rhs, cube_result)
    if tla.const_expr(case.action == "interlock"):
        with tla.vector():
            if tla.const_expr(case.mixed):
                flat = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
            else:
                flat = tla.arch.block_idx()
            if flat == 1:
                tla_dist.remote_copy(target1, src1, qp1, peer=peer)
                tla_dist.quiet(qp1, peer=peer)
        # The wider UDMA completion group must finish before default CQ use.
        scope_point(case.mixed, "local")
    with tla.vector():
        if tla.const_expr(case.mixed):
            flat = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
        else:
            flat = tla.arch.block_idx()
        if flat == 0:
            tla_dist.remote_copy(target, src, qp0, peer=peer)
            if tla.const_expr(case.action == "quiet"):
                tla_dist.quiet()
                if tla.const_expr(case.mode == "get"):
                    observe_tile(out, target, scratch, loaded, completed)
            elif tla.const_expr(case.action == "fence"):
                tla_dist.fence()
                flag = tla.tile_view(control, tla.make_shape(1, 1), tla.make_coord(0, 0))
                flag_src = tla.tile_view(control_source, tla.make_shape(1, 1), tla.make_coord(0, 0))
                tla_dist.remote_copy(flag, flag_src, qp0, peer=peer)
                tla_dist.wait_until(flag, 1.0)
                observe_tile(out, target, scratch, loaded, completed)
                tla_dist.quiet()
        if tla.const_expr(case.action in ("barrier", "interlock")):
            work_in = tla.tile_view(work_source, tla.make_shape(1, 16), tla.make_coord(flat, 0))
            work_target = tla.tile_view(work_dst, tla.make_shape(1, 16), tla.make_coord(flat, 0))
            tla.copy(work, work_in)
            tla.set_flag(work_loaded)
            tla.wait_flag(work_loaded)
            # No per-core MTE3 completion before the collective under test.
            tla.copy(work_target, work)
    if tla.const_expr(case.action in ("barrier", "interlock")):
        scope_point(case.mixed, "barrier")
    else:
        scope_point(case.mixed, "sync")
    with tla.vector():
        if tla.const_expr(case.mixed):
            flat = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
        else:
            flat = tla.arch.block_idx()
        if tla.const_expr(case.action in ("barrier", "interlock")):
            local = tla.tile_view(work_dst, tla.make_shape(1, 16), tla.make_coord(flat, 0))
            snap = tla.tile_view(work_seen, tla.make_shape(1, 16), tla.make_coord(flat, 0))
            # Capture after barrier inside the kernel, before a Host completion
            # could hide failure to complete another AIV's local pipeline.
            observe_tile(snap, local, work, work_loaded, work_completed)
        if flat == 0:
            if tla.const_expr(
                case.action in ("barrier", "interlock") or (case.action == "quiet" and case.mode == "put")
            ):
                observe_tile(out, target, scratch, loaded, completed)
            if tla.const_expr(case.action == "interlock"):
                observe_tile(out1, target1, scratch, loaded, completed)


def manifest():
    yield "network/default-quiet", "quiet"
    yield "network/default-fence", "fence"
    yield "network/barrier-completion", "barrier"
    yield "network/payload-ordering", "publication"
    yield "network/explicit-qp-interlock", "interlock"


def run_case(runtime, action):
    if action == "publication":
        return common._publication(runtime, transport="udma")
    torch = runtime.torch
    symmetric = []

    def allocate(shape, dtype):
        tensor = runtime.shmem.symmetric_tensor(shape, dtype)
        symmetric.append(tensor)
        return tensor

    destination = allocate((14, 32), torch.float32)
    source = allocate((14, 32), torch.float32)
    control = allocate((1, 16), torch.float32)
    control_source = allocate((1, 16), torch.float32)
    sentinel = -1024.0
    launches = 0
    try:
        for mixed in (False, True):
            for mode in ("get", "put") if action == "quiet" else ("put",):
                case = NetworkCase(action, mixed, mode)
                cpu_source = torch.full((14, 32), sentinel, dtype=torch.float32)
                expected = torch.full_like(cpu_source, sentinel)
                for slot in (0, 1):
                    row = slot * 7 + 1
                    values = torch.arange(65, dtype=torch.float32).reshape(5, 13) + runtime.rank * 1048576 + slot * 256
                    peer_values = (
                        torch.arange(65, dtype=torch.float32).reshape(5, 13) + (1 - runtime.rank) * 1048576 + slot * 256
                    )
                    cpu_source[row : row + 5, 8:21] = values
                    if slot == 0 or action == "interlock":
                        expected[row : row + 5, 8:21] = peer_values
                source.copy_(cpu_source)
                destination.fill_(sentinel)
                control.fill_(sentinel)
                control[:, 0] = 0
                control_source.fill_(sentinel)
                control_source[:, 0] = 1
                observed = torch.full_like(destination, sentinel)
                work_source = (torch.arange(1, 9, dtype=torch.float32)[:, None].expand(8, 16).contiguous()).to(
                    destination.device
                )
                work_dst = torch.full_like(work_source, sentinel)
                work_seen = torch.full_like(work_source, sentinel)
                lhs, rhs, cube_result = common._cube_inputs(runtime, 2)
                args = tuple(
                    common._view(t)
                    for t in (
                        destination[:, 8:],
                        source[:, 8:],
                        observed[:, 8:],
                        work_source,
                        work_dst,
                        work_seen,
                        control,
                        control_source,
                        lhs,
                        rhs,
                        cube_result,
                    )
                ) + (tla.Int32(1 - runtime.rank), case)
                artifact = common._compile(runtime, network, args, mixed=mixed)
                common._launch(runtime, artifact, args[:-1], 2)
                launches += 1
                runtime.assert_equal(destination, expected)
                runtime.assert_equal(observed, expected)
                runtime.assert_equal(source, cpu_source)
                expected_control = torch.full((1, 16), sentinel, dtype=torch.float32)
                expected_control[0, 0] = 1 if action == "fence" else 0
                runtime.assert_equal(control, expected_control)
                expected_work = torch.full((8, 16), sentinel, dtype=torch.float32)
                if action in ("barrier", "interlock"):
                    participants = 4 if mixed else 2
                    expected_work[:participants] = torch.arange(1, participants + 1, dtype=torch.float32)[
                        :, None
                    ].expand(participants, 16)
                runtime.assert_equal(work_dst, expected_work)
                runtime.assert_equal(work_seen, expected_work)
                if mixed:
                    common._check_cube(runtime, cube_result)
    finally:
        for tensor in reversed(symmetric):
            runtime.shmem.free(tensor)
    return launches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", default="*")
    args = parser.parse_args()
    cases = [case for case in manifest() if fnmatch.fnmatch(case[0], args.case)]
    if args.list:
        for name, _ in cases:
            print(name)
        return 0
    if not cases:
        raise SystemExit("no matching UDMA acceptance cases")
    if os.environ.get("WORLD_SIZE") != "2":
        raise SystemExit("use two torchrun ranks")
    try:
        from .test_remote_copy_udma import initialized_runtime
    except ImportError:
        from test_remote_copy_udma import initialized_runtime
    with initialized_runtime() as runtime:
        for name, action in cases:
            if runtime.rank == 0:
                print(f"RUN  {name}", flush=True)
            count = run_case(runtime, action)
            print(f"DEVICE_PASS {name} rank={runtime.rank} launches={count}", flush=True)
            runtime.dist.barrier()
            if runtime.rank == 0:
                print(f"PASS {name}", flush=True)
        if runtime.rank == 0:
            print(f"{len(cases)} passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
