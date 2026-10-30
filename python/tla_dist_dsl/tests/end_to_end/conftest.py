"""Real multi-rank NPU fixtures; run this directory with torchrun."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest


@dataclass
class CommRuntime:
    torch: Any
    dist: Any
    dsl: Any
    shmem: Any
    rank: int
    world_size: int
    cache_dir: Path

    def check_errors(self, error: str | None) -> None:
        """Fail every rank together so subsequent collectives stay in step."""
        errors = [None] * self.world_size
        self.dist.all_gather_object(errors, error)
        failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message is not None]
        if failures:
            pytest.fail("\n".join(failures))

    def compile(self, kernel, *args):
        artifact = None
        error = None
        try:
            artifact = self.dsl.compile(kernel, *args, arch_scope="aic.c310", cache_dir=self.cache_dir)
        except Exception as exc:  # noqa: BLE001 - report compile errors to every rank before launch
            error = f"compilation failed: {type(exc).__name__}: {exc}"
        self.check_errors(error)
        return artifact

    def launch(self, artifact, *args) -> None:
        self.torch.npu.synchronize()
        self.shmem.barrier_all()
        # One AIC launches its two paired AIVs and uses SHMEM hardware core sync.
        artifact(*args, block_num=1)
        self.torch.npu.synchronize()
        self.dist.barrier()

    def assert_equal(self, actual, expected) -> None:
        error = None
        try:
            self.torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        except AssertionError as exc:
            error = str(exc)
        self.check_errors(error)


@pytest.fixture(scope="session")
def comm_runtime(tmp_path_factory):
    if int(os.environ.get("WORLD_SIZE", "1")) < 2 or "LOCAL_RANK" not in os.environ:
        pytest.skip("communication E2E tests require torchrun --standalone --nproc-per-node=2 -m pytest")

    # Keep native/NPU imports lazy so ordinary CPU test runs can collect these tests.
    pytest.importorskip("catccos")
    pytest.importorskip("catlass._mlir.ir")
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    pytest.importorskip("shmem")
    pytest.importorskip("acl")
    from catccos import tla_dist
    from catccos.tla_dist.compiler_bridge import resolve_tla_dist_compile

    if resolve_tla_dist_compile() is None:
        pytest.skip("TlaDistCompile is unavailable; build python/tla_dist_dsl first")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")

    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= torch.npu.device_count():
        pytest.fail("torchrun needs one visible NPU per local rank")
    torch.npu.set_device(local_rank)
    device_name = torch.npu.get_device_name(local_rank)
    if "950" not in device_name:
        pytest.skip(f"communication device lowering requires Ascend 950, got {device_name}")

    dist = torch.distributed
    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=180))
    runtime = None
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        tla_dist.distributed.configure_store_tls(False, "")
        uid = [tla_dist.distributed.get_unique_id() if rank == 0 else None]
        dist.broadcast_object_list(uid, src=0)
        runtime = tla_dist.distributed.ShmemRuntime(
            uid=uid[0],
            rank=rank,
            world_size=world_size,
            heap_size=64 * 1024 * 1024,
            device_id=local_rank,
        ).initialize()
        yield CommRuntime(
            torch,
            dist,
            tla_dist,
            runtime,
            rank,
            world_size,
            tmp_path_factory.mktemp("comm-cache"),
        )
    finally:
        try:
            if runtime is not None:
                torch.npu.synchronize()
                runtime.finalize()
        finally:
            dist.destroy_process_group()


@pytest.fixture
def symmetric_tensor(comm_runtime):
    """Every rank must call this allocator with the same shapes in the same order."""
    allocations = []

    def allocate(shape):
        tensor = comm_runtime.shmem.symmetric_tensor(shape, comm_runtime.torch.float32)
        allocations.append(tensor)
        return tensor

    yield allocate
    comm_runtime.torch.npu.synchronize()
    comm_runtime.dist.barrier()
    for tensor in reversed(allocations):
        comm_runtime.shmem.free(tensor)
