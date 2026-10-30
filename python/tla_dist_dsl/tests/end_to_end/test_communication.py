"""Numerical communication tests on two or more Ascend 950 ranks.

Run with ``torchrun --standalone --nproc-per-node=2 -m pytest tests/end_to_end -q``.
"""

import pytest

# (rows, columns, source row stride, destination row stride, dynamic ABI)
GEOMETRIES = [
    pytest.param((8, 16, 16, 16, False), id="aligned-static"),
    pytest.param((5, 13, 24, 32, False), id="tail-padded-static"),
    pytest.param((5, 13, 24, 32, True), id="tail-padded-dynamic"),
]
SENTINEL = -1024.0
INITIAL = 7.0
CROSS_RANK_ATOMIC_PUT_UNAVAILABLE = pytest.mark.skip(
    reason="Cross-rank MTE atomic PUT hangs on the current PCIe PIX topology (P2P DMA_ADD unsupported)"
)


def _tensor(runtime, tensor, dynamic):
    result = runtime.dsl.from_dlpack(tensor, layout_tag=runtime.dsl.arch.RowMajor)
    return result.mark_layout_dynamic() if dynamic else result


def _source(runtime, rank, rows, columns):
    # Exact integer FP32 values distinguish rank, row and column permutations.
    return runtime.torch.arange(rows * columns, dtype=runtime.torch.float32).reshape(rows, columns) + 256 * rank


def _run_copy(
    runtime,
    allocate,
    geometry,
    *,
    mode=None,
    route="gm_to_gm",
    atomic=False,
    reset_atomic=False,
):
    # Import after comm_runtime has checked native dependencies.
    from .comm_kernels import CopyCase, remote_copy

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
        CopyCase(rows, columns, route, mode, atomic, reset_atomic),
    )
    artifact = runtime.compile(remote_copy, *args)
    # CopyCase is a compile-time argument and has no launch ABI slot.
    runtime.launch(artifact, *args[:-1])

    # GET pulls from the next rank; PUT receives the previous rank's data.
    is_get = mode == "get" or route == "gm_to_ub"
    source_rank = (runtime.rank + (1 if is_get else -1)) % runtime.world_size
    expected = runtime.torch.full((rows, dst_stride), SENTINEL)
    values = _source(runtime, source_rank, rows, columns)
    expected[:, :columns] = INITIAL + 2 * values if atomic and not reset_atomic else values
    runtime.assert_equal(destination, expected)
    # Detect accidental writes into either the source or the destination padding.
    expected_source = runtime.torch.full((rows, src_stride), SENTINEL)
    expected_source[:, :columns] = _source(runtime, runtime.rank, rows, columns)
    runtime.assert_equal(source, expected_source)


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize(
    "mode,atomic",
    [
        pytest.param("get", False, id="copy-get"),
        pytest.param("put", False, id="copy-put"),
        pytest.param("get", True, id="atomic-add-get"),
        pytest.param("put", True, id="atomic-add-put", marks=CROSS_RANK_ATOMIC_PUT_UNAVAILABLE),
    ],
)
def test_remote_copy_gm_to_gm(comm_runtime, symmetric_tensor, geometry, mode, atomic):
    _run_copy(comm_runtime, symmetric_tensor, geometry, mode=mode, atomic=atomic)


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("route,mode", [("gm_to_ub", "get"), ("ub_to_gm", "put")])
@pytest.mark.parametrize("infer_mode", [False, True], ids=["explicit-mode", "inferred-mode"])
def test_remote_copy_gm_ub(comm_runtime, symmetric_tensor, geometry, route, mode, infer_mode):
    _run_copy(
        comm_runtime,
        symmetric_tensor,
        geometry,
        route=route,
        mode=None if infer_mode else mode,
    )


@pytest.mark.parametrize("mode", ["get", pytest.param("put", marks=CROSS_RANK_ATOMIC_PUT_UNAVAILABLE)])
def test_remote_copy_restores_non_atomic_store(comm_runtime, symmetric_tensor, mode):
    _run_copy(
        comm_runtime,
        symmetric_tensor,
        (5, 13, 24, 32, True),
        mode=mode,
        atomic=True,
        reset_atomic=True,
    )


def test_barrier_all_publishes_remote_writes(comm_runtime, symmetric_tensor):
    from .comm_kernels import CopyCase, barrier_visibility

    runtime = comm_runtime
    rows, columns, repeats = 5, 13, 3
    source = symmetric_tensor((rows * repeats, columns))
    destination = symmetric_tensor((rows, columns))
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
    artifact = runtime.compile(barrier_visibility, *args)
    runtime.launch(artifact, *args[:-1])
    runtime.assert_equal(observed, expected)
