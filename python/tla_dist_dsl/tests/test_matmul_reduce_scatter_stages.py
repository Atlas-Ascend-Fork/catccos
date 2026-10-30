"""Compile-only coverage for fixed double buffering and tiling."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
import re

import pytest


@pytest.fixture(scope="module")
def example():
    import catccos  # noqa: F401 - bootstrap the pinned Catlass package

    pytest.importorskip("catlass._mlir.ir")
    source = Path(__file__).resolve().parents[1] / "examples/fusion_ops/matmul_reduce_scatter.py"
    spec = importlib.util.spec_from_file_location("mmrs_stages_example", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_double_buffer_allocations(example):
    tiling = example.TILING
    ir = example.dump_tlair(tiling=tiling)
    allocations = re.findall(r"tla.alloc_ptr\{size_bytes = (\d+)\} -> !tla.ptr<\w+, (\w+),", ir)
    usage = {}
    for size, space in allocations:
        usage[space] = usage.get(space, 0) + int(size)
    # f16 operands, f32 accumulators/communication. L0C stays single-buffered.
    assert usage == {
        "l1": (tiling.l1_tm + tiling.l1_tn) * tiling.l1_tk * 2 * 2,
        "l0a": tiling.l0_tm * tiling.l0_tk * 2 * 2,
        "l0b": tiling.l0_tn * tiling.l0_tk * 2 * 2,
        "l0c": tiling.l0_tm * tiling.l0_tn * 4,
        "ub": tiling.comm_tile_m * tiling.comm_tile_n * 4 * 2,
    }


def test_remote_copy_pairs_each_buffer_with_its_submission_and_completion_flags(example):
    from catlass._mlir import ir as mlir_ir
    from catlass.base_dsl import BaseDSL
    from catlass.core_api import _as_value

    def blocks(operation):
        for region in operation.regions:
            for block in region.blocks:
                yield block
                for child in block.operations:
                    yield from blocks(child.operation)

    kernel = example.matmul_reduce_scatter
    lowered = BaseDSL()._lower(
        kernel.fn,
        kind=kernel.kind,
        options=dict(kernel.options),
        type_args=example._compile_only_args(),
        location=kernel.decorator_location,
    )
    with lowered.context:
        buffers = {}
        for block in blocks(lowered.module.operation):
            operations = [op.operation for op in block.operations]
            for position, operation in enumerate(operations):
                if operation.name != "tla_dist.remote_copy":
                    continue
                submission = operation.operands[4].owner.operation
                assert submission.name == "tla.flag"
                slot = mlir_ir.IntegerAttr(submission.attributes["id"]).value
                assert slot in (0, 1)
                assert mlir_ir.StringAttr(submission.attributes["name"]).value == f"remote_submit_{slot}"
                scratch = operation.operands[3].owner.operation
                assert scratch.name == "tla.make_tensor_like"
                buffer = _as_value(scratch.operands[0])
                assert buffer.owner.operation.name == "tla.alloc_ptr"
                assert slot not in buffers
                buffers[slot] = buffer

                def uses_completion_flag(op, name):
                    return (
                        op.name == name
                        and mlir_ir.StringAttr(op.operands[0].owner.operation.attributes["name"]).value
                        == f"remote_available_{slot}"
                    )

                assert any(uses_completion_flag(op, "tla.wait_flag") for op in operations[:position])
                assert any(uses_completion_flag(op, "tla.set_flag") for op in operations[position + 1 :])

        assert set(buffers) == {0, 1}
        assert buffers[0] != buffers[1]


def test_custom_tiling_changes_allocations_and_workspace(example):
    tiling = replace(
        example.TILING,
        l1_tm=256,
        l1_tn=128,
        l1_tk=128,
        l0_tm=256,
        l0_tn=128,
        l0_tk=32,
        comm_block_m=128,
        comm_block_n=64,
        comm_tile_m=16,
        comm_tile_n=32,
        comm_interval=4,
        comm_data_split=4,
        comm_npu_split=2,
    )
    ir = example.dump_tlair(tiling=tiling)
    assert "size_bytes = 2048} -> !tla.ptr<f32, ub" in ir
    assert example._workspace_shape(tiling, 28) == (57344, 128)
    example._validate_tiling(tiling, example.SWIZZLE, 2, 28)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"l0_tm": 64}, "L0 and L1"),
        ({"comm_block_m": 48}, "communication blocks"),
        ({"comm_tile_m": 48}, "communication tiles"),
        ({"comm_data_split": 29}, "core split"),
        ({"comm_interval": 0}, "positive integer"),
        ({"comm_npu_split": 3}, "rank split"),
    ],
)
def test_invalid_tiling_is_rejected(example, changes, match):
    tiling = replace(example.TILING, **changes)
    with pytest.raises(ValueError, match=match):
        example._validate_tiling(tiling, example.SWIZZLE, 2, 28)


@pytest.mark.parametrize("layout_a,layout_b", [("row", "row"), ("row", "col"), ("col", "row"), ("col", "col")])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_cli_shapes_layouts_and_dtypes_compile(example, monkeypatch, layout_a, layout_b, dtype):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    args = example._argument_parser().parse_args(
        [
            "--m",
            "4100",
            "--n",
            "516",
            "--k",
            "257",
            "--world-size",
            "4",
            "--rank",
            "3",
            "--block-num",
            "20",
            "--layout-a",
            layout_a,
            "--layout-b",
            layout_b,
            "--dtype-a",
            dtype,
            "--dtype-b",
            dtype,
        ]
    )
    tensors = example._compile_only_args(args=args)
    assert tuple(tensors[0].origin_shape) == (4100, 257)
    assert tuple(tensors[1].origin_shape) == (257, 516)
    assert tuple(tensors[3].origin_shape) == (1025, 516)
    assert example._launch_context(args) == (3, 4, 0)
    ir = example.dump_tlair(args=args)
    assert f"!tla.ptr<{dtype}, l1" in ir
    assert ("ColumnMajor" in ir) == ("col" in (layout_a, layout_b))


@pytest.mark.parametrize(
    "options,match",
    [
        (["--m", "0"], "m must be positive"),
        (["--m", "33"], "divisible"),
        (["--world-size", "0"], "world_size"),
        (["--rank", "2"], "rank"),
        (["--block-num", "0"], "block_num"),
        (["--dtype-a", "bf16"], "same dtype"),
    ],
)
def test_invalid_cli_inputs(example, monkeypatch, options, match):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    args = example._argument_parser().parse_args(options)
    with pytest.raises(ValueError, match=match):
        example._compile_only_args(args=args)


def test_torchrun_context_overrides_cli(example, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")
    args = example._argument_parser().parse_args([])
    assert example._launch_context(args) == (3, 4, 1)
