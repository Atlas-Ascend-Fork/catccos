"""Read launch signatures from real MLIR modules without compiling device code."""

from unittest.mock import Mock

import catccos  # noqa: F401
import pytest

# Native-dependent imports must follow the optional MLIR availability check.
mlir_ir = pytest.importorskip("catlass._mlir.ir")
from catccos.tla_dist import compiler, compiler_bridge  # noqa: E402
from catlass import _tla_type_bridge  # noqa: E402


@pytest.fixture
def mlir_context():
    with mlir_ir.Context() as context, mlir_ir.Location.unknown():
        context.allow_unregistered_dialects = True
        _tla_type_bridge.load_tla_dialect(context)
        yield context


@pytest.mark.parametrize("generic,entrypoint", [(False, "kernel"), (True, "kernel"), (True, "kernel with spaces")])
def test_module_signature_ignores_comments_and_other_operations(mlir_context, generic, entrypoint):
    module = mlir_ir.Module.parse(
        f"""
        // tla.func @comment_only(%x: i64)
        !payload = !tla.ptr<f32, gm, 16>
        module {{
          "test.decoy"() {{sym_name = "decoy", function_type = (i64) -> ()}} : () -> ()
          tla.func @"{entrypoint}"(%buffer: !payload, %count: i32) {{ tla.return }}
        }}
        """
    )
    module = mlir_ir.Module.parse(module.operation.get_asm(print_generic_op_form=generic, enable_debug_info=True))

    abi = compiler_bridge.build_kernel_abi(module)

    assert abi.entrypoint == entrypoint
    assert abi.schema_version == 3
    assert [argument.kind.value for argument in abi.arguments] == ["pointer", "scalar"]
    assert [argument.offset for argument in abi.arguments] == [0, 8]
    assert abi.arguments[1].scalar.bit_width == 32
    assert abi.total_size == 16


def test_module_signature_preserves_dynamic_memref_and_origin_arguments(mlir_context):
    module = mlir_ir.Module.parse(
        """
        !payload = memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<gm>>
        module {
          tla.func @kernel(%buffer: !payload, %row: index, %column: index, %count: i32) { tla.return }
        }
        """
    )

    abi = compiler_bridge.build_kernel_abi(module)

    assert abi.schema_version == 4
    assert len(abi.arguments) == 14
    assert [argument.logical_index for argument in abi.arguments] == [0] * 13 + [1]
    assert [argument.field for argument in abi.arguments[:3]] == ["allocated", "aligned", "offset"]
    assert [argument.field for argument in abi.arguments[11:13]] == ["originShape0", "originShape1"]
    assert abi.arguments[-1].offset == 104
    assert abi.total_size == 112


def test_module_without_entrypoint_is_rejected(mlir_context):
    module = mlir_ir.Module.parse('module { "test.decoy"() {sym_name = "decoy"} : () -> () }')

    with pytest.raises(compiler_bridge.TlaKernelCompileError, match="no tla.func entrypoint"):
        compiler_bridge.build_kernel_abi(module)


def test_execution_hook_passes_module_abi_and_preprocessed_text(mlir_context, monkeypatch, tmp_path):
    module = mlir_ir.Module.parse('module { tla.func @kernel(%count: i32) { tla.return } }')
    runtime = compiler._execution.TlaCompileOption(
        cce_disable_asc_reserved_ubuf=True,
        cce_disable_vf_stack_reserved_ubuf=True,
    )
    cli = Mock()
    monkeypatch.setattr(compiler, "lower_tlair_to_mlir_cli", cli)
    output = tmp_path / "lowered.mlir"

    with compiler._catccos_execution_hooks():
        result = compiler._execution._run_tla_lowering_to_mlir(
            lowered_module=module,
            tlair_mlir="stale text",
            mlir_path=output,
            compile_option=runtime,
        )

    assert result is cli.return_value
    cli.assert_called_once()
    call = cli.call_args.kwargs
    assert call["kernel_abi"].entrypoint == "kernel"
    assert len(call["kernel_abi"].arguments) == 1
    assert call["kernel_abi"].arguments[0].scalar.bit_width == 32
    assert "tla.ub_reserve_released" in call["tlair_mlir"]
    assert call["mlir_path"] == output
    assert call["compile_option"] is runtime
