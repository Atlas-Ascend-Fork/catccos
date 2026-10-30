"""Compile argument binding without running the native compiler or an NPU."""

# Native-dependent imports must follow the importorskip check.
# ruff: noqa: E402

from contextlib import nullcontext
from unittest.mock import Mock

import catccos  # noqa: F401
import pytest

pytest.importorskip("catlass._mlir.ir")
from catccos import tla_dist
from catccos.tla_dist import compiler


@pytest.fixture
def compile_backend(monkeypatch):
    backend = Mock()
    monkeypatch.setattr(compiler._execution, "compile_and_cache", backend)
    monkeypatch.setattr(compiler, "_catccos_execution_hooks", nullcontext)
    return backend


@tla_dist.kernel
def kernel(a, n=16):
    pass


@pytest.mark.parametrize("type_args", [("b",), ()])
@pytest.mark.parametrize(
    "args,kwargs",
    [
        (("a",), {}),
        ((), {"a": "a"}),
        ((), {"n": 32}),
        (("a",), {"n": 32}),
    ],
)
def test_explicit_type_args_rejects_kernel_arguments(compile_backend, type_args, args, kwargs):
    with pytest.raises(TypeError, match="type_args cannot be combined"):
        assert tla_dist.compile(kernel, *args, **kwargs, type_args=type_args, cache=True)
    compile_backend.assert_not_called()


@pytest.mark.parametrize("type_args", [("b",), ()])
def test_explicit_type_args_allows_compile_options(compile_backend, type_args, tmp_path):
    artifact = tla_dist.compile(
        kernel,
        type_args=type_args,
        options="--npu-arch 3510",
        cache=False,
        cache_dir=tmp_path,
        force_recompile=True,
        arch_scope="aiv.c310",
        mlir_print_ir_after=True,
    )
    compile_backend.assert_called_once()
    assert artifact is compile_backend.return_value
    call = compile_backend.call_args
    assert call.args == (kernel.fn,)
    assert call.kwargs["type_args"] is type_args
    runtime = call.kwargs["compile_option"]
    assert not runtime.cache_enabled
    assert runtime.cache_dir == tmp_path.resolve()
    assert runtime.force_recompile
    assert runtime.arch_scope == "aiv.c310"
    assert runtime.print_ir


@pytest.mark.parametrize("type_args_kw", [{}, {"type_args": None}])
@pytest.mark.parametrize(
    "args,kwargs,expected",
    [
        (("a",), {"n": 32}, ("a", 32)),
        ((), {"a": "a", "n": 32}, ("a", 32)),
        (("a",), {}, ("a", 16)),
    ],
)
def test_call_arguments_are_bound(compile_backend, type_args_kw, args, kwargs, expected):
    artifact = tla_dist.compile(kernel, *args, **kwargs, **type_args_kw, cache=True)
    compile_backend.assert_called_once()
    assert artifact is compile_backend.return_value
    assert compile_backend.call_args.kwargs["type_args"] == expected
    assert compile_backend.call_args.kwargs["compile_option"].cache_enabled
