"""Host-only hard-sync launcher isolation and residency contracts."""

import threading
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import catccos  # noqa: F401

pytest.importorskip("catlass._mlir.ir")
from catlass.execution import TlaUnsupportedAbiError

from catccos.tla_dist import _sync_runtime, compiler


def executor(monkeypatch, *, limit=4):
    obj = compiler._SyncExecutor.__new__(compiler._SyncExecutor)
    obj.device = 0
    obj.jit_module = SimpleNamespace(is_mixed=False)
    obj.function_handle = 10
    monkeypatch.setattr(
        _sync_runtime, "resident_block_limit", lambda device, paired: limit
    )
    return obj


@pytest.mark.parametrize("blocks", [0, -1, True, 1.5, 5])
def test_residency_or_invalid_block_count_prevents_dispatch(monkeypatch, blocks):
    obj = executor(monkeypatch)
    parent = Mock()
    monkeypatch.setattr(compiler.JitExecutor, "__call__", parent)
    with pytest.raises(TlaUnsupportedAbiError):
        obj(block_num=blocks)
    parent.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_exact_handle_dispatch_and_exception_restore(monkeypatch, fail):
    obj = executor(monkeypatch)
    regular = Mock()
    sync = Mock()
    monkeypatch.setattr(compiler._execution, "launch_kernel", regular)
    monkeypatch.setattr(_sync_runtime, "launch_sync_kernel", sync)

    def call(self, *, block_num):
        compiler._execution.launch_kernel(function_handle=11)
        compiler._execution.launch_kernel(function_handle=10)
        if fail:
            raise RuntimeError("launcher failure")
        return block_num

    monkeypatch.setattr(compiler.JitExecutor, "__call__", call)
    if fail:
        with pytest.raises(RuntimeError, match="launcher failure"):
            obj(block_num=2)
    else:
        assert obj(block_num=2) == 2
    regular.assert_called_once_with(function_handle=11)
    sync.assert_called_once_with(function_handle=10)
    assert compiler._execution.launch_kernel is regular


def test_single_vector_handoff_retains_logical_entry_and_one_side(monkeypatch):
    @dataclass
    class Module:
        kernel_mode: str = "mix_aiv"
        is_mixed: bool = False
        entrypoint: str = "logical"

    obj = compiler.SyncCompiledFunction.__new__(compiler.SyncCompiledFunction)
    obj.jit_module = Module()
    obj._executor = None
    obj._executor_lock = threading.Lock()
    factory = Mock()
    monkeypatch.setattr(compiler, "_SyncExecutor", factory)
    assert obj._get_executor() is factory.return_value
    native = factory.call_args.args[0]
    assert native.kernel_mode == "mix"
    assert native.entrypoint == "logical"
    assert native.is_mixed is False
    assert obj.jit_module.kernel_mode == "mix_aiv"
    assert obj._get_executor() is factory.return_value
    factory.assert_called_once()


def test_default_arch_mixed_cache_includes_actual_aic_template(monkeypatch, tmp_path):
    @dataclass
    class Options:
        kernel_mode: str = "aiv"

    aic = tmp_path / "meta_op.mix_aic.c310.bc"
    aiv = tmp_path / "meta_op.mix_aiv.c310.bc"
    aic.write_bytes(b"aic-v1")
    aiv.write_bytes(b"aiv-v1")
    monkeypatch.setattr(
        compiler._execution, "_cache_key", lambda **kw: kw["tlair_mlir"]
    )
    monkeypatch.setattr(
        compiler,
        "_resolve_base_bitcode",
        lambda runtime: f"{aic},{aiv}" if runtime.kernel_mode == "mix" else str(aiv),
    )
    with compiler._catccos_execution_hooks():
        kwargs = {
            "compile_option": Options(),
            "tlair_mlir": '"tla.cube" "tla.vector" "tla_dist.local_sync"',
        }
        first = compiler._execution._cache_key(**kwargs)
        aic.write_bytes(b"aic-v2")
        second = compiler._execution._cache_key(**kwargs)
    assert first != second
