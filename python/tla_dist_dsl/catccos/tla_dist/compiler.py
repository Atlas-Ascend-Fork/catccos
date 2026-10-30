"""Patchless CATCCOS compiler entry layered over ``catlass.execution``."""

from __future__ import annotations

import hashlib
import inspect
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import catlass.execution as _execution
from catlass.base_dsl.jit_executor import JitCompiledFunction
from catlass.dsl import TlaJitFunction, _bind_kernel_call_args
from catlass.execution import TlaRuntimeUnavailableError, TlaUnsupportedAbiError

from .compiler_bridge import bridge_path, build_kernel_abi, lower_tlair_to_mlir_cli

_COMPILE_LOCK = threading.RLock()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_options_from_kwargs(kwargs: dict[str, Any]) -> Any:
    """Preserve CATCCOS's explicit compile controls on Catlass."""
    runtime = _execution.compile_option_from_kwargs(kwargs)
    updates: dict[str, Any] = {}
    if "cache" in kwargs:
        updates["cache_enabled"] = bool(kwargs["cache"])
    if "cache_dir" in kwargs and kwargs["cache_dir"] is not None:
        updates["cache_dir"] = Path(kwargs["cache_dir"]).expanduser().resolve()
    if "force_recompile" in kwargs:
        updates["force_recompile"] = bool(kwargs["force_recompile"])
    if "arch_scope" in kwargs:
        arch_scope = str(kwargs["arch_scope"]).lower()
        updates["arch_scope"] = arch_scope
        updates["kernel_mode"] = "aic" if arch_scope.startswith("aic") else "aiv"
    if any(
        bool(kwargs.get(name))
        for name in (
            "mlir_print_ir_before",
            "mlir_print_ir_after",
            "mlir_print_ir_before_all",
            "mlir_print_ir_after_all",
        )
    ):
        updates["print_ir"] = True
    return replace(runtime, **updates) if updates else runtime


def _resolve_base_bitcode(runtime: Any) -> str:
    """Resolve TLA_DIST templates with SHMEM already linked into the AIV image."""
    explicit = os.getenv("TLA_DIST_DSL_BASE_BITCODE")
    if explicit:
        paths = [Path(item).expanduser().resolve() for item in explicit.split(",")]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise TlaRuntimeUnavailableError(
                "TLA_DIST_DSL_BASE_BITCODE contains missing files: " + ", ".join(str(path) for path in missing)
            )
        return ",".join(str(path) for path in paths)

    package_dir = Path(__file__).resolve().parent
    build_dir = _project_root() / "csrc" / "mlir" / "build"
    package_dirs = (
        package_dir / "bc",
        build_dir / "python" / "catccos" / "tla_dist" / "bc",
    )

    def find_packaged(name: str) -> Path | None:
        for root in package_dirs:
            candidate = root / name
            if candidate.is_file():
                return candidate.resolve()
        return None

    aic = find_packaged("meta_op.aic.c310.bc")
    # A mixed kernel must never fall back to the pure-AIV soft barrier image.
    aiv_kind = "mix_aiv" if runtime.kernel_mode == "mix" else "aiv"
    aiv = find_packaged(f"meta_op.{aiv_kind}.c310.bc")
    if aic is None:
        raw_aic = build_dir / "tla" / "bc" / "meta_op.aic.c310.bc"
        if raw_aic.is_file():
            aic = raw_aic.resolve()
    if aiv is None:
        raw_aiv = build_dir / "bc" / f"meta_op.tla_dist.{aiv_kind}.c310.bc"
        if raw_aiv.is_file():
            aiv = raw_aiv.resolve()

    if runtime.kernel_mode == "mix":
        if aic is not None and aiv is not None:
            return f"{aic},{aiv}"
    else:
        bitcode = aic if str(runtime.arch_scope).startswith("aic") else aiv
        if bitcode is not None:
            return str(bitcode)

    raise TlaRuntimeUnavailableError(
        "C310 base TLA bitcode was not found in the TLA_DIST build/package tree. "
        f"Required AIV template: meta_op.{aiv_kind}.c310.bc. "
        "Run python/tla_dist_dsl/build.sh or set TLA_DIST_DSL_BASE_BITCODE."
    )


def _bitcode_fingerprint(paths: str) -> str:
    digest = hashlib.sha256()
    for value in paths.split(","):
        path = Path(value)
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _catccos_execution_hooks() -> Iterator[None]:
    original_lower_driver = _execution._run_tla_lowering_to_mlir
    original_bridge_path = _execution.resolve_bridge_extension_path
    original_bitcode = _execution._resolve_hivm_template_bitcode
    original_cache_key = _execution._cache_key

    def lower_driver(
        *,
        lowered_module: Any | None,
        tlair_mlir: str,
        mlir_path: Path,
        compile_option: Any | None = None,
    ) -> Any:
        """Use the live module; ``tlair_mlir`` is retained for Catlass hook compatibility."""
        if lowered_module is None:
            raise ValueError("TLA_DIST lowering requires a live MLIR module.")
        _execution._mark_ub_reserve_released(lowered_module, compile_option)
        # Keep the Python-owned MLIR module out of the TLA_DIST compiler process;
        # textual TLAIR is the complete subprocess boundary.
        module_ir = lowered_module.operation.get_asm(print_generic_op_form=True, enable_debug_info=True)
        return lower_tlair_to_mlir_cli(
            tlair_mlir=module_ir,
            kernel_abi=build_kernel_abi(lowered_module),
            mlir_path=mlir_path,
            compile_option=compile_option,
        )

    def resolve_bitcode(runtime: Any, *, variant: str = "normal") -> str:
        # Current TLA_DIST aggregates contain normal compute helpers and SHMEM,
        # but do not include Catlass's device print/debug helpers.
        if variant != "normal" and not os.getenv("TLA_DIST_DSL_BASE_BITCODE"):
            raise TlaRuntimeUnavailableError(
                f"TLA_DIST bitcode variant {variant!r} is not built. "
                "Device print/debug helpers require compatible bitcode supplied "
                "through TLA_DIST_DSL_BASE_BITCODE."
            )
        return _resolve_base_bitcode(runtime)

    def cache_key(**kwargs: Any) -> str:
        runtime = kwargs["compile_option"]
        # Cache lookup precedes mixed-function splitting. An initial AIC target
        # can become MIX, so include its mixed AIV template in the fingerprint.
        template_runtime = replace(runtime, kernel_mode="mix") if runtime.kernel_mode == "aic" else runtime
        template = _resolve_base_bitcode(template_runtime)
        kwargs["tlair_mlir"] = (
            f"{kwargs['tlair_mlir']}\n// tla_dist-template-bitcode-sha256: {_bitcode_fingerprint(template)}"
        )
        return original_cache_key(**kwargs)

    _execution._run_tla_lowering_to_mlir = lower_driver
    _execution.resolve_bridge_extension_path = bridge_path
    _execution._resolve_hivm_template_bitcode = resolve_bitcode
    _execution._cache_key = cache_key
    try:
        yield
    finally:
        _execution._run_tla_lowering_to_mlir = original_lower_driver
        _execution.resolve_bridge_extension_path = original_bridge_path
        _execution._resolve_hivm_template_bitcode = original_bitcode
        _execution._cache_key = original_cache_key


class CompileCallable:
    """Compile a kernel using either call arguments or explicit ``type_args``."""

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> JitCompiledFunction:
        if func is None:
            raise TlaUnsupportedAbiError("Function is not set or invalid.")
        if not isinstance(func, TlaJitFunction):
            raise TlaUnsupportedAbiError("tla_dist.compile expects a @tla_dist.kernel function.")

        type_args = kwargs.pop("type_args", None)
        if type_args is None:
            samples, kwargs = _bind_kernel_call_args(func.fn, args, kwargs)
            type_args = samples if samples else None
        elif args or kwargs.keys() & inspect.signature(func.fn).parameters.keys():
            raise TypeError("type_args cannot be combined with positional or keyword kernel arguments.")
        runtime = _runtime_options_from_kwargs(kwargs)
        with _COMPILE_LOCK, _catccos_execution_hooks():
            artifact = _execution.compile_and_cache(
                func.fn,
                kind=func.kind,
                options=func.options,
                compile_option=runtime,
                type_args=type_args,
                decorator_location=func.decorator_location,
            )
        return artifact


compile = CompileCallable()

__all__ = ["CompileCallable", "compile"]
