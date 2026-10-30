"""CATCCOS communication-enabled downstream TLA DSL.

The compute/type surface is re-exported from the pinned ``catlass`` package.
Only communication operations, compilation, device bitcode and SHMEM runtime
ownership live in CATCCOS.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import catlass.tla as _catlass

# Import CATCCOS-owned modules by their fully qualified names so upstream
# exports cannot shadow the private implementation modules.
_version = import_module("._version", __name__)
_comm = import_module("._comm", __name__)
_compiler = import_module(".compiler", __name__)
distributed = import_module(".distributed", __name__)
params = import_module(".params", __name__)

__version__ = _version.__version__
barrier_all = _comm.barrier_all
remote_copy = _comm.remote_copy
CompileCallable = _compiler.CompileCallable
compile = _compiler.compile

_BASE_EXPORTS = tuple(name for name in getattr(_catlass, "__all__", ()) if name != "comm")
_CATLASS_COMPAT_EXPORTS = {
    # Keep the downstream dtype aliases independent of upstream lazy exports.
    "fp16": "Float16",
    "bf16": "BFloat16",
    "fp32": "Float32",
}
__all__ = [  # noqa: PLE0604 - the pinned upstream export list is intentional
    *_BASE_EXPORTS,
    "CompileCallable",
    "__version__",
    "barrier_all",
    "compile",
    "distributed",
    "params",
    "remote_copy",
]


def __getattr__(name: str) -> Any:
    if name == "comm":
        raise AttributeError("module 'catccos.tla_dist' has no attribute 'comm'")
    upstream_name = _CATLASS_COMPAT_EXPORTS.get(name, name)
    return getattr(_catlass, upstream_name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(dir(_catlass)))
