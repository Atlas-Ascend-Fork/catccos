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
quiet = _comm.quiet
remote_copy = _comm.remote_copy
shmem_ptr = _comm.shmem_ptr
local_sync = _comm.local_sync
sync = _comm.sync
barrier = _comm.barrier
fence = _comm.fence
signal = _comm.signal
wait_until = _comm.wait_until
CoreScope = params.CoreScope
SignalOp = params.SignalOp
CompareOp = params.CompareOp
TEAM_WORLD = params.TEAM_WORLD
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
    "CoreScope",
    "SignalOp",
    "CompareOp",
    "TEAM_WORLD",
    "local_sync",
    "sync",
    "barrier",
    "fence",
    "signal",
    "wait_until",

    "__version__",
    "barrier_all",
    "compile",
    "distributed",
    "params",
    "quiet",
    "remote_copy",
    "shmem_ptr",
]


def __getattr__(name: str) -> Any:
    if name == "comm":
        raise AttributeError("module 'catccos.tla_dist' has no attribute 'comm'")
    upstream_name = _CATLASS_COMPAT_EXPORTS.get(name, name)
    return getattr(_catlass, upstream_name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(dir(_catlass)))
