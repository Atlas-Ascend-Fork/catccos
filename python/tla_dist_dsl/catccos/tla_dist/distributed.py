"""Optional ACLSHMEM host runtime for CATCCOS TLA_DIST kernels."""

from __future__ import annotations

import ctypes
import importlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any


class ShmemUnavailableError(RuntimeError):
    """Raised when the ACLSHMEM Python package cannot be loaded."""


class ShmemRuntimeError(RuntimeError):
    """Raised when an ACLSHMEM runtime contract is violated."""


def _load_shmem() -> Any:
    try:
        return importlib.import_module("shmem")
    except (ImportError, OSError, RuntimeError) as exc:
        raise ShmemUnavailableError(
            "ACLSHMEM Python support is unavailable; build/install CATCCOS "
            "3rdparty/shmem and verify it with 'shmem-config --diagnose'."
        ) from exc


def get_unique_id() -> Any:
    shmem = _load_shmem()
    return shmem.aclshmem_get_unique_id()


def configure_store_tls(enabled: bool, tls_info: str = "") -> None:
    shmem = _load_shmem()
    result = shmem.set_conf_store_tls(bool(enabled), str(tls_info))
    if result not in (None, 0):
        raise ShmemRuntimeError(f"set_conf_store_tls failed with status {result}")


@dataclass
class ShmemRuntime:
    """Own one rank's ACLSHMEM lifecycle and symmetric allocations."""

    uid: Any
    rank: int
    world_size: int
    heap_size: int
    device_id: int = 0
    _shmem: Any | None = field(default=None, init=False, repr=False)
    _allocations: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _initialized: bool = field(default=False, init=False)
    _host_barrier: Any | None = field(default=None, init=False, repr=False)
    _stream_barrier: Any | None = field(default=None, init=False, repr=False)
    _barrier_library: Any | None = field(default=None, init=False, repr=False)

    def initialize(self) -> ShmemRuntime:
        if self._initialized:
            raise ShmemRuntimeError("ACLSHMEM runtime is already initialized")
        if int(self.world_size) <= 0 or int(self.heap_size) <= 0:
            raise ValueError("world_size and heap_size must be positive")
        if not 0 <= int(self.rank) < int(self.world_size):
            raise ValueError(f"rank must be in [0, world_size), got {self.rank}/{self.world_size}")
        shmem = _load_shmem()
        result = shmem.aclshmem_init_using_unique_id(
            int(self.rank), int(self.world_size), int(self.heap_size), self.uid
        )
        if result not in (None, 0):
            raise ShmemRuntimeError(f"aclshmem_init_using_unique_id failed with status {result}")
        self._shmem = shmem
        self._initialized = True
        return self

    def _check_initialized(self) -> None:
        if not self._initialized or self._shmem is None:
            raise ShmemRuntimeError("initialize ACLSHMEM before using the runtime")

    def _require_initialized(self) -> Any:
        self._check_initialized()
        return self._shmem

    def barrier_all(self) -> None:
        """Synchronize initialized ranks through ACLSHMEM's host barrier."""
        self._check_initialized()
        if self._host_barrier is None:
            # The pinned SHMEM package has no Python barrier binding. Resolve
            # the native API through the loaded extension's dependencies so it
            # shares the same SHMEM library as initialization.
            library = self._barrier_library
            try:
                if library is None:
                    extension = importlib.import_module("shmem._pyshmem")
                    library = ctypes.CDLL(extension.__file__)
                barrier = library.aclshmem_barrier_all
            except (ImportError, OSError, AttributeError) as exc:
                raise ShmemUnavailableError("ACLSHMEM host barrier is unavailable") from exc
            barrier.argtypes = []
            barrier.restype = None  # native host API returns void
            self._barrier_library = library
            self._host_barrier = barrier
        self._host_barrier()

    def barrier_all_on_stream(self, stream: Any) -> None:
        """Queue a world barrier on an ACL stream without synchronizing the host."""
        self._check_initialized()
        stream_handle = int(getattr(stream, "npu_stream", stream))
        if stream_handle <= 0:
            raise ValueError("stream must have a positive ACL stream handle")
        if self._stream_barrier is None:
            library = self._barrier_library
            try:
                if library is None:
                    extension = importlib.import_module("shmem._pyshmem")
                    library = ctypes.CDLL(extension.__file__)
                barrier = library.aclshmemx_barrier_all_on_stream
            except (ImportError, OSError, AttributeError) as exc:
                raise ShmemUnavailableError("ACLSHMEM stream barrier is unavailable") from exc
            barrier.argtypes = [ctypes.c_void_p]
            barrier.restype = None
            self._barrier_library = library
            self._stream_barrier = barrier
        self._stream_barrier(stream_handle)

    @property
    def ffts_config(self) -> int:
        shmem = self._require_initialized()
        return int(shmem.get_ffts_config())

    def symmetric_tensor(self, shape: Iterable[int], dtype: Any) -> Any:
        shmem = self._require_initialized()
        normalized = tuple(int(dim) for dim in shape)
        if not normalized or any(dim <= 0 for dim in normalized):
            raise ValueError(f"symmetric tensor shape must be positive: {normalized}")
        tensor = shmem.aclshmem_create_tensor(normalized, dtype=dtype, device_id=int(self.device_id))
        self._allocations[id(tensor)] = tensor
        return tensor

    def free(self, tensor: Any) -> None:
        shmem = self._require_initialized()
        if self._allocations.get(id(tensor)) is not tensor:
            raise ShmemRuntimeError("tensor is not owned by this ACLSHMEM runtime")
        result = shmem.aclshmem_free_tensor(tensor)
        if result not in (None, 0):
            raise ShmemRuntimeError(f"symmetric tensor free failed with status {result}")
        del self._allocations[id(tensor)]

    def finalize(self) -> None:
        shmem = self._require_initialized()
        errors: list[BaseException] = []
        for tensor in reversed(tuple(self._allocations.values())):
            try:
                self.free(tensor)
            except Exception as exc:  # noqa: BLE001 - finalize remaining allocations
                errors.append(exc)
        try:
            result = shmem.aclshmem_finalize()
            if result not in (None, 0):
                raise ShmemRuntimeError(f"ACLSHMEM finalize failed with status {result}")
        finally:
            self._allocations.clear()
            self._initialized = False
            self._shmem = None
            self._host_barrier = None
            self._stream_barrier = None
            self._barrier_library = None
        if errors:
            raise ShmemRuntimeError(f"failed to free {len(errors)} symmetric allocation(s)") from errors[0]

    def __enter__(self) -> ShmemRuntime:  # noqa: PYI034 - returns initialize() result
        return self.initialize()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc, traceback
        self.finalize()
        return False


__all__ = [
    "ShmemRuntime",
    "ShmemRuntimeError",
    "ShmemUnavailableError",
    "configure_store_tls",
    "get_unique_id",
]
