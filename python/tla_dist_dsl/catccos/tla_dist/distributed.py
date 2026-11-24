# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Optional ACLSHMEM host runtime for CATCCOS TLA_DIST kernels."""

from __future__ import annotations

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
    transport: str = "mte"
    qp_num: int = 1
    _shmem: Any | None = field(default=None, init=False, repr=False)
    _allocations: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _initialized: bool = field(default=False, init=False)

    def initialize(self) -> ShmemRuntime:
        if self._initialized:
            raise ShmemRuntimeError("ACLSHMEM runtime is already initialized")
        if int(self.world_size) <= 0 or int(self.heap_size) <= 0:
            raise ValueError("world_size and heap_size must be positive")
        if not 0 <= int(self.rank) < int(self.world_size):
            raise ValueError(f"rank must be in [0, world_size), got {self.rank}/{self.world_size}")
        shmem = _load_shmem()
        if self.transport not in ("mte", "rdma", "udma"):
            raise ValueError(f"unsupported SHMEM transport: {self.transport}")
        if self.transport == "mte":
            result = shmem.aclshmem_init_using_unique_id(
                int(self.rank), int(self.world_size), int(self.heap_size), self.uid
            )
        else:
            engine = shmem.OpEngineType.ROCE if self.transport == "rdma" else shmem.OpEngineType.UDMA
            uid = shmem.UniqueId.from_bytes(bytes(self.uid))
            attr = shmem.InitAttr()
            result = shmem.aclshmemx_set_attr_uniqueid_args(
                int(self.rank), int(self.world_size), int(self.heap_size), uid, attr
            )
            if result != 0:
                raise ShmemRuntimeError(f"ACLSHMEM UID attribute setup failed with status {result}")
            result = shmem.aclshmemx_set_qp_num(engine, int(self.qp_num))
            if result != 0:
                raise ShmemRuntimeError(f"ACLSHMEM QP configuration failed with status {result}")
            attr.option_attr.data_op_engine_type = engine
            result = shmem.aclshmemx_init_attr(shmem.InitMode.UNIQUEID, attr)
        if result not in (None, 0):
            raise ShmemRuntimeError(f"ACLSHMEM UID initialization failed with status {result}")
        self._shmem = shmem
        self._initialized = True
        return self

    def _require_initialized(self) -> Any:
        if not self._initialized or self._shmem is None:
            raise ShmemRuntimeError("initialize ACLSHMEM before using the runtime")
        return self._shmem

    def barrier_all(self) -> None:
        """Synchronize initialized ranks through ACLSHMEM's host barrier."""
        shmem = self._require_initialized()
        shmem.aclshmem_barrier_all()

    def barrier_all_on_stream(self, stream: Any) -> None:
        """Queue a world barrier on an ACL stream without synchronizing the host."""
        shmem = self._require_initialized()
        stream_handle = int(getattr(stream, "npu_stream", stream))
        if stream_handle <= 0:
            raise ValueError("stream must have a positive ACL stream handle")
        shmem.aclshmemx_barrier_all_on_stream(stream_handle)

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
