"""Remote-copy parameters and the upstream compute parameter surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from catlass import params as _params
from catlass.params import AtomicMode


class CopyMode(str, Enum):
    PUT = "put"
    GET = "get"


class CopyTransport(str, Enum):
    MTE = "mte"
    RDMA = "rdma"
    UDMA = "udma"
    SDMA = "sdma"


class CopyWqePipe(str, Enum):
    MTE3 = "mte3"
    S = "s"


@dataclass
class CopyParams(_params.CopyParams):
    """Compile-time settings for ``tla_dist.remote_copy``.

    GET reads peer ``src`` into local ``dst``; PUT writes local ``src`` to
    peer ``dst``. GM->GM requires an explicit mode. ``mode=None`` leaves
    the mode unspecified, selecting GET for GM->UB and PUT for UB->GM.
    An explicit mode must match the GM<->UB route.
    Only MTE has device lowering. Event ids belong to the
    ``tla.flag`` passed to ``remote_copy``, rather than to these parameters.
    """

    mode: CopyMode | None = None
    atomic_mode: AtomicMode = AtomicMode.NONE
    transport: CopyTransport = CopyTransport.MTE
    wqe_pipe: CopyWqePipe | None = None

    def _validate(self) -> None:
        if self.mode is not None and self.mode not in (CopyMode.PUT, CopyMode.GET):
            raise ValueError("CopyParams.mode must be None, 'put', or 'get'")
        if self.transport not in tuple(CopyTransport):
            raise ValueError("CopyParams.transport must be 'mte', 'rdma', 'udma', or 'sdma'")
        if self.atomic_mode not in (AtomicMode.NONE, AtomicMode.ADD):
            raise ValueError("CopyParams.atomic_mode must be AtomicMode.NONE or AtomicMode.ADD")
        if self.wqe_pipe is not None:
            if self.transport != CopyTransport.UDMA:
                raise ValueError("CopyParams.wqe_pipe is only supported for UDMA")
            if self.wqe_pipe not in tuple(CopyWqePipe):
                raise ValueError("CopyParams.wqe_pipe must be 'mte3' or 's'")
        if self.atomic_mode == AtomicMode.ADD and self.transport != CopyTransport.MTE:
            raise ValueError("atomic ADD is only supported for MTE")


def __getattr__(name: str) -> Any:
    return getattr(_params, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_params)))


__all__ = ["AtomicMode", "CopyMode", "CopyParams", "CopyTransport", "CopyWqePipe"]
