# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Remote-copy parameters and the upstream compute parameter surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any

from catlass import params as _params
from catlass.core_api import IndexLike
from catlass.params import AtomicMode


@unique
class CoreScope(Enum):
    AIV = "aiv"
    MIX = "mix"


@unique
class SignalOp(Enum):
    SET = "set"
    ADD = "add"


@unique
class CompareOp(Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"


TEAM_WORLD = 0


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
    """Parameters for ``tla_dist.remote_copy`` and ``tla_dist.quiet``.

    GET reads peer ``src`` into local ``dst``; PUT writes local ``src`` to
    peer ``dst``. GM->GM requires an explicit mode. ``mode=None`` leaves
    the mode unspecified, selecting GET for GM->UB and PUT for UB->GM.
    An explicit mode must match the GM<->UB route.
    RDMA/UDMA GM->GM requests complete through quiet. The compile-time
    ``event_id`` is passed separately to ``remote_copy`` and ``quiet``.
    """

    mode: CopyMode | None = None
    atomic_mode: AtomicMode = AtomicMode.NONE
    transport: CopyTransport = CopyTransport.MTE
    wqe_pipe: CopyWqePipe | None = None
    # Runtime DSL integers are supported for RDMA/UDMA; MTE ignores this field.
    qp_idx: IndexLike = 0

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


__all__ = [
    "CoreScope",
    "SignalOp",
    "CompareOp",
    "TEAM_WORLD",
    "AtomicMode",
    "CopyMode",
    "CopyParams",
    "CopyTransport",
    "CopyWqePipe",
]
