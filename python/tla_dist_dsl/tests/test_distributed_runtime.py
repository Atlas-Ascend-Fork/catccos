# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "catccos/tla_dist/distributed.py"


def _load_distributed() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tla_dist_distributed_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeShmem(ModuleType):
    def __init__(self) -> None:
        super().__init__("shmem")
        self.events: list[object] = []
        self.OpEngineType = SimpleNamespace(ROCE=4, UDMA=8)
        self.InitMode = SimpleNamespace(UNIQUEID=2)
        self.UniqueId = SimpleNamespace(from_bytes=lambda data: SimpleNamespace(data=data))
        self.InitAttr = lambda: SimpleNamespace(option_attr=SimpleNamespace(data_op_engine_type=1))
        self.qp_status = 0
        self.attr_status = 0
        self.init_status = 0

    def aclshmemx_set_attr_uniqueid_args(self, rank, world_size, heap_size, uid, attr):
        # SHMEM fills defaults here; the selected engine must be set afterwards.
        attr.option_attr.data_op_engine_type = 1
        self.events.append(("set_attr", rank, world_size, heap_size, uid, attr))
        return self.attr_status

    def aclshmemx_set_qp_num(self, engine, qp_num):
        self.events.append(("set_qp_num", engine, qp_num))
        return self.qp_status

    def aclshmemx_init_attr(self, mode, attr):
        self.events.append(("init_attr", mode, attr, attr.option_attr.data_op_engine_type))
        return self.init_status

    def aclshmem_barrier_all(self):
        self.events.append("barrier")

    def aclshmemx_barrier_all_on_stream(self, stream):
        self.events.append(("stream_barrier", stream))

    def aclshmem_init_using_unique_id(self, rank, world_size, heap_size, uid):
        self.events.append(("init", rank, world_size, heap_size, uid))
        return 0

    def get_ffts_config(self):
        return 1234

    def aclshmem_create_tensor(self, shape, *, dtype, device_id):
        tensor = object()
        self.events.append(("alloc", shape, dtype, device_id, tensor))
        return tensor

    def aclshmem_free_tensor(self, tensor):
        self.events.append(("free", tensor))
        return 0

    def aclshmem_finalize(self):
        self.events.append("finalize")
        return 0


def test_symmetric_allocations_are_freed_in_reverse_order(monkeypatch) -> None:
    distributed = _load_distributed()
    fake = FakeShmem()
    monkeypatch.setitem(sys.modules, "shmem", fake)
    runtime = distributed.ShmemRuntime(uid="uid", rank=0, world_size=2, heap_size=4096, device_id=0).initialize()
    first = runtime.symmetric_tensor((2, 3), "f32")
    second = runtime.symmetric_tensor((4,), "f16")
    assert runtime.ffts_config == 1234
    runtime.finalize()
    free_events = [event for event in fake.events if isinstance(event, tuple) and event[0] == "free"]
    assert free_events == [("free", second), ("free", first)]
    assert fake.events[-1] == "finalize"


@pytest.mark.parametrize("transport,engine", [("mte", None), ("rdma", 4), ("udma", 8)])
@pytest.mark.parametrize("qp_num", [1, 2])
def test_uid_initialization_selects_transport(monkeypatch, transport, engine, qp_num):
    distributed = _load_distributed()
    fake = FakeShmem()
    monkeypatch.setitem(sys.modules, "shmem", fake)
    uid = b"uid\x00with-null"
    runtime = distributed.ShmemRuntime(uid, 1, 3, 8192, transport=transport, qp_num=qp_num).initialize()
    if transport == "mte":
        assert fake.events == [("init", 1, 3, 8192, uid)]
    else:
        setup, qp_config, init = fake.events
        assert qp_config == ("set_qp_num", engine, qp_num)
        assert setup[:4] == ("set_attr", 1, 3, 8192)
        assert setup[4].data == uid
        assert init == ("init_attr", fake.InitMode.UNIQUEID, setup[5], engine)
    assert runtime._initialized and runtime._shmem is fake
    runtime.finalize()


@pytest.mark.parametrize(
    "method,args,event",
    [
        ("barrier_all", (), "barrier"),
        ("barrier_all_on_stream", (123,), ("stream_barrier", 123)),
    ],
)
def test_barriers_require_live_runtime_and_use_python_api(monkeypatch, method, args, event):
    distributed = _load_distributed()
    fake = FakeShmem()
    monkeypatch.setitem(sys.modules, "shmem", fake)
    runtime = distributed.ShmemRuntime(uid=b"uid", rank=0, world_size=2, heap_size=4096)
    barrier = getattr(runtime, method)
    with pytest.raises(distributed.ShmemRuntimeError, match="initialize"):
        barrier(*args)
    runtime.initialize()
    barrier(*args)
    barrier(*args)
    assert fake.events[-2:] == [event, event]
    runtime.finalize()
    with pytest.raises(distributed.ShmemRuntimeError, match="initialize"):
        barrier(*args)


def test_stream_barrier_queues_on_requested_stream(monkeypatch):
    distributed = _load_distributed()
    fake = FakeShmem()
    monkeypatch.setitem(sys.modules, "shmem", fake)
    runtime = distributed.ShmemRuntime(uid=b"uid", rank=0, world_size=2, heap_size=4096).initialize()
    with pytest.raises(ValueError, match="positive ACL stream handle"):
        runtime.barrier_all_on_stream(0)
    runtime.barrier_all_on_stream(SimpleNamespace(npu_stream=123))
    runtime.barrier_all_on_stream(456)
    assert fake.events[-2:] == [("stream_barrier", 123), ("stream_barrier", 456)]
    runtime.finalize()


@pytest.mark.parametrize("transport", ["rdma", "udma"])
@pytest.mark.parametrize("stage", ["attr", "qp", "init"])
@pytest.mark.parametrize("status", [-1, -6, -9])
def test_uid_initialization_failure_keeps_runtime_uninitialized(monkeypatch, transport, stage, status):
    distributed = _load_distributed()
    fake = FakeShmem()
    setattr(fake, f"{stage}_status", status)
    monkeypatch.setitem(sys.modules, "shmem", fake)
    runtime = distributed.ShmemRuntime(b"uid", 0, 2, 4096, transport=transport)
    message = {"attr": "attribute setup", "qp": "QP configuration", "init": "initialization"}[stage]
    with pytest.raises(distributed.ShmemRuntimeError, match=f"{message} failed with status {status}"):
        runtime.initialize()
    assert not runtime._initialized and runtime._shmem is None
    assert [event[0] for event in fake.events] == {
        "attr": ["set_attr"],
        "qp": ["set_attr", "set_qp_num"],
        "init": ["set_attr", "set_qp_num", "init_attr"],
    }[stage]
