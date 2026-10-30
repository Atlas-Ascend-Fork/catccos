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


@pytest.mark.parametrize(
    "method,args,event",
    [
        ("barrier_all", (), "barrier"),
        ("barrier_all_on_stream", (123,), ("stream_barrier", 123)),
    ],
)
def test_barriers_require_live_runtime_and_use_native_api(monkeypatch, method, args, event):
    distributed = _load_distributed()
    fake = FakeShmem()
    monkeypatch.setitem(sys.modules, "shmem", fake)
    monkeypatch.setitem(sys.modules, "shmem._pyshmem", SimpleNamespace(__file__="/installed/_pyshmem.so"))
    monkeypatch.setattr(
        distributed.ctypes,
        "CDLL",
        lambda path: SimpleNamespace(
            aclshmem_barrier_all=lambda: fake.events.append("barrier"),
            aclshmemx_barrier_all_on_stream=lambda stream: fake.events.append(("stream_barrier", stream)),
        ),
    )
    runtime = distributed.ShmemRuntime(uid="uid", rank=0, world_size=2, heap_size=4096)
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


def test_host_barrier_resolves_same_extension_dependencies_once(monkeypatch):
    distributed = _load_distributed()
    fake = FakeShmem()
    monkeypatch.setitem(sys.modules, "shmem", fake)
    monkeypatch.setitem(sys.modules, "shmem._pyshmem", SimpleNamespace(__file__="/installed/_pyshmem.so"))
    calls = []

    def barrier():
        fake.events.append("barrier")

    def load(path):
        calls.append(path)
        return SimpleNamespace(
            aclshmem_barrier_all=barrier,
            aclshmemx_barrier_all_on_stream=lambda stream: fake.events.append(("stream_barrier", stream)),
        )

    monkeypatch.setattr(distributed.ctypes, "CDLL", load)
    runtime = distributed.ShmemRuntime(uid="uid", rank=0, world_size=2, heap_size=4096).initialize()
    runtime.barrier_all()
    runtime.barrier_all()
    runtime.barrier_all_on_stream(123)
    assert calls == ["/installed/_pyshmem.so"]
    assert barrier.argtypes == [] and barrier.restype is None
    assert fake.events[-3:] == ["barrier", "barrier", ("stream_barrier", 123)]
    runtime.finalize()
    assert runtime._barrier_library is None and runtime._host_barrier is None


@pytest.mark.parametrize(
    "method,args,kind",
    [("barrier_all", (), "host"), ("barrier_all_on_stream", (123,), "stream")],
)
@pytest.mark.parametrize("load_error", [False, True])
def test_barriers_report_unavailable_native_api(monkeypatch, method, args, kind, load_error):
    distributed = _load_distributed()
    fake = FakeShmem()
    monkeypatch.setitem(sys.modules, "shmem", fake)
    monkeypatch.setitem(sys.modules, "shmem._pyshmem", SimpleNamespace(__file__="/installed/_pyshmem.so"))

    def load(path):
        if load_error:
            raise OSError("cannot load SHMEM dependencies")
        return SimpleNamespace()

    monkeypatch.setattr(distributed.ctypes, "CDLL", load)
    runtime = distributed.ShmemRuntime(uid="uid", rank=0, world_size=2, heap_size=4096).initialize()
    with pytest.raises(distributed.ShmemUnavailableError, match=f"{kind} barrier is unavailable") as error:
        getattr(runtime, method)(*args)
    assert isinstance(error.value.__cause__, OSError if load_error else AttributeError)
    runtime.finalize()


def test_stream_barrier_queues_on_requested_stream(monkeypatch):
    distributed = _load_distributed()
    fake = FakeShmem()
    monkeypatch.setitem(sys.modules, "shmem", fake)
    monkeypatch.setitem(sys.modules, "shmem._pyshmem", SimpleNamespace(__file__="/installed/_pyshmem.so"))
    loaded = []

    def barrier(stream):
        fake.events.append(("stream_barrier", stream))

    def load(path):
        loaded.append(path)
        return SimpleNamespace(aclshmemx_barrier_all_on_stream=barrier)

    monkeypatch.setattr(distributed.ctypes, "CDLL", load)
    runtime = distributed.ShmemRuntime(uid="uid", rank=0, world_size=2, heap_size=4096).initialize()
    with pytest.raises(ValueError, match="positive ACL stream handle"):
        runtime.barrier_all_on_stream(0)
    runtime.barrier_all_on_stream(SimpleNamespace(npu_stream=123))
    runtime.barrier_all_on_stream(456)
    assert loaded == ["/installed/_pyshmem.so"]
    assert barrier.argtypes == [distributed.ctypes.c_void_p]
    assert barrier.restype is None
    assert fake.events[-2:] == [("stream_barrier", 123), ("stream_barrier", 456)]
    runtime.finalize()
    assert runtime._stream_barrier is None
