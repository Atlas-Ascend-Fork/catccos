"""C310 hard-barrier launch metadata, using the installed public ACL ABI."""
from __future__ import annotations
import ctypes
from typing import Any

# Exact CANN 9.2 acl_rt.h aclrtDevAttr and launch-attribute identifiers.
_CUBE_CORE_COUNT = 102
_VECTOR_CORE_COUNT = 201
_SCHEM_MODE = 1
_BATCH_SCHEDULE = 1


def resident_block_limit(device: int, *, paired: bool) -> int:
    """Fail closed if the runtime cannot establish simultaneous residency."""
    from catlass.execution import TlaRuntimeUnavailableError
    library = ctypes.CDLL("libascendcl.so")
    query = library.aclrtGetDeviceInfo
    query.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
    query.restype = ctypes.c_int

    def count(attribute: int) -> int:
        result = ctypes.c_int64()
        status = int(query(device, attribute, ctypes.byref(result)))
        if status != 0 or result.value <= 0:
            raise TlaRuntimeUnavailableError(f"cannot verify hard-sync core residency: aclrtGetDeviceInfo({attribute}) status={status}, value={result.value}")
        return int(result.value)
    vector = count(_VECTOR_CORE_COUNT)
    return min(count(_CUBE_CORE_COUNT), vector // 2) if paired else vector


def launch_sync_kernel(**kwargs: Any) -> None:
    """Use batch scheduling with the ordinary Catlass Host-argument ABI."""
    from catlass.base_dsl.runtime.ascend import _launch_config, check_acl_errors
    import acl
    config = _launch_config(kwargs.get("dyn_ubuf_bytes", 0))
    config.append({"id": _SCHEM_MODE, "value": _BATCH_SCHEDULE})
    stream = int(kwargs["stream"])
    blocks = int(kwargs["block_num"])

    def launch(payload: bytes) -> None:
        payload = payload or b"\x00" * ctypes.sizeof(ctypes.c_uint64)
        host = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        check_acl_errors(acl.rt.launch_kernel_with_host_args(int(kwargs["function_handle"]), blocks,
            stream, config, ctypes.addressof(host), len(payload), [], 0),
            "acl.rt.launch_kernel_with_host_args(batch hard-sync)")
    if kwargs.get("uses_scalar_print") or kwargs.get("uses_tensor_print"):
        from catlass.base_dsl.runtime.ascend_debug_fifo import launch_with_debug_fifo
        launch_with_debug_fifo(launch_kernel=launch, payload=kwargs["payload"], block_num=blocks,
            stream=stream, uses_scalar_print=kwargs.get("uses_scalar_print", False),
            uses_tensor_print=kwargs.get("uses_tensor_print", False), is_mixed=kwargs.get("is_mixed", False),
            print_tensor_position=kwargs.get("print_tensor_position"), device=int(kwargs.get("device_id") or 0))
    else:
        launch(kwargs["payload"])
