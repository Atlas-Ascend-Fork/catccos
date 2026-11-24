# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Read and cross-check RDMA settings from an existing SHMEM build."""

import argparse
import json
import shlex
import sys
from pathlib import Path


def read_rdma_config(shmem_root: Path) -> dict[str, str | None]:
    cache_path = shmem_root / "build/CMakeCache.txt"
    cache = {}
    for line in cache_path.read_text().splitlines():
        if not line or line.startswith(("#", "//")):
            continue
        key, separator, value = line.partition("=")
        if separator:
            cache[key.partition(":")[0]] = value

    support = cache.get("ACLSHMEM_RDMA_SUPPORT", "").upper()
    if support not in ("ON", "OFF", "TRUE", "FALSE", "YES", "NO", "1", "0"):
        raise ValueError(f"{cache_path}: missing or invalid ACLSHMEM_RDMA_SUPPORT")
    if cache.get("ACLSHMEM_RELAY_SUPPORT", "OFF").upper() not in ("OFF", "FALSE", "NO", "0"):
        raise ValueError(f"{cache_path}: UDMA QP APIs require ACLSHMEM_RELAY_SUPPORT=OFF")
    soc = cache.get("SOC_TYPE")
    if not soc:
        raise ValueError(f"{cache_path}: missing SOC_TYPE")
    backend = cache.get("ACLSHMEM_RDMA_BACKEND", "").upper()
    if support in ("OFF", "FALSE", "NO", "0"):
        if backend:
            raise ValueError(f"{cache_path}: ACLSHMEM_RDMA_BACKEND is set while ACLSHMEM_RDMA_SUPPORT is disabled")
        backend = "NONE"
    elif soc != "Ascend950" or backend not in ("XSCALE", "HNS_1825"):
        raise ValueError(
            f"{cache_path}: unsupported RDMA configuration SOC_TYPE={soc}, ACLSHMEM_RDMA_BACKEND={backend}"
        )

    database = shmem_root / "build/compile_commands.json"
    entries = json.loads(database.read_text())
    device_root = (shmem_root / "src/device").resolve()
    expected = {} if backend == "NONE" else {f"ACLSHMEMI_RDMA_K_BACKEND_{backend}": "1"}
    modes = set()
    for entry in entries:
        source = (Path(entry["directory"]) / entry["file"]).resolve()
        if not source.is_relative_to(device_root):
            continue
        args = entry.get("arguments")
        if args is None:
            args = shlex.split(entry["command"])
        definitions = {}
        for index, arg in enumerate(args):
            if arg.startswith("-D"):
                definition = args[index + 1] if arg == "-D" else arg[2:]
                name, separator, value = definition.partition("=")
                definitions[name] = value if separator else "1"
        if "ACLSHMEM_RELAY_SUPPORT" in definitions:
            raise ValueError(f"{database}: {source}: UDMA QP APIs require ACLSHMEM_RELAY_SUPPORT=OFF")
        actual = {key: value for key, value in definitions.items() if key.startswith("ACLSHMEMI_RDMA_K_BACKEND_")}
        if actual != expected:
            raise ValueError(
                f"{database}: {source}: RDMA backend definitions {actual} disagree with {cache_path} ({backend})"
            )
        mode = definitions.get("ACLSHMEMI_HNS_1825_PI_ON_CHIP")
        if backend == "HNS_1825":
            if mode not in ("0", "1"):
                raise ValueError(f"{database}: {source}: must define ACLSHMEMI_HNS_1825_PI_ON_CHIP=0 or 1")
            cached_mode = cache.get("ACLSHMEMI_HNS_1825_PI_ON_CHIP")
            if cached_mode is not None and mode != cached_mode:
                raise ValueError(
                    f"{database}: {source}: ACLSHMEMI_HNS_1825_PI_ON_CHIP={mode} "
                    f"disagrees with {cache_path} ({cached_mode})"
                )
        elif mode is not None:
            raise ValueError(f"{database}: {source}: ACLSHMEMI_HNS_1825_PI_ON_CHIP requires the HNS_1825 backend")
        modes.add(mode)
    if not modes:
        raise ValueError(f"{database}: no device compilation commands found under {device_root}")
    if len(modes) != 1:
        raise ValueError(f"{database}: inconsistent ACLSHMEMI_HNS_1825_PI_ON_CHIP across device commands")
    return {"rdma_backend": backend, "pi_on_chip": modes.pop()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shmem_root", type=Path)
    args = parser.parse_args()
    try:
        config = read_rdma_config(args.shmem_root)
    except FileNotFoundError as error:
        print(
            f"Missing SHMEM build configuration: {error.filename}; building CATCCOS DSL without RDMA.",
            file=sys.stderr,
        )
        config = {"rdma_backend": "NONE", "pi_on_chip": None}
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        parser.exit(1, f"{error}\nCheck the SHMEM build configuration under {args.shmem_root / 'build'}.\n")
    print(json.dumps(config))
