#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#
import argparse
from pathlib import Path


def verify(output_path, golden_path):
    output = output_path.read_bytes()
    golden = golden_path.read_bytes()
    if not golden or len(output) != len(golden):
        raise RuntimeError(f"{output_path.name}: size={len(output)}, expected={len(golden)}")
    if output != golden:
        offset = next(i for i, (actual, expected) in enumerate(zip(output, golden)) if actual != expected)
        raise RuntimeError(
            f"{output_path.name}: byte={offset}, actual=0x{output[offset]:02x}, expected=0x{golden[offset]:02x}"
        )


def main():
    parser = argparse.ArgumentParser(description="Verify AllToAll data, scales and slot padding bit for bit")
    parser.add_argument("data_path", type=Path)
    parser.add_argument("rank_id", type=int)
    args = parser.parse_args()
    for name in ("", "scale_"):
        verify(args.data_path / f"output_{name}{args.rank_id}.bin", args.data_path / f"golden_{name}{args.rank_id}.bin")
    print(f"PASS rank={args.rank_id}")


if __name__ == "__main__":
    main()
