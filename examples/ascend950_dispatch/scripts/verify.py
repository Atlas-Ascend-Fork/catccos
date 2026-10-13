#!/usr/bin/env python3

import argparse
import math
import struct
from pathlib import Path


OUTPUTS = {
    "recv_x": ("<b", 1),
    "recv_x_scales": ("<f", 4),
    "recv_topk_idx": ("<q", 8),
    "recv_topk_weights": ("<f", 4),
    "recv_src_idx": ("<i", 4),
}


def main():
    parser = argparse.ArgumentParser(description="Verify Ascend950 Dispatch binary outputs")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("rank", type=int)
    args = parser.parse_args()

    passed = True
    for name, (value_format, value_bytes) in OUTPUTS.items():
        output_path = args.data_dir / f"output_{name}_{args.rank}.bin"
        golden_path = args.data_dir / f"golden_{name}_{args.rank}.bin"
        if not output_path.is_file() or not golden_path.is_file():
            print(f"[FAILED] rank {args.rank} {name}: missing output or golden file")
            passed = False
            continue
        actual = output_path.read_bytes()
        expected = golden_path.read_bytes()
        if len(actual) != len(expected):
            print(f"[FAILED] rank {args.rank} {name} bytes: actual={len(actual)} expected={len(expected)}")
            passed = False
            continue
        if actual == expected:
            continue
        if name == 'recv_x_scales' and all(
            math.isclose(a[0], b[0], rel_tol=1e-5, abs_tol=0.0)
            for a, b in zip(struct.iter_unpack('<f', actual), struct.iter_unpack('<f', expected))
        ):
            continue
        byte_index = next(index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])
        element_index = byte_index // value_bytes
        offset = element_index * value_bytes
        actual_value = struct.unpack_from(value_format, actual, offset)[0]
        expected_value = struct.unpack_from(value_format, expected, offset)[0]
        print(
            f"[FAILED] rank {args.rank} {name} at element {element_index}: "
            f"actual={actual_value} expected={expected_value}"
        )
        passed = False

    if not passed:
        raise SystemExit(1)
    print(f"[PASSED] rank {args.rank} Dispatch binary outputs")


if __name__ == "__main__":
    main()
