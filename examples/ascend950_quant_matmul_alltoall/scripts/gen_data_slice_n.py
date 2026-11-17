#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#
"""Fast input and golden generator for QuantMatmul AllToAll along N."""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np


BLOCK_SIZE = 32
E8M0_BIAS = 127
GEN_ROW_BLOCK = 8192
GOLDEN_MEMORY_CAP = 128 << 20

# E4M3 bytes and E2M1 nibbles for 0.5, 1.0, 1.5, and 2.0.
FP8_PATTERN_BYTES = np.array([0x30, 0x38, 0x3C, 0x40], dtype=np.uint8)
FP4_PATTERN_NIBBLES = np.array([0x1, 0x2, 0x3, 0x4], dtype=np.uint8)
PATTERN_VALUES = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)


def _log(message):
    print(f"[gen] {message}", flush=True)


def _groups(k):
    scale_groups = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
    return scale_groups, scale_groups + (scale_groups & 1)


def _e8m0_values(raw):
    return np.exp2(raw.astype(np.float32) - np.float32(E8M0_BIAS))


def _write_array(handle, array):
    np.ascontiguousarray(array).tofile(handle)


def _a_value_index(rows, groups, source, seed_phase):
    return (rows % 7 + groups + 3 * source + seed_phase) % 4


def _a_scale_raw(rows, groups, source, seed_phase):
    return (125 + (rows // 7 + groups + source + seed_phase) % 3).astype(np.uint8)


def _b_value_index(columns, groups, seed_phase):
    return (columns % 11 + 3 * groups + seed_phase) % 4


def _b_scale_raw(groups, columns, seed_phase):
    return np.broadcast_to(123 + ((groups + seed_phase) % 2), (groups.shape[0], columns.shape[1])).astype(np.uint8)


def _write_a(fmt, path, m, k, scale_groups, padded_groups, source, seed_phase):
    groups = np.arange(scale_groups, dtype=np.int64)[None, :]
    with path.open("wb") as data_file:
        for first in range(0, m, GEN_ROW_BLOCK):
            rows = np.arange(first, min(first + GEN_ROW_BLOCK, m), dtype=np.int64)[:, None]
            index = _a_value_index(rows, groups, source, seed_phase)
            if fmt == 0:
                block = np.repeat(FP8_PATTERN_BYTES[index], BLOCK_SIZE, axis=1)[:, :k]
            else:
                nibble = np.repeat(FP4_PATTERN_NIBBLES[index], BLOCK_SIZE, axis=1)[:, :k]
                block = np.bitwise_or(nibble[:, 0::2], np.left_shift(nibble[:, 1::2], 4))
            _write_array(data_file, block)

    with path.with_name(path.name.replace("_a.bin", "_a_scale.bin")).open("wb") as scale_file:
        for first in range(0, m, GEN_ROW_BLOCK):
            rows = np.arange(first, min(first + GEN_ROW_BLOCK, m), dtype=np.int64)[:, None]
            scale = np.full((rows.shape[0], padded_groups), 127, dtype=np.uint8)
            scale[:, :scale_groups] = _a_scale_raw(rows, groups, source, seed_phase)
            _write_array(scale_file, scale)


def _write_b_data(fmt, path, k, n, scale_groups, seed_phase):
    columns = np.arange(n, dtype=np.int64)
    with path.open("wb") as data_file:
        for group in range(scale_groups):
            index = _b_value_index(columns, np.int64(group), seed_phase)
            if fmt == 0:
                row = FP8_PATTERN_BYTES[index]
                block = np.broadcast_to(row, (BLOCK_SIZE, n))
            else:
                nibble = FP4_PATTERN_NIBBLES[index]
                packed = np.bitwise_or(nibble[0::2], np.left_shift(nibble[1::2], 4))
                block = np.broadcast_to(packed, (BLOCK_SIZE, n // 2))
            rows = min(BLOCK_SIZE, k - group * BLOCK_SIZE)
            _write_array(data_file, block[:rows])


def _write_b_scale(path, n, scale_groups, padded_groups, seed_phase):
    scale = np.full((padded_groups, n), 127, dtype=np.uint8)
    for group in range(scale_groups):
        scale[group, :] = 123 + (group + seed_phase) % 2
    # The kernel consumes scale groups as [pair][N][group-in-pair].
    scale = scale.reshape(padded_groups // 2, 2, n).transpose(0, 2, 1)
    np.ascontiguousarray(scale).tofile(path)


def _link_or_copy(source, destination):
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def _a_factors(rows, scale_groups, source, seed_phase):
    rows = rows[:, None]
    groups = np.arange(scale_groups, dtype=np.int64)[None, :]
    value = PATTERN_VALUES[_a_value_index(rows, groups, source, seed_phase)]
    scale = _e8m0_values(_a_scale_raw(rows, groups, source, seed_phase))
    return value * scale


def _build_b_factors(k, n, scale_groups, seed_phase):
    groups = np.arange(scale_groups, dtype=np.int64)[:, None]
    columns = np.arange(n, dtype=np.int64)[None, :]
    value = PATTERN_VALUES[_b_value_index(columns, groups, seed_phase)]
    scale = _e8m0_values(_b_scale_raw(groups, columns, seed_phase))
    return value * scale


def _create_golden_files(data_dir, rank_size, m, chunk_n):
    shape = (rank_size * m, chunk_n)
    return [
        np.memmap(data_dir / f"golden_{rank}.bin", dtype=np.float32, mode="w+", shape=shape)
        for rank in range(rank_size)
    ]


def _generate_fast_golden(data_dir, rank_size, m, n, k, seed_phase):
    chunk_n = n // rank_size
    scale_groups, _ = _groups(k)
    golden = _create_golden_files(data_dir, rank_size, m, chunk_n)
    b_factors = _build_b_factors(k, n, scale_groups, seed_phase)
    row_block = min(GEN_ROW_BLOCK, max(1, GOLDEN_MEMORY_CAP // max(4 * n, 1)))

    for source in range(rank_size):
        for first in range(0, m, row_block):
            last = min(first + row_block, m)
            rows = np.arange(first, last, dtype=np.int64)
            partial = _a_factors(rows, scale_groups, source, seed_phase) @ b_factors
            partial *= np.float32(BLOCK_SIZE)
            destination_row = source * m + first
            destination_row_end = source * m + last
            for rank in range(rank_size):
                first_column = rank * chunk_n
                golden[rank][destination_row:destination_row_end, :] = partial[:, first_column : first_column + chunk_n]
        _log(f"golden source {source + 1}/{rank_size}")

    for rank, output in enumerate(golden):
        output.flush()
        _log(f"golden rank={rank} written")
    del golden


def _generate_fast(fmt, data_dir, rank_size, m, n, k, seed):
    seed_phase = seed % 4
    scale_groups, padded_groups = _groups(k)
    _log(f"periodic fast path fmt={fmt} P={rank_size} M={m} N={n} K={k} scale_groups={scale_groups}")

    for source in range(rank_size):
        _log(f"A rank {source}: stream E4M3/E2M1 pattern")
        _write_a(
            fmt,
            data_dir / f"rank_{source}_a.bin",
            m,
            k,
            scale_groups,
            padded_groups,
            source,
            seed_phase,
        )

    b_data = data_dir / "rank_0_b.bin"
    b_scale = data_dir / "rank_0_b_scale.bin"
    _log("B: stream shared pattern")
    _write_b_data(fmt, b_data, k, n, scale_groups, seed_phase)
    _write_b_scale(b_scale, n, scale_groups, padded_groups, seed_phase)
    for rank in range(1, rank_size):
        _link_or_copy(b_data, data_dir / f"rank_{rank}_b.bin")
        _link_or_copy(b_scale, data_dir / f"rank_{rank}_b_scale.bin")

    _log("golden: grouped 32-element factors, blockwise output")
    _generate_fast_golden(data_dir, rank_size, m, n, k, seed_phase)


def _tensor_to_file(tensor, file_name):
    import torch

    if tensor.dtype == torch.bfloat16:
        tensor.view(torch.uint16).numpy().tofile(file_name)
    else:
        tensor.numpy().tofile(file_name)


def _generate_bf16(data_dir, rank_size, m, n, k, seed):
    import torch

    torch.manual_seed(seed)
    _log("BF16 path: preserve legacy random generation")
    a_dequant = []
    for rank in range(rank_size):
        a = torch.randn((m, k), dtype=torch.float32).to(torch.bfloat16)
        _tensor_to_file(a, data_dir / f"rank_{rank}_a.bin")
        a_dequant.append(a.float())

    b = torch.randn((k, n), dtype=torch.float32).to(torch.bfloat16)
    for rank in range(rank_size):
        _tensor_to_file(b, data_dir / f"rank_{rank}_b.bin")

    b_fp32 = b.float()
    chunk_n = n // rank_size
    for rank in range(rank_size):
        golden = []
        for source in range(rank_size):
            partial = torch.matmul(a_dequant[source], b_fp32)
            golden.append(partial[:, rank * chunk_n : (rank + 1) * chunk_n])
        _tensor_to_file(torch.cat(golden, dim=0), data_dir / f"golden_{rank}.bin")
        _log(f"golden rank={rank} written")


def main():
    parser = argparse.ArgumentParser(description="Generate QuantMatmulAllToAll-N data (fast MXFP8/MXFP4, legacy BF16)")
    parser.add_argument("fmt", type=int, help="Input format: 0=MXFP8, 1=MXFP4, 2=BF16")
    parser.add_argument("rank_size", type=int, help="Number of ranks")
    parser.add_argument("m", type=int, help="Per-rank M dimension")
    parser.add_argument("n", type=int, help="Matrix N dimension")
    parser.add_argument("k", type=int, help="Matrix K dimension")
    parser.add_argument("data_dir", type=Path, help="Directory for generated files")
    parser.add_argument("--seed", type=int, default=20261024, help="Deterministic generation seed")
    args = parser.parse_args()

    if args.fmt not in (0, 1, 2):
        parser.error("fmt must be 0 (MXFP8), 1 (MXFP4), or 2 (BF16)")
    if args.rank_size < 2:
        parser.error("rank_size must be at least 2")
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        parser.error("M, N, and K must be positive")
    if args.n % args.rank_size:
        parser.error("N must be divisible by rank_size")
    if args.k % BLOCK_SIZE:
        parser.error("K must be a multiple of 32")
    if args.fmt == 1 and args.n % 2:
        parser.error("N must be even for packed MXFP4")

    seed = int(os.environ.get("QMMA_SEED", args.seed))
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    _log(f"fmt={args.fmt} P={args.rank_size} M={args.m} N={args.n} K={args.k} seed={seed} -> {data_dir}")
    if args.fmt in (0, 1):
        _generate_fast(args.fmt, data_dir, args.rank_size, args.m, args.n, args.k, seed)
    else:
        _generate_bf16(data_dir, args.rank_size, args.m, args.n, args.k, seed)
    _log(f"done fmt={args.fmt} P={args.rank_size} M={args.m} N={args.n} K={args.k} chunkN={args.n // args.rank_size}")


if __name__ == "__main__":
    main()
