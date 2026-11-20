#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""File inputs and independent CPU checks for the AllToAll-MatMul example."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

PATTERN_BYTES = np.array([0x30, 0x38, 0x3C, 0x40], dtype=np.uint8)
PATTERN_VALUES = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)


def sample_rows(chunk_m):
    # Keep all 256-row compute-tile boundaries, including every 512/1024-row communication boundary.
    starts = np.arange(0, chunk_m, 256, dtype=np.uint32)
    return np.unique(np.concatenate((starts, np.minimum(starts + 256, chunk_m) - 1)))


def expected_pattern(world, destination, local_k=256):
    # Sum independent FP32 dot products over source/local-K groups. No device layout code is imported.
    result = np.zeros((21, 11), dtype=np.float32)
    rows = np.arange(21)[:, None]
    columns = np.arange(11)[None, :]
    for source in range(world):
        scale_a = np.exp2(-2.0 + (source + destination + rows // 7) % 3)
        for group in range(local_k // 32):
            a = PATTERN_VALUES[(source + 3 * destination + rows % 7 + group) % 4]
            b = PATTERN_VALUES[(2 * source + columns + 3 * group) % 4]
            scale_b = 2.0 ** (-4 + (source + group) % 2)
            result += 32 * a * scale_a * b * scale_b
    return result


def metadata_for(world, chunk_m, local_k, n):
    return dict(pattern='periodic', world=world, chunk_m=chunk_m, local_k=local_k, n=n, version=4)


def generate_periodic(directory, world, chunk_m, local_k=256, n=4096):
    scale_k = local_k // 32
    directory.mkdir(parents=True, exist_ok=True)
    metadata = metadata_for(world, chunk_m, local_k, n)
    manifest = directory / 'data.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != metadata:
            raise ValueError('existing data.json does not match the requested case')
        check_inputs(directory, world, chunk_m, local_k, n)
        print(f'reusing complete inputs: {directory}')
        return
    if any(directory.iterdir()):
        raise ValueError('generate into an empty directory to avoid mixing datasets')
    group = np.arange(local_k // 32)[None, :]
    for source in range(world):
        with (
            (directory / f'input_a_{source}.bin').open('wb') as afile,
            (directory / f'input_a_scale_{source}.bin').open('wb') as sfile,
        ):
            for destination in range(world):
                for first in range(0, chunk_m, 8192):
                    rows = np.arange(first, min(first + 8192, chunk_m))[:, None]
                    a = PATTERN_BYTES[(source + 3 * destination + rows % 7 + group) % 4]
                    np.repeat(a, 32, axis=1).tofile(afile)
                for first in range(0, chunk_m, 8192):
                    rows = np.arange(first, min(first + 8192, chunk_m))[:, None]
                    scales = (
                        np.broadcast_to(125 + (source + destination + (rows // 7) % 3) % 3, (len(rows), scale_k))
                        .astype(np.uint8)
                        .copy()
                    )
                    scales.tofile(sfile)
        print(f'generated source {source}/{world - 1}', flush=True)
    columns = np.arange(n)[:, None, None]
    sources = np.arange(world)[None, :, None]
    groups = np.arange(local_k // 32)[None, None, :]
    b = PATTERN_BYTES[(2 * sources + columns % 11 + 3 * groups) % 4]
    np.repeat(b, 32, axis=2).tofile(directory / 'input_b_0.bin')
    scales = np.broadcast_to(123 + (sources + groups) % 2, (n, world, scale_k)).astype(np.uint8)
    scales.tofile(directory / 'input_b_scale_0.bin')
    for rank in range(1, world):
        os.link(directory / 'input_b_0.bin', directory / f'input_b_{rank}.bin')
        os.link(directory / 'input_b_scale_0.bin', directory / f'input_b_scale_{rank}.bin')
    sample_rows(chunk_m).astype('<u4').tofile(directory / 'sample_rows.bin')
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    check_inputs(directory, world, chunk_m, local_k, n)


def check_inputs(directory, world, chunk_m, local_k=256, n=4096):
    scale_k = local_k // 32
    manifest = directory / 'data.json'
    if not manifest.is_file():
        raise ValueError('data.json is required; generate periodic inputs first')
    meta = json.loads(manifest.read_text())
    if meta != metadata_for(world, chunk_m, local_k, n):
        raise ValueError('data.json shape, world or format mismatch')
    files = {}
    for rank in range(world):
        files[f'input_a_{rank}.bin'] = world * chunk_m * local_k
        files[f'input_a_scale_{rank}.bin'] = world * chunk_m * scale_k
        files[f'input_b_{rank}.bin'] = world * local_k * n
        files[f'input_b_scale_{rank}.bin'] = world * scale_k * n
    files['sample_rows.bin'] = len(sample_rows(chunk_m)) * 4
    for name, size in files.items():
        if not (directory / name).is_file() or (directory / name).stat().st_size != size:
            raise ValueError(f'missing or wrong-size input: {directory / name}, expected {size} bytes')
    if not np.array_equal(np.fromfile(directory / 'sample_rows.bin', dtype='<u4'), sample_rows(chunk_m)):
        raise ValueError('sample_rows.bin must cover the first and last row of every 256-row compute tile')


def check(directory, world, chunk_m, rank_start=0, rank_count=None, local_k=256, n=4096):
    check_inputs(directory, world, chunk_m, local_k, n)
    if rank_count is None:
        rank_count = world - rank_start
    if rank_start < 0 or rank_count < 1 or rank_start + rank_count > world:
        raise ValueError('rank range must be within world')
    rows = sample_rows(chunk_m)
    for rank in range(rank_start, rank_start + rank_count):
        # BF16 stores the high 16 bits of an FP32 value; NumPy has no built-in BF16 dtype.
        bits = np.fromfile(directory / f'output_samples_{rank}.bin', dtype='<u2')
        actual = (bits.astype('<u4') << 16).view('<f4')
        if actual.size != len(rows) * n:
            raise ValueError(f'rank {rank}: sampled output has incorrect size')
        actual = actual.reshape(len(rows), n)
        # Round the finite FP32 reference to BF16 using round-to-nearest, ties-to-even.
        bits = expected_pattern(world, rank, local_k).view(np.uint32)
        rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).view(np.float32)
        reference = rounded[rows[:, None] % 21, np.arange(n)[None, :] % 11]
        difference = np.abs(actual - reference)
        failures = ~np.isfinite(actual) | (difference > 0.01 + 0.0006 * np.abs(reference))
        print(
            f'VALIDATE rank={rank} M={chunk_m} sampled_rows={len(rows)} elements={actual.size} '
            f'failures={failures.sum()} max_abs={difference.max():.6g}'
        )
        if np.any(failures):
            r, c = np.argwhere(failures)[0]
            raise AssertionError(
                f'rank {rank}, row {rows[r]}, column {c}: got {actual[r, c]}, expected {reference[r, c]}'
            )


def bf16_metadata(args):
    return {
        "world": args.world,
        "M": args.m,
        "N": args.n,
        "K": args.k,
        "seed": args.seed,
        "dtype": "BF16",
        "b_layout": "ColumnMajor",
    }


def bf16(float_values):
    bits = np.asarray(float_values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def fp32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def bf16_values(indices, salt):
    # Integer / 16 inputs make the FP32 dot products exact for the supported test shapes.
    x = np.asarray(indices, dtype=np.uint32) + np.uint32(salt & 0xFFFFFFFF)
    x = (x ^ (x >> 16)) * np.uint32(0x7FEB352D)
    x = (x ^ (x >> 15)) * np.uint32(0x846CA68B)
    x ^= x >> 16
    return ((x % 17).astype(np.float32) - 8) / 16


def a_rows(args, source, dest, begin, end):
    width = args.k // args.world
    indices = np.arange(begin * width, end * width, dtype=np.uint32).reshape(end - begin, width)
    return bf16_values(indices, args.seed * 1000003 + source * 65537 + dest * 8191)


def b_matrix(args, rank):
    # The file is [N,K] contiguous: logical B[K,N] is column-major.
    indices = np.arange(args.n * args.k, dtype=np.uint32).reshape(args.n, args.k)
    return bf16_values(indices, args.seed * 2000003 + rank * 104729 + 17)


def generate_bf16(args):
    args.path.mkdir(parents=True, exist_ok=True)
    for rank in range(args.rank_begin, args.rank_end):
        with (args.path / f"input_a_{rank}.bin").open("wb") as output:
            for dest in range(args.world):
                bf16(a_rows(args, rank, dest, 0, args.m)).tofile(output)
        bf16(b_matrix(args, rank)).tofile(args.path / f"input_b_{rank}.bin")
        print(
            f"GENERATED rank={rank} M={args.m} N={args.n} K={args.k} seed={args.seed}",
            flush=True,
        )
    (args.path / "manifest.json").write_text(json.dumps(bf16_metadata(args), indent=2) + "\n", encoding="utf-8")


def check_bf16_inputs(args):
    metadata = json.loads((args.path / "manifest.json").read_text(encoding="utf-8"))
    if metadata != bf16_metadata(args):
        raise ValueError("manifest does not match BF16 inputs")
    for rank in range(args.rank_begin, args.rank_end):
        for name, size in (("a", args.m * args.k * 2), ("b", args.n * args.k * 2)):
            path = args.path / f"input_{name}_{rank}.bin"
            if path.stat().st_size != size:
                raise ValueError(f"wrong input size: {path}")


def check_bf16(args):
    check_bf16_inputs(args)
    failed = False
    for rank in range(args.rank_begin, args.rank_end):
        output_path = args.path / f"output_{rank}.bin"
        if output_path.stat().st_size != args.m * args.n * 2:
            raise ValueError(f"wrong output size: {output_path}")
        actual = np.memmap(output_path, dtype=np.uint16, mode="r", shape=(args.m, args.n))
        b = b_matrix(args, rank).T
        bad = 0
        max_error = 0.0
        for begin in range(0, args.m, 256):
            end = min(begin + 256, args.m)
            a = np.concatenate(
                [a_rows(args, source, rank, begin, end) for source in range(args.world)],
                axis=1,
            )
            reference = bf16(a @ b)
            got = actual[begin:end]
            bad += int(np.count_nonzero(got != reference))
            error = np.abs(fp32(got) - fp32(reference))
            max_error = max(max_error, float(np.max(error)))
        print(
            f"CHECK rank={rank} elements={args.m * args.n} bad={bad} max_abs={max_error} "
            f"reference=cpu_fp32_to_bf16 full=1 status={'PASS' if bad == 0 else 'FAIL'}",
            flush=True,
        )
        failed |= bad != 0
    return int(failed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('generate', 'check', 'check-inputs'))
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--world', type=int, choices=(1, 2, 4, 8, 16, 32), default=16)
    parser.add_argument('--rank-start', type=int, default=0)
    parser.add_argument('--rank-count', type=int)
    parser.add_argument('--dtype', choices=('mxfp8', 'bf16'), default='mxfp8')
    parser.add_argument('--seed', type=int, default=1, help='BF16 input seed')
    parser.add_argument('--m', type=int, help='output rows per rank')
    parser.add_argument('--k', type=int, help='full K after AllToAll; default world*256')
    parser.add_argument('--n', type=int)
    args = parser.parse_args()
    use_bf16 = args.dtype == 'bf16'
    if args.m is None:
        args.m = 3667 if use_bf16 else 10000
    if args.n is None:
        args.n = 3328 if use_bf16 else 4096
    if not 1 <= args.m <= 300000:
        parser.error('--m must be in [1,300000]')
    full_k = args.k if args.k is not None else (12288 if use_bf16 else args.world * 256)
    k_multiple = args.world * (1 if use_bf16 else 64)
    if full_k <= 0 or full_k > 0xFFFFFFFF or full_k % k_multiple:
        parser.error(f'K must be a positive uint32 multiple of {k_multiple}')
    if args.n <= 0 or args.n > 0xFFFFFFFF or args.n % 16:
        parser.error('N must be a positive uint32 multiple of 16')
    if use_bf16:
        args.k = full_k
        args.path = args.out_dir
        args.rank_begin = args.rank_start
        args.rank_end = args.world if args.rank_count is None else args.rank_start + args.rank_count
        if not 0 <= args.rank_begin < args.rank_end <= args.world or args.seed < 0:
            parser.error('invalid BF16 rank range or seed')
        if args.mode == 'generate':
            generate_bf16(args)
        elif args.mode == 'check-inputs':
            check_bf16_inputs(args)
        else:
            return check_bf16(args)
        return 0
    local_k = full_k // args.world
    if args.mode == 'generate':
        generate_periodic(args.out_dir, args.world, args.m, local_k, args.n)
    elif args.mode == 'check':
        check(args.out_dir, args.world, args.m, args.rank_start, args.rank_count, local_k, args.n)
    else:
        check_inputs(args.out_dir, args.world, args.m, local_k, args.n)


if __name__ == '__main__':
    raise SystemExit(main())
