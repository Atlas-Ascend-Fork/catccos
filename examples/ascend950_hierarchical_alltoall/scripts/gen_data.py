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

import torch


PADDING_VALUE = 0xA5


def peer_tensors(world, m, n, source, destination):
    generator = torch.Generator().manual_seed(2026 + source * world + destination)
    data = torch.randn((m, n), generator=generator).to(torch.float8_e4m3fn)
    # E8M0 stores the unsigned exponent with bias 127. These encode 2**[-8, 8].
    scale = torch.randint(-8, 9, (m, n // 32), generator=generator).add(127).to(torch.uint8)
    return data.view(torch.uint8).flatten(), scale.flatten()


def write_slot(file, tensor):
    file.write(bytes(tensor.tolist()))
    padding = (-tensor.numel()) % 32
    file.write(bytes([PADDING_VALUE]) * padding)


def generate_rank(directory, world, m, n, rank):
    # A rerun must not accept output files left by an earlier execution.
    for name in (f"output_{rank}.bin", f"output_scale_{rank}.bin"):
        (directory / name).unlink(missing_ok=True)
    with (
        (directory / f"rank_{rank}_a.bin").open("wb") as data_file,
        (directory / f"rank_{rank}_a_scale.bin").open("wb") as scale_file,
        (directory / f"golden_{rank}.bin").open("wb") as golden_file,
        (directory / f"golden_scale_{rank}.bin").open("wb") as golden_scale_file,
    ):
        for peer in range(world):
            data, scale = peer_tensors(world, m, n, rank, peer)
            write_slot(data_file, data)
            write_slot(scale_file, scale)
            # output[peer] receives input[rank] from source peer.
            data, scale = peer_tensors(world, m, n, peer, rank)
            write_slot(golden_file, data)
            write_slot(golden_scale_file, scale)


def main():
    parser = argparse.ArgumentParser(description="Generate E4M3 data, E8M0 scales and AllToAll golden files")
    parser.add_argument("rank_size", type=int)
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--rank-start", type=int, default=0)
    parser.add_argument("--rank-count", type=int)
    args = parser.parse_args()
    if not 1 <= args.rank_size <= 32 or args.m <= 0 or args.n <= 0 or args.n % 32:
        parser.error("rank_size must be 1..32; m/n must be positive and n divisible by 32")
    count = args.rank_size if args.rank_count is None else args.rank_count
    if args.rank_start < 0 or count <= 0 or args.rank_start + count > args.rank_size:
        parser.error("invalid local rank range")
    data_stride = (args.m * args.n + 31) // 32 * 32
    scale_stride = (args.m * (args.n // 32) + 31) // 32 * 32
    if 2 * args.rank_size * (data_stride + scale_stride) > (512 - 16) * 1024 * 1024:
        parser.error("data and scale buffers exceed the example SHMEM heap budget")
    torch.set_num_threads(1)
    args.data_path.mkdir(parents=True, exist_ok=True)
    for rank in range(args.rank_start, args.rank_start + count):
        generate_rank(args.data_path, args.rank_size, args.m, args.n, rank)


if __name__ == "__main__":
    main()
