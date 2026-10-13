#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import torch
import os

from utils import DataType, tensor_to_file

BIAS_LOW = -65536
BIAS_HIGH = 65536
WORKSPACE = os.getcwd()

os.environ["WORKSPACE"] = WORKSPACE


def gen_random_data(size, dtype, debug=False):
    if dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32:
        return torch.empty(size=size, dtype=dtype).uniform_(-2, 2)
    elif dtype == torch.int8:
        if not debug:
            return torch.randint(-16, 16, size=size, dtype=dtype)
        else:
            return torch.ones(size=size, dtype=dtype)
    else:
        print(f"Invalid dtype: {dtype}.")
        exit(1)


def dequantize(golden, perchannel_scale_gm, per_token_scale_gm, pertensor_scale=False):
    if not pertensor_scale:
        golden = golden.to(torch.float32) * (per_token_scale_gm * perchannel_scale_gm)
    else:
        fused_scale = perchannel_scale_gm * per_token_scale_gm[0]
        golden = golden.to(torch.float32) * fused_scale

    return golden


def gen_golden_data():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('kernel_name', type=str)
    parser.add_argument('out_dtype', type=DataType.from_str, choices=[DataType.FLOAT16, DataType.BF16, DataType.INT8])
    parser.add_argument('rank_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('transA', type=int)
    parser.add_argument('transB', type=int)
    parser.add_argument('data_dir', type=str, help='Directory to save the data files', default="./output")
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k
    rankSize = args.rank_size
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    debug = int(os.getenv("debug", 0)) == 1

    if not debug:
        per_channel_scale_gm = torch.empty(size=[1, N], dtype=torch.float32).uniform_(0.1, 0.5)
        per_tensor_scale = torch.rand(1).item()
    else:
        per_tensor_scale = 0.1
        per_channel_scale_gm = torch.full(size=[1, N], fill_value=0.1, dtype=torch.float32)

    # use all same value as perTensorScale.
    per_token_scale_gm = torch.full(size=[M, 1], fill_value=per_tensor_scale, dtype=torch.float32)
    fused_scale_gm = per_channel_scale_gm * per_tensor_scale
    print(f"{per_tensor_scale=}")
    print(f"{per_channel_scale_gm=}")
    print(f"{fused_scale_gm=}")
    d_gm = torch.zeros((M * rankSize, N), dtype=torch.float32)

    if debug:
        print('[warning]: in debug mode, use special bias input.')
        bias_gm = torch.zeros(size=(N,), dtype=torch.int32)
    else:
        bias_gm = torch.randint(low=BIAS_LOW, high=BIAS_HIGH + 1, size=(N,), dtype=torch.int32)

    tensor_to_file(bias_gm, os.path.join(data_dir, "bias_gm.bin"))

    A = []
    for i in range(args.rank_size):
        # Generate rank-specific inputs x1 and x2,
        a_gm_rank = gen_random_data([M, K], dtype=torch.int8, debug=debug)
        tensor_to_file(a_gm_rank, os.path.join(data_dir, f"a_gm_rank{i}.bin"))
        print(f"Generated data a for rank {i}:")
        print(f"  a_gm_rank{i}.bin: shape={a_gm_rank.shape}")
        A.append(a_gm_rank)

    allgathered_a = torch.cat(A, dim=0)

    for i in range(args.rank_size):
        b_gm_rank = gen_random_data([K, N], dtype=torch.int8, debug=debug)
        tensor_to_file(b_gm_rank, os.path.join(data_dir, f"b_gm_rank{i}.bin"))
        print(f"Generated data b for rank {i}:")
        print(f"  b_gm_rank{i}.bin: shape={b_gm_rank.shape}")

        # Calculate this rank's contribution to the matmul sum
        accumulator_rank = torch.matmul(allgathered_a.to(torch.float32), b_gm_rank.to(torch.float32))
        accumulator_rank += bias_gm.to(torch.float32)
        rank_golden = dequantize(
            accumulator_rank,
            per_channel_scale_gm.view(1, N),
            per_token_scale_gm=per_token_scale_gm,
            pertensor_scale=True,
        )
        if args.out_dtype == DataType.FLOAT16:
            # Fixpipe VDEQF16 saturates values outside the finite FP16 range.
            limits = torch.finfo(torch.float16)
            rank_golden = rank_golden.clamp(limits.min, limits.max)
        tensor_to_file(rank_golden, os.path.join(data_dir, f"golden_rank{i}.bin"))
        print(f"Generated golden for rank {i}:")
        print(f"golden_rank{i}.bin: shape={rank_golden.shape}")

    per_channel_scale = torch.zeros(size=(1, N), dtype=torch.int64)
    per_channel_scale.view(torch.float32)[:, ::2] = fused_scale_gm

    tensor_to_file(d_gm, os.path.join(data_dir, "d_gm.bin"))
    tensor_to_file(per_channel_scale, os.path.join(data_dir, "scale_gm.bin"))


if __name__ == "__main__":
    gen_golden_data()
