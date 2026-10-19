#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
import shutil
import sys
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Tuple

import numpy as np
import torch

try:
    import torch_npu
except ImportError as err:
    raise ImportError("gen_data.py requires torch_npu to generate NPU golden data.") from err

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
from utils import DataType, tensor_to_file


MX_SCALE_GROUP_NUM = 32
KERNEL_LOCAL_K_ALIGN = 512
FP8_DTYPE = torch.float8_e4m3fn
MX_SCALE_DTYPE = getattr(torch, "float8_e8m0fnu", None)
if MX_SCALE_DTYPE is None:
    MX_SCALE_DTYPE = getattr(torch_npu, "float8_e8m0fnu")


@dataclass(frozen=True)
class Config:
    out_dtype: DataType
    rank_size: int
    m: int
    n: int
    local_k: int
    trans_a: int
    trans_b: int
    data_dir: str
    tmp_dir: str
    seed: int
    scale_exp_low: int
    scale_exp_high: int
    device_ids: Tuple[int, ...]
    packed_pertoken_scale: bool
    keep_intermediate: bool
    generate_local_atomic_golden: bool

    @property
    def chunk_m(self) -> int:
        return self.m // self.rank_size

    @property
    def full_k(self) -> int:
        return self.local_k * self.rank_size

    @property
    def local_scale_k(self) -> int:
        return ceil_div(self.local_k, MX_SCALE_GROUP_NUM)

    @property
    def aligned_local_scale_k(self) -> int:
        return round_up(self.local_scale_k, 2)

    @property
    def full_scale_k(self) -> int:
        return ceil_div(self.full_k, MX_SCALE_GROUP_NUM)

    @property
    def aligned_full_scale_k(self) -> int:
        return round_up(self.full_scale_k, 2)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def round_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


def element_count(shape) -> int:
    return reduce(mul, shape, 1)


def storage_to_file_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8).reshape(-1)


def tensor_from_storage(file_name: str, shape, dtype: torch.dtype) -> torch.Tensor:
    raw = np.fromfile(file_name, dtype=np.uint8)
    expected = element_count(shape)
    if raw.size != expected:
        raise ValueError(f"{file_name}: expected {expected} bytes for shape {shape}, got {raw.size}")

    tensor = torch.empty(shape, dtype=dtype)
    tensor.view(torch.uint8).flatten().copy_(torch.from_numpy(raw))
    return tensor


def pack_a_scale(scale: torch.Tensor) -> torch.Tensor:
    return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2).contiguous()


def unpack_a_scale(scale_packed: torch.Tensor) -> torch.Tensor:
    return scale_packed.reshape(scale_packed.shape[0], scale_packed.shape[1] * 2).contiguous()


def pack_b_scale_transposed(scale: torch.Tensor) -> torch.Tensor:
    return scale.reshape(scale.shape[0] // 2, 2, scale.shape[1]).permute(2, 0, 1).contiguous()


def pack_b_scale(scale: torch.Tensor) -> torch.Tensor:
    return scale.reshape(scale.shape[0] // 2, 2, scale.shape[1]).permute(0, 2, 1).contiguous()


def unpack_b_scale_transposed(scale_packed: torch.Tensor) -> torch.Tensor:
    return scale_packed.permute(1, 2, 0).reshape(scale_packed.shape[1] * 2, scale_packed.shape[0]).contiguous()


def make_random_fp8(shape, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.randint(-8, 9, shape, generator=generator, dtype=torch.int16).to(torch.float32) / 8.0
    return values.to(FP8_DTYPE)


def make_random_mx_scale(shape, seed: int, exp_low: int, exp_high: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    exponents = torch.randint(exp_low, exp_high + 1, shape, generator=generator, dtype=torch.int32)
    return torch.pow(torch.tensor(2.0, dtype=torch.float32), exponents.to(torch.float32)).to(MX_SCALE_DTYPE)


def set_npu_device(device_id: int) -> None:
    torch_npu.npu.set_device(device_id)


def npu_quant_matmul(input_after_a2a, weight_t, b_scale, a_scale, output_dtype):
    scale_dtype = getattr(torch_npu, "float8_e8m0fnu", MX_SCALE_DTYPE)
    kwargs = {
        "pertoken_scale": a_scale,
        "output_dtype": output_dtype,
        "group_sizes": [1, 1, MX_SCALE_GROUP_NUM],
        "scale_dtype": scale_dtype,
        "pertoken_scale_dtype": scale_dtype,
    }
    try:
        return torch_npu.npu_quant_matmul(input_after_a2a, weight_t, b_scale, **kwargs)
    except TypeError as err:
        message = str(err)
        if "scale_dtype" not in message and "pertoken_scale_dtype" not in message:
            raise
        kwargs.pop("scale_dtype")
        kwargs.pop("pertoken_scale_dtype")
        return torch_npu.npu_quant_matmul(input_after_a2a, weight_t, b_scale, **kwargs)


def input_a_path(cfg: Config, rank_idx: int) -> str:
    return os.path.join(cfg.data_dir, f"input_a_{rank_idx}.bin")


def input_a_scale_path(cfg: Config, rank_idx: int) -> str:
    return os.path.join(cfg.data_dir, f"input_a_scale_{rank_idx}.bin")


def input_b_path(cfg: Config, rank_idx: int) -> str:
    return os.path.join(cfg.data_dir, f"input_b_{rank_idx}.bin")


def input_b_scale_path(cfg: Config, rank_idx: int) -> str:
    return os.path.join(cfg.data_dir, f"input_b_scale_{rank_idx}.bin")


def a2a_a_path(cfg: Config, src_rank: int, dst_rank: int) -> str:
    return os.path.join(cfg.tmp_dir, f"a_from_rank{src_rank}_to_rank{dst_rank}.bin")


def a2a_scale_path(cfg: Config, src_rank: int, dst_rank: int) -> str:
    return os.path.join(cfg.tmp_dir, f"a_scale_from_rank{src_rank}_to_rank{dst_rank}.bin")


def a2a_concat_a_path(cfg: Config, dst_rank: int) -> str:
    return os.path.join(cfg.tmp_dir, f"input_after_a2a_rank{dst_rank}.bin")


def a2a_concat_scale_path(cfg: Config, dst_rank: int) -> str:
    return os.path.join(cfg.tmp_dir, f"a_scale_after_a2a_rank{dst_rank}.bin")


def golden_path(cfg: Config, rank_idx: int) -> str:
    return os.path.join(cfg.data_dir, f"golden_{rank_idx}.bin")


def local_atomic_golden_path(cfg: Config, rank_idx: int) -> str:
    return os.path.join(cfg.data_dir, f"golden_local_atomic_{rank_idx}.bin")


def generate_input_files(cfg: Config) -> None:
    for rank_idx in range(cfg.rank_size):
        a_fp8 = make_random_fp8((cfg.m, cfg.local_k), cfg.seed + 1009 * rank_idx)
        a_scale = make_random_mx_scale(
            (cfg.m, cfg.aligned_local_scale_k),
            cfg.seed + 2003 * rank_idx,
            cfg.scale_exp_low,
            cfg.scale_exp_high,
        )
        tensor_to_file(storage_to_file_bytes(a_fp8), input_a_path(cfg, rank_idx))
        tensor_to_file(storage_to_file_bytes(pack_a_scale(a_scale)), input_a_scale_path(cfg, rank_idx))

    for rank_idx in range(cfg.rank_size):
        b_fp8 = make_random_fp8((cfg.full_k, cfg.n), cfg.seed + 3001 * rank_idx)
        b_scale = make_random_mx_scale(
            (cfg.aligned_full_scale_k, cfg.n),
            cfg.seed + 4001 * rank_idx,
            cfg.scale_exp_low,
            cfg.scale_exp_high,
        )
        weight_t = b_fp8.t().contiguous()
        b_scale_kn2 = pack_b_scale_transposed(b_scale)
        tensor_to_file(storage_to_file_bytes(weight_t), input_b_path(cfg, rank_idx))
        tensor_to_file(storage_to_file_bytes(b_scale_kn2), input_b_scale_path(cfg, rank_idx))


def write_rank_a2a_files(cfg: Config, src_rank: int) -> None:
    a = tensor_from_storage(input_a_path(cfg, src_rank), (cfg.m, cfg.local_k), FP8_DTYPE)
    a_scale = tensor_from_storage(
        input_a_scale_path(cfg, src_rank),
        (cfg.m, cfg.aligned_local_scale_k // 2, 2),
        MX_SCALE_DTYPE,
    )

    for dst_rank in range(cfg.rank_size):
        m_begin = dst_rank * cfg.chunk_m
        m_end = m_begin + cfg.chunk_m
        tensor_to_file(storage_to_file_bytes(a[m_begin:m_end, :].contiguous()), a2a_a_path(cfg, src_rank, dst_rank))
        tensor_to_file(
            storage_to_file_bytes(a_scale[m_begin:m_end, :, :].contiguous()),
            a2a_scale_path(cfg, src_rank, dst_rank),
        )


def generate_rank_golden(cfg: Config, dst_rank: int) -> None:
    set_npu_device(cfg.device_ids[dst_rank])

    a_chunks = []
    a_scale_chunks = []
    for src_rank in range(cfg.rank_size):
        a_chunks.append(tensor_from_storage(a2a_a_path(cfg, src_rank, dst_rank), (cfg.chunk_m, cfg.local_k), FP8_DTYPE))
        scale_packed = tensor_from_storage(
            a2a_scale_path(cfg, src_rank, dst_rank),
            (cfg.chunk_m, cfg.aligned_local_scale_k // 2, 2),
            MX_SCALE_DTYPE,
        )
        a_scale_chunks.append(unpack_a_scale(scale_packed)[:, : cfg.local_scale_k])

    input_after_a2a = torch.cat(a_chunks, dim=1).contiguous()
    a_scale_after_a2a_logical = torch.cat(a_scale_chunks, dim=1).contiguous()
    a_scale_after_a2a_packed = pack_a_scale(a_scale_after_a2a_logical)
    a_scale_after_a2a = (
        a_scale_after_a2a_packed if cfg.packed_pertoken_scale else a_scale_after_a2a_logical.unsqueeze(1)
    )
    weight_t = tensor_from_storage(input_b_path(cfg, dst_rank), (cfg.n, cfg.full_k), FP8_DTYPE)
    b_scale_kn2 = tensor_from_storage(
        input_b_scale_path(cfg, dst_rank),
        (cfg.n, cfg.aligned_full_scale_k // 2, 2),
        MX_SCALE_DTYPE,
    )
    weight = weight_t.t().contiguous()
    b_scale_logical = unpack_b_scale_transposed(b_scale_kn2)
    b_scale = pack_b_scale(b_scale_logical)

    tensor_to_file(storage_to_file_bytes(input_after_a2a), a2a_concat_a_path(cfg, dst_rank))
    tensor_to_file(storage_to_file_bytes(a_scale_after_a2a_packed), a2a_concat_scale_path(cfg, dst_rank))

    output_dtype = cfg.out_dtype.torch_type
    torch.npu.synchronize()
    output = npu_quant_matmul(
        input_after_a2a.npu(),
        weight.npu(),
        b_scale.npu(),
        a_scale_after_a2a.npu(),
        output_dtype=output_dtype,
    )
    torch.npu.synchronize()

    output_fp32 = output.cpu().to(torch.float32).contiguous()
    if output_fp32.shape != (cfg.chunk_m, cfg.n):
        raise ValueError(
            f"rank {dst_rank}: expected output shape {(cfg.chunk_m, cfg.n)}, got {tuple(output_fp32.shape)}"
        )
    tensor_to_file(output_fp32, golden_path(cfg, dst_rank))
    print(
        f"Generated torch_npu golden for rank {dst_rank}: shape={tuple(output_fp32.shape)}, dtype={output_fp32.dtype}"
    )

    if not cfg.generate_local_atomic_golden:
        return

    local_k_begin = dst_rank * cfg.local_k
    local_k_end = local_k_begin + cfg.local_k
    local_scale_begin = dst_rank * cfg.local_scale_k
    local_scale_end = local_scale_begin + cfg.local_scale_k

    local_a_scale_logical = a_scale_chunks[dst_rank]
    local_a_scale = (
        pack_a_scale(local_a_scale_logical) if cfg.packed_pertoken_scale else local_a_scale_logical.unsqueeze(1)
    )
    local_b_scale = pack_b_scale(b_scale_logical[local_scale_begin:local_scale_end, :])
    local_output = npu_quant_matmul(
        a_chunks[dst_rank].npu(),
        weight[local_k_begin:local_k_end, :].contiguous().npu(),
        local_b_scale.contiguous().npu(),
        local_a_scale.contiguous().npu(),
        output_dtype=output_dtype,
    ).to(output_dtype)

    remote_ranks = [rank_idx for rank_idx in range(cfg.rank_size) if rank_idx != dst_rank]
    if remote_ranks:
        remote_a = torch.cat([a_chunks[rank_idx] for rank_idx in remote_ranks], dim=1).contiguous()
        remote_a_scale_logical = torch.cat([a_scale_chunks[rank_idx] for rank_idx in remote_ranks], dim=1).contiguous()
        remote_a_scale = (
            pack_a_scale(remote_a_scale_logical) if cfg.packed_pertoken_scale else remote_a_scale_logical.unsqueeze(1)
        )
        remote_weight = torch.cat(
            [weight[rank_idx * cfg.local_k : (rank_idx + 1) * cfg.local_k, :] for rank_idx in remote_ranks],
            dim=0,
        ).contiguous()
        remote_b_scale_logical = torch.cat(
            [
                b_scale_logical[
                    rank_idx * cfg.local_scale_k : (rank_idx + 1) * cfg.local_scale_k,
                    :,
                ]
                for rank_idx in remote_ranks
            ],
            dim=0,
        ).contiguous()
        remote_b_scale = pack_b_scale(remote_b_scale_logical)
        remote_output = npu_quant_matmul(
            remote_a.npu(),
            remote_weight.npu(),
            remote_b_scale.npu(),
            remote_a_scale.contiguous().npu(),
            output_dtype=output_dtype,
        ).to(output_dtype)
        local_atomic_output = (local_output + remote_output).to(output_dtype)
    else:
        local_atomic_output = local_output

    torch.npu.synchronize()
    local_atomic_output = local_atomic_output.cpu().contiguous()
    tensor_to_file(local_atomic_output, local_atomic_golden_path(cfg, dst_rank))
    print(
        f"Generated local-atomic low golden for rank {dst_rank}: "
        f"shape={tuple(local_atomic_output.shape)}, dtype={local_atomic_output.dtype}"
    )


def run_jobs(worker, cfg: Config, serial: bool) -> None:
    if serial:
        for rank_idx in range(cfg.rank_size):
            worker(cfg, rank_idx)
        return

    ctx = mp.get_context("spawn")
    processes = [ctx.Process(target=worker, args=(cfg, rank_idx)) for rank_idx in range(cfg.rank_size)]
    for process in processes:
        process.start()
    failed = []
    for rank_idx, process in enumerate(processes):
        process.join()
        if process.exitcode != 0:
            failed.append((rank_idx, process.exitcode))
    if failed:
        raise RuntimeError(f"{worker.__name__} failed: {failed}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate random MXFP8 split-K AllToAll golden with torch_npu.")
    parser.add_argument("out_dtype", type=DataType.from_str, choices=[DataType.FLOAT16, DataType.BF16])
    parser.add_argument("rank_size", type=int)
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int, help="Local K owned by each rank.")
    parser.add_argument("transA", type=int)
    parser.add_argument("transB", type=int)
    parser.add_argument("data_dir", type=str, default="./out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id-list", type=str, default=None)
    parser.add_argument("--scale-exp-low", type=int, default=-2)
    parser.add_argument("--scale-exp-high", type=int, default=1)
    parser.add_argument("--tmp-dir", type=str, default=None)
    parser.add_argument(
        "--packed-pertoken-scale",
        dest="packed_pertoken_scale",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--logical-pertoken-scale",
        dest="packed_pertoken_scale",
        action="store_false",
        help="Pass A scale to torch_npu as logical [M, 1, K/32] instead of default packed [M, K/64, 2].",
    )
    parser.set_defaults(packed_pertoken_scale=True)
    parser.add_argument("--serial", action="store_true", help="Run file-communication and golden workers serially.")
    parser.add_argument("--clean-intermediate", action="store_true", help="Delete simulated A2A files after success.")
    parser.add_argument(
        "--local-atomic-golden",
        action="store_true",
        help="Also generate an FP16 low golden matching local write plus remote atomic add.",
    )
    args = parser.parse_args()

    if args.transA != 0 or args.transB != 1:
        raise ValueError("This torch_npu golden generator targets transA=0, transB=1.")
    if args.rank_size <= 0:
        raise ValueError(f"rank_size must be positive, got {args.rank_size}")
    if args.m % args.rank_size != 0:
        raise ValueError(f"m={args.m} must be divisible by rank_size={args.rank_size}")
    if args.k % KERNEL_LOCAL_K_ALIGN != 0:
        raise ValueError(f"local k={args.k} must be divisible by {KERNEL_LOCAL_K_ALIGN}")
    if args.scale_exp_low > args.scale_exp_high:
        raise ValueError("scale-exp-low must be <= scale-exp-high")
    if args.local_atomic_golden and args.out_dtype != DataType.FLOAT16:
        raise ValueError("local-atomic golden currently requires FLOAT16 output")

    if args.device_id_list:
        device_ids = tuple(int(item) for item in args.device_id_list.split(",") if item != "")
    else:
        device_ids = tuple(range(args.rank_size))
    if len(device_ids) < args.rank_size:
        raise ValueError(f"need at least {args.rank_size} device ids, got {device_ids}")

    data_dir = os.path.abspath(args.data_dir)
    tmp_dir = os.path.abspath(args.tmp_dir) if args.tmp_dir else os.path.join(data_dir, "a2a_tmp")
    return Config(
        out_dtype=args.out_dtype,
        rank_size=args.rank_size,
        m=args.m,
        n=args.n,
        local_k=args.k,
        trans_a=args.transA,
        trans_b=args.transB,
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        seed=args.seed,
        scale_exp_low=args.scale_exp_low,
        scale_exp_high=args.scale_exp_high,
        device_ids=device_ids,
        packed_pertoken_scale=args.packed_pertoken_scale,
        keep_intermediate=not args.clean_intermediate,
        generate_local_atomic_golden=args.local_atomic_golden,
    ), args.serial


def main() -> None:
    cfg, serial = parse_args()
    os.makedirs(cfg.data_dir, exist_ok=True)
    if os.path.exists(cfg.tmp_dir):
        shutil.rmtree(cfg.tmp_dir)
    os.makedirs(cfg.tmp_dir, exist_ok=True)

    generate_input_files(cfg)
    run_jobs(write_rank_a2a_files, cfg, serial)
    run_jobs(generate_rank_golden, cfg, serial)

    if not cfg.keep_intermediate:
        shutil.rmtree(cfg.tmp_dir)

    print(
        "Generated random torch_npu MXFP8 split-K URMA data: "
        f"M={cfg.m}, N={cfg.n}, localK={cfg.local_k}, fullK={cfg.full_k}, "
        f"rank_size={cfg.rank_size}, chunkM={cfg.chunk_m}, data_dir={cfg.data_dir}"
    )


if __name__ == "__main__":
    main()
