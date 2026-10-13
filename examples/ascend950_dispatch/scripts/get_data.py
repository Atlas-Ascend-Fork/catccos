#!/usr/bin/env python3

import argparse
import math
import struct
import sys
from array import array
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Ascend950 Dispatch inputs and golden outputs")
    parser.add_argument("rank_size", type=int)
    parser.add_argument("num_tokens", type=int)
    parser.add_argument("hidden", type=int)
    parser.add_argument("group_size", type=int)
    parser.add_argument("top_k", type=int)
    parser.add_argument("experts_per_rank", type=int)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--route-mode", choices=("balanced", "duplicate", "skewed"), default="balanced")
    return parser.parse_args()


def expert_for(src_rank, token, route, rank_size, group_size, experts_per_rank, route_mode):
    if route_mode == "balanced":
        dst_rank = (src_rank + token + route) % rank_size
    elif route_mode == "duplicate":
        dst_rank = (src_rank + token + route // 2) % rank_size
    else:
        # Concentrate traffic on the first group; other ranks receive no rows.
        dst_rank = (route // 2) % group_size
    local_expert = (token * 3 + route) % experts_per_rank
    return dst_rank * experts_per_rank + local_expert


def save(data_dir, kind, name, rank, values):
    with (data_dir / f"{kind}_{name}_{rank}.bin").open("wb") as file:
        if isinstance(values, bytearray):
            file.write(values)
        else:
            values.tofile(file)


def bf16_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def quantize_row(bits):
    """Reference per-token BF16-to-INT8 quantization; returns raw bytes and scale."""
    values = [struct.unpack('<f', struct.pack('<I', value << 16))[0] for value in bits]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError('A nonempty finite BF16 row is required')
    amax = max(map(abs, values))
    if amax == 0:
        return bytearray(len(values)), 0.0
    try:
        multiplier = fp32(127.0 / amax)
    except OverflowError as error:
        raise ValueError('INT8 reciprocal scale overflows FP32') from error
    # Preserve FP32 division/multiply and RINT (ties to even), then the exact
    # integer-to-half-to-int8 conversion used by ops-transformer.
    quantized = [round(fp32(value * multiplier)) for value in values]
    return bytearray(struct.pack(f'<{len(values)}b', *quantized)), fp32(1.0 / multiplier)


def main():
    args = parse_args()
    if not (0 < args.rank_size <= 128):
        raise ValueError("rank_size must be in [1, 128]")
    if not (0 < args.group_size <= args.rank_size):
        raise ValueError("group_size must be in [1, rank_size]")
    if not (0 < args.top_k <= min(args.rank_size, 128)):
        raise ValueError("top_k must be in [1, min(rank_size, 128)]")
    if args.num_tokens <= 0 or args.hidden <= 0 or args.experts_per_rank <= 0:
        raise ValueError("num_tokens, hidden, and experts_per_rank must be positive")
    if array("H").itemsize != 2 or array("i").itemsize != 4 or array("q").itemsize != 8 or array("f").itemsize != 4:
        raise RuntimeError("Python native array sizes do not match the Dispatch binary contract")
    if sys.byteorder != "little":
        raise RuntimeError("Dispatch binary files require little-endian byte order")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    rank_size = args.rank_size
    num_tokens = args.num_tokens
    num_experts = rank_size * args.experts_per_rank

    all_x = []
    all_quant_x = []
    all_scales = []
    all_topk_idx = []
    all_topk_weights = []
    all_routes = []
    all_tokens_per_rank = []

    for src_rank in range(rank_size):
        x = array("H", [0]) * (num_tokens * args.hidden)
        topk_idx = array("q", [0]) * (num_tokens * args.top_k)
        topk_weights = array("f", [0.0]) * (num_tokens * args.top_k)
        routes = bytearray(num_tokens * rank_size)
        tokens_per_rank = array("i", [0]) * rank_size

        for token in range(num_tokens):
            for hidden_idx in range(args.hidden):
                value = ((src_rank + 1) * 17 + (token + 1) * 3 + hidden_idx) % 127 - 63
                if src_rank == 0 and token == 0:
                    value = 0  # Exercise the zero-amax path, including M=1.
                x[token * args.hidden + hidden_idx] = bf16_bits(float(value))
            for route in range(args.top_k):
                expert = expert_for(
                    src_rank, token, route, rank_size, args.group_size, args.experts_per_rank, args.route_mode
                )
                topk_idx[token * args.top_k + route] = expert
                topk_weights[token * args.top_k + route] = src_rank * 10000 + token * 100 + route + 0.5
                routes[token * rank_size + expert // args.experts_per_rank] = 1
        for token in range(num_tokens):
            for dst_rank in range(rank_size):
                tokens_per_rank[dst_rank] += routes[token * rank_size + dst_rank]

        all_x.append(x)
        quant_x = bytearray()
        scales = array('f')
        for token in range(num_tokens):
            q, scale = quantize_row(x[token * args.hidden : (token + 1) * args.hidden])
            quant_x.extend(q)
            scales.append(scale)
        all_quant_x.append(quant_x)
        all_scales.append(scales)
        all_topk_idx.append(topk_idx)
        all_topk_weights.append(topk_weights)
        all_routes.append(routes)
        all_tokens_per_rank.append(tokens_per_rank)

    rank_prefix = array("i", [0]) * (rank_size * rank_size)
    accum = [0] * rank_size
    for src_rank in range(rank_size):
        for dst_rank in range(rank_size):
            accum[dst_rank] += all_tokens_per_rank[src_rank][dst_rank]
            rank_prefix[src_rank * rank_size + dst_rank] = accum[dst_rank]

    for rank in range(rank_size):
        save(args.data_dir, "input", "x", rank, all_x[rank])
        save(args.data_dir, "input", "topk_idx", rank, all_topk_idx[rank])
        save(args.data_dir, "input", "topk_weights", rank, all_topk_weights[rank])
        save(args.data_dir, "input", "num_tokens_per_rank", rank, all_tokens_per_rank[rank])
        save(args.data_dir, "input", "is_token_in_rank", rank, all_routes[rank])
        save(args.data_dir, "input", "rank_prefix_matrix", rank, rank_prefix)

        # The receive region is the concatenation over source ranks of the
        # tokens addressed to this rank, in input token order.  There is no
        # channel dimension any more: a source's rows are contiguous.
        num_recv_tokens = rank_prefix[(rank_size - 1) * rank_size + rank]
        num_output_tokens = num_recv_tokens
        recv_x = bytearray(num_output_tokens * args.hidden)
        recv_scales = array('f', [0.0]) * num_output_tokens
        recv_topk_idx = array("q", [-1]) * (num_output_tokens * args.top_k)
        recv_topk_weights = array("f", [0.0]) * (num_output_tokens * args.top_k)
        recv_src_idx = array("i", [0]) * num_output_tokens

        recv_position = 0
        for src_rank in range(rank_size):
            for token in range(num_tokens):
                if all_routes[src_rank][token * rank_size + rank] == 0:
                    continue
                x_begin = token * args.hidden
                out_x_begin = recv_position * args.hidden
                recv_x[out_x_begin : out_x_begin + args.hidden] = all_quant_x[src_rank][x_begin : x_begin + args.hidden]
                recv_scales[recv_position] = all_scales[src_rank][token]
                recv_src_idx[recv_position] = token
                for route in range(args.top_k):
                    expert = all_topk_idx[src_rank][token * args.top_k + route]
                    if expert // args.experts_per_rank == rank:
                        recv_topk_idx[recv_position * args.top_k + route] = expert - rank * args.experts_per_rank
                        recv_topk_weights[recv_position * args.top_k + route] = all_topk_weights[src_rank][
                            token * args.top_k + route
                        ]
                recv_position += 1
        if recv_position != num_recv_tokens:
            raise RuntimeError("golden receive count does not match rank_prefix")

        save(args.data_dir, "golden", "recv_x", rank, recv_x)
        save(args.data_dir, "golden", "recv_x_scales", rank, recv_scales)
        save(args.data_dir, "golden", "recv_topk_idx", rank, recv_topk_idx)
        save(args.data_dir, "golden", "recv_topk_weights", rank, recv_topk_weights)
        save(args.data_dir, "golden", "recv_src_idx", rank, recv_src_idx)

    print(
        f"Generated Dispatch data: ranks={rank_size}, M={num_tokens}, K={args.hidden}, "
        f"topK={args.top_k}, groupSize={args.group_size}, experts={num_experts}, route={args.route_mode}"
    )


if __name__ == "__main__":
    main()
