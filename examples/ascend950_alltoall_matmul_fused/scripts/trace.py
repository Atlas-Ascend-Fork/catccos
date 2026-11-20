#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Convert this example's AscendTimer CSV to Chrome tracing JSON (rank-local clocks)."""

import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time


LABELS = {
    'AIC': 'AIC tile scope (Tik/Tok pipeline barriers)',
    'AIC_WAIT': 'AIC wait frontier',
    'AIV': 'AIV send window scope',
    'AIV_WAIT': 'AIV communication wait',
}
NOTE = (
    'Each rank is independently aligned to its earliest device kernel entry. '
    'Cross-rank launch offsets are NOT measured. Within a rank, AIC/AIV share the device clock. '
    'Stages use the original Tik/Tok with PIPE_ALL barriers; tile scopes are not pure matrix-engine utilization. '
    'Sender AIV scopes include submission and Ready.Notify; RDMA completion is in the final AIV_WAIT. '
    'Coordinator AIV_WAIT marks all-source wait and frontier publication. '
    'The coordinator wait end is an upper bound on observed incoming completion.'
)


def metadata(pid, tid, name):
    return dict(
        ph='M', name='thread_name' if tid is not None else 'process_name', pid=pid, tid=tid or 0, args={'name': name}
    )


def read_rank(path, world, group_size, chunk_m, cycles_per_us, window_rows=512, k=None, n=4096):
    match = re.fullmatch(r'rank(\d+)_iter(\d+)_raw.csv', path.name)
    rank, iteration = map(int, match.groups())
    if rank >= world:
        raise ValueError(f'{path}: rank outside world')
    with path.open(encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    aic_count = sum(row['core_type'] == 'AIC' for row in rows)
    if aic_count == 0 or len(rows) != aic_count * 3:
        raise ValueError(f'{path}: incomplete core records')
    windows = (chunk_m + window_rows - 1) // window_rows
    coordinator = world
    tile_cols = (n + 255) // 256
    tile_count = ((chunk_m + 255) // 256) * tile_cols
    fixed = [
        (int(r['KERNEL_TIMING_IDX_start_cycle']), int(r['KERNEL_TIMING_IDX_end_cycle']))
        for r in rows
        if int(r['KERNEL_TIMING_IDX_start_cycle']) > 0
    ]
    if not fixed or any(end <= start for start, end in fixed):
        raise ValueError(f'{path}: incomplete kernel timestamps')
    origin = min(start for start, _ in fixed)
    pid = rank
    events = [metadata(pid, None, f'rank {rank}, iter {iteration} (local time origin)')]
    stages = {name: [] for name in LABELS}
    for row in rows:
        group, sub = int(row['group_id']), int(row['sub_group_id'])
        is_aic = row['core_type'] == 'AIC'
        core = group * 2 + sub
        peer = core
        tid = group if is_aic else aic_count + core
        if is_aic:
            name = f'AIC {group}'
            expected = {'AIC': len(range(group, tile_count, aic_count))}
            expected['AIC_WAIT'] = expected['AIC']
        elif core < coordinator:
            route = 'MTE' if peer == rank else ('UDMA' if peer // group_size == rank // group_size else 'RDMA')
            name = f'AIV {core} -> peer {peer} QP 0 ({route})'
            expected = {'AIV': windows, 'AIV_WAIT': int(route == 'RDMA')}
        elif core == coordinator:
            name = f'AIV {core} coordinator'
            expected = {'AIV_WAIT': windows}
        else:
            expected = {}
            name = f'AIV {core} inactive'
        for stage in LABELS:
            count = int(row.get(f'{stage}_IterCount', 0))
            if count != expected.get(stage, 0):
                raise ValueError(f'{path}: {name} {stage} count={count}, expected={expected.get(stage, 0)}')
        if not expected:
            continue
        events.append(metadata(pid, tid, name))
        for stage, count in expected.items():
            for index in range(count):
                begin = int(row[f'{stage}_{index}_start_cycle'])
                end = int(row[f'{stage}_{index}_end_cycle'])
                if begin < origin or end < begin:
                    raise ValueError(f'{path}: invalid {stage}_{index} interval')
                start = (begin - origin) / cycles_per_us
                finish = (end - origin) / cycles_per_us
                details = {'index': index}
                if is_aic:
                    tile = group + index * aic_count
                    tile_row = (tile // tile_cols) * 256
                    tile_rows = min(256, chunk_m - tile_row)
                    details.update(
                        tile=tile,
                        window=(tile_row + tile_rows - 1) // window_rows,
                        column_tile=tile % tile_cols,
                        row_start=tile_row,
                        rows=tile_rows,
                    )
                elif stage == 'AIV':
                    details.update(peer=peer, qp=0, transport=route, window=index)
                elif stage == 'AIV_WAIT' and core < coordinator:
                    details.update(peer=peer, qp=0, transport=route, role='final_quiet')
                else:
                    details.update(window=index, role='receive_window')
                stages[stage].append((start, finish, tid, index))
                event_name = LABELS[stage]
                if stage == 'AIV_WAIT':
                    event_name = (
                        'AIV wait all sources + publish frontier' if core == coordinator else 'AIV final RDMA quiet'
                    )
                events.append(
                    dict(
                        ph='X',
                        cat=stage,
                        name=event_name,
                        pid=pid,
                        tid=tid,
                        ts=round(start, 3),
                        dur=round(finish - start, 3),
                        args=details,
                    )
                )
        # Keep whole-kernel timestamps in the raw CSV/report; avoid a duplicate lane per core.

    frontiers = sorted(
        (event for event in stages['AIV_WAIT'] if event[2] == aic_count + coordinator), key=lambda event: event[3]
    )
    first_ready, last_ready = frontiers[0][1], frontiers[-1][1]
    receive_begin = frontiers[0][0]
    compute = stages['AIC']
    tile_sum = sum(end - begin for begin, end, _, _ in compute)
    wait_sum = sum(end - begin for begin, end, _, _ in stages['AIC_WAIT'])
    overlap = sum(max(0, min(end, last_ready) - max(begin, receive_begin)) for begin, end, _, _ in compute)
    report = dict(
        rank=rank,
        iteration=iteration,
        M=chunk_m,
        N=n,
        K=k or world * 256,
        window_rows=window_rows,
        rdma_qps=1,
        timing_method='Tik/Tok with PIPE_ALL',
        windows=windows,
        aic_cores=aic_count,
        kernel_span_us=(max(end for _, end in fixed) - origin) / cycles_per_us,
        first_frontier_us=first_ready,
        last_frontier_us=last_ready,
        first_tile_us=min(begin for begin, _, _, _ in compute),
        last_tile_scope_end_us=max(end for _, end, _, _ in compute),
        mean_aic_frontier_wait_us=wait_sum / aic_count,
        mean_aic_tile_scope_us=tile_sum / aic_count,
        aic_wait_fraction_of_recorded_scopes=wait_sum / (wait_sum + tile_sum) if wait_sum + tile_sum else 0,
        tile_scope_during_receive_fraction=overlap / tile_sum if tile_sum else 0,
    )
    return rank, iteration, events, report


def write_trace(path, events, cycles_per_us):
    with path.open('w', encoding='utf-8') as output:
        json.dump(
            {
                'traceEvents': events,
                'displayTimeUnit': 'ms',
                'otherData': {'clock_note': NOTE, 'cycles_per_us': cycles_per_us},
            },
            output,
            separators=(',', ':'),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, help='trace directories from either or both nodes')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--world', type=int, default=32)
    parser.add_argument('--dtype', choices=('mxfp8', 'bf16'), default='mxfp8')
    parser.add_argument('--group-size', type=int, default=16)
    parser.add_argument('--m', required=True, type=int, help='output rows per rank')
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--k', type=int, help='full K after AllToAll; default world*256')
    parser.add_argument(
        '--window-rows',
        type=int,
        choices=(256, 512, 1024),
        default=512,
        help='communication window rows used by this run; compute tiles remain 256 rows',
    )
    parser.add_argument('--cycles-per-us', type=float, default=1000.0, help='Ascend950 GetSystemCycle: 1 GHz')
    parser.add_argument(
        '--ranks', default='0,16', help='ranks included in combined JSON; each rank also has its own JSON'
    )
    args = parser.parse_args()
    if (
        args.cycles_per_us <= 0
        or args.group_size not in (1, 2, 4, 8, 16)
        or args.world not in (args.group_size, 2 * args.group_size)
    ):
        parser.error('invalid clock/topology')
    if not 1 <= args.m <= 300000:
        parser.error('--m must be in [1,300000]')
    args.k = args.k if args.k is not None else args.world * 256
    k_multiple = args.world * (1 if args.dtype == 'bf16' else 64)
    if not 0 < args.k <= 0xFFFFFFFF or args.k % k_multiple:
        parser.error(f'--k must be a positive uint32 multiple of {k_multiple}')
    if not 0 < args.n <= 0xFFFFFFFF or args.n % 16:
        parser.error('--n must be a positive uint32 multiple of 16')
    selected = {int(value) for value in args.ranks.split(',')}
    paths = sorted(path for directory in args.inputs for path in directory.rglob('rank*_iter*_raw.csv'))
    if not paths:
        parser.error('no raw timer CSV found')
    args.output.mkdir(parents=True, exist_ok=True)
    reports, combined, seen = [], {}, set()
    for path in paths:
        rank, iteration, events, report = read_rank(
            path, args.world, args.group_size, args.m, args.cycles_per_us, args.window_rows, args.k, args.n
        )
        if (rank, iteration) in seen:
            raise ValueError(f'duplicate rank/iteration: {path}')
        seen.add((rank, iteration))
        reports.append(report)
        write_trace(args.output / f'rank{rank}_iter{iteration}.json', events, args.cycles_per_us)
        if rank in selected:
            combined.setdefault(iteration, []).extend(events)
    for iteration, events in combined.items():
        write_trace(args.output / f'combined_iter{iteration}.json', events, args.cycles_per_us)
    with (args.output / 'summary.csv').open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(sorted(reports, key=lambda r: (r['iteration'], r['rank'])))
    (args.output / 'README.txt').write_text(NOTE + '\n', encoding='utf-8')
    print(f'Converted {len(reports)} rank/iteration records to {args.output}. {NOTE}')


MODES = ("hccl", "torch", "serial")
OWN_MODES = ("fused", "comm", "matmul", "serial")
TIMING_SCOPE = "gm_input_to_output_host"


def parse_shape(value):
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use M,N,K, for example 17,4096,12288") from error
    if len(shape) != 3 or min(shape) <= 0:
        raise argparse.ArgumentTypeError("M,N,K must be positive")
    return shape


def summarize(rank_samples, iterations):
    if not rank_samples or any(len(samples) != iterations for samples in rank_samples):
        raise ValueError("Every rank must provide exactly the requested number of samples")
    if any(not math.isfinite(value) or value <= 0 for samples in rank_samples for value in samples):
        raise ValueError("Timings must be positive and finite")
    maxima = [max(samples[index] for samples in rank_samples) for index in range(iterations)]
    return {
        "mean_us": statistics.mean(maxima),
        "stdev_us": statistics.stdev(maxima) if iterations > 1 else 0.0,
        "min_us": min(maxima),
        "max_us": max(maxima),
        "round_max_us": maxima,
        "rank_samples_us": rank_samples,
    }


def read_own_rank(log_dir, shape, rank, world, warmup, iterations):
    samples = {mode: {} for mode in OWN_MODES}
    configurations = set()
    path = log_dir / f"rank{rank}.log"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("TIMING "):
            continue
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        mode = fields.get("mode")
        if mode not in OWN_MODES:
            raise ValueError(f"Expected OVERLAP=1 logs: {path}")
        if tuple(int(fields[key]) for key in ("M", "N", "K")) != tuple(shape):
            raise ValueError(f"Shape mismatch: {path}")
        if (int(fields["rank"]), int(fields["world"])) != (rank, world):
            raise ValueError(f"Rank/world mismatch: {path}")
        scope = "received_input_to_output" if mode == "matmul" else "gm_input_to_output"
        if fields.get("timing_scope") != scope:
            raise ValueError(f"Expected {scope} host timing: {path}")
        iteration = int(fields["iter"])
        if not warmup <= iteration < warmup + iterations:
            continue
        if iteration in samples[mode]:
            raise ValueError(f"Duplicate {mode}/{iteration}: {path}")
        configurations.add((fields["window_rows"], fields["group_size"]))
        samples[mode][iteration] = float(fields["total_ms"]) * 1000
    expected = set(range(warmup, warmup + iterations))
    if any(item.keys() != expected for item in samples.values()) or len(configurations) != 1:
        raise ValueError(f"Missing samples or mixed configurations: {path}")
    return {
        "samples": {mode: [samples[mode][i] for i in sorted(expected)] for mode in OWN_MODES},
        "configuration": configurations.pop(),
    }


def summarize_own(ranks, iterations):
    configuration = ranks[0]["configuration"]
    if any(rank["configuration"] != configuration for rank in ranks):
        raise ValueError("Own configurations differ across ranks")
    return {
        "own_stats": {mode: summarize([rank["samples"][mode] for rank in ranks], iterations) for mode in OWN_MODES},
        "own_configuration": {"window_rows": int(configuration[0]), "group_size": int(configuration[1])},
    }


def make_report(baseline):
    if baseline.get("timing_scope") != TIMING_SCOPE or baseline.get("dtype") not in ("BF16", "MXFP8"):
        raise ValueError("Expected BF16 or MXFP8 complete host timings")
    has_hccl = baseline["dtype"] == "BF16"
    report = {
        "baseline": baseline,
        "formulas": {"own_overlap_percent": "(own_serial - fused) / min(own_comm, own_matmul) * 100"},
        "rows": [],
    }
    if has_hccl:
        report["formulas"]["hccl_torch_overlap_percent"] = "(serial - fused) / min(hccl, torch) * 100"
    for row in baseline["rows"]:
        own = row["own_stats"]
        stats = row["stats"] if has_hccl else {}
        if any(not math.isfinite(item["mean_us"]) or item["mean_us"] <= 0 for item in (*own.values(), *stats.values())):
            raise ValueError("Baseline means must be positive and finite")
        fused = own["fused"]["mean_us"]
        own_eta = (own["serial"]["mean_us"] - fused) / min(own["comm"]["mean_us"], own["matmul"]["mean_us"]) * 100
        timings = {
            "shape": row["shape"],
            "stats": {"fused": own["fused"]},
            "own_stats": own,
            "own_overlap_percent": own_eta,
        }
        if has_hccl:
            denominator = min(stats["hccl"]["mean_us"], stats["torch"]["mean_us"])
            timings["stats"].update(stats)
            timings["hccl_torch_overlap_percent"] = (stats["serial"]["mean_us"] - fused) / denominator * 100
        report["rows"].append(timings)
    report["interpretation"] = (
        "Own overlap compares identical communication/MatMul code in serial and fused execution. "
        "It includes launch/synchronization effects. Reported values retain results outside [0,100]."
    )
    if has_hccl:
        report["interpretation"] += (
            " HCCL/Torch overlap is a cross-implementation end-to-end score that also includes framework differences."
        )
    return report


def print_rows(rows):
    has_hccl = "hccl_torch_overlap_percent" in rows[0]
    hccl_columns = "HCCL/ms  Torch/ms  HCCL_serial/ms  " if has_hccl else ""
    hccl_score = "  HCCL_Torch/%" if has_hccl else ""
    print(f"M,N,K  {hccl_columns}own_comm/ms  own_mm/ms  own_serial/ms  fused/ms  own/%{hccl_score}", flush=True)
    for row in rows:
        cells = [",".join(map(str, row["shape"]))]
        if has_hccl:
            cells.extend(f"{row['stats'][mode]['mean_us'] / 1000:.6f}" for mode in MODES)
        cells.extend(
            f"{row['own_stats'][mode]['mean_us'] / 1000:.6f}" for mode in ("comm", "matmul", "serial", "fused")
        )
        cells.append(f"{row['own_overlap_percent']:.2f}")
        if has_hccl:
            cells.append(f"{row['hccl_torch_overlap_percent']:.2f}")
        print("  ".join(cells), flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_tensor(torch, path, shape):
    count = math.prod(shape)
    if path.stat().st_size != count * 2:
        raise ValueError(f"Incorrect BF16 file size: {path}")
    storage = torch.from_file(str(path), shared=False, size=count, dtype=torch.int16)
    return storage.view(torch.bfloat16).reshape(shape)


def check_output(torch, actual, expected):
    if not torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16)):
        raise RuntimeError("Full BF16 output comparison failed")


def measure_case(torch, dist, args, rank, world, device, shape):
    m, n, k = shape
    if k % world:
        raise ValueError("K must be divisible by WORLD_SIZE")
    local_k = k // world
    data = args.data_path
    own = read_own_rank(args.own_logs, shape, rank, world, args.warmup, args.iterations)
    meta = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if (meta["world"], meta["M"], meta["N"], meta["K"], meta["dtype"], meta["b_layout"]) != (
        world,
        m,
        n,
        k,
        "BF16",
        "ColumnMajor",
    ):
        raise ValueError(f"Input manifest mismatch: {data}")
    a = load_tensor(torch, data / f"input_a_{rank}.bin", (world * m, local_k)).to(device)
    b = load_tensor(torch, data / f"input_b_{rank}.bin", (n, k)).t().to(device)
    reference = load_tensor(torch, data / f"output_{rank}.bin", (m, n))
    if b.stride() != (1, k):
        raise ValueError("B must retain column-major layout")
    send, receive = torch.empty_like(a), torch.empty_like(a)
    compact = torch.empty((m, k), dtype=torch.bfloat16, device=device)
    received_view = receive.view(world, m, local_k)
    compact_view = compact.view(m, world, local_k).permute(1, 0, 2)
    output = torch.empty((m, n), dtype=torch.bfloat16, device=device)

    def execute(mode):
        if mode in ("hccl", "serial"):
            dist.all_to_all_single(receive, send, async_op=False)
        if mode == "serial":
            compact_view.copy_(received_view)
        if mode in ("torch", "serial"):
            torch.mm(compact, b, out=output)
        torch.npu.synchronize()

    # Correctness calls change A and compare every C element with a supplied reference.
    for scale in (1, 2):
        torch.mul(a, scale, out=send)
        expected = reference * scale
        torch.npu.synchronize()
        dist.barrier()
        execute("serial")
        check_output(torch, output, expected)
        execute("torch")
        check_output(torch, output, expected)
    print(f"VALIDATION rank={rank} shape={m},{n},{k} full=1 calls=4 status=PASS", flush=True)
    send.copy_(a)
    execute("serial")
    check_output(torch, output, reference)

    timings = {mode: [] for mode in MODES}
    # Fixed inputs, no CPU copies, file IO or validation between timed repetitions.
    for iteration in range(args.warmup + args.iterations):
        for offset in range(len(MODES)):
            mode = MODES[(iteration + offset) % len(MODES)]
            dist.barrier()
            torch.npu.synchronize()
            begin = time.perf_counter_ns()
            execute(mode)
            elapsed = (time.perf_counter_ns() - begin) / 1000
            if iteration >= args.warmup:
                timings[mode].append(elapsed)
    check_output(torch, output, reference)
    gathered = [None] * world
    dist.all_gather_object(gathered, {"timings": timings, "own": own})
    return {
        "shape": list(shape),
        "stats": {mode: summarize([item["timings"][mode] for item in gathered], args.iterations) for mode in MODES},
        **summarize_own([item["own"] for item in gathered], args.iterations),
    }


def overlap_baseline(args, world):
    return {
        "dtype": args.dtype.upper(),
        "world": world,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timing_scope": TIMING_SCOPE,
        "statistic": "Arithmetic mean of per-round maximum rank host durations",
        "rows": [],
    }


def emit_overlap(args, baseline):
    report = make_report(baseline)
    write_json(args.output, report)
    print_rows(report["rows"])
    print(report["interpretation"])


def measure_own(args):
    import torch.distributed as dist

    rank, world = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE"))
    own = read_own_rank(args.own_logs, args.shape, rank, world, args.warmup, args.iterations)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=180))
    gathered = [None] * world
    try:
        dist.all_gather_object(gathered, own)
    finally:
        dist.destroy_process_group()
    if rank == 0:
        result = overlap_baseline(args, world)
        result["rows"].append({"shape": list(args.shape), **summarize_own(gathered, args.iterations)})
        emit_overlap(args, result)


def measure(args):
    import torch
    import torch.distributed as dist
    import torch_npu

    rank, world, local_rank = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device_id = args.devices[local_rank]
    torch.npu.set_device(device_id)
    torch.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=180))
    result = {
        **overlap_baseline(args, world),
        "B_layout": "ColumnMajor",
        "serial_path": "HCCL -> receive-layout copy -> torch.mm -> NPU synchronize",
        "environment": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "cann": os.environ.get("ASCEND_HOME_PATH"),
            "hccl_expansion_mode": os.environ.get("HCCL_OP_EXPANSION_MODE", "default"),
        },
    }
    with torch.inference_mode():
        result["rows"].append(measure_case(torch, dist, args, rank, world, f"npu:{device_id}", args.shape))
    dist.destroy_process_group()
    if rank == 0:
        emit_overlap(args, result)


def overlap_main(argv):
    parser = argparse.ArgumentParser(description="Measure Ascend950 AllToAll MatMul overlap")
    parser.add_argument("--dtype", choices=("mxfp8", "bf16"), default="bf16")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--own-logs", type=Path, required=True)
    parser.add_argument("--shape", type=parse_shape, required=True, help="M,N,K")
    parser.add_argument("--devices", type=lambda value: tuple(map(int, value.split(","))))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.iterations < 2:
        parser.error("Use at least one warmup and two measured repetitions")
    if args.dtype == "mxfp8":
        measure_own(args)
        return
    if args.data_path is None:
        parser.error("--data-path is required for BF16")
    if not args.devices or len(set(args.devices)) != len(args.devices) or any(not 0 <= i < 16 for i in args.devices):
        parser.error("Use distinct device IDs in [0,15]")
    measure(args)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'overlap':
        overlap_main(sys.argv[2:])
    else:
        main()
