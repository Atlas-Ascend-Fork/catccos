#!/usr/bin/env python3
"""Profile the native CATCCOS Dispatch; no torch_npu operator is substituted."""

import argparse
import csv
import json
import math
import os
import re
import signal
import shlex
import shutil
import statistics
import subprocess  # nosec B404 - Required for local benchmark processes; shell execution is disabled.
import sys
import time
from pathlib import Path

HIDDEN, TOPK, DTYPE_BYTES, NUM_EXPERTS = 3584, 8, 1, 384
SAMPLES_PER_RANK, FASTEST_SAMPLES = 100, 70
TOKENS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)
FIELDS = ('OP', 'token per rank', 'EP', 'p50_us', 'p95_us', 'p99_us', '数据量KB', '平均耗时', '算法带宽', '带宽利用率')
SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[2]


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def read_samples(directory, pattern, count):
    # One fresh profile per rank/case. Refuse ambiguous duplicate exports.
    paths = sorted(directory.rglob('op_summary*.csv'))
    if not paths:
        raise ValueError(
            f'No op_summary*.csv found under {directory}. '
            f'Inspect {directory.parent / (directory.name + ".log")} and the profiler export.'
        )
    name_columns = ('Op Name', 'OP Name', 'Name', 'Op Type', 'OP Type')
    candidates = []
    for path in paths:
        values = []
        with path.open(encoding='utf-8-sig', newline='') as stream:
            header = stream.readline()
            stream.seek(0)
            try:
                delimiter = csv.Sniffer().sniff(header, delimiters=',\t').delimiter
            except csv.Error as error:
                raise ValueError(f'Unrecognized op_summary delimiter: {path}; header={header!r}') from error
            reader = csv.DictReader(stream, delimiter=delimiter)
            reader.fieldnames = [key.strip() for key in (reader.fieldnames or [])]
            if (
                'Task Duration(us)' not in reader.fieldnames
                or 'Task Type' not in reader.fieldnames
                or not any(key in reader.fieldnames for key in name_columns)
            ):
                raise ValueError(f'Unexpected op_summary columns: {path}; columns={reader.fieldnames}')
            for row in reader:
                name = ' '.join(row.get(key, '') for key in name_columns)
                if not re.search(pattern, name, re.IGNORECASE):
                    continue
                if row.get('Task Type') in ('AI_CPU', 'N/A', 'HCCL'):
                    continue
                value = float(row['Task Duration(us)'])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f'Invalid duration in {path}: {value}')
                values.append(value)
        if values:
            candidates.append((path, values))
    if len(candidates) != 1 or len(candidates[0][1]) != count:
        details = [(str(p), len(v)) for p, v in candidates]
        raise ValueError(
            f'Expected one Dispatch export with {count} launches; got {details}. '
            f'Exports inspected: {[str(path) for path in paths]}. '
            'Inspect op_summary and set --op-pattern to the kernel name.'
        )
    return candidates[0][1][-SAMPLES_PER_RANK:]


def result_row(tokens, selected_per_rank, ep):
    samples = []
    for rank_samples in selected_per_rank:
        samples.extend(rank_samples)
    average = statistics.mean(statistics.mean(rank_samples) for rank_samples in selected_per_rank)
    size = tokens * HIDDEN * TOPK * DTYPE_BYTES
    bandwidth = size * 1_000_000 / average / (1024**3)  # GB/s uses binary GB; average is in us.
    return [
        'CATCCOS Dispatch BF16-INT8',
        tokens,
        ep,
        *(percentile(samples, q) for q in (0.50, 0.95, 0.99)),
        size / 1024,
        average,
        bandwidth,
        bandwidth / 196 * 100,
    ]


def summarize(inputs, output):
    records = {}
    configuration = None
    for directory in inputs:
        if not directory.is_dir():
            raise ValueError(f'Input directory not found: {directory}')
        for path in directory.rglob('samples_rank_*.json'):
            record = json.loads(path.read_text())
            ep, group_size = record['ep'], record['group_size']
            if (
                not TOPK <= ep <= 128
                or NUM_EXPERTS % ep
                or group_size <= 0
                or ep % group_size
                or not 0 <= record['rank'] < ep
                or record['hidden'] != HIDDEN
                or record['topk'] != TOPK
                or record.get('num_experts') != NUM_EXPERTS
                or record['dtype'] != 'int8'
            ):
                raise ValueError(f'Unexpected configuration: {path}')
            test_tokens = tuple(record.get('test_tokens', TOKENS))
            if (
                not test_tokens
                or min(test_tokens) <= 0
                or len(set(test_tokens)) != len(test_tokens)
                or record['tokens'] not in test_tokens
                or record['iters'] < SAMPLES_PER_RANK
            ):
                raise ValueError(f'Invalid sampling configuration: {path}')
            devices = tuple(record.get('devices', ()))
            if devices and (len(devices) != ep or len(set(devices)) != ep or min(devices) < 0):
                raise ValueError(f'Invalid device mapping: {path}')
            if 'device_id' in record and (not devices or record['device_id'] != devices[record['rank']]):
                raise ValueError(f'Invalid rank/device mapping: {path}')
            current = (
                ep,
                group_size,
                record['iters'],
                test_tokens,
                devices,
                record.get('comm_interval'),
                record.get('comm_block_m'),
            )
            if configuration is not None and current != configuration:
                raise ValueError('Cannot combine different EP or run configurations')
            configuration = current
            key = (record['tokens'], record['rank'])
            if key in records:
                raise ValueError(f'Duplicate rank/case: {key}')
            records[key] = record
    if not records:
        raise ValueError('No samples_rank_*.json found')
    ep, _, _, test_tokens, *_ = configuration
    rows = []
    for tokens in test_tokens:
        per_rank_samples = []
        counts = set()
        for rank in range(ep):
            record = records.get((tokens, rank))
            if record is None:
                raise ValueError(f'Missing tokens={tokens}, rank={rank}; all {ep} ranks are required')
            values = record['samples_us']
            if not values or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f'Invalid samples: tokens={tokens}, rank={rank}')
            counts.add(len(values))
            per_rank_samples.append(values)
        if counts != {SAMPLES_PER_RANK}:
            raise ValueError(f'Expected {SAMPLES_PER_RANK} samples per rank for tokens={tokens}; got {sorted(counts)}')
        selected_per_rank = [sorted(values)[:FASTEST_SAMPLES] for values in per_rank_samples]
        rows.append(result_row(tokens, selected_per_rank, ep))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows([[f'{v:.6f}' if isinstance(v, float) else v for v in row] for row in rows])
    print(f'Performance results: {output}')


def run(args):
    ep = args.ep
    if not TOPK <= ep <= 128 or NUM_EXPERTS % ep:
        raise ValueError(f'ep must be in [{TOPK}, 128] and divide {NUM_EXPERTS} experts')
    experts_per_rank = NUM_EXPERTS // ep if args.experts_per_rank is None else args.experts_per_rank
    devices = list(range(ep)) if args.devices is None else [int(value) for value in args.devices.split(',')]
    if (
        args.nodes != 1
        or args.node_rank != 0
        or len(devices) != ep
        or len(set(devices)) != len(devices)
        or min(devices) < 0
    ):
        raise ValueError(f'This benchmark requires one node with {ep} unique local device IDs')
    if (
        args.iters < SAMPLES_PER_RANK
        or args.group_size <= 0
        or ep % args.group_size
        or min(experts_per_rank, args.comm_interval, args.comm_block_m) <= 0
        or min(args.tprs) <= 0
        or len(set(args.tprs)) != len(args.tprs)
    ):
        raise ValueError(
            f'iters must be >={SAMPLES_PER_RANK}; tiling/tokens must be positive; '
            f'group_size must divide {ep}; tprs must not contain duplicates'
        )
    if ep * experts_per_rank != NUM_EXPERTS:
        raise ValueError(f'ep * experts_per_rank must equal {NUM_EXPERTS}; expected {NUM_EXPERTS // ep} per rank')
    msprof = shutil.which('msprof')
    if not args.binary.is_file() or msprof is None:
        raise ValueError('Build ascend950_dispatch and source CANN/SHMEM environment (msprof required)')
    msprof = str(Path(msprof).resolve(strict=True))
    # Never silently mix an old trace with this run.
    node_dir = args.output.resolve() / f'node_{args.node_rank}'
    if node_dir.exists():
        raise ValueError(f'Output already exists: {node_dir}; use a fresh --output directory')
    node_dir.mkdir(parents=True, exist_ok=False)
    device_map = ','.join(map(str, devices * args.nodes))
    for tokens in args.tprs:
        case = node_dir / f'tprs_{tokens}'
        data = case / 'data'
        # Fixed repository generator, absolute interpreter, and separate argv entries.
        subprocess.run(  # nosec B603 - Trusted local generator; no shell command interpolation.
            [
                str(Path(sys.executable).resolve(strict=True)),
                str(SCRIPT / 'get_data.py'),
                str(ep),
                str(tokens),
                str(HIDDEN),
                str(args.group_size),
                str(TOPK),
                str(experts_per_rank),
                str(data),
            ],
            check=True,
            shell=False,
        )
        processes, handles = [], []
        try:
            for local_rank in range(len(devices)):
                rank = args.node_rank * len(devices) + local_rank
                trace = case / f'rank_{rank}'
                command = [
                    str(args.binary.resolve()),
                    str(ep),
                    str(rank),
                    args.master,
                    str(tokens),
                    str(HIDDEN),
                    str(args.group_size),
                    str(TOPK),
                    str(experts_per_rank),
                    str(args.comm_interval),
                    str(args.comm_block_m),
                    str(data),
                    device_map,
                    '0',
                    str(args.iters),
                ]
                log = (case / f'rank_{rank}.log').open('w')
                handles.append(log)
                processes.append(
                    # CANN profiler from the configured environment; application argv is shell-quoted.
                    subprocess.Popen(  # nosec B603 - Explicit profiler path and argv; shell=False.
                        [
                            msprof,
                            '--application=' + shlex.join(command),
                            '--output=' + str(trace),
                            '--aic-metrics=PipeUtilization',
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        shell=False,
                    )
                )
            deadline = time.monotonic() + args.timeout
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError(f'Profiling failed; inspect {case}/*.log')
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Profiling timeout; inspect {case}/*.log')
                time.sleep(0.2)
            if any(p.returncode != 0 for p in processes):
                raise RuntimeError(f'Profiling failed; inspect {case}/*.log')
        finally:
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for handle in handles:
                handle.close()
        for local_rank in range(len(devices)):
            rank = args.node_rank * len(devices) + local_rank
            samples = read_samples(case / f'rank_{rank}', args.op_pattern, args.iters)
            record = dict(
                rank=rank,
                device_id=devices[local_rank],
                devices=devices,
                tokens=tokens,
                test_tokens=args.tprs,
                ep=ep,
                hidden=HIDDEN,
                topk=TOPK,
                dtype='int8',
                group_size=args.group_size,
                iters=args.iters,
                num_experts=ep * experts_per_rank,
                comm_interval=args.comm_interval,
                comm_block_m=args.comm_block_m,
                samples_us=samples,
            )
            (case / f'samples_rank_{rank}.json').write_text(json.dumps(record, indent=2))
        print(f'[PERF] node={args.node_rank} tokens={tokens} complete', flush=True)
    summarize([node_dir], args.output.resolve() / 'performance.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    launch = commands.add_parser('run')
    launch.add_argument('--node-rank', type=int, choices=[0], default=0)
    launch.add_argument('--nodes', type=int, choices=[1], default=1)
    launch.add_argument('--ep', type=int, default=16, help='Number of ranks on this node (default: 16)')
    launch.add_argument('--devices', help='Exactly ep device IDs; defaults to 0,...,ep-1')
    launch.add_argument('--master', default='tcp://127.0.0.1:27029', help='SHMEM rendezvous')
    launch.add_argument('--output', type=Path, required=True)
    launch.add_argument('--binary', type=Path, default=ROOT / 'build/bin/ascend950_dispatch')
    launch.add_argument('--group-size', type=int, default=4, help='Ranks per group; must divide ep')
    launch.add_argument('--experts-per-rank', type=int, help=f'Default: {NUM_EXPERTS} / ep; total experts stay fixed')
    launch.add_argument(
        '--tprs', nargs='+', type=int, default=list(TOKENS), help='Token counts; default: all 13 shapes'
    )
    launch.add_argument('--comm-interval', type=int, default=1)
    launch.add_argument('--comm-block-m', type=int, default=2)
    launch.add_argument('--iters', type=int, default=100)
    launch.add_argument('--timeout', type=float, default=1800)
    launch.add_argument('--op-pattern', default='dispatch')
    merge = commands.add_parser('summarize')
    merge.add_argument('inputs', nargs='+', type=Path)
    merge.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run(args)
    else:
        summarize(args.inputs, args.output)


if __name__ == '__main__':
    main()
