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
import json
from pathlib import Path
import re


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
    'UDMA and pipelined RDMA send scopes can end with ready still in flight; '
    'Coordinator AIV_WAIT marks all-source wait and frontier publication; sender AIV_WAIT marks final quiet. '
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
            expected = {'AIV': windows, 'AIV_WAIT': int(route != 'MTE')}
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
                    details.update(peer=peer, qp=0, transport=route, window=windows - 1, role='final_quiet')
                else:
                    details.update(window=index, role='receive_window')
                stages[stage].append((start, finish, tid, index))
                event_name = LABELS[stage]
                if stage == 'AIV_WAIT':
                    event_name = (
                        'AIV wait all sources + publish frontier' if core == coordinator else f'AIV final {route} quiet'
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
    if not 0 < args.k <= 0xFFFFFFFF or args.k % (args.world * 64):
        parser.error('--k must be a positive uint32 multiple of world*64')
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


if __name__ == '__main__':
    main()
