#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0.

"""Convert AscendTimer raw CSV files to rank-local Chrome tracing JSON."""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


EVENT_RE = re.compile(r"^(?P<stage>.+)_(?P<index>[0-9]+)_(?P<kind>start|end)_cycle$")
RANK_FILE_RE = re.compile(r"^timer_rank_(?P<rank>[0-9]+)(?P<suffix>.*)_timeline\.csv$")
ROUND_SOURCE_RE = re.compile(r"^round(?P<round>[0-9]+)_(?P<phase>.+)$")
SUBMISSION_STAGES = ("UDMA_COPY", "RDMA_COPY")
STAGE_DISPLAY_ALIASES = {
    "AIV_LOCAL_COPY": "UDMA_COPY",
    "AIV_AG": "RDMA_COPY",
}
QP_TRANSFER_STAGES = ("UDMA_TRANSFER", "RDMA_TRANSFER")
COMPUTE_STAGES = ("AIC", "AIC_COMPUTE")
AIV_ENVELOPE_STAGES = ("AIV",)
SUBMISSION_TO_TRANSFER = {
    "UDMA_COPY": "UDMA_TRANSFER",
    "RDMA_COPY": "RDMA_TRANSFER",
}
SUBMISSION_TO_GAP = {
    "UDMA_COPY": "UDMA_SUBMIT_GAP",
    "RDMA_COPY": "RDMA_SUBMIT_GAP",
}
AIV_STAGE_TRACKS = {
    "AIV": "AIV transport envelope",
    "AIV_WAIT": "AIV waits",
    "UDMA_COPY": "AIV UDMA schedule envelope",
    "RDMA_COPY": "AIV RDMA schedule envelope",
    "UDMA_SUBMIT_GAP": "AIV UDMA issue timeline",
    "RDMA_SUBMIT_GAP": "AIV RDMA issue timeline",
    "UDMA_TRANSFER": "AIV UDMA scheduling envelope",
    "RDMA_TRANSFER": "AIV RDMA scheduling envelope",
    "QUIET": "AIV transport quiet",
    "AIV_LOCAL_COPY": "AIV local copy",
}
CLOCK_NOTE = (
    "Each rank is aligned independently to its earliest KERNEL_TIMING_IDX start. "
    "Cross-rank launch skew is not measured. Files with a rank-local filename suffix are merged "
    "on the same rank/device clock. AIC is one complete logical-round compute span. "
    "AIV is the complete AIV transport state-machine envelope through the final quiet. "
    "UDMA_COPY and RDMA_COPY mark AIV issue windows inside the AIV transport envelope. "
    "UDMA_SUBMIT_GAP and RDMA_SUBMIT_GAP mark issue-side gaps and are not engine-idle "
    "measurements. AIV_WAIT records per-peer producer-ticket waits (no all-rank barrier). "
    "The envelopes support scheduling-overlap analysis, but not hardware-engine-busy utilization. "
    "copy_round is only a nominal predecessor label, not a request index."
)


def merge_intervals(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if end <= begin:
            continue
        if not merged or begin > merged[-1][1]:
            merged.append([begin, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def interval_union_us(intervals):
    return sum(end - begin for begin, end in merge_intervals(intervals))


def interval_intersection_us(left, right):
    left_merged = merge_intervals(left)
    right_merged = merge_intervals(right)
    total = 0.0
    left_index = 0
    right_index = 0
    while left_index < len(left_merged) and right_index < len(right_merged):
        begin = max(left_merged[left_index][0], right_merged[right_index][0])
        end = min(left_merged[left_index][1], right_merged[right_index][1])
        if end > begin:
            total += end - begin
        if left_merged[left_index][1] <= right_merged[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return total


def percentage(numerator_us, denominator_us):
    return 100.0 * numerator_us / denominator_us if denominator_us > 0 else 0.0


def median(values):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def interval_duration_stats(intervals):
    durations = [end - begin for begin, end in intervals if end > begin]
    return {
        "count": len(durations),
        "total_us": sum(durations),
        "mean_us": sum(durations) / len(durations) if durations else 0.0,
        "median_us": median(durations),
        "max_us": max(durations, default=0.0),
    }


def stage_track_label(core_type, stage):
    if core_type != "AIV":
        return "timeline"
    return AIV_STAGE_TRACKS.get(stage, "AIV control and waits")


def transport_aiv_counts(summaries):
    aiv_transport_domains = defaultdict(set)
    for (source, core_type, group, sub_group, stage), values in summaries.items():
        if core_type != "AIV" or values["count"] == 0:
            continue
        if stage == "UDMA_COPY":
            aiv_transport_domains[(source, group, sub_group)].add("UDMA")
        elif stage == "RDMA_COPY":
            aiv_transport_domains[(source, group, sub_group)].add("RDMA")
    return {
        "udma": sum("UDMA" in domains for domains in aiv_transport_domains.values()),
        "rdma": sum("RDMA" in domains for domains in aiv_transport_domains.values()),
        "mixed": sum({"UDMA", "RDMA"}.issubset(domains) for domains in aiv_transport_domains.values()),
    }


def discover_stages(fieldnames):
    stages = set()
    for name in fieldnames:
        match = EVENT_RE.match(name)
        if match:
            stages.add(match.group("stage"))
    return sorted(stages)


def source_name(path):
    match = RANK_FILE_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"unexpected timer filename: {path}")
    suffix = match.group("suffix")
    if suffix.startswith("_"):
        suffix = suffix[1:]
    return suffix or "base"


def collect_intervals(stage_intervals, core_type, stages, source=None):
    selected = []
    for (entry_source, entry_core, stage), intervals in stage_intervals.items():
        if entry_core != core_type or stage not in stages:
            continue
        if source is not None and entry_source != source:
            continue
        selected.extend(intervals)
    return selected


def classify_transport(stage_intervals):
    current_compute = collect_intervals(stage_intervals, "AIC", ("AIC",))
    aiv_envelopes = collect_intervals(stage_intervals, "AIV", ("AIV",))
    qp_transfers = collect_intervals(stage_intervals, "AIV", QP_TRANSFER_STAGES)
    local_copies = collect_intervals(stage_intervals, "AIV", ("AIV_LOCAL_COPY",))
    submissions = collect_intervals(stage_intervals, "AIV", SUBMISSION_STAGES)
    missing_transfers = []
    for submission_stage, transfer_stage in SUBMISSION_TO_TRANSFER.items():
        submission_present = bool(collect_intervals(stage_intervals, "AIV", (submission_stage,)))
        transfer_present = bool(collect_intervals(stage_intervals, "AIV", (transfer_stage,)))
        if submission_present and not transfer_present:
            missing_transfers.append(transfer_stage)

    # Current traces use AIC/AIV plus one compact COPY marker per engine. The
    # parser turns those markers into engine-specific schedule envelopes below.
    if current_compute and aiv_envelopes:
        envelopes = qp_transfers if qp_transfers else (submissions if submissions else aiv_envelopes)
        return envelopes + local_copies, "aiv_schedule_envelope", []
    if qp_transfers and not missing_transfers:
        return qp_transfers + local_copies, "qp_completion_envelope", []
    if aiv_envelopes:
        return aiv_envelopes + local_copies, "aiv_schedule_envelope", []
    if qp_transfers:
        return qp_transfers + local_copies, "partial_qp_completion_envelope", missing_transfers
    if submissions:
        return (
            submissions + local_copies,
            "legacy_submission",
            [
                SUBMISSION_TO_TRANSFER[stage]
                for stage in SUBMISSION_STAGES
                if collect_intervals(stage_intervals, "AIV", (stage,))
            ],
        )
    return local_copies, "missing_qp_transfer", []


def read_rank(paths, cycles_per_us):
    loaded = []
    ranks = set()
    positive_starts = []
    for path in paths:
        match = RANK_FILE_RE.fullmatch(path.name)
        if not match:
            raise ValueError(f"unexpected timer filename: {path}")
        ranks.add(int(match.group("rank")))
        source = source_name(path)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
        if not rows:
            raise ValueError(f"empty timer CSV: {path}")
        if "KERNEL_TIMING_IDX_start_cycle" not in fieldnames or "KERNEL_TIMING_IDX_end_cycle" not in fieldnames:
            raise ValueError(f"kernel envelope missing from {path}")
        for row in rows:
            try:
                begin = int(row.get("KERNEL_TIMING_IDX_start_cycle", "0"))
                end = int(row.get("KERNEL_TIMING_IDX_end_cycle", "0"))
            except ValueError as exc:
                raise ValueError(f"invalid kernel cycle in {path}") from exc
            if begin > 0 and end >= begin:
                positive_starts.append(begin)
        loaded.append((path, source, rows, discover_stages(fieldnames)))

    if len(ranks) != 1:
        raise ValueError(f"timer files for multiple ranks were grouped together: {sorted(ranks)}")
    rank = next(iter(ranks))
    if not positive_starts:
        raise ValueError(f"no positive kernel timestamps for rank {rank}")
    origin = min(positive_starts)

    events = []
    stage_counts = Counter()
    summaries = defaultdict(
        lambda: {
            "count": 0,
            "sum_us": 0.0,
            "max_us": 0.0,
            "first_us": None,
            "last_us": None,
        }
    )
    kernel_intervals = []
    kernel_intervals_by_type = defaultdict(list)
    stage_intervals = defaultdict(list)
    indexed_stage_intervals = defaultdict(list)
    lane_names = {}
    lane_tids = {}
    submission_copy_cycles = defaultdict(list)
    aiv_envelopes_by_domain = defaultdict(list)
    transport_engines_by_domain = defaultdict(set)
    engine_submission_intervals_by_domain = defaultdict(list)

    def get_track(source, core_type, group, sub_group, label):
        lane_key = (source, core_type, group, sub_group, label)
        if lane_key not in lane_tids:
            lane_tids[lane_key] = len(lane_tids)
            lane_names[lane_tids[lane_key]] = f"{source} {core_type} group={group} sub={sub_group} {label}"
        return lane_tids[lane_key]

    for path, source, rows, stages in loaded:
        for row in rows:
            group = int(row["group_id"])
            sub_group = int(row["sub_group_id"])
            core_type = row["core_type"]
            tid = get_track(source, core_type, group, sub_group, "timeline")

            kernel_begin = int(row["KERNEL_TIMING_IDX_start_cycle"])
            kernel_end = int(row["KERNEL_TIMING_IDX_end_cycle"])
            if kernel_begin > 0 and kernel_end >= kernel_begin:
                kernel_start_us = (kernel_begin - origin) / cycles_per_us
                kernel_duration_us = (kernel_end - kernel_begin) / cycles_per_us
                kernel_tid = 10000 + tid
                lane_names[kernel_tid] = f"{source} {core_type} group={group} sub={sub_group} kernel envelope"
                stage_counts["KERNEL_TIMING_IDX"] += 1
                summary = summaries[(source, core_type, group, sub_group, "KERNEL_TIMING_IDX")]
                summary["count"] += 1
                summary["sum_us"] += kernel_duration_us
                summary["max_us"] = max(summary["max_us"], kernel_duration_us)
                summary["first_us"] = kernel_start_us
                summary["last_us"] = kernel_start_us + kernel_duration_us
                interval = (kernel_start_us, kernel_start_us + kernel_duration_us)
                kernel_intervals.append(interval)
                kernel_intervals_by_type[core_type].append(interval)
                events.append(
                    {
                        "ph": "X",
                        "cat": "KERNEL_TIMING_IDX",
                        "name": "KERNEL_TIMING_IDX",
                        "pid": rank,
                        "tid": kernel_tid,
                        "ts": round(kernel_start_us, 3),
                        "dur": round(kernel_duration_us, 3),
                        "args": {
                            "source": source,
                            "group_id": group,
                            "sub_group_id": sub_group,
                            "core_type": core_type,
                            "csv": str(path),
                        },
                    }
                )

            for stage in stages:
                start_prefix = f"{stage}_"
                end_suffix = "_start_cycle"
                indices = sorted(
                    int(key[len(start_prefix) : -len(end_suffix)])
                    for key in row
                    if isinstance(key, str)
                    and key.startswith(start_prefix)
                    and key.endswith(end_suffix)
                    and key[len(start_prefix) : -len(end_suffix)].isdigit()
                )
                for index in indices:
                    start_key = f"{stage}_{index}_start_cycle"
                    end_key = f"{stage}_{index}_end_cycle"
                    if not row[start_key].strip() or not row.get(end_key, "").strip():
                        continue
                    begin = int(row[start_key])
                    end = int(row[end_key])
                    if begin > 0 and end >= begin:
                        start_us = (begin - origin) / cycles_per_us
                        duration_us = (end - begin) / cycles_per_us
                        last_us = start_us + duration_us
                        display_stage = STAGE_DISPLAY_ALIASES.get(stage, stage)
                        stage_counts[display_stage] += 1
                        summary_key = (source, core_type, group, sub_group, display_stage)
                        summary = summaries[summary_key]
                        summary["count"] += 1
                        summary["sum_us"] += duration_us
                        summary["max_us"] = max(summary["max_us"], duration_us)
                        summary["first_us"] = (
                            start_us if summary["first_us"] is None else min(summary["first_us"], start_us)
                        )
                        summary["last_us"] = last_us if summary["last_us"] is None else max(summary["last_us"], last_us)
                        interval = (start_us, last_us)
                        stage_intervals[(source, core_type, display_stage)].append(interval)
                        indexed_stage_intervals[(source, core_type, display_stage, index)].append(interval)
                        if core_type == "AIV" and display_stage == "AIV":
                            aiv_envelopes_by_domain[(source, group, sub_group)].append(interval)
                        if display_stage in SUBMISSION_STAGES:
                            submission_copy_cycles[(source, group, sub_group, display_stage)].append((begin, end))
                            engine_submission_intervals_by_domain[(source, group, sub_group, display_stage)].append(
                                interval
                            )
                            transport_engines_by_domain[(source, group, sub_group)].add(
                                "UDMA" if display_stage == "UDMA_COPY" else "RDMA"
                            )
                        track_label = stage_track_label(core_type, display_stage)
                        stage_tid = get_track(source, core_type, group, sub_group, track_label)
                        events.append(
                            {
                                "ph": "X",
                                "cat": display_stage,
                                "name": display_stage,
                                "pid": rank,
                                "tid": stage_tid,
                                "ts": round(start_us, 3),
                                "dur": round(duration_us, 3),
                                "args": {
                                    "source": source,
                                    "index": index,
                                    "group_id": group,
                                    "sub_group_id": sub_group,
                                    "core_type": core_type,
                                    "csv": str(path),
                                },
                            }
                        )

    # Keep one compact set of device timer types: AIV is the transport envelope and
    # UDMA_COPY/RDMA_COPY are AIV issue windows. Derive engine-specific schedule
    # tracks without adding more GPU timer columns.
    for (source, group, sub_group), intervals in sorted(aiv_envelopes_by_domain.items()):
        engines = transport_engines_by_domain.get((source, group, sub_group), set())
        for engine in sorted(engines):
            transfer_stage = f"{engine}_TRANSFER"
            copy_stage = f"{engine}_COPY"
            copy_intervals = engine_submission_intervals_by_domain.get((source, group, sub_group, copy_stage), [])
            derived_intervals = []
            for envelope_begin, envelope_end in intervals:
                copies = [
                    interval
                    for interval in copy_intervals
                    if envelope_begin <= interval[0] and interval[1] <= envelope_end
                ]
                if copies:
                    derived_intervals.append((min(begin for begin, _ in copies), max(end for _, end in copies)))
            stage_intervals[(source, "AIV", transfer_stage)].extend(derived_intervals)
            stage_counts[transfer_stage] += len(derived_intervals)
            summary_key = (source, "AIV", group, sub_group, transfer_stage)
            summary = summaries[summary_key]
            for begin, end in derived_intervals:
                duration_us = end - begin
                summary["count"] += 1
                summary["sum_us"] += duration_us
                summary["max_us"] = max(summary["max_us"], duration_us)
                summary["first_us"] = begin if summary["first_us"] is None else min(summary["first_us"], begin)
                summary["last_us"] = end if summary["last_us"] is None else max(summary["last_us"], end)
            transfer_tid = get_track(source, "AIV", group, sub_group, AIV_STAGE_TRACKS[transfer_stage])
            for begin, end in derived_intervals:
                events.append(
                    {
                        "ph": "X",
                        "cat": "DERIVED",
                        "name": transfer_stage,
                        "pid": rank,
                        "tid": transfer_tid,
                        "ts": round(begin, 3),
                        "dur": round(end - begin, 3),
                        "args": {
                            "source": source,
                            "group_id": group,
                            "sub_group_id": sub_group,
                            "core_type": "AIV",
                            "derived_from": f"AIV+{copy_stage}",
                            "engine": engine,
                        },
                    }
                )

    # SHMEM exposes no per-request completion point before the final quiet. Expose the
    # issue cadence separately and add explicit gaps between consecutive submissions
    # on the same AIV, so scheduler stalls cannot hide inside the derived envelope.
    for (source, group, sub_group, copy_stage), cycles in sorted(submission_copy_cycles.items()):
        cycles.sort()
        gap_stage = SUBMISSION_TO_GAP[copy_stage]
        gap_tid = get_track(source, "AIV", group, sub_group, AIV_STAGE_TRACKS[gap_stage])
        for previous, current in zip(cycles, cycles[1:]):
            _, previous_end = previous
            current_begin, _ = current
            if current_begin <= previous_end:
                continue
            start_us = (previous_end - origin) / cycles_per_us
            duration_us = (current_begin - previous_end) / cycles_per_us
            stage_counts[gap_stage] += 1
            summary_key = (source, "AIV", group, sub_group, gap_stage)
            summary = summaries[summary_key]
            summary["count"] += 1
            summary["sum_us"] += duration_us
            summary["max_us"] = max(summary["max_us"], duration_us)
            summary["first_us"] = start_us if summary["first_us"] is None else min(summary["first_us"], start_us)
            summary["last_us"] = (
                start_us + duration_us
                if summary["last_us"] is None
                else max(summary["last_us"], start_us + duration_us)
            )
            interval = (start_us, start_us + duration_us)
            stage_intervals[(source, "AIV", gap_stage)].append(interval)
            events.append(
                {
                    "ph": "X",
                    "cat": "DERIVED",
                    "name": gap_stage,
                    "pid": rank,
                    "tid": gap_tid,
                    "ts": round(start_us, 3),
                    "dur": round(duration_us, 3),
                    "args": {
                        "source": source,
                        "group_id": group,
                        "sub_group_id": sub_group,
                        "core_type": "AIV",
                        "derived_from": copy_stage,
                    },
                }
            )

    for tid, name in sorted(lane_names.items()):
        events.append(
            {
                "ph": "M",
                "name": "thread_name",
                "pid": rank,
                "tid": tid,
                "args": {"name": name},
            }
        )

    return (
        rank,
        events,
        stage_counts,
        summaries,
        kernel_intervals,
        kernel_intervals_by_type,
        stage_intervals,
        indexed_stage_intervals,
        tuple(sorted(source_name(path) for path in paths)),
    )


def build_steady_rounds(
    rank, paths, indexed_stage_intervals, stage_intervals, transfer_intervals, transfer_semantics, sources
):
    rows = []
    compute_by_round = defaultdict(list)

    # New round-scoped files name the logical compute round explicitly. Gather
    # every AIC core from files such as round3_compute[_peerN] into round 3.
    for source in sources:
        match = ROUND_SOURCE_RE.fullmatch(source)
        if not match:
            continue
        intervals = collect_intervals(stage_intervals, "AIC", COMPUTE_STAGES, source)
        if intervals:
            compute_by_round[int(match.group("round"))].extend(intervals)

    # Legacy traces use AIC_ROUND anchors around per-peer AIC_COMPUTE spans.
    if not compute_by_round:
        round_anchors = []
        for (source, core_type, stage, index), intervals in indexed_stage_intervals.items():
            if core_type == "AIC" and stage == "AIC_ROUND":
                round_anchors.extend((begin, end, index) for begin, end in intervals)
        round_anchors.sort()

        for (source, core_type, stage, index), intervals in indexed_stage_intervals.items():
            if core_type == "AIC" and stage == "AIC_COMPUTE":
                for interval in intervals:
                    begin = interval[0]
                    logical_round = None
                    for anchor_begin, anchor_end, anchor_round in round_anchors:
                        if anchor_begin <= begin <= anchor_end:
                            logical_round = anchor_round
                            break
                    if logical_round is not None:
                        compute_by_round[logical_round].append(interval)

    # The current kernel records one AIC interval per logical round on every
    # AIC core. Timer iteration index 0 is therefore the first round, index 1
    # the second round, and so on across all cores.
    if not compute_by_round:
        for (source, core_type, stage, index), intervals in indexed_stage_intervals.items():
            if core_type == "AIC" and stage == "AIC":
                compute_by_round[index].extend(intervals)

    source_text = ";".join(str(path) for path in paths)
    for compute_round in sorted(compute_by_round):
        if compute_round < 1:
            continue
        compute_intervals = merge_intervals(compute_by_round[compute_round])
        compute_us = interval_union_us(compute_intervals)
        if compute_us <= 0.0:
            continue

        compute_begin = min(begin for begin, _ in compute_intervals)
        compute_end = max(end for _, end in compute_intervals)
        active_transfers = [
            (begin, end) for begin, end in transfer_intervals if end > compute_begin and begin < compute_end
        ]
        copy_us = interval_union_us(active_transfers)
        covered_us = interval_intersection_us(compute_intervals, active_transfers)
        rows.append(
            {
                # Kept for run_overlap_validation.sh schema compatibility. It is
                # a nominal predecessor label; covered_us uses wall-clock overlap.
                "rank": rank,
                "source": source_text,
                "compute_round": compute_round,
                "copy_round": compute_round - 1,
                "compute_us": f"{compute_us:.6f}",
                "copy_us": f"{copy_us:.6f}",
                "covered_us": f"{covered_us:.6f}",
                "uncovered_us": f"{compute_us - covered_us:.6f}",
                "cover_pct": f"{percentage(covered_us, compute_us):.3f}",
                "transport_semantics": transfer_semantics,
                "coverage_valid": (
                    "yes" if transfer_semantics in ("aiv_schedule_envelope", "qp_completion_envelope") else "no"
                ),
            }
        )
    return rows


def write_trace(path, events, cycles_per_us):
    # Keep thread metadata first, then order timed events by the device clock.
    # Chrome trace viewers commonly expect timestamp order; CSV row order is
    # grouped by core/stage and otherwise makes later-stage submissions appear
    # before earlier compute intervals.
    metadata = [event for event in events if event.get("ph") == "M"]
    timed = [event for event in events if event.get("ph") != "M"]
    timed.sort(key=lambda event: (float(event.get("ts", 0.0)), int(event.get("tid", 0)), str(event.get("name", ""))))
    payload = {
        "traceEvents": metadata + timed,
        "displayTimeUnit": "ms",
        "otherData": {"clock_note": CLOCK_NOTE, "cycles_per_us": cycles_per_us},
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="directories or files containing timer_rank_*_timeline.csv"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cycles-per-us", type=float, default=1000.0, help="Ascend950 GetSystemCycle frequency (default: 1000)"
    )
    parser.add_argument(
        "--dynamic-cap", type=int, default=150, help="AscendTimer per-stage counter capacity (default: 150)"
    )
    args = parser.parse_args()
    if args.cycles_per_us <= 0:
        parser.error("--cycles-per-us must be positive")
    if args.dynamic_cap <= 0:
        parser.error("--dynamic-cap must be positive")

    candidates = set()
    for root in args.inputs:
        if root.is_file():
            candidates.add(root.resolve())
        else:
            candidates.update(path.resolve() for path in root.rglob("timer_rank_*_timeline.csv"))
    paths = sorted(path for path in candidates if RANK_FILE_RE.fullmatch(path.name))
    if not paths:
        searched = ", ".join(str(root) for root in args.inputs)
        parser.error(
            "no timer_rank_*_timeline.csv files found\n"
            f"  searched: {searched}\n"
            "  expected: <timer_dir>/data/timer_rank_<rank>[_<suffix>]_timeline.csv\n"
            "  hint: pass the collected timer directory (for example <log>/shape*/timer). A rank "
            "that failed before the kernel finished, or an ENABLE_TIMER=OFF build, "
            "produces no CSV."
        )

    by_rank = defaultdict(list)
    for path in paths:
        match = RANK_FILE_RE.fullmatch(path.name)
        by_rank[int(match.group("rank"))].append(path)

    args.output.mkdir(parents=True, exist_ok=True)
    for stale in args.output.glob("rank*.json"):
        stale.unlink()
    stale_steady = args.output / "steady_overlap.csv"
    if stale_steady.exists():
        stale_steady.unlink()

    summary_rows = []
    stage_rows = []
    stage_lookup = {}
    performance_rows = []
    steady_round_rows = []
    total_counts = Counter()

    for rank, rank_paths in sorted(by_rank.items()):
        (
            parsed_rank,
            events,
            counts,
            summaries,
            kernel_intervals,
            kernel_intervals_by_type,
            stage_intervals,
            indexed_stage_intervals,
            sources,
        ) = read_rank(rank_paths, args.cycles_per_us)
        if parsed_rank != rank:
            raise SystemExit(f"internal rank mismatch: {parsed_rank} != {rank}")
        write_trace(args.output / f"rank{rank}.json", events, args.cycles_per_us)
        total_counts.update(counts)

        compute_intervals = collect_intervals(stage_intervals, "AIC", COMPUTE_STAGES)
        udma_copy_intervals = collect_intervals(stage_intervals, "AIV", ("UDMA_COPY",))
        rdma_copy_intervals = collect_intervals(stage_intervals, "AIV", ("RDMA_COPY",))
        udma_transfer_intervals = collect_intervals(stage_intervals, "AIV", ("UDMA_TRANSFER",))
        rdma_transfer_intervals = collect_intervals(stage_intervals, "AIV", ("RDMA_TRANSFER",))
        udma_gap_intervals = collect_intervals(stage_intervals, "AIV", ("UDMA_SUBMIT_GAP",))
        rdma_gap_intervals = collect_intervals(stage_intervals, "AIV", ("RDMA_SUBMIT_GAP",))
        local_copy_intervals = collect_intervals(stage_intervals, "AIV", ("AIV_LOCAL_COPY",))
        qp_comm_intervals = collect_intervals(stage_intervals, "AIV", AIV_ENVELOPE_STAGES)
        transport_intervals, transport_semantics, missing_transfers = classify_transport(stage_intervals)
        submission_intervals = collect_intervals(stage_intervals, "AIV", SUBMISSION_STAGES)
        aiv_scope_intervals = collect_intervals(stage_intervals, "AIV", ("AIV",))
        aiv_wait_intervals = collect_intervals(stage_intervals, "AIV", ("AIV_WAIT",))

        kernel_union_us = interval_union_us(kernel_intervals)
        compute_union_us = interval_union_us(compute_intervals)
        transport_union_us = interval_union_us(transport_intervals)
        submission_union_us = interval_union_us(submission_intervals)
        qp_comm_union_us = interval_union_us(qp_comm_intervals)
        overlap_us = interval_intersection_us(compute_intervals, transport_intervals)
        submission_overlap_us = interval_intersection_us(compute_intervals, submission_intervals)
        qp_comm_overlap_us = interval_intersection_us(compute_intervals, qp_comm_intervals)
        compute_or_transport_union_us = interval_union_us(compute_intervals + transport_intervals)
        udma_gap_stats = interval_duration_stats(udma_gap_intervals)
        rdma_gap_stats = interval_duration_stats(rdma_gap_intervals)
        aiv_transport_counts = transport_aiv_counts(summaries)

        round_rows = build_steady_rounds(
            rank,
            rank_paths,
            indexed_stage_intervals,
            stage_intervals,
            transport_intervals,
            transport_semantics,
            sources,
        )
        steady_round_rows.extend(round_rows)
        steady_compute_us = sum(float(row["compute_us"]) for row in round_rows)
        steady_covered_us = sum(float(row["covered_us"]) for row in round_rows)
        steady_uncovered_us = steady_compute_us - steady_covered_us

        performance_rows.append(
            {
                "rank": rank,
                "source": ";".join(str(path) for path in rank_paths),
                "kernel_union_us": f"{kernel_union_us:.6f}",
                "aic_compute_union_us": f"{compute_union_us:.6f}",
                "udma_copy_union_us": f"{interval_union_us(udma_copy_intervals):.6f}",
                "rdma_copy_union_us": f"{interval_union_us(rdma_copy_intervals):.6f}",
                "udma_transfer_union_us": f"{interval_union_us(udma_transfer_intervals):.6f}",
                "rdma_transfer_union_us": f"{interval_union_us(rdma_transfer_intervals):.6f}",
                "aiv_local_copy_union_us": f"{interval_union_us(local_copy_intervals):.6f}",
                "qp_comm_union_us": f"{qp_comm_union_us:.6f}",
                "transport_copy_union_us": f"{transport_union_us:.6f}",
                "transport_semantics": transport_semantics,
                "transport_measurement_status": (
                    "PRECISE_SCHEDULING_ENVELOPE"
                    if transport_semantics in ("aiv_schedule_envelope", "qp_completion_envelope")
                    else "INVALID"
                ),
                "hardware_engine_busy_status": "NOT_MEASURED",
                "missing_transfer_stages": ";".join(missing_transfers),
                "request_submission_union_us": f"{submission_union_us:.6f}",
                "udma_submit_gap_count": udma_gap_stats["count"],
                "udma_submit_gap_total_us": f"{udma_gap_stats['total_us']:.6f}",
                "udma_submit_gap_mean_us": f"{udma_gap_stats['mean_us']:.6f}",
                "udma_submit_gap_median_us": f"{udma_gap_stats['median_us']:.6f}",
                "udma_submit_gap_max_us": f"{udma_gap_stats['max_us']:.6f}",
                "rdma_submit_gap_count": rdma_gap_stats["count"],
                "rdma_submit_gap_total_us": f"{rdma_gap_stats['total_us']:.6f}",
                "rdma_submit_gap_mean_us": f"{rdma_gap_stats['mean_us']:.6f}",
                "rdma_submit_gap_median_us": f"{rdma_gap_stats['median_us']:.6f}",
                "rdma_submit_gap_max_us": f"{rdma_gap_stats['max_us']:.6f}",
                "aiv_scope_union_us": f"{interval_union_us(aiv_scope_intervals):.6f}",
                "udma_aiv_count": aiv_transport_counts["udma"],
                "rdma_aiv_count": aiv_transport_counts["rdma"],
                "mixed_transport_aiv_count": aiv_transport_counts["mixed"],
                "compute_copy_overlap_us": f"{overlap_us:.6f}",
                "compute_transfer_overlap_us": f"{overlap_us:.6f}",
                "compute_request_submission_overlap_us": f"{submission_overlap_us:.6f}",
                "compute_qp_comm_overlap_us": f"{qp_comm_overlap_us:.6f}",
                "compute_only_us": f"{compute_union_us - overlap_us:.6f}",
                "copy_only_us": f"{transport_union_us - overlap_us:.6f}",
                "compute_or_copy_union_us": f"{compute_or_transport_union_us:.6f}",
                "neither_compute_nor_copy_us": f"{max(0.0, kernel_union_us - compute_or_transport_union_us):.6f}",
                "compute_cover_pct": f"{percentage(compute_union_us, kernel_union_us):.3f}",
                "copy_cover_pct": f"{percentage(transport_union_us, kernel_union_us):.3f}",
                "overlap_of_compute_pct": f"{percentage(overlap_us, compute_union_us):.3f}",
                "overlap_of_copy_pct": f"{percentage(overlap_us, transport_union_us):.3f}",
                "qp_comm_overlap_of_compute_pct": f"{percentage(qp_comm_overlap_us, compute_union_us):.3f}",
                "aic_wait_union_us": f"{interval_union_us(collect_intervals(stage_intervals, 'AIC', ('AIC_WAIT',))):.6f}",
                "aiv_idle_wait_union_us": f"{interval_union_us(aiv_wait_intervals):.6f}",
                "aic_compute_events": len(compute_intervals),
                "udma_copy_events": len(udma_copy_intervals),
                "rdma_copy_events": len(rdma_copy_intervals),
                "udma_transfer_events": len(udma_transfer_intervals),
                "rdma_transfer_events": len(rdma_transfer_intervals),
                "aiv_local_copy_events": len(local_copy_intervals),
                "qp_comm_events": len(qp_comm_intervals),
                "steady_compute_rounds": len(round_rows),
                "steady_compute_union_us": f"{steady_compute_us:.6f}",
                "steady_compute_covered_us": f"{steady_covered_us:.6f}",
                "steady_compute_uncovered_us": f"{steady_uncovered_us:.6f}",
                "steady_compute_cover_pct": f"{percentage(steady_covered_us, steady_compute_us):.3f}",
                "steady_compute_cover_status": (
                    "VALID" if transport_semantics in ("aiv_schedule_envelope", "qp_completion_envelope") else "INVALID"
                ),
            }
        )

        if transport_semantics not in ("aiv_schedule_envelope", "qp_completion_envelope"):
            detail = ", ".join(missing_transfers) if missing_transfers else "no completion envelope"
            print(
                f"[WARN] rank={rank}: transport timing is {transport_semantics} ({detail}); "
                "masking percentages are diagnostic only and must not be accepted as overlap",
                file=sys.stderr,
            )

        saturated = sorted(
            {row["stage"] for row in summary_rows if row["rank"] == rank and int(row["count"]) >= args.dynamic_cap}
        )
        if saturated:
            print(
                f"[WARN] rank={rank}: dynamic timer stage(s) reached the configured per-stage "
                f"capacity {args.dynamic_cap}: {', '.join(saturated)}; late intervals may have been truncated",
                file=sys.stderr,
            )

        for (source, core_type, group, sub_group, stage), values in sorted(summaries.items()):
            summary_rows.append(
                {
                    "rank": rank,
                    "source": source,
                    "core_type": core_type,
                    "group_id": group,
                    "sub_group_id": sub_group,
                    "stage": stage,
                    "count": values["count"],
                    "sum_us": f"{values['sum_us']:.6f}",
                    "mean_us": f"{values['sum_us'] / values['count']:.6f}",
                    "max_us": f"{values['max_us']:.6f}",
                    "first_us": f"{values['first_us']:.6f}",
                    "last_us": f"{values['last_us']:.6f}",
                }
            )
            stage_key = (rank, core_type, stage)
            aggregate = stage_lookup.get(stage_key)
            if aggregate is None:
                aggregate = {
                    "key": stage_key,
                    "rank": rank,
                    "core_type": core_type,
                    "stage": stage,
                    "count": 0,
                    "sum_us": 0.0,
                    "max_us": 0.0,
                    "first_us": values["first_us"],
                    "last_us": values["last_us"],
                }
                stage_rows.append(aggregate)
                stage_lookup[stage_key] = aggregate
            aggregate["count"] += values["count"]
            aggregate["sum_us"] += values["sum_us"]
            aggregate["max_us"] = max(aggregate["max_us"], values["max_us"])
            aggregate["first_us"] = min(aggregate["first_us"], values["first_us"])
            aggregate["last_us"] = max(aggregate["last_us"], values["last_us"])

        for (_, core_type, stage), intervals in stage_intervals.items():
            stage_key = (rank, core_type, stage)
            if stage_key in stage_lookup:
                stage_lookup[stage_key].setdefault("intervals", []).extend(intervals)
        for core_type, intervals in kernel_intervals_by_type.items():
            stage_key = (rank, core_type, "KERNEL_TIMING_IDX")
            if stage_key in stage_lookup:
                stage_lookup[stage_key].setdefault("intervals", []).extend(intervals)

    if not summary_rows or not stage_rows or not performance_rows:
        raise SystemExit("timer data did not produce any summary rows")

    summary_path = args.output / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    stage_summary_rows = []
    for item in sorted(stage_rows, key=lambda value: value["key"]):
        union_us = interval_union_us(item.get("intervals", []))
        stage_summary_rows.append(
            {
                "rank": item["rank"],
                "core_type": item["core_type"],
                "stage": item["stage"],
                "count": item["count"],
                "sum_core_us": f"{item['sum_us']:.6f}",
                "mean_us": f"{item['sum_us'] / item['count']:.6f}",
                "max_us": f"{item['max_us']:.6f}",
                "first_us": f"{item['first_us']:.6f}",
                "last_us": f"{item['last_us']:.6f}",
                "wall_us": f"{item['last_us'] - item['first_us']:.6f}",
                "union_us": f"{union_us:.6f}",
                "parallelism": f"{item['sum_us'] / union_us:.6f}" if union_us > 0 else "0.000000",
            }
        )
    stage_summary_path = args.output / "stage_summary.csv"
    with stage_summary_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(stage_summary_rows[0]))
        writer.writeheader()
        writer.writerows(stage_summary_rows)

    performance_path = args.output / "performance_summary.csv"
    with performance_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(performance_rows[0]))
        writer.writeheader()
        writer.writerows(performance_rows)
    if steady_round_rows:
        steady_path = args.output / "steady_overlap.csv"
        with steady_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(steady_round_rows[0]))
            writer.writeheader()
            writer.writerows(steady_round_rows)

    (args.output / "README.txt").write_text(CLOCK_NOTE + "\n", encoding="utf-8")
    print("timer events: " + ", ".join(f"{name}={total_counts[name]}" for name in sorted(total_counts)))
    for item in performance_rows:
        valid_transport = item["transport_semantics"] in ("aiv_schedule_envelope", "qp_completion_envelope")
        transport_name = (
            "AIV schedule envelope"
            if item["transport_semantics"] == "aiv_schedule_envelope"
            else "legacy QP completion envelope"
            if valid_transport
            else "LEGACY submission - coverage INVALID"
            if item["transport_semantics"] == "legacy_submission"
            else f"INVALID transport ({item['transport_semantics']})"
        )
        print(
            f"rank={item['rank']} kernel={item['kernel_union_us']} us | "
            f"AIC compute={item['aic_compute_union_us']} us ({item['compute_cover_pct']}% of kernel) | "
            f"{transport_name}/local-copy={item['transport_copy_union_us']} us "
            f"({item['copy_cover_pct']}% of kernel) | "
            f"interval intersection={item['compute_copy_overlap_us']} us ({item['overlap_of_compute_pct']}% compute, "
            f"{item['overlap_of_copy_pct']}% recorded intervals)"
        )
        print(
            f"rank={item['rank']} UDMA schedule envelope={item['udma_transfer_union_us']} us "
            f"({item['udma_transfer_events']} events), submit={item['udma_copy_union_us']} us | "
            f"RDMA schedule envelope={item['rdma_transfer_union_us']} us "
            f"({item['rdma_transfer_events']} events), submit={item['rdma_copy_union_us']} us | "
            f"local-copy={item['aiv_local_copy_union_us']} us | AIV scope={item['qp_comm_union_us']} us"
        )
        print(
            f"rank={item['rank']} submission gaps (issue-side): UDMA "
            f"count={item['udma_submit_gap_count']} total={item['udma_submit_gap_total_us']} us "
            f"median={item['udma_submit_gap_median_us']} us max={item['udma_submit_gap_max_us']} us | "
            f"RDMA count={item['rdma_submit_gap_count']} total={item['rdma_submit_gap_total_us']} us "
            f"median={item['rdma_submit_gap_median_us']} us max={item['rdma_submit_gap_max_us']} us"
        )
        print(
            f"rank={item['rank']} AIV transport ownership from trace: UDMA={item['udma_aiv_count']} "
            f"RDMA={item['rdma_aiv_count']} mixed={item['mixed_transport_aiv_count']}"
        )
        print(
            f"rank={item['rank']} AIV overlap={item['compute_qp_comm_overlap_us']} us "
            f"({item['qp_comm_overlap_of_compute_pct']}% compute) | "
            f"AIV scope={item['aiv_scope_union_us']} us | AIC_WAIT={item['aic_wait_union_us']} us | "
            f"AIV_WAIT={item['aiv_idle_wait_union_us']} us | "
            f"compute-only={item['compute_only_us']} us | "
            f"copy-only={item['copy_only_us']} us"
        )
        if valid_transport:
            coverage_suffix = ""
        elif item["transport_semantics"] == "legacy_submission":
            coverage_suffix = " [INVALID: legacy submission timing]"
        else:
            coverage_suffix = f" [INVALID: {item['transport_semantics']}]"
        print(
            f"rank={item['rank']} steady r1+: rounds={item['steady_compute_rounds']} "
            f"compute={item['steady_compute_union_us']} us, recorded intersection={item['steady_compute_covered_us']} us "
            f"({item['steady_compute_cover_pct']}%), uncovered={item['steady_compute_uncovered_us']} us"
            f"{coverage_suffix}"
        )
    for item in stage_summary_rows:
        print(
            f"rank={item['rank']} {item['core_type']} {item['stage']}: events={item['count']} "
            f"sum_core={item['sum_core_us']} us union={item['union_us']} us wall={item['wall_us']} us "
            f"max={item['max_us']} us"
        )
    print(
        f"trace: {args.output}; summary: {summary_path}; stage summary: {stage_summary_path}; "
        f"performance: {performance_path}"
    )


if __name__ == "__main__":
    main()
