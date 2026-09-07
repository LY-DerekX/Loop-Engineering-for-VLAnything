#!/usr/bin/env python3
"""Profile the data distribution of a LeRobot v3 dataset.

The dataset is read only. Two scan depths are available:

  --scan meta    metadata only (meta/info.json, meta/episodes/*, meta/tasks.parquet).
                 Reports task balance, episode-length distribution and video packing.
  --scan frames  additionally streams every data parquet to build exact frame-level
                 distributions for action, observation.state, per-step motion,
                 action/state tracking error and timestamp spacing. Videos are never
                 decoded, so this stays cheap relative to the dataset size.

Outputs a condensed console report plus, in the output directory:
  distribution_summary.json   full machine-readable result
  distribution_report.md      the same content as a readable report
  task_distribution.csv       one row per task
  episode_distribution.csv    one row per episode
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from utils.dataset_io import read_json
from utils.dataset_metadata import data_path_for_episode, video_keys_from_info

DEFAULT_REPORT_DIR = "distribution_reports"
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
QUANTILE_KEYS = ("q01", "q05", "q25", "q50", "q75", "q95", "q99")
# A step whose per-joint motion stays under this fraction of the joint's own
# observed range counts as static; used only for the near-static frame ratio.
STATIC_MOTION_FRACTION = 0.001
# A frame counts as saturated when a joint sits within this fraction of the
# joint's observed range from that range's min or max.
SATURATION_FRACTION = 0.005
# Timestamp steps deviating from 1/fps by more than this fraction are irregular.
DT_TOLERANCE = 0.1
# Trailing session suffixes stripped when grouping task names into families.
_TASK_SUFFIX = re.compile(r"(?:[-/.](?:\d+|new))+$")


@dataclass
class Config:
    root: Path
    out_dir: Path
    scan: str
    stride: int
    bins: int
    hist_width: int
    top_tasks: int
    max_episodes: int | None


@dataclass
class FrameScan:
    """Frame-level pools gathered while streaming the data parquets."""

    action: list[np.ndarray] = field(default_factory=list)
    state: list[np.ndarray] = field(default_factory=list)
    motion: list[np.ndarray] = field(default_factory=list)
    dt: list[np.ndarray] = field(default_factory=list)
    episodes_scanned: int = 0
    frames_scanned: int = 0
    frames_pooled: int = 0
    length_mismatch: list[int] = field(default_factory=list)
    read_errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# generic statistics helpers
# --------------------------------------------------------------------------- #


def describe(values: np.ndarray) -> dict[str, Any]:
    """Summary statistics for a 1-D sample; empty input yields a null record."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    quantiles = np.quantile(values, QUANTILES)
    summary: dict[str, Any] = {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }
    summary.update({key: float(value) for key, value in zip(QUANTILE_KEYS, quantiles)})
    summary["iqr"] = summary["q75"] - summary["q25"]
    summary["shape"] = shape_stats(values)
    return summary


def shape_stats(values: np.ndarray) -> dict[str, Any]:
    """Describe how far a sample departs from normality, as effect sizes.

    Formal normality tests are useless at these sample sizes: with millions of
    frames, Shapiro-Wilk or KS reject on any trivial deviation, so the p-value
    carries no information. These descriptors say *how much* the sample departs
    instead, against the normal reference values noted per key.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size < 4:
        return {"status": "too_few_samples"}
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std <= 0:
        return {"status": "constant"}
    z = (values - mean) / std
    skewness = float((z**3).mean())
    excess_kurtosis = float((z**4).mean() - 3.0)
    # Sarle's bimodality coefficient: normal 0.333, uniform 0.555, higher is
    # increasingly bimodal. Undefined when the kurtosis denominator vanishes.
    denominator = excess_kurtosis + 3.0
    bimodality = float((skewness**2 + 1.0) / denominator) if denominator > 0 else None
    return {
        "skewness": round(skewness, 4),
        "excess_kurtosis": round(excess_kurtosis, 4),
        "bimodality_coefficient": round(bimodality, 4) if bimodality is not None else None,
        "tvd_vs_normal": round(total_variation_vs_normal(values, mean, std), 4),
        "normal_reference": {"skewness": 0.0, "excess_kurtosis": 0.0, "bimodality_coefficient": 0.333},
    }


def total_variation_vs_normal(values: np.ndarray, mean: float, std: float, bins: int = 200) -> float:
    """Total variation distance between the sample and its best-fit normal.

    0.0 means the normal fit is indistinguishable from the data; 1.0 means the
    two place their mass on disjoint sets. Computed on a fixed 200-bin grid
    spanning the sample, so values are comparable across signals.
    """
    low, high = float(values.min()), float(values.max())
    # Exact-grid signals (e.g. timestamps written as i/fps) have a span of only
    # a few float ulps; histogram bins would collapse to zero width.
    if high - low <= np.spacing(max(abs(low), abs(high), 1.0)) * bins:
        return 0.0
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    empirical = counts / counts.sum()
    from math import erf, sqrt

    cdf = np.array([0.5 * (1.0 + erf((edge - mean) / (std * sqrt(2.0)))) for edge in edges])
    fitted = np.diff(cdf)
    return float(0.5 * np.abs(empirical - fitted).sum())


def histogram(values: np.ndarray, bins: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"edges": [], "counts": []}
    low = float(values.min())
    high = float(values.max())
    # A span of a few float ulps cannot be split into finite bins; collapse to
    # the same single-bin fallback as an exactly constant sample.
    if high - low <= np.spacing(max(abs(low), abs(high), 1.0)) * bins:
        high = low + 1.0
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    return {"edges": [float(edge) for edge in edges], "counts": [int(count) for count in counts]}


def ascii_histogram(hist: dict[str, Any], width: int, max_rows: int = 20) -> list[str]:
    counts = hist.get("counts") or []
    edges = hist.get("edges") or []
    if not counts:
        return ["  (no data)"]
    if len(counts) > max_rows:
        group = math.ceil(len(counts) / max_rows)
        counts = [int(sum(counts[i : i + group])) for i in range(0, len(counts), group)]
        edges = edges[::group] + [edges[-1]]
    total = sum(counts) or 1
    peak = max(counts) or 1
    lines = []
    for i, count in enumerate(counts):
        bar = "#" * max(1 if count else 0, round(width * count / peak))
        lines.append(
            f"  [{edges[i]:>11.3f}, {edges[i + 1]:>11.3f})  {bar:<{width}}  "
            f"{count:>9,d}  {100.0 * count / total:5.1f}%"
        )
    return lines


def gini(counts: list[int]) -> float:
    """Gini coefficient of a count vector; 0.0 is perfectly balanced."""
    values = np.sort(np.asarray(counts, dtype=np.float64))
    n = values.size
    total = values.sum()
    if n == 0 or total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * (index * values).sum()) / (n * total) - (n + 1.0) / n)


def normalized_entropy(counts: list[int]) -> float:
    """Shannon entropy over the count vector, scaled to [0, 1]; 1.0 is uniform."""
    values = np.asarray([c for c in counts if c > 0], dtype=np.float64)
    if values.size <= 1:
        return 1.0 if values.size == 1 else 0.0
    probabilities = values / values.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy / math.log(values.size)


def imbalance(counts: list[int]) -> dict[str, Any]:
    ordered = sorted(counts, reverse=True)
    total = sum(ordered) or 1
    return {
        "groups": len(ordered),
        "total": sum(ordered),
        "gini": round(gini(ordered), 4),
        "normalized_entropy": round(normalized_entropy(ordered), 4),
        "top1_share": round(ordered[0] / total, 4) if ordered else 0.0,
        "top5_share": round(sum(ordered[:5]) / total, 4) if ordered else 0.0,
        "max_over_min_ratio": (
            round(ordered[0] / ordered[-1], 2) if ordered and ordered[-1] > 0 else None
        ),
    }


def outlier_bounds(values: np.ndarray) -> tuple[float, float]:
    """Tukey fences at 1.5 IQR."""
    q25, q75 = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.75])
    iqr = float(q75 - q25)
    return float(q25) - 1.5 * iqr, float(q75) + 1.5 * iqr


def task_family(task: str) -> str:
    """Collapse a raw task name to its recording family.

    Task names in this dataset carry a trailing session suffix (a date, a time or
    a take counter), so `pick_up_cups-0526-2030` and `pick_up_cups-0527-2030` are
    the same underlying task recorded twice. Stripping the suffix groups them.
    The rule is a naming heuristic, not ground truth: families are reported
    alongside, never instead of, the raw per-task counts.
    """
    base = task.strip().lower()
    stripped = _TASK_SUFFIX.sub("", base)
    return stripped or base


# --------------------------------------------------------------------------- #
# metadata loading
# --------------------------------------------------------------------------- #


def load_tasks(root: Path) -> dict[int, str]:
    table = pq.read_table(root / "meta" / "tasks.parquet")
    columns = table.column_names
    text_column = "task" if "task" in columns else columns[-1]
    if "task_index" in columns:
        indices = table.column("task_index").to_pylist()
    else:
        indices = list(range(table.num_rows))
    return {int(i): str(t) for i, t in zip(indices, table.column(text_column).to_pylist())}


def load_episodes(root: Path) -> pa.Table:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {root / 'meta' / 'episodes'}")
    return pa.concat_tables([pq.read_table(path) for path in files])


def episode_task_lists(episodes: pa.Table) -> list[list[str] | None]:
    """Per-episode task labels; None entries when the column is absent."""
    if "tasks" in episodes.column_names:
        return episodes.column("tasks").to_pylist()
    return [None] * episodes.num_rows


def joint_names(info: dict[str, Any], feature: str) -> list[str]:
    spec = info.get("features", {}).get(feature, {})
    names = spec.get("names")
    if isinstance(names, list) and names and all(isinstance(n, str) for n in names):
        return list(names)
    shape = spec.get("shape") or [0]
    return [f"{feature}[{i}]" for i in range(int(shape[0]))]


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# --------------------------------------------------------------------------- #
# metadata-level distributions
# --------------------------------------------------------------------------- #


def build_overview(root: Path, info: dict[str, Any], episodes: pa.Table) -> dict[str, Any]:
    fps = float(info.get("fps") or 0.0) or 30.0
    lengths = np.asarray(episodes.column("length").to_pylist(), dtype=np.float64)
    features = info.get("features", {})
    videos = {}
    for key in video_keys_from_info(info):
        spec = features.get(key, {})
        video_info = spec.get("info", {}) if isinstance(spec.get("info"), dict) else {}
        videos[key] = {
            "shape": spec.get("shape"),
            "codec": video_info.get("video.codec"),
            "fps": video_info.get("video.fps"),
            "pix_fmt": video_info.get("video.pix_fmt"),
            "is_depth_map": video_info.get("video.is_depth_map"),
        }
    return {
        "dataset": str(root),
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": fps,
        "total_episodes": int(info.get("total_episodes", episodes.num_rows)),
        "total_frames": int(info.get("total_frames", int(lengths.sum()))),
        "total_tasks": int(info.get("total_tasks", 0)),
        "splits": info.get("splits"),
        "episodes_in_metadata": episodes.num_rows,
        "frames_in_metadata": int(lengths.sum()),
        "total_duration_hours": round(float(lengths.sum()) / fps / 3600.0, 3),
        "numeric_features": {
            name: spec.get("shape")
            for name, spec in features.items()
            if spec.get("dtype") in {"float32", "float64", "int64"}
        },
        "video_features": videos,
        "bytes_on_disk": {
            "data": directory_bytes(root / "data"),
            "videos": directory_bytes(root / "videos"),
            "meta": directory_bytes(root / "meta"),
        },
    }


def build_task_distribution(
    episodes: pa.Table, tasks: dict[int, str], fps: float
) -> dict[str, Any]:
    task_lists = episode_task_lists(episodes)
    lengths = episodes.column("length").to_pylist()
    episode_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    length_by_task: defaultdict[str, list[int]] = defaultdict(list)
    multi_task_episodes = 0
    for names, length in zip(task_lists, lengths):
        labels = [str(n) for n in (names or [])] or ["<unlabelled>"]
        if len(labels) > 1:
            multi_task_episodes += 1
        for label in labels:
            episode_counts[label] += 1
            frame_counts[label] += int(length)
            length_by_task[label].append(int(length))

    total_episodes = sum(episode_counts.values())
    total_frames = sum(frame_counts.values())
    rows = []
    for label, count in episode_counts.most_common():
        task_lengths = np.asarray(length_by_task[label], dtype=np.float64)
        rows.append(
            {
                "task": label,
                "family": task_family(label),
                "episodes": count,
                "episode_share": round(count / total_episodes, 5) if total_episodes else 0.0,
                "frames": frame_counts[label],
                "frame_share": round(frame_counts[label] / total_frames, 5) if total_frames else 0.0,
                "minutes": round(frame_counts[label] / fps / 60.0, 2),
                "length_mean": round(float(task_lengths.mean()), 1),
                "length_min": int(task_lengths.min()),
                "length_max": int(task_lengths.max()),
                "length_std": round(float(task_lengths.std(ddof=0)), 1),
            }
        )

    family_episodes: Counter[str] = Counter()
    family_frames: Counter[str] = Counter()
    family_tasks: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        family_episodes[row["family"]] += row["episodes"]
        family_frames[row["family"]] += row["frames"]
        family_tasks[row["family"]].add(row["task"])
    families = [
        {
            "family": name,
            "raw_tasks": len(family_tasks[name]),
            "episodes": count,
            "episode_share": round(count / total_episodes, 5) if total_episodes else 0.0,
            "frames": family_frames[name],
            "frame_share": round(family_frames[name] / total_frames, 5) if total_frames else 0.0,
            "minutes": round(family_frames[name] / fps / 60.0, 2),
        }
        for name, count in family_episodes.most_common()
    ]

    declared = set(tasks.values())
    observed = set(episode_counts) - {"<unlabelled>"}
    return {
        "tasks_declared": len(declared),
        "tasks_observed_in_episodes": len(observed),
        "tasks_declared_but_unused": sorted(declared - observed),
        "tasks_used_but_undeclared": sorted(observed - declared),
        "duplicate_task_strings": sorted(
            name for name, count in Counter(tasks.values()).items() if count > 1
        ),
        "multi_task_episodes": multi_task_episodes,
        "per_task": rows,
        "per_family": families,
        "episode_imbalance": imbalance([r["episodes"] for r in rows]),
        "frame_imbalance": imbalance([r["frames"] for r in rows]),
        "family_episode_imbalance": imbalance([f["episodes"] for f in families]),
    }


def build_length_distribution(episodes: pa.Table, fps: float, bins: int) -> dict[str, Any]:
    lengths = np.asarray(episodes.column("length").to_pylist(), dtype=np.float64)
    indices = np.asarray(episodes.column("episode_index").to_pylist(), dtype=np.int64)
    low, high = outlier_bounds(lengths)
    outliers = np.nonzero((lengths < low) | (lengths > high))[0]
    order = np.argsort(lengths)
    return {
        "frames": describe(lengths),
        "seconds": describe(lengths / fps),
        "histogram": histogram(lengths, bins),
        "outlier_bounds": {"low": round(low, 1), "high": round(high, 1)},
        "outlier_count": int(outliers.size),
        "outlier_share": round(float(outliers.size) / max(lengths.size, 1), 5),
        "shortest_episodes": [
            {"episode_index": int(indices[i]), "length": int(lengths[i])} for i in order[:10]
        ],
        "longest_episodes": [
            {"episode_index": int(indices[i]), "length": int(lengths[i])} for i in order[-10:][::-1]
        ],
    }


def build_video_packing(episodes: pa.Table, root: Path, info: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in video_keys_from_info(info):
        chunk_column = f"videos/{key}/chunk_index"
        file_column = f"videos/{key}/file_index"
        if chunk_column not in episodes.column_names or file_column not in episodes.column_names:
            result[key] = {"status": "not_in_episode_metadata"}
            continue
        chunks = episodes.column(chunk_column).to_pylist()
        files = episodes.column(file_column).to_pylist()
        packing = Counter(zip(chunks, files))
        counts = list(packing.values())
        sizes = [
            path.stat().st_size for path in sorted((root / "videos" / key).rglob("*.mp4"))
        ]
        result[key] = {
            "video_files_referenced": len(packing),
            "video_files_on_disk": len(sizes),
            "episodes_per_file": describe(np.asarray(counts, dtype=np.float64)),
            "file_size_mb": describe(np.asarray(sizes, dtype=np.float64) / (1024 * 1024)),
            "total_size_mb": round(sum(sizes) / (1024 * 1024), 1),
        }
    return result


# --------------------------------------------------------------------------- #
# frame-level scan
# --------------------------------------------------------------------------- #


def list_column_to_array(table: pa.Table, name: str) -> np.ndarray:
    """Materialise a fixed_size_list / list column as a 2-D float array."""
    column = table.column(name).combine_chunks()
    if isinstance(column, pa.ChunkedArray):
        column = column.chunk(0) if column.num_chunks else pa.array([], type=column.type)
    flat = np.asarray(column.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
    rows = table.num_rows
    if rows == 0:
        return flat.reshape(0, 0)
    return flat.reshape(rows, -1)


def slice_episode(table: pa.Table, from_index: int, to_index: int, length: int) -> pa.Table:
    """Select one episode's rows from a possibly multi-episode data file."""
    if table.num_rows == length or "index" not in table.column_names:
        return table
    mask = pc.and_(
        pc.greater_equal(table.column("index"), from_index),
        pc.less(table.column("index"), to_index),
    )
    return table.filter(mask)


def scan_frames(config: Config, episodes: pa.Table, fps: float) -> tuple[FrameScan, list[dict]]:
    """Stream every episode's numeric columns and collect frame-level pools."""
    columns = ["action", "observation.state", "timestamp", "index"]
    wanted = [
        "episode_index",
        "length",
        "tasks",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    rows = episodes.select([c for c in wanted if c in episodes.column_names]).to_pylist()
    if config.max_episodes is not None:
        rows = rows[: config.max_episodes]

    scan = FrameScan()
    per_episode: list[dict[str, Any]] = []
    cache_path: Path | None = None
    cache_table: pa.Table | None = None
    expected_dt = 1.0 / fps

    for position, row in enumerate(rows):
        path = data_path_for_episode(config.root, row)
        if path != cache_path:
            try:
                available = pq.ParquetFile(path).schema_arrow.names
                cache_table = pq.read_table(path, columns=[c for c in columns if c in available])
                cache_path = path
            except Exception as exc:  # unreadable parquet: recorded, scan continues
                scan.read_errors.append(f"{path.name}:{type(exc).__name__}:{exc}")
                cache_path, cache_table = None, None
                continue
        assert cache_table is not None
        length = int(row["length"])
        table = slice_episode(
            cache_table, int(row["dataset_from_index"]), int(row["dataset_to_index"]), length
        )
        if table.num_rows != length:
            scan.length_mismatch.append(int(row["episode_index"]))
        if table.num_rows == 0:
            continue

        action = list_column_to_array(table, "action")
        state = list_column_to_array(table, "observation.state")
        motion = np.abs(np.diff(state, axis=0)) if state.shape[0] > 1 else np.empty((0, state.shape[1]))
        if "timestamp" in table.column_names:
            stamps = np.asarray(table.column("timestamp").to_numpy(zero_copy_only=False), dtype=np.float64)
            dt = np.diff(stamps) if stamps.size > 1 else np.empty(0)
        else:
            dt = np.empty(0)

        scan.episodes_scanned += 1
        scan.frames_scanned += table.num_rows
        stride = config.stride
        scan.action.append(action[::stride])
        scan.state.append(state[::stride])
        scan.frames_pooled += action[::stride].shape[0]
        if motion.size:
            scan.motion.append(motion[::stride])
        if dt.size:
            scan.dt.append(dt[::stride])

        tracking = np.abs(action - state)
        irregular = (
            int(np.count_nonzero(np.abs(dt - expected_dt) > DT_TOLERANCE * expected_dt))
            if dt.size
            else 0
        )
        per_episode.append(
            {
                "episode_index": int(row["episode_index"]),
                "task": (row.get("tasks") or ["<unlabelled>"])[0],
                "length": length,
                "rows_read": int(table.num_rows),
                "seconds": round(length / fps, 2),
                "mean_step_motion": round(float(motion.mean()), 6) if motion.size else 0.0,
                "max_step_motion": round(float(motion.max()), 6) if motion.size else 0.0,
                "path_length": round(float(motion.sum()), 3) if motion.size else 0.0,
                "tracking_error_mean": round(float(tracking.mean()), 5),
                "tracking_error_max": round(float(tracking.max()), 5),
                "gripper_mean": round(float(state[:, -1].mean()), 4) if state.shape[1] else 0.0,
                "gripper_range": (
                    round(float(state[:, -1].max() - state[:, -1].min()), 4) if state.shape[1] else 0.0
                ),
                "irregular_dt_steps": irregular,
            }
        )
        if position % 250 == 0 or position == len(rows) - 1:
            print(f"  scanned episode {position + 1}/{len(rows)}", flush=True)
    return scan, per_episode


def build_signal_distribution(
    scan: FrameScan, names_action: list[str], names_state: list[str], fps: float, bins: int
) -> dict[str, Any]:
    if not scan.action:
        return {"status": "no_frames_scanned"}
    action = np.concatenate(scan.action, axis=0)
    state = np.concatenate(scan.state, axis=0)
    motion = np.concatenate(scan.motion, axis=0) if scan.motion else np.empty((0, action.shape[1]))
    dt = np.concatenate(scan.dt) if scan.dt else np.empty(0)
    tracking = action - state if action.shape == state.shape else np.empty((0, action.shape[1]))

    def per_joint(values: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
        entries = []
        for i in range(values.shape[1] if values.ndim == 2 else 0):
            column = values[:, i]
            summary = describe(column)
            summary["name"] = names[i] if i < len(names) else f"dim{i}"
            summary["index"] = i
            summary["histogram"] = histogram(column, bins)
            if summary["count"]:
                span = summary["max"] - summary["min"]
                margin = SATURATION_FRACTION * span
                saturated_low = int(np.count_nonzero(column <= summary["min"] + margin))
                saturated_high = int(np.count_nonzero(column >= summary["max"] - margin))
                summary["range"] = span
                summary["saturated_low_share"] = round(saturated_low / summary["count"], 5)
                summary["saturated_high_share"] = round(saturated_high / summary["count"], 5)
            entries.append(summary)
        return entries

    result: dict[str, Any] = {
        "frames_pooled": int(action.shape[0]),
        "frames_scanned": scan.frames_scanned,
        "action": per_joint(action, names_action),
        "observation_state": per_joint(state, names_state),
        "step_motion_abs": per_joint(motion, names_state),
        "tracking_error": per_joint(tracking, names_action),
    }

    if state.shape[1]:
        gripper = state[:, -1]
        midpoint = float((gripper.min() + gripper.max()) / 2.0)
        result["gripper"] = {
            "name": names_state[-1] if names_state else "gripper",
            "midpoint": round(midpoint, 4),
            "below_midpoint_share": round(float(np.count_nonzero(gripper < midpoint)) / gripper.size, 5),
            "above_midpoint_share": round(float(np.count_nonzero(gripper >= midpoint)) / gripper.size, 5),
            "histogram": histogram(gripper, bins),
        }

    if motion.size:
        span = np.maximum(state.max(axis=0) - state.min(axis=0), 1e-9)
        static_frames = np.all(motion < STATIC_MOTION_FRACTION * span, axis=1)
        result["static_frame_share"] = round(float(np.count_nonzero(static_frames)) / motion.shape[0], 5)
        result["step_motion_norm"] = describe(np.linalg.norm(motion, axis=1))

    if dt.size:
        expected = 1.0 / fps
        result["timestamp_delta"] = describe(dt)
        result["timestamp_delta"]["expected"] = round(expected, 6)
        result["timestamp_delta"]["irregular_share"] = round(
            float(np.count_nonzero(np.abs(dt - expected) > DT_TOLERANCE * expected)) / dt.size, 6
        )
        result["timestamp_delta"]["negative_steps"] = int(np.count_nonzero(dt <= 0))
        result["timestamp_delta"]["histogram"] = histogram(dt, bins)

    result["integrity"] = {
        "episodes_scanned": scan.episodes_scanned,
        "frames_scanned": scan.frames_scanned,
        "episodes_with_row_count_mismatch": scan.length_mismatch[:50],
        "episodes_with_row_count_mismatch_total": len(scan.length_mismatch),
        "unreadable_data_files": scan.read_errors[:20],
        "unreadable_data_files_total": len(scan.read_errors),
    }
    return result


# --------------------------------------------------------------------------- #
# flags
# --------------------------------------------------------------------------- #


def build_flags(report: dict[str, Any]) -> list[str]:
    """Distribution observations worth a human look. Not a pass/fail verdict."""
    flags: list[str] = []
    task = report["tasks"]
    if task["episode_imbalance"]["gini"] > 0.3:
        flags.append(
            f"task episode counts are imbalanced (gini={task['episode_imbalance']['gini']}, "
            f"top-5 tasks hold {task['episode_imbalance']['top5_share']:.1%} of episodes)"
        )
    if task["duplicate_task_strings"]:
        flags.append(
            f"meta/tasks.parquet holds {len(task['duplicate_task_strings'])} duplicate task "
            f"string(s): {', '.join(task['duplicate_task_strings'])}"
        )
    if task["tasks_declared_but_unused"]:
        flags.append(f"{len(task['tasks_declared_but_unused'])} declared task(s) have no episodes")
    if task["multi_task_episodes"]:
        flags.append(f"{task['multi_task_episodes']} episode(s) carry more than one task label")

    lengths = report["episode_lengths"]
    if lengths["outlier_share"] > 0.05:
        flags.append(
            f"{lengths['outlier_count']} episodes ({lengths['outlier_share']:.1%}) fall outside "
            "the 1.5-IQR length fence"
        )
    frames = lengths["frames"]
    if frames.get("count") and frames["std"] / max(frames["mean"], 1e-9) > 0.3:
        flags.append(
            f"episode length is highly variable (mean={frames['mean']:.0f}, std={frames['std']:.0f} frames)"
        )
    length_shape = frames.get("shape", {})
    if (length_shape.get("bimodality_coefficient") or 0.0) > 0.555:
        flags.append(
            f"episode length is multimodal (bimodality={length_shape['bimodality_coefficient']}, "
            f"TVD to best-fit normal={length_shape['tvd_vs_normal']}); mean and std do not "
            "summarise it, read the histogram"
        )

    signals = report.get("signals", {})
    if signals.get("status") == "no_frames_scanned" or not signals:
        return flags
    for group in ("action", "observation_state"):
        for joint in signals.get(group, []):
            share = joint.get("saturated_low_share", 0.0) + joint.get("saturated_high_share", 0.0)
            if share > 0.25:
                flags.append(
                    f"{group}.{joint['name']} spends {share:.1%} of frames at its observed range "
                    "limits (possible clipping or a resting pose)"
                )
    if signals.get("static_frame_share", 0.0) > 0.2:
        flags.append(
            f"{signals['static_frame_share']:.1%} of steps show near-zero motion on every joint"
        )
    delta = signals.get("timestamp_delta", {})
    if delta.get("irregular_share", 0.0) > 0.01:
        flags.append(
            f"{delta['irregular_share']:.2%} of timestamp steps deviate more than "
            f"{DT_TOLERANCE:.0%} from 1/fps"
        )
    if delta.get("negative_steps"):
        flags.append(f"{delta['negative_steps']} non-increasing timestamp step(s)")
    integrity = signals.get("integrity", {})
    if integrity.get("episodes_with_row_count_mismatch_total"):
        flags.append(
            f"{integrity['episodes_with_row_count_mismatch_total']} episode(s) read a row count "
            "different from their declared length"
        )
    if integrity.get("unreadable_data_files_total"):
        flags.append(f"{integrity['unreadable_data_files_total']} data parquet(s) could not be read")
    return flags


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def shape_rows(report: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Collect every signal that carries shape statistics, in report order."""
    rows: list[tuple[str, dict[str, Any]]] = [
        ("episode_length", report["episode_lengths"]["frames"].get("shape", {}))
    ]
    signals = report.get("signals")
    if signals and not signals.get("status"):
        for key, label in (
            ("action", "action"),
            ("observation_state", "state"),
            ("step_motion_abs", "|dstate|"),
            ("tracking_error", "act-state"),
        ):
            for joint in signals.get(key, []):
                name = joint.get("name", "?").replace(".pos", "")
                rows.append((f"{label}.{name}", joint.get("shape", {})))
    return [(name, shape) for name, shape in rows if shape and not shape.get("status")]


def print_shape_section(report: dict[str, Any]) -> None:
    rows = shape_rows(report)
    if not rows:
        return
    print()
    print("-" * 100)
    print("DISTRIBUTION SHAPE  (departure from normality, as effect sizes)")
    print("-" * 100)
    print("  normal reference: skew 0.00, excess kurtosis 0.00, bimodality 0.333, TVD 0.000")
    print(f"  {'signal':<28}{'skew':>10}{'ex.kurt':>10}{'bimodality':>13}{'TVD':>8}   verdict")
    for name, shape in rows:
        bimodality = shape.get("bimodality_coefficient")
        print(
            f"  {name:<28}{shape['skewness']:>10.2f}{shape['excess_kurtosis']:>10.2f}"
            f"{(bimodality if bimodality is not None else float('nan')):>13.3f}"
            f"{shape['tvd_vs_normal']:>8.3f}   {shape_verdict(shape)}"
        )


def shape_verdict(shape: dict[str, Any]) -> str:
    """Short reading of the shape descriptors; advisory, not a test result.

    Sarle's bimodality coefficient is inflated by skewness alone - its numerator
    is skew^2 + 1 - so a one-sided heavy tail scores as high as a genuine second
    mode. Tail shape is therefore checked first, and the coefficient is trusted
    only for samples that are not strongly skewed.
    """
    bimodality = shape.get("bimodality_coefficient")
    kurtosis = shape.get("excess_kurtosis", 0.0)
    skewness = abs(shape.get("skewness", 0.0))
    if kurtosis > 3.0 and skewness > 2.0:
        return "heavy one-sided tail (zero-inflated)"
    if kurtosis > 3.0:
        return "bell-shaped but heavy-tailed"
    if kurtosis < -0.5 and bimodality is not None and bimodality > 0.555:
        return "multimodal / plateaued"
    if skewness > 1.0:
        return "skewed, peaked"
    if bimodality is not None and bimodality > 0.555:
        return "multimodal"
    if shape.get("tvd_vs_normal", 0.0) < 0.05:
        return "close to normal"
    return "unimodal, non-normal"


def format_stats_row(name: str, stats: dict[str, Any]) -> str:
    if not stats.get("count"):
        return f"  {name:<20} (no data)"
    return (
        f"  {name:<20} min={stats['min']:>10.3f}  q01={stats['q01']:>10.3f}  "
        f"q50={stats['q50']:>10.3f}  q99={stats['q99']:>10.3f}  max={stats['max']:>10.3f}  "
        f"mean={stats['mean']:>10.3f}  std={stats['std']:>9.3f}"
    )


def print_report(config: Config, report: dict[str, Any]) -> None:
    overview = report["overview"]
    print("=" * 100)
    print("DATASET DISTRIBUTION REPORT")
    print("=" * 100)
    print(f"Dataset        : {overview['dataset']}")
    print(f"Format         : LeRobot {overview['codebase_version']}   robot={overview['robot_type']}")
    print(
        f"Scale          : {overview['total_episodes']:,} episodes  "
        f"{overview['total_frames']:,} frames  {overview['total_tasks']} tasks  "
        f"{overview['fps']:g} fps  ({overview['total_duration_hours']:.2f} h)"
    )
    sizes = overview["bytes_on_disk"]
    print(
        f"On disk        : data {sizes['data'] / 1e6:,.0f} MB   videos {sizes['videos'] / 1e9:,.1f} GB"
        f"   meta {sizes['meta'] / 1e6:,.1f} MB"
    )
    for key, spec in overview["video_features"].items():
        shape = spec.get("shape") or []
        print(f"  video {key:<28} {shape}  codec={spec.get('codec')}  fps={spec.get('fps')}")

    tasks = report["tasks"]
    print()
    print("-" * 100)
    print(f"TASK DISTRIBUTION  ({tasks['tasks_observed_in_episodes']} tasks with episodes)")
    print("-" * 100)
    ep_imb = tasks["episode_imbalance"]
    print(
        f"  balance: gini={ep_imb['gini']}  normalized_entropy={ep_imb['normalized_entropy']}  "
        f"top1={ep_imb['top1_share']:.1%}  top5={ep_imb['top5_share']:.1%}  "
        f"max/min={ep_imb['max_over_min_ratio']}"
    )
    print()
    print(f"  {'task':<34}{'episodes':>10}{'share':>9}{'frames':>12}{'minutes':>10}{'mean len':>10}")
    for row in tasks["per_task"][: config.top_tasks]:
        print(
            f"  {row['task'][:33]:<34}{row['episodes']:>10,}{row['episode_share']:>8.1%}"
            f"{row['frames']:>12,}{row['minutes']:>10.1f}{row['length_mean']:>10.0f}"
        )
    if len(tasks["per_task"]) > config.top_tasks:
        rest = tasks["per_task"][config.top_tasks :]
        print(
            f"  {'... ' + str(len(rest)) + ' more tasks':<34}"
            f"{sum(r['episodes'] for r in rest):>10,}"
            f"{sum(r['episode_share'] for r in rest):>8.1%}"
            f"{sum(r['frames'] for r in rest):>12,}"
            f"{sum(r['minutes'] for r in rest):>10.1f}"
        )
    print()
    print(f"  task families (session suffix stripped): {len(tasks['per_family'])}")
    for row in tasks["per_family"][:15]:
        print(
            f"  {row['family'][:33]:<34}{row['episodes']:>10,}{row['episode_share']:>8.1%}"
            f"{row['frames']:>12,}{row['minutes']:>10.1f}{row['raw_tasks']:>10} raw"
        )

    lengths = report["episode_lengths"]
    print()
    print("-" * 100)
    print("EPISODE LENGTH DISTRIBUTION")
    print("-" * 100)
    print(format_stats_row("length (frames)", lengths["frames"]))
    print(format_stats_row("length (seconds)", lengths["seconds"]))
    print(
        f"  1.5-IQR fence [{lengths['outlier_bounds']['low']:.0f}, "
        f"{lengths['outlier_bounds']['high']:.0f}] frames -> {lengths['outlier_count']} outliers "
        f"({lengths['outlier_share']:.1%})"
    )
    print()
    for line in ascii_histogram(lengths["histogram"], config.hist_width):
        print(line)

    packing = report.get("video_packing", {})
    if packing:
        print()
        print("-" * 100)
        print("VIDEO PACKING")
        print("-" * 100)
        for key, spec in packing.items():
            if spec.get("status"):
                print(f"  {key:<32} {spec['status']}")
                continue
            per_file = spec["episodes_per_file"]
            print(
                f"  {key:<32} files={spec['video_files_on_disk']:>4}  "
                f"episodes/file min={per_file.get('min', 0):.0f} med={per_file.get('q50', 0):.0f} "
                f"max={per_file.get('max', 0):.0f}  total={spec['total_size_mb'] / 1024:.1f} GB"
            )

    signals = report.get("signals")
    if not signals or signals.get("status"):
        print()
        print("Frame-level signals not scanned (use --scan frames).")
    else:
        print()
        print("-" * 100)
        print(f"FRAME-LEVEL SIGNALS  ({signals['frames_pooled']:,} frames pooled)")
        print("-" * 100)
        for label, key in (
            ("action", "action"),
            ("observation.state", "observation_state"),
            ("|state[t+1]-state[t]|", "step_motion_abs"),
            ("action - state", "tracking_error"),
        ):
            print(f"  {label}:")
            for joint in signals[key]:
                print(format_stats_row(joint["name"], joint))
            print()
        if "static_frame_share" in signals:
            print(f"  near-static steps      : {signals['static_frame_share']:.2%}")
        if "gripper" in signals:
            gripper = signals["gripper"]
            print(
                f"  gripper open/close     : {gripper['below_midpoint_share']:.1%} below / "
                f"{gripper['above_midpoint_share']:.1%} above midpoint {gripper['midpoint']:.2f}"
            )
        delta = signals.get("timestamp_delta")
        if delta:
            print(
                f"  timestamp step         : expected={delta['expected']:.5f}s  "
                f"median={delta['q50']:.5f}s  irregular={delta['irregular_share']:.3%}  "
                f"non-increasing={delta['negative_steps']}"
            )
        print()
        print("  observation.state histogram per joint:")
        for joint in signals["observation_state"]:
            print(f"    {joint['name']}")
            for line in ascii_histogram(joint["histogram"], config.hist_width, max_rows=12):
                print("  " + line)

    print_shape_section(report)

    print()
    print("-" * 100)
    print("FLAGS")
    print("-" * 100)
    if report["flags"]:
        for flag in report["flags"]:
            print(f"  - {flag}")
    else:
        print("  none")


def markdown_report(config: Config, report: dict[str, Any]) -> str:
    overview = report["overview"]
    tasks = report["tasks"]
    lengths = report["episode_lengths"]
    out = [
        "# Dataset distribution report",
        "",
        f"- **Dataset**: `{overview['dataset']}`",
        f"- **Format**: LeRobot {overview['codebase_version']}, robot `{overview['robot_type']}`",
        f"- **Scale**: {overview['total_episodes']:,} episodes, {overview['total_frames']:,} frames, "
        f"{overview['total_tasks']} tasks, {overview['fps']:g} fps "
        f"({overview['total_duration_hours']:.2f} h)",
        f"- **Scan depth**: `{config.scan}`" + (f" (stride {config.stride})" if config.stride > 1 else ""),
        "",
        "## Flags",
        "",
    ]
    out.extend([f"- {flag}" for flag in report["flags"]] or ["- none"])

    out += [
        "",
        "## Task distribution",
        "",
        f"Episode-count balance: gini `{tasks['episode_imbalance']['gini']}`, "
        f"normalized entropy `{tasks['episode_imbalance']['normalized_entropy']}`, "
        f"top-5 share `{tasks['episode_imbalance']['top5_share']:.1%}`.",
        "",
        "| task | family | episodes | share | frames | minutes | mean len |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in tasks["per_task"]:
        out.append(
            f"| `{row['task']}` | `{row['family']}` | {row['episodes']:,} | "
            f"{row['episode_share']:.2%} | {row['frames']:,} | {row['minutes']:.1f} | "
            f"{row['length_mean']:.0f} |"
        )

    out += [
        "",
        "### Task families",
        "",
        "| family | raw tasks | episodes | share | frames | minutes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in tasks["per_family"]:
        out.append(
            f"| `{row['family']}` | {row['raw_tasks']} | {row['episodes']:,} | "
            f"{row['episode_share']:.2%} | {row['frames']:,} | {row['minutes']:.1f} |"
        )

    out += [
        "",
        "## Episode length",
        "",
        "| statistic | frames | seconds |",
        "| --- | ---: | ---: |",
    ]
    for key in ("min", "q01", "q25", "q50", "q75", "q99", "max", "mean", "std"):
        out.append(
            f"| {key} | {lengths['frames'].get(key, float('nan')):.1f} | "
            f"{lengths['seconds'].get(key, float('nan')):.2f} |"
        )
    out += [
        "",
        "```",
        *ascii_histogram(lengths["histogram"], config.hist_width),
        "```",
    ]

    signals = report.get("signals")
    if signals and not signals.get("status"):
        out += ["", "## Frame-level signals", "", f"Frames pooled: {signals['frames_pooled']:,}.", ""]
        for label, key in (
            ("action", "action"),
            ("observation.state", "observation_state"),
            ("per-step |dstate|", "step_motion_abs"),
            ("action - state", "tracking_error"),
        ):
            out += [
                f"### {label}",
                "",
                "| joint | min | q01 | q25 | q50 | q75 | q99 | max | mean | std |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for joint in signals[key]:
                if not joint.get("count"):
                    out.append(f"| `{joint['name']}` | | | | | | | | | |")
                    continue
                out.append(
                    f"| `{joint['name']}` | {joint['min']:.3f} | {joint['q01']:.3f} | "
                    f"{joint['q25']:.3f} | {joint['q50']:.3f} | {joint['q75']:.3f} | "
                    f"{joint['q99']:.3f} | {joint['max']:.3f} | {joint['mean']:.3f} | "
                    f"{joint['std']:.3f} |"
                )
            out.append("")
        if "static_frame_share" in signals:
            out.append(f"- Near-static steps: {signals['static_frame_share']:.2%}")
        if "gripper" in signals:
            gripper = signals["gripper"]
            out.append(
                f"- Gripper: {gripper['below_midpoint_share']:.1%} below / "
                f"{gripper['above_midpoint_share']:.1%} above midpoint {gripper['midpoint']:.2f}"
            )
        delta = signals.get("timestamp_delta")
        if delta:
            out.append(
                f"- Timestamp step: median {delta['q50']:.5f}s vs expected {delta['expected']:.5f}s, "
                f"{delta['irregular_share']:.3%} irregular, {delta['negative_steps']} non-increasing"
            )

    rows = shape_rows(report)
    if rows:
        out += [
            "",
            "## Distribution shape",
            "",
            "Departure from normality as effect sizes, not hypothesis tests: at these sample "
            "sizes Shapiro-Wilk and KS reject on any trivial deviation, so their p-values carry "
            "no information. Normal reference: skew 0, excess kurtosis 0, bimodality 0.333, "
            "TVD 0. Bimodality above 0.555 indicates a multimodal or plateaued shape; TVD is the "
            "total variation distance to the best-fit normal.",
            "",
            "| signal | skew | excess kurtosis | bimodality | TVD vs normal | reading |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for name, shape in rows:
            bimodality = shape.get("bimodality_coefficient")
            out.append(
                f"| `{name}` | {shape['skewness']:.2f} | {shape['excess_kurtosis']:.2f} | "
                f"{bimodality:.3f} | {shape['tvd_vs_normal']:.3f} | {shape_verdict(shape)} |"
                if bimodality is not None
                else f"| `{name}` | {shape['skewness']:.2f} | {shape['excess_kurtosis']:.2f} | | "
                f"{shape['tvd_vs_normal']:.3f} | {shape_verdict(shape)} |"
            )

    packing = report.get("video_packing")
    if packing:
        out += [
            "",
            "## Video packing",
            "",
            "| video key | files on disk | episodes/file (min/med/max) | total GB |",
            "| --- | ---: | ---: | ---: |",
        ]
        for key, spec in packing.items():
            if spec.get("status"):
                out.append(f"| `{key}` | | {spec['status']} | |")
                continue
            per_file = spec["episodes_per_file"]
            out.append(
                f"| `{key}` | {spec['video_files_on_disk']} | "
                f"{per_file.get('min', 0):.0f} / {per_file.get('q50', 0):.0f} / "
                f"{per_file.get('max', 0):.0f} | {spec['total_size_mb'] / 1024:.1f} |"
            )
    return "\n".join(out) + "\n"


def write_outputs(config: Config, report: dict[str, Any], per_episode: list[dict[str, Any]]) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "distribution_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=False), encoding="utf-8"
    )
    (config.out_dir / "distribution_report.md").write_text(
        markdown_report(config, report), encoding="utf-8"
    )
    task_rows = report["tasks"]["per_task"]
    if task_rows:
        with (config.out_dir / "task_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
            writer.writeheader()
            writer.writerows(task_rows)
    if per_episode:
        with (config.out_dir / "episode_distribution.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_episode[0]))
            writer.writeheader()
            writer.writerows(per_episode)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def run_report(config: Config) -> int:
    info = read_json(config.root / "meta" / "info.json")
    episodes = load_episodes(config.root)
    tasks = load_tasks(config.root)
    fps = float(info.get("fps") or 30.0)

    report: dict[str, Any] = {
        "overview": build_overview(config.root, info, episodes),
        "scan": {
            "depth": config.scan,
            "stride": config.stride,
            "bins": config.bins,
            "max_episodes": config.max_episodes,
        },
        "tasks": build_task_distribution(episodes, tasks, fps),
        "episode_lengths": build_length_distribution(episodes, fps, config.bins),
        "video_packing": build_video_packing(episodes, config.root, info),
    }

    per_episode: list[dict[str, Any]] = []
    if config.scan == "frames":
        print("Scanning data parquets...", flush=True)
        scan, per_episode = scan_frames(config, episodes, fps)
        report["signals"] = build_signal_distribution(
            scan,
            joint_names(info, "action"),
            joint_names(info, "observation.state"),
            fps,
            config.bins,
        )

    report["flags"] = build_flags(report)
    print_report(config, report)
    write_outputs(config, report, per_episode)
    print()
    print(f"Written to: {config.out_dir}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile the data distribution of a LeRobot v3 dataset (read only)."
    )
    parser.add_argument("dataset", type=Path, help="dataset root, e.g. data/lerobot_rgb_vla")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="report directory; defaults to <dataset>/../distribution_reports/<dataset name>",
    )
    parser.add_argument(
        "--scan",
        choices=["meta", "frames"],
        default="frames",
        help="metadata only, or also stream every data parquet for frame-level statistics",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="keep every Nth frame in the pooled frame-level statistics (memory control)",
    )
    parser.add_argument("--bins", type=int, default=40, help="histogram bins stored in the JSON")
    parser.add_argument("--hist-width", type=int, default=48, help="ASCII histogram bar width")
    parser.add_argument("--top-tasks", type=int, default=25, help="tasks listed in the console report")
    parser.add_argument(
        "--max-episodes", type=int, default=None, help="scan only the first N episodes (debugging)"
    )
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be >= 1")
    if args.bins < 1:
        parser.error("--bins must be >= 1")

    config = Config(
        root=args.dataset,
        out_dir=args.out_dir or args.dataset.parent / DEFAULT_REPORT_DIR / args.dataset.name,
        scan=args.scan,
        stride=args.stride,
        bins=args.bins,
        hist_width=args.hist_width,
        top_tasks=args.top_tasks,
        max_episodes=args.max_episodes,
    )
    raise SystemExit(run_report(config))


if __name__ == "__main__":
    main()
