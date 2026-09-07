"""Tests for the distribution report tool and the RGB export pipeline step."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.data_quality_audit import audit_dataset
from pipeline.rgbd_export_rgb import run_export
from tools.dataset_distribution_report import Config as ReportConfig, run_report
from tests.test_dataset_etl_pipeline import DEPTH_KEY, RGB_KEY, write_dataset, write_rgb_dataset


def add_tasks_column(dataset: Path, task: str) -> None:
    """Mirror the recorder's per-episode `tasks` metadata column."""
    episode_path = dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(episode_path)
    tasks = pa.array([[task]] * table.num_rows, type=pa.list_(pa.string()))
    pq.write_table(table.append_column("tasks", tasks), episode_path)


class DistributionReportTests(unittest.TestCase):
    def test_frame_scan_profiles_a_synthetic_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_rgb_dataset(root, "dist_set", [30, 40], 10)
            add_tasks_column(dataset, "pick up cups")
            out = root / "report"
            config = ReportConfig(
                root=dataset, out_dir=out, scan="frames", stride=1,
                bins=10, hist_width=20, top_tasks=5, max_episodes=None,
            )
            self.assertEqual(run_report(config), 0)

            summary = json.loads((out / "distribution_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["overview"]["total_episodes"], 2)
            self.assertEqual(summary["signals"]["frames_scanned"], 70)
            self.assertEqual(summary["signals"]["integrity"]["episodes_with_row_count_mismatch_total"], 0)
            self.assertEqual(summary["tasks"]["per_task"][0]["task"], "pick up cups")
            self.assertEqual(summary["tasks"]["per_task"][0]["episodes"], 2)
            self.assertTrue((out / "distribution_report.md").exists())
            self.assertTrue((out / "task_distribution.csv").exists())
            self.assertTrue((out / "episode_distribution.csv").exists())

    def test_meta_scan_survives_a_missing_tasks_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = write_rgb_dataset(root, "no_tasks_set", [30], 10)
            out = root / "report"
            config = ReportConfig(
                root=dataset, out_dir=out, scan="meta", stride=1,
                bins=10, hist_width=20, top_tasks=5, max_episodes=None,
            )
            self.assertEqual(run_report(config), 0)
            summary = json.loads((out / "distribution_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["tasks"]["per_task"][0]["task"], "<unlabelled>")
            self.assertNotIn("signals", summary)


class RgbExportTests(unittest.TestCase):
    def test_export_strips_depth_and_survives_the_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = write_dataset(root, "rgbd_set", [2, 3], 10)
            out = root / "rgb_set"
            run_export(src, out, overwrite=False)

            info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertNotIn(DEPTH_KEY, info["features"])
            self.assertIn(RGB_KEY, info["features"])
            self.assertFalse((out / "depth_sidecar").exists())
            self.assertFalse((out / "meta" / "rgbd_vla_depth_recording.json").exists())
            for parquet in sorted((out / "data").rglob("*.parquet")):
                self.assertNotIn(DEPTH_KEY, pq.read_table(parquet).column_names)
            # Source untouched.
            self.assertTrue((src / "depth_sidecar").exists())

            exit_code = audit_dataset(
                root=out,
                out_dir=root / "audit",
                depth_check="header",
                video_check="metadata",
                video_content_check="none",
                sample_frames=0,
                depth_sample_frames=0,
                duplicate_check="none",
                video_sample_stride=30,
            )
            self.assertEqual(exit_code, 0)

    def test_export_refuses_an_output_that_contains_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = write_dataset(root / "container", "rgbd_set", [2], 10)
            with self.assertRaises(ValueError):
                run_export(src, root / "container", overwrite=True)
            self.assertTrue((src / "meta" / "info.json").exists())

    def test_export_rejects_a_dataset_that_is_already_rgb_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = write_rgb_dataset(root, "rgb_set", [2], 10)
            with self.assertRaises(ValueError):
                run_export(src, root / "out", overwrite=False)


if __name__ == "__main__":
    unittest.main()
