from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from task_semantic_studio import build_audit_command
from utils.task_description_parser import HeuristicTaskDescriptionParser
from utils.task_semantic_profile import TaskProfileRegistry, TaskSemanticProfile


class TaskSemanticProfileTests(unittest.TestCase):
    def test_parser_builds_editable_chinese_profile_and_registry_matches_alias(self) -> None:
        description = (
            "机器人接近桌面上的红色杯子，抓住杯子，抬起杯子，"
            "把杯子放入右侧收纳箱并松开夹爪。"
            "杯子最终必须完全位于收纳箱内部并保持稳定。"
        )
        parser = HeuristicTaskDescriptionParser(review_threshold=0.62)
        profile = parser.parse(description, task_name="place red cup in bin")

        stage_names = [stage.name for stage in profile.stages]
        self.assertEqual(stage_names, ["approach", "grasp", "lift", "place", "release"])
        self.assertEqual(profile.review_threshold, 0.62)
        objects = {item.name: item.attributes for item in profile.objects}
        self.assertIn("red", objects["cup"])
        self.assertNotIn("right", objects["cup"])
        self.assertIn("right", objects["storage_bin"])
        self.assertTrue(profile.success_criteria)

        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = Path(tmp)
            profile.save(profile_dir / "place_red_cup_in_bin.json")
            registry = TaskProfileRegistry.from_directory(profile_dir)
            matched = registry.match("place red cup in bin")
            self.assertIsNotNone(matched)
            self.assertEqual(matched.task_key, profile.task_key)

    def test_profile_validation_rejects_missing_observable_condition(self) -> None:
        payload = {
            "task_key": "open_drawer",
            "task_name": "open drawer",
            "instruction": "Open the drawer",
            "stages": [
                {
                    "name": "pull",
                    "description": "Pull the drawer",
                    "observable_condition": "",
                }
            ],
            "success_criteria": ["The drawer is open"],
            "failure_criteria": [],
        }
        with self.assertRaisesRegex(ValueError, "observable_condition"):
            TaskSemanticProfile.from_dict(payload)

    def test_studio_builds_safe_argument_list(self) -> None:
        command = build_audit_command(
            dataset=Path("dataset"),
            profile_dir=Path("task_profiles"),
            evaluator_spec="my_backend:create_evaluator",
            out_dir=Path("reports"),
            evaluator_config=Path("evaluator.json"),
            review_threshold=0.55,
            python_executable="python",
        )
        self.assertEqual(command[0], "python")
        self.assertIn("--task-profile-dir", command)
        self.assertIn("--semantic-evaluator", command)
        self.assertIn("my_backend:create_evaluator", command)
        self.assertIn("--out-dir", command)


if __name__ == "__main__":
    unittest.main()
