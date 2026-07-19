"""Model-agnostic contracts for optional semantic episode evaluation.

The base audit remains usable without any model dependencies. A semantic
backend is loaded only when explicitly requested and receives task text plus
episode-scoped video references.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class VideoSegment:
    """A video file and the time interval belonging to one episode."""

    key: str
    path: Path
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class SemanticEpisode:
    """Inputs exposed to a semantic evaluator."""

    dataset_root: Path
    episode_index: int
    length: int
    tasks: tuple[str, ...]
    videos: tuple[VideoSegment, ...]


@dataclass(frozen=True)
class SemanticResult:
    """Normalized semantic assessment returned by an evaluator."""

    progress_curve: tuple[float, ...] = ()
    success_probability: float | None = None
    failure_stage: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def final_progress(self) -> float | None:
        return self.progress_curve[-1] if self.progress_curve else None

    @property
    def score(self) -> float:
        """Prefer explicit success probability, otherwise final progress."""
        if self.success_probability is not None:
            return self.success_probability
        if self.final_progress is not None:
            return self.final_progress
        raise ValueError("semantic result has neither success probability nor progress")


@runtime_checkable
class SemanticEvaluator(Protocol):
    """Interface implemented by optional video-language scoring backends."""

    name: str

    def evaluate(self, episode: SemanticEpisode) -> SemanticResult | Mapping[str, Any]:
        """Return progress and/or success information for one episode."""


def evaluator_name(evaluator: SemanticEvaluator) -> str:
    name = str(getattr(evaluator, "name", "")).strip()
    return name or evaluator.__class__.__name__


def load_evaluator(spec: str, config: Mapping[str, Any] | None = None) -> SemanticEvaluator:
    """Load an evaluator from module:attribute.

    The attribute may be an evaluator instance, a class initialized with
    keyword arguments, or a factory accepting one configuration dictionary.
    """
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("semantic evaluator must use module:attribute syntax")

    module = importlib.import_module(module_name)
    target = getattr(module, attribute_name)
    options = dict(config or {})
    if inspect.isclass(target):
        candidate = target(**options)
    elif hasattr(target, "evaluate"):
        candidate = target
    elif callable(target):
        candidate = target(options)
    else:
        raise TypeError(f"semantic evaluator target is not usable: {spec}")

    if not callable(getattr(candidate, "evaluate", None)):
        raise TypeError(f"semantic evaluator does not define evaluate(): {spec}")
    return candidate


def load_evaluator_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("semantic evaluator config must contain one JSON object")
    return payload


def normalize_result(value: SemanticResult | Mapping[str, Any]) -> SemanticResult:
    """Validate backend output and return a stable result object."""
    if isinstance(value, SemanticResult):
        result = value
    elif isinstance(value, Mapping):
        raw_curve = value.get("progress_curve", ())
        if raw_curve is None:
            raw_curve = ()
        if isinstance(raw_curve, (str, bytes)) or not isinstance(raw_curve, Sequence):
            raise TypeError("progress_curve must be a numeric sequence")
        raw_details = value.get("details", {})
        if raw_details is None:
            raw_details = {}
        if not isinstance(raw_details, Mapping):
            raise TypeError("semantic result details must be a mapping")
        result = SemanticResult(
            progress_curve=tuple(float(item) for item in raw_curve),
            success_probability=(
                None if value.get("success_probability") is None else float(value["success_probability"])
            ),
            failure_stage=(
                None if value.get("failure_stage") in {None, ""} else str(value["failure_stage"])
            ),
            details=dict(raw_details),
        )
    else:
        raise TypeError("semantic evaluator must return SemanticResult or a mapping")

    for item in result.progress_curve:
        if not math.isfinite(item) or not 0.0 <= item <= 1.0:
            raise ValueError("progress_curve values must be finite and within [0, 1]")
    if result.success_probability is not None:
        probability = result.success_probability
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("success_probability must be finite and within [0, 1]")
    if not result.progress_curve and result.success_probability is None:
        raise ValueError("semantic result must provide progress_curve or success_probability")

    json.dumps(result.details, ensure_ascii=False)
    return result


def infer_failure_stage(progress_curve: Sequence[float]) -> str:
    """Infer a coarse failure stage when a backend does not provide one."""
    if not progress_curve:
        return "final_outcome"
    peak = max(float(value) for value in progress_curve)
    final = float(progress_curve[-1])
    if peak - final >= 0.15:
        return "regressed_after_progress"
    if peak < 1.0 / 3.0:
        return "early_progress"
    if peak < 2.0 / 3.0:
        return "middle_progress"
    return "late_progress"
