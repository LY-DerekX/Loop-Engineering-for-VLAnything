#!/usr/bin/env python
"""Collect RGB-D policy feedback with safe, explicitly labelled human control."""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from pprint import pformat
from typing import Any, Mapping

import numpy as np
import rerun as rr


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
TOOLKIT_CANDIDATES = (PROJECT_ROOT.parents[1], PROJECT_ROOT.parents[2])
for path in (SCRIPT_ROOT, SRC_ROOT, *TOOLKIT_CANDIDATES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import record as base_record
import lerobot_policy_rgbd_vla_depth  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from policy_improvement.capture import (
    AsyncTriggerMonitor,
    BoundedCaptureBuffer,
    CapturedPacket,
    EpisodeRotator,
    EventCaptureController,
    TriggerSignal,
    load_trigger_config,
    load_window_trigger,
)
from policy_improvement.contracts import (
    CaptureMode,
    ControlPhase,
    EpisodeFeedback,
    FeedbackFrame,
    FeedbackSidecarWriter,
    InterventionSegment,
    TriggerSource,
)
from policy_improvement.intervention import (
    CapabilityHandoverAdapter,
    InterventionCommand,
    InterventionController,
    InterventionEventQueue,
    InterventionKeyConfig,
    create_intervention_key_listener,
)


@dataclass
class AutomaticTriggerConfig:
    plugin: str | None = None
    config_path: Path | None = None
    interval_frames: int = 30
    post_trigger_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.interval_frames <= 0 or self.post_trigger_seconds <= 0:
            raise ValueError("automatic trigger interval and post-trigger duration must be positive")


@dataclass
class FeedbackCollectionConfig:
    mode: str = CaptureMode.CORRECTIONS_ONLY.value
    sidecar_dir: Path | None = None
    session_id: str | None = None
    source_policy: str = ""
    overwrite_sidecar: bool = False
    keyboard_enabled: bool = True
    toggle_control_key: str = "i"
    begin_correction_key: str = "c"
    cancel_key: str = "x"
    save_event_key: str = "h"
    stop_key: str = "esc"
    buffer_seconds: float = 15.0
    buffer_memory_mb: float = 2048.0
    require_feedback_alignment: bool = True
    synchronization_seconds: float = 0.5
    automatic_trigger: AutomaticTriggerConfig = field(default_factory=AutomaticTriggerConfig)

    def __post_init__(self) -> None:
        CaptureMode(self.mode)
        if self.buffer_seconds <= 0 or self.buffer_memory_mb <= 0:
            raise ValueError("feedback buffer limits must be positive")


@dataclass
class FeedbackRecordConfig(RecordConfig):
    feedback: FeedbackCollectionConfig = field(default_factory=FeedbackCollectionConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.policy is None or self.teleop is None:
            raise ValueError("feedback collection requires both a policy and a teleoperator")
        if self.resume:
            raise ValueError("feedback collection resume is not supported; start a new traced session")


@dataclass(frozen=True)
class DatasetPacket:
    timestamp: float
    observation_frame: Mapping[str, Any]
    observation_processed: Mapping[str, Any]
    action_values: Any
    policy_action: Any
    phase: ControlPhase
    intervention_id: int
    trigger_source: TriggerSource
    training_eligible: bool
    details: Mapping[str, Any] = field(default_factory=dict)


def clone_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def episode_buffer_size(dataset: LeRobotDataset) -> int:
    return int(dataset.episode_buffer.get("size", 0))


def reset_policy_runtime(policy: Any, preprocessor: Any, postprocessor: Any) -> None:
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()


def default_sidecar_dir(dataset: LeRobotDataset, session_id: str) -> Path:
    dataset_root = Path(dataset.root).resolve()
    return dataset_root.parent / "feedback_outputs" / dataset_root.name / session_id


def make_feedback_frame(
    dataset: LeRobotDataset,
    packet: DatasetPacket,
    episode_origin: float,
) -> FeedbackFrame:
    return FeedbackFrame(
        episode_index=dataset.num_episodes,
        frame_index=episode_buffer_size(dataset),
        timestamp=max(0.0, packet.timestamp - episode_origin),
        phase=packet.phase,
        intervention_id=packet.intervention_id,
        trigger_source=packet.trigger_source,
        policy_action=packet.policy_action,
        executed_action=packet.action_values,
        training_eligible=packet.training_eligible,
        details=packet.details,
    )


def store_packet(
    dataset: LeRobotDataset,
    packet: DatasetPacket,
    episode_origin: float,
    feedback_frames: list[FeedbackFrame],
    task: str,
) -> None:
    feedback_frame = make_feedback_frame(dataset, packet, episode_origin)
    action_frame = build_dataset_frame(dataset.features, packet.action_values, prefix=ACTION)
    frame = {**packet.observation_frame, **action_frame, "task": task}
    base_record.persist_depth_sidecars(dataset, dict(packet.observation_frame))
    dataset.add_frame(frame)
    feedback_frames.append(feedback_frame)


def feedback_segment(episode_index: int, window: Any) -> InterventionSegment:
    return InterventionSegment(
        episode_index=episode_index,
        intervention_id=window.intervention_id,
        recovery_start_frame=window.recovery_start_frame,
        recovery_end_frame=window.recovery_end_frame,
        correction_start_frame=window.correction_start_frame,
        correction_end_frame=window.correction_end_frame,
        trigger_source=window.trigger_source,
        outcome=window.outcome,
    )


def save_feedback_episode(
    dataset: LeRobotDataset,
    writer: FeedbackSidecarWriter,
    frames: list[FeedbackFrame],
    segments: list[InterventionSegment],
    *,
    task: str,
    source_policy: str,
    mode: CaptureMode,
    outcome: str,
    trigger_details: Mapping[str, Any] | None = None,
) -> None:
    if not frames:
        return
    episode_index = dataset.num_episodes
    base_record.flush_depth_writes()
    dataset.save_episode()
    writer.append_episode(
        frames,
        EpisodeFeedback(
            episode_index=episode_index,
            length=len(frames),
            source_policy=source_policy,
            capture_mode=mode,
            task=task,
            outcome=outcome,
            intervention_count=len(segments),
            details=dict(trigger_details or {}),
        ),
        segments,
    )


def run_feedback_loop(
    *,
    cfg: FeedbackRecordConfig,
    robot: Any,
    teleop: Any,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    teleop_action_processor: Any,
    robot_action_processor: Any,
    robot_observation_processor: Any,
    dataset: LeRobotDataset,
    writer: FeedbackSidecarWriter,
) -> None:
    mode = CaptureMode(cfg.feedback.mode)
    task = cfg.dataset.single_task
    source_policy = cfg.feedback.source_policy or str(cfg.policy.pretrained_path or "")
    controller = InterventionController()
    handover = CapabilityHandoverAdapter(
        synchronization_seconds=cfg.feedback.synchronization_seconds,
        require_feedback_alignment=cfg.feedback.require_feedback_alignment,
    )
    events = InterventionEventQueue()
    listener = None
    if cfg.feedback.keyboard_enabled:
        if is_headless():
            raise RuntimeError("keyboard feedback controls are unavailable in headless mode")
        listener = create_intervention_key_listener(
            events,
            InterventionKeyConfig(
                toggle_control=cfg.feedback.toggle_control_key,
                begin_correction=cfg.feedback.begin_correction_key,
                cancel=cfg.feedback.cancel_key,
                save_event=cfg.feedback.save_event_key,
                stop=cfg.feedback.stop_key,
            ),
        )

    ring = BoundedCaptureBuffer(
        max_seconds=cfg.feedback.buffer_seconds,
        fps=cfg.dataset.fps,
        max_memory_mb=cfg.feedback.buffer_memory_mb,
    )
    event_capture = EventCaptureController(ring)
    rotator = EpisodeRotator(cfg.dataset.episode_time_s)
    trigger_monitor = None
    if cfg.feedback.automatic_trigger.plugin:
        if mode != CaptureMode.EVENT_BUFFER:
            raise ValueError("automatic triggers are only valid in event_buffer mode")
        trigger_monitor = AsyncTriggerMonitor(
            load_window_trigger(
                cfg.feedback.automatic_trigger.plugin,
                load_trigger_config(cfg.feedback.automatic_trigger.config_path),
            )
        )

    feedback_frames: list[FeedbackFrame] = []
    segments: list[InterventionSegment] = []
    episode_origin: float | None = None
    event_deadline: float | None = None
    last_robot_action: Any = None
    last_policy_action: Any = None
    loop_index = 0
    stop_requested = False

    def flush_event_buffer(signal: TriggerSignal) -> None:
        nonlocal episode_origin, event_deadline
        buffered = event_capture.start(signal)
        if buffered:
            episode_origin = buffered[0].timestamp
            for captured in buffered:
                store_packet(dataset, captured.payload, episode_origin, feedback_frames, task)
        elif episode_origin is None:
            episode_origin = time.perf_counter()
        event_deadline = time.perf_counter() + cfg.feedback.automatic_trigger.post_trigger_seconds

    def finish_episode(outcome: str) -> None:
        nonlocal feedback_frames, segments, episode_origin, event_deadline
        signal = event_capture.stop() if mode == CaptureMode.EVENT_BUFFER else None
        trigger_details = None
        if signal is not None:
            trigger_details = {
                "trigger_source": TriggerSource(signal.source).value,
                "trigger_score": signal.score,
                "trigger_reason": signal.reason,
                "trigger_details": dict(signal.details),
            }
        save_feedback_episode(
            dataset,
            writer,
            feedback_frames,
            segments,
            task=task,
            source_policy=source_policy,
            mode=mode,
            outcome=outcome,
            trigger_details=trigger_details,
        )
        feedback_frames = []
        segments = []
        episode_origin = None
        event_deadline = None
        controller.reset_episode()
        rotator.reset()

    reset_policy_runtime(policy, preprocessor, postprocessor)
    try:
        with VideoEncodingManager(dataset):
            while dataset.num_episodes < cfg.dataset.num_episodes:
                loop_start = time.perf_counter()

                for command in events.drain():
                    phase = controller.phase
                    if command == InterventionCommand.STOP:
                        stop_requested = True
                    elif command == InterventionCommand.TOGGLE_CONTROL:
                        if phase == ControlPhase.AUTONOMOUS:
                            try:
                                controller.request_takeover(TriggerSource.MANUAL)
                                handover.prepare_takeover(robot, teleop, last_robot_action)
                                if mode == CaptureMode.EVENT_BUFFER and not event_capture.recording:
                                    flush_event_buffer(
                                        TriggerSignal(True, TriggerSource.MANUAL, reason="human_takeover")
                                    )
                                controller.confirm_synchronized(episode_buffer_size(dataset))
                                handover.begin_human_control(robot, teleop)
                                log_say("Human recovery started", cfg.play_sounds)
                            except Exception:
                                controller.cancel()
                                raise
                        elif phase == ControlPhase.CORRECTION:
                            window = controller.complete(episode_buffer_size(dataset))
                            handover.end_human_control(robot, teleop)
                            handover.resume_autonomous(robot, teleop)
                            reset_policy_runtime(policy, preprocessor, postprocessor)
                            segments.append(feedback_segment(dataset.num_episodes, window))
                            log_say("Human correction completed", cfg.play_sounds)
                            if mode in {CaptureMode.CORRECTIONS_ONLY, CaptureMode.EVENT_BUFFER}:
                                finish_episode("success_after_intervention")
                    elif command == InterventionCommand.BEGIN_CORRECTION:
                        controller.begin_correction(episode_buffer_size(dataset))
                        log_say("Correction phase started", cfg.play_sounds)
                    elif command == InterventionCommand.SAVE_EVENT and mode == CaptureMode.EVENT_BUFFER:
                        if not event_capture.recording:
                            flush_event_buffer(
                                TriggerSignal(True, TriggerSource.MANUAL, reason="manual_event")
                            )
                        elif controller.phase == ControlPhase.AUTONOMOUS:
                            finish_episode("captured_event")
                    elif command == InterventionCommand.CANCEL:
                        logging.warning(
                            "Cancellation is ignored after human motion begins; complete the correction "
                            "to preserve an auditable episode."
                        )

                if stop_requested and controller.phase == ControlPhase.AUTONOMOUS:
                    break

                if trigger_monitor is not None:
                    signal = trigger_monitor.poll()
                    if signal is not None and signal.triggered and not event_capture.recording:
                        flush_event_buffer(signal)

                observation = robot.get_observation()
                observation_processed = robot_observation_processor(observation)
                observation_frame = build_dataset_frame(
                    dataset.features, observation_processed, prefix=OBS_STR
                )
                phase = controller.phase
                if phase == ControlPhase.AUTONOMOUS:
                    action_values = predict_action(
                        observation=observation_frame,
                        policy=policy,
                        device=get_safe_torch_device(policy.config.device),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy.config.use_amp,
                        task=task,
                        robot_type=robot.robot_type,
                    )
                    processed_policy_action = make_robot_action(action_values, dataset.features)
                    decision = controller.select_action(processed_policy_action)
                    last_policy_action = clone_value(processed_policy_action)
                else:
                    raw_human_action = teleop.get_action()
                    human_action = teleop_action_processor((raw_human_action, observation))
                    decision = controller.select_action(last_policy_action, human_action)

                if decision.hold_position:
                    if last_robot_action is not None:
                        robot.send_action(last_robot_action)
                    precise_sleep(max(1 / cfg.dataset.fps - (time.perf_counter() - loop_start), 0))
                    continue

                robot_action = robot_action_processor((decision.executed_action, observation))
                robot.send_action(robot_action)
                last_robot_action = clone_value(robot_action)
                timestamp = time.perf_counter()
                packet = DatasetPacket(
                    timestamp=timestamp,
                    observation_frame=clone_value(observation_frame),
                    observation_processed=clone_value(observation_processed),
                    action_values=clone_value(decision.executed_action),
                    policy_action=clone_value(last_policy_action),
                    phase=decision.phase,
                    intervention_id=decision.intervention_id,
                    trigger_source=(
                        controller.trigger_source
                        if decision.intervention_id >= 0
                        else TriggerSource.NONE
                    ),
                    training_eligible=decision.training_eligible,
                    details={
                        "policy_action_is_last_proposal": decision.phase
                        in {ControlPhase.RECOVERY, ControlPhase.CORRECTION}
                    },
                )

                should_store = mode == CaptureMode.CONTINUOUS or (
                    mode == CaptureMode.CORRECTIONS_ONLY
                    and decision.phase in {ControlPhase.RECOVERY, ControlPhase.CORRECTION}
                )
                if mode == CaptureMode.EVENT_BUFFER:
                    should_store = event_capture.observe(CapturedPacket(timestamp, packet))
                if should_store:
                    if episode_origin is None:
                        episode_origin = timestamp
                    store_packet(dataset, packet, episode_origin, feedback_frames, task)

                if cfg.display_data:
                    rr.set_time("frame_index", sequence=dataset.num_frames)
                    rr.set_time("episode_time", duration=(timestamp - (episode_origin or timestamp)))
                    log_rerun_data(
                        observation=observation_processed,
                        action=decision.executed_action,
                        compress_images=cfg.display_compressed_images,
                    )

                loop_index += 1
                if (
                    trigger_monitor is not None
                    and not event_capture.recording
                    and loop_index % cfg.feedback.automatic_trigger.interval_frames == 0
                ):
                    trigger_monitor.submit(ring.snapshot())
                if (
                    mode == CaptureMode.EVENT_BUFFER
                    and event_capture.recording
                    and event_deadline is not None
                    and time.perf_counter() >= event_deadline
                    and controller.phase == ControlPhase.AUTONOMOUS
                ):
                    finish_episode("automatic_event_capture")
                if (
                    mode == CaptureMode.CONTINUOUS
                    and feedback_frames
                    and rotator.should_rotate(controller.phase)
                ):
                    finish_episode("continuous_observation")

                elapsed = time.perf_counter() - loop_start
                sleep_time = 1 / cfg.dataset.fps - elapsed
                if sleep_time < 0:
                    logging.warning(
                        "Feedback control loop is slower than target FPS: %.1f Hz < %d Hz",
                        1 / max(elapsed, 1e-9),
                        cfg.dataset.fps,
                    )
                precise_sleep(max(sleep_time, 0.0))

            if controller.phase != ControlPhase.AUTONOMOUS:
                logging.warning("Stop requested during intervention; session waits for a clean handoff")
            if feedback_frames and controller.phase == ControlPhase.AUTONOMOUS:
                finish_episode("session_end")
    finally:
        if trigger_monitor is not None:
            trigger_monitor.close()
        if listener is not None:
            listener.stop()


@parser.wrap()
def record_feedback(cfg: FeedbackRecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="policy_feedback", ip=cfg.display_ip, port=cfg.display_port)

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = base_record.build_rgbd_dataset_features(
        cfg, robot, teleop_action_processor, robot_observation_processor
    )
    dataset = None
    writer = None
    try:
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        base_record.write_depth_manifest(dataset)
        policy = make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )
        robot.connect()
        teleop.connect()

        session_id = cfg.feedback.session_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        sidecar_dir = cfg.feedback.sidecar_dir or default_sidecar_dir(dataset, session_id)
        source_policy = cfg.feedback.source_policy or str(cfg.policy.pretrained_path or "")
        writer = FeedbackSidecarWriter(
            sidecar_dir,
            session_id=session_id,
            source_dataset=Path(dataset.root),
            source_policy=source_policy,
            capture_mode=cfg.feedback.mode,
            fps=cfg.dataset.fps,
            overwrite=cfg.feedback.overwrite_sidecar,
            details={
                "robot_type": robot.name,
                "task": cfg.dataset.single_task,
                "automatic_trigger_plugin": cfg.feedback.automatic_trigger.plugin,
            },
        )
        run_feedback_loop(
            cfg=cfg,
            robot=robot,
            teleop=teleop,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            dataset=dataset,
            writer=writer,
        )
    finally:
        log_say("Stop feedback collection", cfg.play_sounds, blocking=True)
        base_record.flush_depth_writes()
        base_record._depth_writer_pool.shutdown(wait=True)
        if writer is not None:
            writer.close()
        if dataset is not None:
            dataset.finalize()
        if getattr(robot, "is_connected", False):
            robot.disconnect()
        if getattr(teleop, "is_connected", False):
            teleop.disconnect()
        if dataset is not None and cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
    return dataset


def main() -> None:
    register_third_party_plugins()
    record_feedback()


if __name__ == "__main__":
    main()
