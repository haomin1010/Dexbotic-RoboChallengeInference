"""
Local control loop adapter: get_state -> Dexbotic InferenceRunner.infer -> execute_actions.

Supports keyboard: [Space] e-stop | [H] home | [B] teach | [N] record | [I] next chunk | [M] goto | [R] resume | [Q] quit
"""

import atexit
import json
import logging
import math
import queue
import sys
import termios
import threading
import time
import tty
from enum import Enum, auto
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SAFE_MODE_MAX_TRANSLATION_METERS = 0.10


class LoopState(Enum):
    RUNNING = auto()
    STOPPED = auto()
    TEACHING = auto()


class KeyboardListener:
    """Non-blocking keyboard input (Linux)."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        atexit.register(self.restore)
        t = threading.Thread(target=self._listen, daemon=True)
        t.start()

    def _listen(self):
        try:
            while True:
                ch = sys.stdin.read(1)
                if ch:
                    self._queue.put(ch.lower())
        except (EOFError, OSError):
            pass

    def get_key(self) -> str | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def restore(self):
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        except Exception:
            pass


def _prepare_safe_action(
    actions: list,
    current_action: list | None,
    max_translation_m: float = SAFE_MODE_MAX_TRANSLATION_METERS,
) -> list:
    """Cap each action's xyz step length to max_translation_m (no limit on number of actions)."""
    if not actions:
        return []

    prev_xyz = None
    if current_action and len(current_action) >= 3:
        prev_xyz = [float(current_action[i]) for i in range(3)]

    result = []
    for action in actions:
        safe = list(action)
        if len(safe) < 7:
            safe.extend([0.0] * (7 - len(safe)))

        target_xyz = [float(safe[i]) for i in range(3)]
        if prev_xyz is not None:
            delta = [target_xyz[i] - prev_xyz[i] for i in range(3)]
            distance = math.sqrt(sum(v * v for v in delta))
            if distance > max_translation_m and distance > 0:
                scale = max_translation_m / distance
                limited_xyz = [prev_xyz[i] + delta[i] * scale for i in range(3)]
                logger.warning(
                    "SAFE MODE: capped ee translation from %.4fm to %.4fm",
                    distance,
                    max_translation_m,
                )
                safe[:3] = limited_xyz
                target_xyz = limited_xyz

        prev_xyz = target_xyz
        result.append(safe)

    return result


def _save_inference_io(record_dir: Path, round_idx: int, robot_state: dict, actions: list) -> None:
    """Save input (robot_state) and output (actions) to record_dir/round_NNN/.

    Saved structure:
      input/ - current_ee_state.json [x,y,z,euler_x,euler_y,euler_z,gripper], images, state.json
      output/ - actions.json (model-returned actions, EE format)
    """
    rd = record_dir / f"round_{round_idx:04d}"
    rd.mkdir(parents=True, exist_ok=True)
    input_dir = rd / "input"
    input_dir.mkdir(exist_ok=True)
    # Images
    images = robot_state.get("images", {})
    for name, data in images.items():
        if isinstance(data, bytes):
            (input_dir / f"{name}.jpg").write_bytes(data)
    # Current robot state in EE format [x, y, z, euler_x, euler_y, euler_z, gripper]
    current_ee = robot_state.get("action")
    if current_ee is not None:
        (input_dir / "current_ee_state.json").write_text(
            json.dumps(
                {"current_ee_state": current_ee, "format": "[x, y, z, euler_x, euler_y, euler_z, gripper]"},
                indent=2,
            ),
            encoding="utf-8",
        )
    # Metadata
    state_meta = {
        "job_id": robot_state.get("job_id"),
        "timestamp": robot_state.get("timestamp"),
        "state": robot_state.get("state"),
        "pending_actions": robot_state.get("pending_actions"),
    }
    (input_dir / "state.json").write_text(json.dumps(state_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    # Output: model-returned actions (EE format, [x,y,z,euler_x,euler_y,euler_z,gripper] per step)
    output_dir = rd / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "actions.json").write_text(json.dumps(actions, indent=2), encoding="utf-8")
    logger.info("Saved inference I/O to %s", rd)


def local_control_loop_dexbotic(
    interface,
    runner=None,
    inference_client=None,
    image_size: list[int] | None = None,
    image_type: list[str] | None = None,
    action_type: str = "",
    duration: float = 1 / 15,
    use_keyboard: bool = True,
    safe_mode: bool = False,
    record_dir: Optional[str | Path] = None,
) -> None:
    """
    Main loop: get local state -> infer (runner or remote) -> execute actions locally.

    Args:
        interface: LocalRobotInterface (get_state, execute_actions) from arx5_client
        runner: Dexbotic InferenceRunner with infer(state). Ignored if inference_client is set.
        inference_client: RemoteInferenceClient (WebSocket). If set, used instead of runner.
        image_size: [width, height] for images
        image_type: e.g. ["high", "left_hand", "right_hand"]
        action_type: e.g. "leftpos"
        duration: Seconds per action step (e.g. 1/15)
        use_keyboard: If True, Space=e-stop, R=resume, Q=quit, etc.
        safe_mode: If True, cap xyz step to 10cm and require [I] before each inference request.
        record_dir: If set, save input/output to record_dir/round_NNN/ and skip execution (不执行机械臂).
    """
    inferrer = inference_client if inference_client is not None else runner
    if inferrer is None:
        raise ValueError("Either runner or inference_client must be provided")
    record_path = Path(record_dir) if record_dir else None
    round_counter = 0
    state = LoopState.STOPPED if use_keyboard else LoopState.RUNNING
    running = True
    kb = KeyboardListener() if use_keyboard else None
    request_next_chunk = not safe_mode

    if record_path:
        logger.info("RECORD MODE: saving I/O to %s, no execution", record_path)

    # Startup: go home and hold (when keyboard enabled)
    if kb:
        logger.info("Moving arm to home ...")
        interface.go_home()
        time.sleep(2.0)
        interface.hold_position()
        if safe_mode:
            logger.info(
                "Keyboard: [Space] e-stop | [H] home | [B] teach | [N] record (save to file) | [I] next chunk | [M] goto recorded | [R] resume | [Q] quit"
            )
            logger.info("SAFE MODE enabled: press [R] to arm control, then [I] before each inference request.")
        else:
            logger.info(
                "Keyboard: [Space] e-stop | [H] home | [B] teach | [N] record (save to file) | [M] goto recorded | [R] resume | [Q] quit"
            )

    if not kb:
        logger.info("Starting control loop (no keyboard)")

    try:
        while running:
            # Handle keyboard
            if kb:
                key = kb.get_key()
                if key == " ":
                    interface.hold_position()
                    state = LoopState.STOPPED
                    logger.warning("EMERGENCY STOP — position locked")
                elif key == "q":
                    running = False
                    logger.info("Quit requested")
                    break
                elif state == LoopState.STOPPED:
                    if key == "h":
                        logger.info("Resetting to home position ...")
                        interface.hold_position()
                        time.sleep(0.1)
                        interface.go_home()
                        time.sleep(2.0)
                        interface.hold_position()
                        logger.info("Home position reached")
                    elif key == "b":
                        interface.enter_teach_mode()
                        state = LoopState.TEACHING
                        logger.info("TEACH MODE — drag the arm, then press [N] to record")
                    elif key == "n":
                        if safe_mode:
                            logger.info("SAFE MODE is armed only after [R] — press [R] first.")
                        else:
                            logger.info("Not in teach mode — press [B] first.")
                    elif key == "m":
                        if interface.has_recorded_position():
                            interface.move_to_recorded()
                            logger.info("Reached recorded position")
                        else:
                            logger.warning("No position recorded — press [B] to teach, then [N] to record")
                    elif key == "r":
                        interface.hold_position()
                        time.sleep(0.1)
                        state = LoopState.RUNNING
                        request_next_chunk = not safe_mode
                        if safe_mode:
                            logger.info("RESUMED — SAFE MODE active. Press [I] to request the next action chunk.")
                        else:
                            logger.info("RESUMED — VLA control active. Press [Space] to stop.")
                elif state == LoopState.RUNNING and safe_mode:
                    if key == "i":
                        request_next_chunk = True
                        logger.info("SAFE MODE: next action chunk requested")
                elif state == LoopState.TEACHING:
                    if key == "n":
                        interface.exit_teach_and_record()
                        state = LoopState.STOPPED
                        logger.info("Position recorded and saved to file")

            if state != LoopState.RUNNING:
                time.sleep(0.05)
                continue
            if safe_mode and not request_next_chunk:
                time.sleep(0.05)
                continue

            # Get state
            robot_state = interface.get_state(image_size, image_type, action_type)
            if not robot_state or robot_state.get("state") != "normal":
                time.sleep(0.1)
                continue
            if robot_state.get("pending_actions", 0) != 0:
                time.sleep(0.1)
                continue

            robot_state["job_id"] = "local"
            request_next_chunk = False

            # Infer (local runner or remote WebSocket client)
            try:
                actions = inferrer.infer(robot_state)
            except Exception as e:
                logger.error("Inference failed: %s", e)
                time.sleep(0.5)
                continue

            if not actions:
                logger.warning("No actions from inference runner")
                time.sleep(0.1)
                continue

            if safe_mode:
                actions = _prepare_safe_action(actions, robot_state.get("action"))
                if not actions:
                    logger.warning("SAFE MODE filtered out all actions")
                    time.sleep(0.1)
                    continue

            # [执行侧] 即将发送给机械臂的 gripper 日志：排查夹爪不张开
            _grippers = [a[6] if len(a) > 6 else "N/A" for a in actions]
            logger.info("[执行侧] 发送给机械臂 gripper: %s (共 %d 步)", _grippers, len(actions))

            if record_path:
                try:
                    _save_inference_io(record_path, round_counter, robot_state, actions)
                    round_counter += 1
                except OSError as e:
                    logger.warning("Failed to save I/O: %s", e)
                if safe_mode:
                    logger.info("RECORD MODE: press [I] to request next inference.")
            else:
                # Execute
                interface.execute_actions(actions, duration, action_type)
                logger.info("Executed %d actions", len(actions))
                if safe_mode:
                    logger.info("SAFE MODE: press [I] to request the next action chunk.")
    finally:
        if kb:
            logger.info("Shutting down — protect mode ...")
            try:
                interface.protect_mode()
            except Exception:
                pass
