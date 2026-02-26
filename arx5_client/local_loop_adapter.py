"""
Local control loop adapter: get_state -> Dexbotic InferenceRunner.infer -> execute_actions.

Supports keyboard: [Space] e-stop | [H] home | [B] teach | [N] record | [M] goto | [R] resume | [Q] quit
"""

import atexit
import logging
import queue
import sys
import termios
import threading
import time
import tty
from enum import Enum, auto

logger = logging.getLogger(__name__)


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


def local_control_loop_dexbotic(
    interface,
    runner=None,
    inference_client=None,
    image_size: list[int] | None = None,
    image_type: list[str] | None = None,
    action_type: str = "",
    duration: float = 1 / 15,
    use_keyboard: bool = True,
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
    """
    inferrer = inference_client if inference_client is not None else runner
    if inferrer is None:
        raise ValueError("Either runner or inference_client must be provided")
    state = LoopState.STOPPED if use_keyboard else LoopState.RUNNING
    running = True
    kb = KeyboardListener() if use_keyboard else None

    # Startup: go home and hold (when keyboard enabled)
    if kb:
        logger.info("Moving arm to home ...")
        interface.go_home()
        time.sleep(2.0)
        interface.hold_position()
        logger.info(
            "Keyboard: [Space] e-stop | [H] home | [B] teach | [M] goto recorded | [R] resume | [Q] quit"
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
                        logger.info("RESUMED — VLA control active. Press [Space] to stop.")
                elif state == LoopState.TEACHING:
                    if key == "n":
                        interface.exit_teach_and_record()
                        state = LoopState.STOPPED
                        logger.info("Position recorded")

            if state != LoopState.RUNNING:
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

            # [执行侧] 即将发送给机械臂的 gripper 日志：排查夹爪不张开
            _grippers = [a[6] if len(a) > 6 else "N/A" for a in actions]
            logger.info("[执行侧] 发送给机械臂 gripper: %s (共 %d 步)", _grippers, len(actions))

            # Execute
            interface.execute_actions(actions, duration, action_type)
            logger.info("Executed %d actions", len(actions))
    finally:
        if kb:
            logger.info("Shutting down — protect mode ...")
            try:
                interface.protect_mode()
            except Exception:
                pass
