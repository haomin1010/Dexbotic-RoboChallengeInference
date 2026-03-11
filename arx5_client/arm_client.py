"""
ARX5 arm connection layer for local execution.

Provides get_state() and execute_actions() compatible with robochallenge
ARX5 action format [x, y, z, euler_x, euler_y, euler_z, gripper].
Borrows SDK setup and SingleArm usage from dexbotic/hardware/arx_x5.
"""

import ctypes
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 教示位置保存文件（[N] 保存，[M] 读取；重启后仍有效）
_RECORDED_POSITION_FILE = Path(__file__).resolve().parent.parent / "recorded_position.json"

# ---------------------------------------------------------------------------
# ARX X5 SDK — auto-discover path, pre-load native libs
# ---------------------------------------------------------------------------

_ARX_SDK_ROOT = os.path.expanduser(
    os.environ.get("ARX_SDK_ROOT", "~/workspace/ARX_X5/py/arx_x5_python")
)


def _setup_arx_sdk(sdk_root: str) -> None:
    """Add SDK to sys.path and pre-load native .so files."""
    if not os.path.isdir(sdk_root):
        return

    if sdk_root not in sys.path:
        sys.path.insert(0, sdk_root)

    _so_search_dirs = [
        os.path.join(sdk_root, "bimanual", "api", "arx_x5_src"),
        os.path.join(sdk_root, "bimanual", "api"),
    ]
    for d in _so_search_dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".so") and not fn.endswith("-arm64.so"):
                so_path = os.path.join(d, fn)
                try:
                    ctypes.cdll.LoadLibrary(so_path)
                except OSError as e:
                    logger.debug("Pre-load skipped %s: %s", so_path, e)


_setup_arx_sdk(_ARX_SDK_ROOT)

try:
    from bimanual import SingleArm

    HAS_ARX_SDK = True
except ImportError:
    SingleArm = None
    HAS_ARX_SDK = False
    logger.warning(
        "ARX X5 SDK (bimanual) not found — running in dry-run mode. "
        "Searched: %s",
        _ARX_SDK_ROOT,
    )


# ---------------------------------------------------------------------------
# Stub arm (dry-run without real hardware)
# ---------------------------------------------------------------------------


class _StubArm:
    """Fake arm for testing without hardware."""

    def __init__(self, config):
        self._joints = np.zeros(7)
        self._ee_pose = np.zeros(7)  # [x, y, z, roll, pitch, yaw, gripper]
        logger.info("[StubArm] config=%s", config)

    def go_home(self):
        logger.info("[StubArm] go_home")

    def protect_mode(self):
        logger.info("[StubArm] protect_mode")

    def gravity_compensation(self):
        logger.info("[StubArm] gravity_compensation")

    def set_joint_positions(self, positions):
        self._joints[: len(positions)] = positions

    def set_ee_pose_xyzrpy(self, xyzrpy):
        self._ee_pose[: len(xyzrpy)] = xyzrpy

    def set_catch_pos(self, pos):
        self._ee_pose[6] = pos
        self._joints[6] = pos

    def get_joint_positions(self):
        return self._joints.tolist()

    def get_ee_pose_xyzrpy(self):
        return self._ee_pose[:6].tolist()


# ---------------------------------------------------------------------------
# ARX5 Arm Client
# ---------------------------------------------------------------------------


class ARX5ArmClient:
    """
    ARX5 arm client for local execution.

    State format: [x, y, z, euler_x, euler_y, euler_z, gripper]
    (matches robochallenge ARX5 / leftpos)
    """

    now_state = []

    def __init__(self, can_port: str = "can0", arm_type: int = 0, use_stub: bool = False):
        """
        Args:
            can_port: CAN interface (e.g. "can0")
            arm_type: URDF type (0=x5, 1=x5_master, 2=x5_2025)
            use_stub: If True, use StubArm instead of real hardware
        """
        if use_stub or not HAS_ARX_SDK:
            logger.info("Using StubArm (dry-run)")
            self.arm = _StubArm({"can_port": can_port, "type": arm_type})
        else:
            logger.info("Initialising ARX5 arm (can_port=%s, arm_type=%s)", can_port, arm_type)
            self.arm = SingleArm({"can_port": can_port, "type": arm_type})
        self._last_executed = self.get_state()
        self._recorded_position: list | None = None

    def hold_position(self) -> None:
        """
        Lock the arm at its current position (EE-space position control).
        Use after e-stop or to exit gravity compensation safely.
        """
        state = self.get_state()
        self.arm.set_ee_pose_xyzrpy(list(state[:6]))
        state = self.get_state()
        self.arm.set_catch_pos(float(state[6]))
        self._last_executed = state
        time.sleep(0.5)

    def go_home(self) -> None:
        """Move arm to home position (blocking)."""
        self.arm.go_home()

    def protect_mode(self) -> None:
        """Enter protect mode (e.g. on shutdown)."""
        self.arm.protect_mode()

    def enter_teach_mode(self) -> None:
        """Enter gravity compensation — user can drag the arm freely."""
        self.arm.gravity_compensation()

    def exit_teach_and_record(self) -> None:
        """Record current position, save to file, and exit teach mode (hold position)."""
        self._recorded_position = self.get_state()
        try:
            _RECORDED_POSITION_FILE.write_text(
                json.dumps({"recorded_position": self._recorded_position}, indent=2),
                encoding="utf-8",
            )
            logger.info("Recorded position saved to %s", _RECORDED_POSITION_FILE)
        except OSError as e:
            logger.warning("Failed to save recorded position: %s", e)
        self.hold_position()

    def has_recorded_position(self) -> bool:
        """True if a position was recorded (file exists or in-memory)."""
        return (
            self._recorded_position is not None
            or _RECORDED_POSITION_FILE.exists()
        )

    def _load_recorded_position(self) -> list:
        """Load recorded position from file or memory. Raises if none."""
        if self._recorded_position is not None:
            return self._recorded_position
        if _RECORDED_POSITION_FILE.exists():
            try:
                data = json.loads(_RECORDED_POSITION_FILE.read_text(encoding="utf-8"))
                return data["recorded_position"]
            except (json.JSONDecodeError, KeyError, OSError) as e:
                raise ValueError(f"Cannot load recorded position from {_RECORDED_POSITION_FILE}: {e}") from e
        raise ValueError("No recorded position — press [B] to teach, then [N] to record first")

    def move_to_recorded(
        self, num_steps: int = 20, step_interval: float = 0.1
    ) -> None:
        """Move to the recorded position (from file or memory).
        Uses linear interpolation over num_steps with step_interval seconds between steps.
        """
        target_list = self._load_recorded_position()
        self.hold_position()
        time.sleep(0.1)

        start = np.array(self.get_state(), dtype=np.float64)
        target = np.array(target_list, dtype=np.float64)

        for i in range(1, num_steps + 1):
            t = i / num_steps
            interp = start + (target - start) * t
            self.arm.set_ee_pose_xyzrpy(interp[:6].tolist())
            self.arm.set_catch_pos(float(interp[6]))
            if i < num_steps:
                time.sleep(step_interval)

        self._last_executed = self.get_state()

    def get_state(self) -> list:
        """
        Return 7-dim state [x, y, z, euler_x, euler_y, euler_z, gripper].

        Matches robochallenge ARX5 format for _prepare_batch / _post_process_action.
        """
        ee_pose = np.array(self.arm.get_ee_pose_xyzrpy(), dtype=np.float64)
        joints = np.array(self.arm.get_joint_positions(), dtype=np.float64)
        gripper = float(joints[6]) if len(joints) > 6 else 0.0

        state = np.zeros(7, dtype=np.float64)
        state[:6] = ee_pose[:6]
        state[6] = gripper
        return state.tolist()

    def execute_actions(self, actions: list, duration: float) -> None:
        """
        Execute a list of actions on the arm.

        Each action: [x, y, z, euler_x, euler_y, euler_z, gripper]
        Sleeps duration seconds between each action step.

        Args:
            actions: List of 7-element lists
            duration: Time per action step in seconds (e.g. 1/15 for 15 Hz)
        """
        for i, action in enumerate(actions):
            if len(action) < 7:
                action = list(action) + [0.0] * (7 - len(action))
            xyzrpy = action[:6]
            gripper = 50 * float(action[6])

            self.arm.set_ee_pose_xyzrpy(xyzrpy)
            self.arm.set_catch_pos(gripper)
            self._last_executed = action
            if i < len(actions) - 1 and duration > 0:
                time.sleep(duration)

        self.now_state = self.get_state()
