"""
Local robot interface: get_state from local ARX5 + cameras, execute_actions locally.

Replaces robochallenge InterfaceClient for local execution. Returns state dict
in the same format expected by RoboChallengeExecutor.infer().
"""

import logging
import time
from typing import Any, Optional

import cv2
import numpy as np

from .arm_client import ARX5ArmClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RealSense camera support
# ---------------------------------------------------------------------------

try:
    import pyrealsense2 as rs

    HAS_REALSENSE = True
except ImportError:
    rs = None
    HAS_REALSENSE = False
    logger.debug("pyrealsense2 not found; use use_usb_cams for USB cameras")


def list_realsense_devices() -> list:
    """Discover connected RealSense devices."""
    if not HAS_REALSENSE:
        return []
    ctx = rs.context()
    return [
        {
            "name": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number),
        }
        for dev in ctx.query_devices()
    ]


class RealSenseCameraManager:
    """Manage RealSense cameras by serial number."""

    def __init__(self, camera_map: dict, width: int = 640, height: int = 480, fps: int = 30):
        if not HAS_REALSENSE:
            raise RuntimeError("pyrealsense2 required. Install with: pip install pyrealsense2")
        self.pipelines = {}
        self.camera_order = list(camera_map.keys())
        for name, serial in camera_map.items():
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(str(serial))
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            pipeline.start(config)
            self.pipelines[name] = pipeline
            logger.info("RealSense '%s' started (serial=%s)", name, serial)

    def read(self) -> dict[str, np.ndarray]:
        """Read BGR frames from each camera."""
        frames = {}
        for name in self.camera_order:
            try:
                frameset = self.pipelines[name].wait_for_frames(timeout_ms=1000)
                cf = frameset.get_color_frame()
                frames[name] = np.asanyarray(cf.get_data()) if cf else np.zeros((480, 640, 3), dtype=np.uint8)
            except RuntimeError:
                frames[name] = np.zeros((480, 640, 3), dtype=np.uint8)
        return frames

    def release(self) -> None:
        for name, p in self.pipelines.items():
            try:
                p.stop()
            except Exception:
                pass


class USBCameraManager:
    """Manage generic USB cameras via OpenCV."""

    def __init__(self, cam_ids: dict, width: int = 640, height: int = 480, fps: int = 30):
        self.captures = {}
        self.camera_order = list(cam_ids.keys())
        for name, dev_id in cam_ids.items():
            cap = cv2.VideoCapture(int(dev_id))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera '{name}' (device {dev_id})")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            self.captures[name] = cap

    def read(self) -> dict[str, np.ndarray]:
        frames = {}
        for name in self.camera_order:
            ret, frame = self.captures[name].read()
            frames[name] = frame if ret else np.zeros((480, 640, 3), dtype=np.uint8)
        return frames

    def release(self) -> None:
        for cap in self.captures.values():
            cap.release()


# ---------------------------------------------------------------------------
# Local Robot Interface
# ---------------------------------------------------------------------------

# Default mapping: robochallenge image_type -> local camera name
# ARX5 TASK_INFO: cam_high->right_hand, cam_left_wrist->left_hand, cam_right_wrist->high
DEFAULT_IMAGE_TYPE_TO_CAMERA = {
    "high": "side",
    "left_hand": "wrist",
    "right_hand": "front",
}


def _encode_image_bgr_to_jpeg(bgr: np.ndarray, width: int, height: int) -> bytes:
    """Resize BGR array and encode as JPEG bytes."""
    if width > 0 and height > 0 and (bgr.shape[1], bgr.shape[0]) != (width, height):
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    _, buf = cv2.imencode(".jpg", bgr)
    return buf.tobytes()


class LocalRobotInterface:
    """
    Local interface compatible with robochallenge state/action flow.

    get_state() returns dict in format expected by RoboChallengeExecutor.infer().
    execute_actions() sends actions to local ARX5 arm.
    """

    def __init__(
        self,
        arm_client: ARX5ArmClient,
        camera_manager: Any,
        image_type_to_camera: Optional[dict[str, str]] = None,
    ):
        """
        Args:
            arm_client: ARX5ArmClient instance
            camera_manager: RealSenseCameraManager or USBCameraManager
            image_type_to_camera: Mapping from robochallenge image_type (high, left_hand, right_hand)
                                 to local camera name. Default: high->side, left_hand->wrist, right_hand->front
        """
        self.arm_client = arm_client
        self.camera_manager = camera_manager
        self.image_type_to_camera = image_type_to_camera or DEFAULT_IMAGE_TYPE_TO_CAMERA.copy()

    def get_state(
        self,
        image_size: list[int],
        image_type: list[str],
        action_type: str,
    ) -> dict:
        """
        Get robot state from local arm and cameras.

        Returns dict compatible with RoboChallengeExecutor.infer():
          - images: {image_type: bytes} (JPEG-encoded, resized to image_size)
          - action: [x, y, z, euler_x, euler_y, euler_z, gripper]
          - state: "normal"
          - pending_actions: 0
          - timestamp: float
        """
        width, height = image_size[0], image_size[1]
        state_list = self.arm_client.get_state()
        frames = self.camera_manager.read()

        images = {}
        for itype in image_type:
            cam_name = self.image_type_to_camera.get(itype, itype)
            if cam_name in frames:
                bgr = frames[cam_name]
                img_bytes = _encode_image_bgr_to_jpeg(bgr, width, height)
                images[itype] = img_bytes
            else:
                # Fallback: black image
                blank = np.zeros((height or 240, width or 320, 3), dtype=np.uint8)
                images[itype] = _encode_image_bgr_to_jpeg(blank, width, height)
                logger.warning("Camera '%s' not found for image_type '%s', using blank", cam_name, itype)

        return {
            "images": images,
            "action": state_list,
            "state": "normal",
            "pending_actions": 0,
            "timestamp": time.time(),
        }

    def execute_actions(self, actions: list, duration: float, action_type: str) -> None:
        """Send actions to local ARX5 arm."""
        self.arm_client.execute_actions(actions, duration)

    # ---- Arm mode control (delegate to arm_client) ----

    def hold_position(self) -> None:
        """Lock arm at current position."""
        self.arm_client.hold_position()

    def go_home(self) -> None:
        """Move arm to home position."""
        self.arm_client.go_home()

    def protect_mode(self) -> None:
        """Enter protect mode (e.g. shutdown)."""
        self.arm_client.protect_mode()

    def enter_teach_mode(self) -> None:
        """Enter gravity compensation (drag arm freely)."""
        self.arm_client.enter_teach_mode()

    def exit_teach_and_record(self) -> None:
        """Record current position and exit teach mode."""
        self.arm_client.exit_teach_and_record()

    def has_recorded_position(self) -> bool:
        """True if a position was recorded."""
        return self.arm_client.has_recorded_position()

    def move_to_recorded(self) -> None:
        """Move to the last recorded position."""
        self.arm_client.move_to_recorded()
