"""
Dexbotic local ARX5 bridge: send VLA inference results to local ARX5 arm.

Uses Dexbotic InferenceRunner and policies for inference.
Provides ARX5ArmClient, LocalRobotInterface, camera managers (no external spirit dependency).

Run with: python -m local_arx5.run_local_arx5 --task_name open_the_drawer --checkpoint ./checkpoints/...
"""

from .arm_client import ARX5ArmClient
from .local_loop_adapter import local_control_loop_dexbotic
from .local_robot_interface import (
    DEFAULT_IMAGE_TYPE_TO_CAMERA,
    LocalRobotInterface,
    RealSenseCameraManager,
    USBCameraManager,
    list_realsense_devices,
)

__all__ = [
    "ARX5ArmClient",
    "DEFAULT_IMAGE_TYPE_TO_CAMERA",
    "LocalRobotInterface",
    "RealSenseCameraManager",
    "USBCameraManager",
    "list_realsense_devices",
    "local_control_loop_dexbotic",
]
