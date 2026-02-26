#!/usr/bin/env python3
"""
Spirit VLA local ARX5 execution.

Runs Spirit VLA policy locally, gets observations from local ARX5 arm + cameras,
and sends actions to the local arm instead of remote RoboChallenge API.

Usage
-----
  # CAN setup (run once after each USB plug-in):
  #   sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up

  # List RealSense cameras:
  python -m local_arx5.run_local_arx5 --list_cameras

  # Run with RealSense (ARX5 tasks only):
  python -m local_arx5.run_local_arx5 \\
      --single_task open_the_drawer \\
      --ckpt_path /path/to/checkpoint \\
      --cameras side:352122274400 wrist:254522071216 front:409122272986

  # Run with USB cameras:
  python -m local_arx5.run_local_arx5 \\
      --single_task open_the_drawer \\
      --ckpt_path /path/to/checkpoint \\
      --use_usb_cams --cameras side:0 wrist:2 front:4

  # Dry-run (StubArm, no real hardware):
  python -m local_arx5.run_local_arx5 \\
      --single_task open_the_drawer \\
      --ckpt_path /path/to/checkpoint \\
      --use_stub

  # Remote inference (server runs VLA, client runs ARX5):
  python -m local_arx5.run_local_arx5 \\
      --single_task open_the_drawer \\
      --server_url ws://192.168.1.100:8765 \\
      --cameras side:SN1 wrist:SN2 front:SN3
"""

import argparse
import logging
import sys
from argparse import Namespace
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robochallenge.runner.task_info import TASK_INFO

from .arm_client import ARX5ArmClient
from .ws_client import RemoteInferenceClient
from .local_loop import local_control_loop
from .local_robot_interface import (
    DEFAULT_IMAGE_TYPE_TO_CAMERA,
    LocalRobotInterface,
    RealSenseCameraManager,
    USBCameraManager,
    list_realsense_devices,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_cameras(specs: list[str]) -> dict[str, str]:
    """Parse 'name:value' pairs into ordered dict."""
    out = {}
    for s in specs:
        if ":" not in s:
            raise ValueError(f"Invalid camera spec '{s}', expected name:value")
        k, v = s.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spirit VLA local ARX5 execution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Task & model
    parser.add_argument("--single_task", type=str, required=True, help="Task name from TASK_INFO (ARX5 only)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Spirit VLA checkpoint path (required when not using --server_url)")
    parser.add_argument("--server_url", type=str, default=None, help="WebSocket inference server URL (e.g. ws://host:8765). If set, no local checkpoint.")
    parser.add_argument("--used_chunk_size", type=int, default=60, help="Action chunk size (used only for local inference)")
    parser.add_argument(
        "--raw_embodiment_stats_json_path",
        type=str,
        default=None,
        help="Path to raw_embodiment_stats JSON (optional)",
    )
    # Arm
    parser.add_argument("--can_port", type=str, default="can0", help="CAN interface")
    parser.add_argument("--arm_type", type=int, default=0, choices=[0, 1, 2], help="ARX5 URDF type")
    parser.add_argument("--use_stub", action="store_true", help="Use StubArm (no real hardware)")
    # Cameras: map camera name -> serial or device id
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["side:352122274400", "wrist:254522071216", "front:409122272986"],
        help="Camera specs as name:serial_or_id (e.g. side:SN wrist:SN front:SN). "
        "Names must match DEFAULT_IMAGE_TYPE_TO_CAMERA (side, wrist, front).",
    )
    parser.add_argument("--use_usb_cams", action="store_true", help="Use USB cameras (values = device indices)")
    parser.add_argument("--cam_width", type=int, default=640)
    parser.add_argument("--cam_height", type=int, default=480)
    # Image
    parser.add_argument("--image_size", type=int, nargs=2, default=[320, 240], metavar=("W", "H"))
    # Control
    parser.add_argument("--duration", type=float, default=1 / 15, help="Seconds per action step")
    parser.add_argument("--no_keyboard", action="store_true", help="Disable keyboard control (start immediately)")
    # Debug
    parser.add_argument("--list_cameras", action="store_true", help="List RealSense devices and exit")

    args = parser.parse_args()

    if args.list_cameras:
        devices = list_realsense_devices()
        if not devices:
            print("No RealSense devices found.")
        else:
            print(f"Found {len(devices)} RealSense device(s):")
            for d in devices:
                print(f"  {d['name']}  serial={d['serial']}")
        sys.exit(0)

    task_name = args.single_task
    if task_name not in TASK_INFO:
        logger.error("Unknown task: %s", task_name)
        sys.exit(1)
    if TASK_INFO[task_name]["robot_type"] != "ARX5":
        logger.error("Only ARX5 tasks supported. Task %s has robot_type=%s", task_name, TASK_INFO[task_name]["robot_type"])
        sys.exit(1)

    use_remote = bool(args.server_url)
    if use_remote:
        if not args.server_url.startswith(("ws://", "wss://")):
            logger.error("--server_url must be ws:// or wss://")
            sys.exit(1)
        inference_client = RemoteInferenceClient(server_url=args.server_url, task_name=task_name)
        executor = None
        logger.info("Using remote inference: %s", args.server_url)
    else:
        if not args.ckpt_path:
            logger.error("--ckpt_path is required when not using --server_url")
            sys.exit(1)
        from robochallenge.runner.executor import RoboChallengeExecutor
        cfg = Namespace(
            single_task=task_name,
            robochallenge_job_id="local",
            ckpt_path=args.ckpt_path,
            used_chunk_size=args.used_chunk_size,
            raw_embodiment_stats_json_path=args.raw_embodiment_stats_json_path,
            use_embodiment_specific_norm=False,
            embodiment_stats_path=None,
        )
        logger.info("Loading checkpoint: %s", args.ckpt_path)
        executor = RoboChallengeExecutor(cfg)
        inference_client = None
        logger.info("Task=%s checkpoint loaded.", task_name)

    # Arm
    arm_client = ARX5ArmClient(
        can_port=args.can_port,
        arm_type=args.arm_type,
        use_stub=args.use_stub,
    )

    # Cameras
    camera_map = _parse_cameras(args.cameras)
    if args.use_usb_cams:
        camera_map = {k: int(v) for k, v in camera_map.items()}
        camera_mgr = USBCameraManager(
            camera_map,
            width=args.cam_width,
            height=args.cam_height,
        )
    else:
        camera_mgr = RealSenseCameraManager(
            camera_map,
            width=args.cam_width,
            height=args.cam_height,
        )

    # Ensure camera names match DEFAULT_IMAGE_TYPE_TO_CAMERA
    for cam_name in DEFAULT_IMAGE_TYPE_TO_CAMERA.values():
        if cam_name not in camera_map:
            logger.warning(
                "Camera '%s' not in --cameras. Default mapping: %s. Your cameras: %s",
                cam_name,
                DEFAULT_IMAGE_TYPE_TO_CAMERA,
                list(camera_map.keys()),
            )

    interface = LocalRobotInterface(
        arm_client=arm_client,
        camera_manager=camera_mgr,
        image_type_to_camera=DEFAULT_IMAGE_TYPE_TO_CAMERA,
    )

    image_type = ["high", "left_hand", "right_hand"]
    action_type = TASK_INFO[task_name]["action_type"]
    image_size = list(args.image_size)

    logger.info("Starting local control loop (image_size=%s, duration=%s)", image_size, args.duration)

    local_control_loop(
        interface=interface,
        executor=executor,
        task_name=task_name,
        image_size=image_size,
        image_type=image_type,
        action_type=action_type,
        duration=args.duration,
        use_keyboard=not args.no_keyboard,
        inference_client=inference_client,
    )


if __name__ == "__main__":
    main()
