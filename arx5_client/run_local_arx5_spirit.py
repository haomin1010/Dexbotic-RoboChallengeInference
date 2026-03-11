#!/usr/bin/env python3
"""
ARX5 本地执行（Spirit v1.5 远程推理）。

与 DM0 的使用方式尽量一致：
- 服务端：spirit-v1.5/server/run_ws_inference_server.py（GPU 服务器）
- 客户端：本文件（机械臂+相机的本地机器）
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent
for p in (REPO_ROOT, _THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_mode_enabled() -> bool:
    """SAFE_MODE defaults to on; only SAFE_MODE=0 disables it."""
    return os.environ.get("SAFE_MODE", "1") != "0"


def _parse_cameras(specs: list[str]) -> dict:
    out = {}
    for s in specs:
        if ":" not in s:
            raise ValueError(f"Invalid camera spec '{s}', expected name:value")
        k, v = s.split(":", 1)
        out[k.strip()] = v.strip()
    return out


# Spirit 的 ARX5 TASK_INFO 中对三路图像的对应关系与 DM0 不同：
# cam_high <- right_hand, cam_left_wrist <- left_hand, cam_right_wrist <- high
# 为了让模型看到“语义上正确”的三路视角，这里把 image_type 重新映射到本地相机名。
SPIRIT_IMAGE_TYPE_TO_CAMERA = {
    "high": "front",
    "left_hand": "wrist",
    "right_hand": "side",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spirit v1.5 remote inference client for local ARX5 execution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument(
        "--server_url",
        type=str,
        required=True,
        help="WebSocket inference server URL (e.g. ws://192.168.1.100:8765)",
    )

    # Arm
    parser.add_argument("--can_port", type=str, default="can0")
    parser.add_argument("--arm_type", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--use_stub", action="store_true")

    # Cameras
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["side:352122274400", "wrist:254522071216", "front:409122272986"],
        help="Camera specs as name:serial_or_id (e.g. side:SN wrist:SN front:SN)",
    )
    parser.add_argument("--use_usb_cams", action="store_true", help="Use USB cameras (values = device indices)")
    parser.add_argument("--cam_width", type=int, default=640)
    parser.add_argument("--cam_height", type=int, default=480)
    parser.add_argument("--list_cameras", action="store_true")

    parser.add_argument(
        "--flip_cameras",
        nargs="*",
        default=[],
        help="Camera names to rotate 180° (镜头安反时使用，e.g. side wrist front)",
    )
    # Spirit 默认使用 320x240
    parser.add_argument("--image_size", type=int, nargs=2, default=[320, 240], metavar=("W", "H"))
    parser.add_argument("--duration", type=float, default=1 / 15)
    parser.add_argument("--no_keyboard", action="store_true")
    parser.add_argument(
        "--record_dir",
        type=str,
        default=None,
        help="Save inference input/output to dir (不执行机械臂)",
    )

    args = parser.parse_args()
    safe_mode = _safe_mode_enabled()

    try:
        from .arm_client import ARX5ArmClient
        from .local_robot_interface import (
            LocalRobotInterface,
            RealSenseCameraManager,
            USBCameraManager,
            list_realsense_devices,
        )
    except ImportError:
        from arm_client import ARX5ArmClient
        from local_robot_interface import (
            LocalRobotInterface,
            RealSenseCameraManager,
            USBCameraManager,
            list_realsense_devices,
        )

    if args.list_cameras:
        devices = list_realsense_devices()
        if not devices:
            print("No RealSense devices found.")
        else:
            print(f"Found {len(devices)} RealSense device(s):")
            for d in devices:
                print(f"  {d['name']}  serial={d['serial']}")
        sys.exit(0)

    if not args.server_url.startswith(("ws://", "wss://")):
        parser.error("--server_url must be ws:// or wss://")
    if safe_mode and args.no_keyboard:
        parser.error("SAFE_MODE requires keyboard control; remove --no_keyboard or run with SAFE_MODE=0")

    # Arm
    arm_client = ARX5ArmClient(can_port=args.can_port, arm_type=args.arm_type, use_stub=args.use_stub)

    # Cameras
    camera_map = _parse_cameras(args.cameras)
    if args.use_usb_cams:
        camera_map = {k: int(v) for k, v in camera_map.items()}
        camera_mgr = USBCameraManager(camera_map, width=args.cam_width, height=args.cam_height)
    else:
        camera_mgr = RealSenseCameraManager(camera_map, width=args.cam_width, height=args.cam_height)

    interface = LocalRobotInterface(
        arm_client=arm_client,
        camera_manager=camera_mgr,
        image_type_to_camera=SPIRIT_IMAGE_TYPE_TO_CAMERA,
        flip_cameras=args.flip_cameras,
    )

    image_size = list(args.image_size)
    image_type = ["high", "left_hand", "right_hand"]
    action_type = "leftpos"

    from ws_client import RemoteInferenceClient
    from local_loop_adapter import local_control_loop_dexbotic

    inference_client = RemoteInferenceClient(server_url=args.server_url, task_name=args.task_name)

    logger.info(
        "Starting Spirit local control loop (task=%s, image_size=%s, safe_mode=%s, flip_cameras=%s, record_dir=%s)",
        args.task_name,
        image_size,
        safe_mode,
        args.flip_cameras or "none",
        args.record_dir or "none",
    )
    local_control_loop_dexbotic(
        interface=interface,
        runner=None,
        inference_client=inference_client,
        image_size=image_size,
        image_type=image_type,
        action_type=action_type,
        duration=args.duration,
        use_keyboard=not args.no_keyboard,
        safe_mode=safe_mode,
        record_dir=args.record_dir,
    )


if __name__ == "__main__":
    main()
