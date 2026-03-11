#!/usr/bin/env python3
"""
ARX5 客户端：通过 WebSocket 连接 PI05 推理服务，执行 VLA 控制循环。

不依赖评测框架（TASK_METADATA / InferenceRunner 等），prompt 从命令行传入。

Server: run_pi05_ws_server.py (GPU 端)
Client: 本文件 (机械臂端，arm + cameras)

Usage
-----
  python arx5_client/run_pi05_client.py \\
      --prompt "pick up the red cup" \\
      --server_url ws://192.168.1.100:8765 \\
      --cameras side:352122274400 wrist:254522071216 front:409122272986
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
    return os.environ.get("SAFE_MODE", "1") != "0"


def _parse_cameras(specs: list[str]) -> dict:
    out = {}
    for s in specs:
        if ":" not in s:
            raise ValueError(f"Invalid camera spec '{s}', expected name:value")
        k, v = s.split(":", 1)
        out[k.strip()] = v.strip()
    return out


IMAGE_TYPE_TO_CAMERA = {
    "high": "side",
    "left_hand": "wrist",
    "right_hand": "front",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PI05 remote inference client for local ARX5 execution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="VLA task instruction (e.g. 'pick up the red cup')",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        required=True,
        help="PI05 WebSocket server URL (e.g. ws://192.168.1.100:8765)",
    )

    # Arm
    parser.add_argument("--can_port", type=str, default="can0")
    parser.add_argument("--arm_type", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--use_stub", action="store_true", help="Use StubArm (no real hardware)")

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
    parser.add_argument("--list_cameras", action="store_true", help="List RealSense devices and exit")

    parser.add_argument(
        "--flip_cameras",
        nargs="*",
        default=[],
        help="Camera names to rotate 180 degrees (e.g. side wrist front)",
    )
    parser.add_argument(
        "--image_type_to_camera",
        nargs="*",
        default=None,
        help="Override image_type->camera mapping as key:value pairs (e.g. high:side left_hand:wrist right_hand:front)",
    )
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], metavar=("W", "H"))
    parser.add_argument("--duration", type=float, default=0.1, help="Seconds per action step")
    parser.add_argument("--no_keyboard", action="store_true")
    parser.add_argument(
        "--record_dir",
        type=str,
        default=None,
        help="Save inference I/O to dir (no arm execution)",
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

    # image_type -> camera mapping
    it2c = dict(IMAGE_TYPE_TO_CAMERA)
    if args.image_type_to_camera:
        it2c = dict(_parse_cameras(args.image_type_to_camera))

    interface = LocalRobotInterface(
        arm_client=arm_client,
        camera_manager=camera_mgr,
        image_type_to_camera=it2c,
        flip_cameras=args.flip_cameras,
    )

    image_size = list(args.image_size)
    image_type = list(it2c.keys())
    action_type = "leftpos"

    from ws_client_pi05 import PI05RemoteInferenceClient
    from local_loop_adapter import local_control_loop_dexbotic

    inference_client = PI05RemoteInferenceClient(server_url=args.server_url, prompt=args.prompt)

    logger.info(
        "Starting PI05 client (prompt='%s', server=%s, image_size=%s, safe_mode=%s)",
        args.prompt,
        args.server_url,
        image_size,
        safe_mode,
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
