#!/usr/bin/env python3
"""
Dexbotic local ARX5 execution.

Sends VLA inference results from Dexbotic to local ARX5 arm.
Uses built-in hardware interface (arm + cameras)

Usage
-----
  # CAN setup (run once after each USB plug-in):
  #   sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up

  # List RealSense cameras:
  python -m local_arx5.run_local_arx5 --list_cameras

  # Run with RealSense (ARX5 tasks only):
  python -m local_arx5.run_local_arx5 \\
      --task_name open_the_drawer \\
      --checkpoint ./checkpoints/DM0-table30_generalist_arx5 \\
      --cameras side:352122274400 wrist:254522071216 front:409122272986

  # Run with USB cameras:
  python -m local_arx5.run_local_arx5 \\
      --task_name put_cup_on_coaster \\
      --checkpoint ./checkpoints/DM0-table30_generalist_arx5 \\
      --use_usb_cams --cameras side:0 wrist:2 front:4

  # Dry-run (StubArm, no real hardware):
  python -m local_arx5.run_local_arx5 \\
      --task_name open_the_drawer \\
      --checkpoint ./checkpoints/DM0-table30_generalist_arx5 \\
      --use_stub

  # 远程推理（策略与机械臂分离，WebSocket 通信）：
  # 1) 在 GPU 机器启动推理服务：python -m local_arx5.run_ws_inference_server ...
  # 2) 在机械臂机器启动客户端：
  python -m local_arx5.run_local_arx5 \\
      --task_name open_the_drawer \\
      --server_url ws://192.168.1.100:8765 \\
      --cameras side:SN1 wrist:SN2 front:SN3
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure Dexbotic repo root and local_arx5 dir on sys.path (support both -m and direct run)
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
        description="Dexbotic VLA local ARX5 execution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Task & model
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Task name from TASK_METADATA (ARX5 only)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Dexbotic checkpoint path (ignored when --server_url is set)",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default=None,
        help="WebSocket inference server URL (e.g. ws://192.168.1.100:8765). "
        "If set, inference runs remotely; checkpoint not needed.",
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default="dm0",
        choices=["dm0", "dm0_prog"],
        help="Policy type",
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=15,
        help="Action horizon for inference",
    )
    # Arm
    parser.add_argument("--can_port", type=str, default="can0", help="CAN interface")
    parser.add_argument(
        "--arm_type",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="ARX5 URDF type",
    )
    parser.add_argument("--use_stub", action="store_true", help="Use StubArm (no real hardware)")
    # Cameras
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["side:352122274400", "wrist:254522071216", "front:409122272986"],
        help="Camera specs as name:serial_or_id (e.g. side:SN wrist:SN front:SN)",
    )
    parser.add_argument(
        "--use_usb_cams",
        action="store_true",
        help="Use USB cameras (values = device indices)",
    )
    parser.add_argument("--cam_width", type=int, default=640)
    parser.add_argument("--cam_height", type=int, default=480)
    parser.add_argument(
        "--flip_cameras",
        nargs="*",
        default=[],
        help="Camera names to rotate 180° (镜头安反时使用，e.g. side wrist front)",
    )
    # Image & control
    parser.add_argument("--image_size", type=int, nargs=2, default=[728, 728], metavar=("W", "H"))
    parser.add_argument("--duration", type=float, default=0.1, help="Seconds per action step")
    parser.add_argument("--no_keyboard", action="store_true", help="Disable keyboard control")
    parser.add_argument(
        "--record_dir",
        type=str,
        default=None,
        help="Save inference input/output to dir (不执行机械臂)",
    )
    parser.add_argument(
        "--gripper_scale",
        type=float,
        default=50,
        help="gripper 缩放：norm_stats 模型 max≈0.087 → ARX5 0-5。0 表示不缩放（仅本地推理）",
    )
    parser.add_argument(
        "--gripper_max",
        type=float,
        default=5.0,
        help="ARX5 夹爪最大值（0 闭合，5 张开）",
    )
    # Debug
    parser.add_argument("--list_cameras", action="store_true", help="List RealSense devices and exit")

    args = parser.parse_args()
    safe_mode = _safe_mode_enabled()

    # Import from built-in package
    try:
        from .arm_client import ARX5ArmClient
        from .local_robot_interface import (
            DEFAULT_IMAGE_TYPE_TO_CAMERA,
            LocalRobotInterface,
            RealSenseCameraManager,
            USBCameraManager,
            list_realsense_devices,
        )
    except ImportError:
        from arm_client import ARX5ArmClient
        from local_robot_interface import (
            DEFAULT_IMAGE_TYPE_TO_CAMERA,
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

    use_remote = bool(args.server_url)
    if not args.task_name:
        parser.error("--task_name is required (unless --list_cameras)")
    if not use_remote and not args.checkpoint:
        parser.error("--checkpoint is required when not using --server_url")
    if safe_mode and args.no_keyboard:
        parser.error("SAFE_MODE requires keyboard control; remove --no_keyboard or run with SAFE_MODE=0")

    # Validate task (ARX5 only)
    from utils.constants import TASK_METADATA, IMAGE_TYPE_MAP

    if args.task_name not in TASK_METADATA:
        logger.error("Unknown task: %s. Available: %s", args.task_name, list(TASK_METADATA.keys()))
        sys.exit(1)

    metadata = TASK_METADATA[args.task_name]
    robot_type = metadata["robot_type"]
    if robot_type != "arx5":
        logger.error("Only ARX5 tasks supported. Task %s has robot_type=%s", args.task_name, robot_type)
        sys.exit(1)

    prompt = metadata["prompt"]
    image_type = IMAGE_TYPE_MAP[robot_type]
    action_type = "leftpos"

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
        flip_cameras=args.flip_cameras,
    )

    image_size = list(args.image_size)

    if use_remote:
        # 远程推理：使用 WebSocket 客户端
        if not args.server_url.startswith(("ws://", "wss://")):
            logger.error("--server_url must be ws:// or wss://")
            sys.exit(1)
        from ws_client import RemoteInferenceClient

        inference_client = RemoteInferenceClient(server_url=args.server_url, task_name=args.task_name)
        runner = None
        logger.info("Using remote inference: %s", args.server_url)
    else:
        # 本地推理：加载 policy 和 InferenceRunner
        from policies import get_policy
        from runner import InferenceRunner

        policy = get_policy(
            ckpt_path=args.checkpoint,
            policy_type=args.policy_type,
            prompt=prompt,
            robot_type=robot_type,
            action_type=action_type,
            action_horizon=args.action_horizon,
            task_name=args.task_name,
            image_shape=tuple(args.image_size),
        )

        runner = InferenceRunner(
            policy=policy,
            robot_type=robot_type,
            action_type=action_type,
            task_name=args.task_name,
            image_type=image_type,
            action_horizon=args.action_horizon,
            postprocess_args={
                "gripper_threshold": 0,
                "gripper_scale": args.gripper_scale if args.gripper_scale > 0 else None,
                "gripper_max": args.gripper_max,
            },
        )
        inference_client = None

    logger.info(
        "Starting local control loop (task=%s, image_size=%s, safe_mode=%s, flip_cameras=%s, record_dir=%s)",
        args.task_name,
        image_size,
        safe_mode,
        args.flip_cameras or "none",
        args.record_dir or "none",
    )

    from local_loop_adapter import local_control_loop_dexbotic

    local_control_loop_dexbotic(
        interface=interface,
        runner=runner,
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
