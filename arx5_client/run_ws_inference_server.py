#!/usr/bin/env python3
"""
Dexbotic 推理服务入口：加载 DM0 模型，通过 WebSocket 提供动作推理。

运行在带 GPU 的机器上；机械臂端 (run_robot_client.py) 发送观测、接收动作。

Usage
-----
  python -m local_arx5.run_ws_inference_server \\
      --host 0.0.0.0 --port 8765 \\
      --task_name open_the_drawer \\
      --checkpoint ./checkpoints/DM0-table30_generalist_arx5
"""

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_arx5.ws_inference_server import run_ws_inference_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dexbotic WebSocket inference server (action generation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Task name from TASK_METADATA (ARX5 only)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Dexbotic checkpoint path",
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default="dm0",
        choices=["dm0", "dm0_prog"],
        help="Policy type",
    )
    parser.add_argument("--action_horizon", type=int, default=15)
    parser.add_argument("--image_size", type=int, nargs=2, default=[728, 728], metavar=("W", "H"))
    parser.add_argument(
        "--gripper_threshold",
        type=float,
        default=0.0,
        help="gripper 阈值，小于此值会被裁成 close。默认 0 以配合 gripper_scale",
    )
    parser.add_argument(
        "--gripper_scale",
        type=float,
        default=40.0,
        help="gripper 缩放：模型输出 ~0.03 时缩放到 1.0。0 表示不缩放",
    )

    args = parser.parse_args()

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

    from policies import get_policy
    from runner import InferenceRunner

    logger.info("Loading policy (task=%s, ckpt=%s)", args.task_name, args.checkpoint)
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
            "gripper_threshold": args.gripper_threshold,
            "gripper_scale": args.gripper_scale if args.gripper_scale > 0 else None,
        },
    )

    logger.info("Inference server ready.")
    run_ws_inference_server(runner, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
