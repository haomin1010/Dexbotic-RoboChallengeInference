#!/usr/bin/env python3
"""
PI05 + ARX5 最简对接：命令行传入 prompt，循环执行「获取 obs → PI05 推理 → 下发动作」。

支持单机模式（推理+机械臂同机）和 WebSocket 模式（GPU 机跑推理，机械臂机跑 client）。

Usage
-----
  # CAN 配置（每次插拔 USB 后执行一次）：
  #   sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up

  # 列出 RealSense 相机：
  python -m local_arx5.run_pi05_arx5 --list_cameras

  # 运行（RealSense 相机）：
  python -m local_arx5.run_pi05_arx5 \\
      --prompt "open the drawer" \\
      --policy_path lerobot/pi05_base \\
      --cameras side:352122274400 wrist:254522071216 front:409122272986

  # 运行（USB 相机）：
  python -m local_arx5.run_pi05_arx5 \\
      --prompt "pick up the cup" \\
      --policy_path lerobot/pi05_base \\
      --use_usb_cams --cameras side:0 wrist:2 front:4

  # 干跑（StubArm）：
  python -m local_arx5.run_pi05_arx5 --prompt "open the drawer" --policy_path lerobot/pi05_base --use_stub

  # 仅 2 相机（正面+左腕），第 3 路自动 mask：
  python -m local_arx5.run_pi05_arx5 --prompt "open the drawer" --server_url ws://IP:8765 --cameras front:SN1 wrist:SN2

  # ---- WebSocket 模式 ----
  # 1) GPU：python -m local_arx5.run_pi05_ws_server --policy_path lerobot/pi05_base --host 0.0.0.0 --port 8765
  # 2) 机械臂：python -m local_arx5.run_pi05_arx5 --prompt "open the drawer" --server_url ws://<GPU_IP>:8765 --cameras ...
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# 确保 arx5_client 和 Dexbotic 根目录在 path 中
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for p in (_REPO_ROOT, _THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# lerobot：单机模式需要，远程模式不需要。在加载 policy 时通过 _ensure_lerobot() 导入
def _ensure_lerobot():
    try:
        import lerobot  # noqa: F401
    except ImportError:
        _lerobot_src = Path(__file__).resolve().parents[2] / "lerobot" / "src"
        if _lerobot_src.exists():
            sys.path.insert(0, str(_lerobot_src))
        else:
            raise ImportError("lerobot not found. Install: pip install -e /path/to/lerobot") from None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 相机名 -> image_type（DEFAULT_IMAGE_TYPE_TO_CAMERA 的反向）
CAMERA_NAME_TO_IMAGE_TYPE = {
    "side": "high",
    "wrist": "left_hand",
    "front": "right_hand",
}


def _image_types_from_cameras(camera_map: dict) -> list[str]:
    """从 camera_map 的 key 推导 image_type 列表，保持用户传入顺序。"""
    return [CAMERA_NAME_TO_IMAGE_TYPE.get(name, name) for name in camera_map.keys()]


def _parse_cameras(specs: list[str]) -> dict:
    """解析 'name:value' 为有序字典"""
    out = {}
    for s in specs:
        if ":" not in s:
            raise ValueError(f"Invalid camera spec '{s}', expected name:value")
        k, v = s.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _jpeg_bytes_to_rgb_np(data: bytes) -> np.ndarray:
    """JPEG bytes -> numpy RGB (H,W,3) uint8"""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_pi05_observation(
    robot_state: dict,
    policy,
    prompt: str,
    device: torch.device,
    image_types: list[str],
) -> dict:
    """
    将 arx5 的 robot_state 转为 PI05 preprocessor 所需的 batch。

    robot_state: {images: {image_type: bytes}, action: [7D]}
    image_types: 当前使用的 image_type 列表，如 ["right_hand","left_hand"] 表示只传 2 路，
                第 3 路不加入 batch，PI05 会对其 mask。
    """
    from lerobot.configs.types import FeatureType
    from lerobot.utils.constants import OBS_STATE as OBS_STATE_KEY

    image_keys = sorted(
        k for k, v in policy.config.input_features.items()
        if v.type == FeatureType.VISUAL and "empty" not in k
    )
    obs = {}
    # 仅将已有的 image_types 映射到前 N 个 PI05 槽位，其余不加入（由 PI05 自动 mask）
    for i, arx5_key in enumerate(image_types):
        if i >= len(image_keys):
            break
        pi05_key = image_keys[i]
        if arx5_key in robot_state.get("images", {}):
            img_bytes = robot_state["images"][arx5_key]
            img_rgb = _jpeg_bytes_to_rgb_np(img_bytes)
        else:
            img_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        obs[pi05_key] = img_rgb

    # 状态：7D pad 到 max_state_dim
    state = np.array(robot_state.get("action", [0] * 7), dtype=np.float32)
    max_state_dim = getattr(policy.config, "max_state_dim", 32)
    if len(state) < max_state_dim:
        state = np.pad(state, (0, max_state_dim - len(state)), constant_values=0)
    else:
        state = state[:max_state_dim]
    obs[OBS_STATE_KEY] = state

    # 转为 tensor，与 eval_with_real_robot 一致
    policy_device = next(policy.parameters()).device
    batch = {}
    for name, val in obs.items():
        t = torch.from_numpy(val) if isinstance(val, np.ndarray) else torch.tensor(val)
        if "image" in name:
            t = t.float() / 255.0
            t = t.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        t = t.unsqueeze(0).to(policy_device)
        batch[name] = t
    batch["task"] = [prompt]
    batch["robot_type"] = ""
    return batch


def infer_pi05(policy, preprocessor, postprocessor, batch, action_steps: int) -> list:
    """PI05 推理，返回前 action_steps 步的 7D 动作列表"""
    preprocessed = preprocessor(batch)
    with torch.no_grad():
        actions = policy.predict_action_chunk(preprocessed)
    postprocessed = postprocessor(actions)
    # postprocessed: (1, chunk_size, action_dim)
    actions_np = postprocessed.squeeze(0).cpu().numpy()
    actions_np = actions_np[:action_steps]
    # 取前 7 维 [x,y,z,rx,ry,rz,g]
    action_dim = min(7, actions_np.shape[-1])
    return [a[:action_dim].tolist() for a in actions_np]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PI05 + ARX5 最简对接：prompt -> obs -> infer -> execute",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", type=str, required=True, help="任务描述文本")
    parser.add_argument(
        "--server_url",
        type=str,
        default=None,
        help="WebSocket 推理服务地址，如 ws://192.168.1.100:8765。若设置则使用远程推理，无需 --policy_path",
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        default="lerobot/pi05_base",
        help="PI05 预训练路径（单机模式时使用，远程模式可忽略）",
    )
    parser.add_argument(
        "--action_steps",
        type=int,
        default=15,
        help="每次推理后执行的步数",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0 / 15,
        help="每步动作的执行时长（秒）",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu / mps")
    parser.add_argument("--can_port", type=str, default="can0")
    parser.add_argument("--arm_type", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--use_stub", action="store_true", help="使用 StubArm（无真实硬件）")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["side:352122274400", "wrist:254522071216", "front:409122272986"],
        help="相机配置 name:serial_or_id",
    )
    parser.add_argument("--use_usb_cams", action="store_true", help="使用 USB 相机（值为设备 ID）")
    parser.add_argument("--cam_width", type=int, default=640)
    parser.add_argument("--cam_height", type=int, default=480)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], metavar=("W", "H"))
    parser.add_argument("--list_cameras", action="store_true", help="列出 RealSense 设备后退出")

    args = parser.parse_args()

    # 导入 arx5_client
    try:
        from arm_client import ARX5ArmClient
        from local_robot_interface import (
            DEFAULT_IMAGE_TYPE_TO_CAMERA,
            LocalRobotInterface,
            RealSenseCameraManager,
            USBCameraManager,
            list_realsense_devices,
        )
    except ImportError:
        from local_arx5.arm_client import ARX5ArmClient
        from local_arx5.local_robot_interface import (
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
    if not use_remote and not args.policy_path:
        parser.error("--policy_path required when not using --server_url")

    # 加载 PI05（仅单机模式）或远程客户端
    policy = preprocessor = postprocessor = None
    remote_client = None

    if use_remote:
        if not args.server_url.startswith(("ws://", "wss://")):
            logger.error("--server_url must be ws:// or wss://")
            sys.exit(1)
        try:
            from ws_client_pi05 import PI05RemoteInferenceClient
        except ImportError:
            from local_arx5.ws_client_pi05 import PI05RemoteInferenceClient
        remote_client = PI05RemoteInferenceClient(args.server_url)
        logger.info("Using remote PI05 server: %s", args.server_url)
    else:
        _ensure_lerobot()
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        logger.info("Loading PI05 policy from %s", args.policy_path)
        config = PreTrainedConfig.from_pretrained(args.policy_path)
        policy_cls = get_policy_class("pi05")
        policy = policy_cls.from_pretrained(args.policy_path, config=config)
        policy = policy.to(args.device)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=args.policy_path,
            dataset_stats=None,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        logger.info("PI05 loaded")

    # 机械臂 + 相机
    arm_client = ARX5ArmClient(
        can_port=args.can_port,
        arm_type=args.arm_type,
        use_stub=args.use_stub,
    )
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

    interface = LocalRobotInterface(
        arm_client=arm_client,
        camera_manager=camera_mgr,
        image_type_to_camera=DEFAULT_IMAGE_TYPE_TO_CAMERA,
    )

    image_size = list(args.image_size)
    image_type = _image_types_from_cameras(camera_map)

    # 启动前回零
    logger.info("Moving arm to home...")
    interface.go_home()
    time.sleep(2.0)
    interface.hold_position()
    logger.info("Press Ctrl+C to stop. Control loop starting...")

    try:
        while True:
            robot_state = interface.get_state(
                image_size=image_size,
                image_type=image_type,
                action_type="leftpos",
            )
            if robot_state.get("state") != "normal":
                time.sleep(0.1)
                continue
            robot_state["job_id"] = "local"

            if use_remote:
                actions = remote_client.infer(robot_state, args.prompt)
            else:
                batch = build_pi05_observation(
                    robot_state,
                    policy,
                    args.prompt,
                    torch.device(args.device),
                    image_types=image_type,
                )
                actions = infer_pi05(
                    policy, preprocessor, postprocessor, batch, args.action_steps
                )

            if not actions:
                logger.warning("No actions from inference")
                time.sleep(0.1)
                continue

            # 补齐 7 维（远程返回可能已是 7 维）
            for i, a in enumerate(actions):
                if len(a) < 7:
                    actions[i] = list(a) + [0.0] * (7 - len(a))

            interface.execute_actions(actions, args.duration, "leftpos")
            logger.info("Executed %d actions", len(actions))
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        try:
            interface.protect_mode()
        except Exception:
            pass
        camera_mgr.release()


if __name__ == "__main__":
    main()
