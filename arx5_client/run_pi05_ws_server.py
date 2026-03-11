#!/usr/bin/env python3
"""
PI05 WebSocket 推理服务（GPU 端）：
加载 PI05 模型，接收 obs（images + state + prompt），返回动作序列。

运行在带 GPU 的机器上；机械臂端通过 --server_url 连接。

Usage
-----
  python arx5_client/run_pi05_ws_server.py \\
      --policy_path lerobot/pi05_base \\
      --host 0.0.0.0 --port 8765
"""

import argparse
import asyncio
import base64
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for p in (_REPO_ROOT, _THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

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


def _jpeg_bytes_to_rgb_np(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_pi05_batch(robot_state: dict, prompt: str, policy) -> dict:
    """Build an unbatched observation dict for the PI05 preprocessor pipeline.

    The preprocessor (loaded from pretrained) handles: AddBatchDim, Normalize,
    PadState + Discretize + BuildPrompt, Tokenize, and ToDevice.
    So we only do format conversion here (numpy -> torch, HWC -> CHW, uint8 -> float).
    """
    from lerobot.configs.types import FeatureType
    from lerobot.utils.constants import OBS_STATE as OBS_STATE_KEY

    image_keys = sorted(
        k for k, v in policy.config.input_features.items()
        if v.type == FeatureType.VISUAL and "empty" not in k
    )

    present = list(robot_state.get("images", {}).keys())
    batch: dict = {}
    for i, arx5_key in enumerate(present):
        if i >= len(image_keys):
            break
        pi05_key = image_keys[i]
        img_rgb = _jpeg_bytes_to_rgb_np(robot_state["images"][arx5_key])
        t = torch.from_numpy(img_rgb).float() / 255.0
        batch[pi05_key] = t.permute(2, 0, 1)  # (3, H, W), no batch dim

    state = np.array(robot_state.get("action", [0.0] * 7), dtype=np.float32)
    batch[OBS_STATE_KEY] = torch.from_numpy(state)  # (state_dim,), no pad, no batch dim

    batch["task"] = prompt  # plain string; AddBatchDim wraps it in a list
    return batch


def infer_pi05(policy, preprocessor, postprocessor, batch: dict, action_steps: int) -> list:
    preprocessed = preprocessor(batch)
    with torch.no_grad():
        actions = policy.predict_action_chunk(preprocessed)
    postprocessed = postprocessor(actions)
    actions_np = postprocessed.squeeze(0).cpu().numpy()
    actions_np = actions_np[:action_steps]
    action_dim = min(7, actions_np.shape[-1])
    return [a[:action_dim].tolist() for a in actions_np]


def _decode_request(data: dict) -> dict:
    images_b64 = data.get("images", {})
    images = {}
    for k, v in images_b64.items():
        if isinstance(v, str):
            images[k] = base64.standard_b64decode(v)
        else:
            images[k] = v
    return {
        "images": images,
        "action": data.get("action", []),
    }


async def _handle_client(websocket, policy, preprocessor, postprocessor, action_steps: int):
    try:
        raw = await websocket.recv()
        data = json.loads(raw)
    except Exception as e:
        logger.exception("Receive/parse error: %s", e)
        await websocket.send(json.dumps({"ok": False, "error": str(e)}))
        return

    try:
        robot_state = _decode_request(data)
        prompt = data.get("prompt", "")
        if not prompt:
            await websocket.send(json.dumps({"ok": False, "error": "prompt is required"}))
            return

        batch = build_pi05_batch(robot_state, prompt, policy)
        actions = infer_pi05(policy, preprocessor, postprocessor, batch, action_steps)
        for i, a in enumerate(actions):
            if len(a) < 7:
                actions[i] = list(a) + [0.0] * (7 - len(a))

        logger.info("Inference done, returning %d actions", len(actions))
        await websocket.send(json.dumps({"ok": True, "actions": actions}))
    except Exception as e:
        logger.exception("Inference error: %s", e)
        await websocket.send(json.dumps({"ok": False, "error": str(e)}))


async def _run_server(policy, preprocessor, postprocessor, action_steps: int, host: str, port: int):
    import websockets

    async def handler(websocket):
        await _handle_client(websocket, policy, preprocessor, postprocessor, action_steps)

    async with websockets.serve(handler, host, port, max_size=2**26):
        logger.info("PI05 WebSocket server listening on %s:%s", host, port)
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PI05 WebSocket 推理服务（GPU 端）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy_path", type=str, default="lerobot/pi05_base")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--action_steps", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    logger.info("Loading PI05 from %s", args.policy_path)
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
    logger.info("PI05 loaded, starting WebSocket server")

    asyncio.run(_run_server(policy, preprocessor, postprocessor, args.action_steps, args.host, args.port))


if __name__ == "__main__":
    main()
