"""
PI05 WebSocket 推理客户端：将 obs + prompt 发送到 PI05 服务端，接收动作序列。

与 run_pi05_ws_server 通信，使用 prompt 而非 task_name。
"""

import asyncio
import base64
import json
import logging

logger = logging.getLogger(__name__)


def _encode_robot_state(robot_state: dict) -> dict:
    """images bytes -> base64"""
    images = robot_state.get("images", {})
    images_b64 = {k: base64.standard_b64encode(v).decode("ascii") for k, v in images.items()}
    return {
        "images": images_b64,
        "action": robot_state.get("action", []),
        "timestamp": robot_state.get("timestamp", 0.0),
    }


async def _infer_async(server_url: str, prompt: str, robot_state: dict) -> list:
    import websockets

    payload = _encode_robot_state(robot_state)
    payload["prompt"] = prompt

    async with websockets.connect(server_url, max_size=2**26) as ws:
        await ws.send(json.dumps(payload))
        raw = await ws.recv()

    data = json.loads(raw)
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "Server returned error"))
    return data.get("actions", [])


class PI05RemoteInferenceClient:
    """PI05 远程推理客户端：infer(robot_state, prompt) -> actions"""

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    def infer(self, robot_state: dict, prompt: str) -> list:
        """发送 obs + prompt，返回 [[x,y,z,rx,ry,rz,g], ...]"""
        return asyncio.run(_infer_async(self.server_url, prompt, robot_state))
