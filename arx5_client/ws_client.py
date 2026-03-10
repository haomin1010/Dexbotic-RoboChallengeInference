"""
WebSocket 推理客户端：将观测发送到远程推理服务，接收动作序列。

供机械臂端 (run_local_arx5 --server_url) 使用，与 ws_inference_server 通信。
"""

import asyncio
import base64
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _encode_robot_state(robot_state: dict) -> dict:
    """Encode robot_state for JSON: images bytes -> base64."""
    images = robot_state.get("images", {})
    images_b64 = {k: base64.standard_b64encode(v).decode("ascii") for k, v in images.items()}
    return {
        "images": images_b64,
        "action": robot_state.get("action", []),
        "job_id": robot_state.get("job_id", "local"),
        "task_name": robot_state.get("task_name"),
        "timestamp": robot_state.get("timestamp", 0.0),
        "new_execution": bool(robot_state.get("new_execution", False)),
    }


async def _infer_async(server_url: str, task_name: str, robot_state: dict) -> list:
    import websockets

    payload = _encode_robot_state(robot_state)
    payload["task_name"] = task_name

    async with websockets.connect(server_url, max_size=2**26) as ws:
        await ws.send(json.dumps(payload))
        raw = await ws.recv()

    data = json.loads(raw)
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "Server returned error"))

    actions = data.get("actions", [])
    # [客户端] 从服务端收到的 gripper 日志：排查夹爪不张开
    _grippers = [a[6] if len(a) > 6 else "N/A" for a in actions]
    logger.info("[客户端] 收到 gripper: %s (共 %d 步)", _grippers, len(actions))
    return actions


class RemoteInferenceClient:
    """
    远程推理客户端：将 robot_state 发送到 WebSocket 推理服务，返回动作列表。

    接口与 InferenceRunner.infer(state) 一致，可在 local_control_loop 中作为 inferrer 使用。
    """

    def __init__(self, server_url: str, task_name: str):
        """
        Args:
            server_url: e.g. ws://192.168.1.100:8765
            task_name: 任务名，每次请求会携带
        """
        self.server_url = server_url.rstrip("/")
        self.task_name = task_name

    def infer(self, robot_state: dict, new_execution: bool = False) -> list:
        """
        发送观测到服务端，返回动作列表。

        Args:
            robot_state: dict with keys images (bytes), action (list), job_id, etc.
            new_execution: 忽略，接口兼容用

        Returns:
            List of actions [[x,y,z,rx,ry,rz,g], ...]
        """
        if new_execution:
            robot_state = dict(robot_state)
            robot_state["new_execution"] = True
        return asyncio.run(_infer_async(self.server_url, self.task_name, robot_state))
