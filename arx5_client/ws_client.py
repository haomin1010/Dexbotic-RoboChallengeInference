"""
WebSocket 推理客户端：将观测发送到远程推理服务，接收动作序列。

供机械臂端 (run_local_arx5 --server_url) 使用，与 ws_inference_server 通信。
"""

import asyncio
import base64
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 推理耗时日志文件（单独保存）
_LATENCY_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_LATENCY_LOG_FILE = _LATENCY_LOG_DIR / "inference_latency.log"


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
        t0 = time.perf_counter()
        await ws.send(json.dumps(payload))
        raw = await ws.recv()
        elapsed = time.perf_counter() - t0
        line = "%s 推理往返耗时: %.3fs (发送请求 -> 收到动作)\n" % (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            elapsed,
        )
        logger.info("[客户端] %s", line.strip())
        try:
            _LATENCY_LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(_LATENCY_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            logger.warning("写入推理耗时日志失败: %s", e)

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
