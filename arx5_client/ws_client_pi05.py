"""
PI05 WebSocket 推理客户端：将 obs + prompt 发送到 PI05 服务端，接收动作序列。

与 run_pi05_ws_server 通信，使用 prompt 而非 task_name。
接口兼容 local_control_loop_dexbotic 的 inferrer.infer(robot_state) 调用方式。
"""

import asyncio
import base64
import datetime
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_LATENCY_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_LATENCY_LOG_FILE = _LATENCY_LOG_DIR / "pi05_inference_latency.log"


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
        t0 = time.perf_counter()
        await ws.send(json.dumps(payload))
        raw = await ws.recv()
        elapsed = time.perf_counter() - t0
        line = "%s PI05 推理往返耗时: %.3fs\n" % (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            elapsed,
        )
        logger.info("[PI05 客户端] %s", line.strip())
        try:
            _LATENCY_LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(_LATENCY_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass

    data = json.loads(raw)
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "Server returned error"))

    actions = data.get("actions", [])
    _grippers = [a[6] if len(a) > 6 else "N/A" for a in actions]
    logger.info("[PI05 客户端] 收到 gripper: %s (共 %d 步)", _grippers, len(actions))
    return actions


class PI05RemoteInferenceClient:
    """PI05 远程推理客户端。

    prompt 在构造时传入，infer(robot_state) 只需一个参数，
    兼容 local_control_loop_dexbotic 的 inferrer.infer(robot_state) 调用方式。
    """

    def __init__(self, server_url: str, prompt: str):
        self.server_url = server_url.rstrip("/")
        self.prompt = prompt

    def infer(self, robot_state: dict, **kwargs) -> list:
        """发送 obs + prompt，返回 [[x,y,z,rx,ry,rz,g], ...]"""
        return asyncio.run(_infer_async(self.server_url, self.prompt, robot_state))
