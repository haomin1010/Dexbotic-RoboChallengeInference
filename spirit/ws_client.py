"""
WebSocket inference client: sends observations to remote server, receives actions.
"""

import asyncio
import base64
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RemoteInferenceClient:
    """
    Client that sends robot_state to a WebSocket inference server and returns actions.

    Implements the same interface as infer(robot_state) -> list for use in local_loop.
    """

    def __init__(self, server_url: str, task_name: str):
        """
        Args:
            server_url: e.g. ws://192.168.1.100:8765
            task_name: Task name sent with each request (e.g. open_the_drawer)
        """
        self.server_url = server_url.rstrip("/")
        self.task_name = task_name

    def infer(self, robot_state: dict, new_execution: bool = False) -> list:
        """
        Send observation to server, return action list.

        Args:
            robot_state: dict with keys images (bytes), action (list), job_id, etc.
            new_execution: Ignored (for interface compatibility with RoboChallengeExecutor).

        Returns:
            List of actions [[x,y,z,rx,ry,rz,g], ...]
        """
        return asyncio.run(_infer_async(self.server_url, self.task_name, robot_state))


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
    return actions
