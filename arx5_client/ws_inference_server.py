"""
WebSocket 推理服务：加载 Dexbotic InferenceRunner，接收观测（图像+状态），返回动作序列。

与 run_robot_client.py 通过 WebSocket 通信，实现动作策略与机械臂连接的分离。
"""

import asyncio
import base64
import json
import logging

logger = logging.getLogger(__name__)


def _decode_request(data: dict) -> dict:
    """Decode JSON request: base64 images -> bytes."""
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
        "job_id": data.get("job_id", "local"),
        "state": "normal",
        "pending_actions": 0,
        "timestamp": data.get("timestamp", 0.0),
    }


async def _handle_client(websocket, runner):
    """Handle one client: receive obs JSON, infer, send actions JSON."""
    try:
        raw = await websocket.recv()
        data = json.loads(raw)
    except Exception as e:
        logger.exception("Receive/parse error: %s", e)
        await websocket.send(json.dumps({"ok": False, "error": str(e)}))
        return

    try:
        robot_state = _decode_request(data)
        task_name = data.get("task_name")
        if task_name and task_name != runner.task_name:
            await websocket.send(
                json.dumps({
                    "ok": False,
                    "error": f"task_name mismatch: client {task_name} vs server {runner.task_name}",
                })
            )
            return

        actions = runner.infer(robot_state)
        # [策略侧] 推理输出 gripper 日志：排查夹爪不张开
        _grippers = [a[6] if len(a) > 6 else "N/A" for a in actions]
        logger.info("[策略侧] inference 输出 gripper: %s (共 %d 步)", _grippers, len(actions))
        await websocket.send(json.dumps({"ok": True, "actions": actions}))
    except Exception as e:
        logger.exception("Inference error: %s", e)
        await websocket.send(json.dumps({"ok": False, "error": str(e)}))


async def _run_server(runner, host: str, port: int):
    import websockets

    async def handler(websocket):
        await _handle_client(websocket, runner)

    async with websockets.serve(handler, host, port, max_size=2**26):
        logger.info("WebSocket inference server listening on %s:%s", host, port)
        await asyncio.Future()


def run_ws_inference_server(runner, host: str = "0.0.0.0", port: int = 8765):
    """Run WebSocket inference server (blocking)."""
    asyncio.run(_run_server(runner, host, port))
