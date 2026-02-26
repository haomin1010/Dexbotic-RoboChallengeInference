"""
WebSocket inference server: loads Spirit VLA executor, receives observations,
returns actions. For use with remote inference (client runs ARX5 locally).
"""

import asyncio
import base64
import json
import logging
from argparse import Namespace
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_executor(args) -> "RoboChallengeExecutor":
    """Build RoboChallengeExecutor (import here to avoid loading model on client)."""
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from robochallenge.runner.executor import RoboChallengeExecutor

    cfg = Namespace(
        single_task=args.single_task,
        robochallenge_job_id="ws_server",
        ckpt_path=args.ckpt_path,
        used_chunk_size=args.used_chunk_size,
        raw_embodiment_stats_json_path=getattr(args, "raw_embodiment_stats_json_path", None),
        use_embodiment_specific_norm=False,
        embodiment_stats_path=None,
    )
    return RoboChallengeExecutor(cfg)


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


async def _handle_client(websocket, executor):
    """Handle one client connection: receive obs JSON, infer, send actions JSON."""
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
        if task_name and task_name != executor.task_name:
            await websocket.send(
                json.dumps({"ok": False, "error": f"task_name mismatch: client {task_name} vs server {executor.task_name}"})
            )
            return

        actions = executor.infer(robot_state)
        await websocket.send(json.dumps({"ok": True, "actions": actions}))
    except Exception as e:
        logger.exception("Inference error: %s", e)
        await websocket.send(json.dumps({"ok": False, "error": str(e)}))


async def _run_server(executor, host: str, port: int):
    import websockets

    async def handler(websocket):
        await _handle_client(websocket, executor)

    async with websockets.serve(handler, host, port, max_size=2**26):
        logger.info("WebSocket server listening on %s:%s", host, port)
        await asyncio.Future()


def run_ws_server(executor, host: str = "0.0.0.0", port: int = 8765):
    """Run WebSocket server (blocking)."""
    asyncio.run(_run_server(executor, host, port))
