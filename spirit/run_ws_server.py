#!/usr/bin/env python3
"""
WebSocket inference server entry: load Spirit VLA and serve inference over WebSocket.

Run on a machine with GPU; clients (e.g. run_local_arx5 --server_url ws://...)
send observations and receive actions.

Usage
-----
  python -m local_arx5.run_ws_server \\
      --host 0.0.0.0 --port 8765 \\
      --single_task open_the_drawer \\
      --ckpt_path /path/to/checkpoint
"""

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_arx5.ws_server import _build_executor, run_ws_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WebSocket Spirit VLA inference server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--single_task", type=str, required=True, help="Task name (ARX5)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Spirit VLA checkpoint path")
    parser.add_argument("--used_chunk_size", type=int, default=60, help="Action chunk size")
    parser.add_argument("--raw_embodiment_stats_json_path", type=str, default=None)
    args = parser.parse_args()

    logger.info("Loading executor (task=%s, ckpt=%s)", args.single_task, args.ckpt_path)
    executor = _build_executor(args)
    logger.info("Executor ready.")
    run_ws_server(executor, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
