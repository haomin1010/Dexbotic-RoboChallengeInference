#!/usr/bin/env python3
"""
夹爪测试脚本：创建 ARX5ArmClient 后直接控制夹爪张开/闭合，用于排查 ArmClient 侧问题。

Usage
-----
  # 真实硬件：先张开再闭合
  python -m arx5_client.test_gripper

  # StubArm 测试（无硬件）
  python -m arx5_client.test_gripper --use_stub
"""

import argparse
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent
for p in (REPO_ROOT, _THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main():
    parser = argparse.ArgumentParser(description="ARX5 夹爪直接控制测试")
    parser.add_argument("--can_port", default="can0")
    parser.add_argument("--use_stub", action="store_true", help="StubArm，无真实硬件")
    args = parser.parse_args()

    try:
        from arx5_client.arm_client import ARX5ArmClient
    except ImportError:
        from arm_client import ARX5ArmClient

    client = ARX5ArmClient(can_port=args.can_port, use_stub=args.use_stub)

    # 获取当前位姿
    state = client.get_state()
    xyzrpy = state[:6]
    current_gripper = state[6]
    print(f"当前夹爪值: {current_gripper}")
    print(f"当前位姿 xyzrpy: {xyzrpy}")

    # 保持位姿不变，仅将夹爪设为 5.0（张开，ARX5 范围 0-5）
    action_open = list(xyzrpy) + [0.3]
    print("\n发送 action gripper=5.0 张开夹爪（ARX5 范围 0-5）...")
    client.execute_actions([action_open], duration=0)
    time.sleep(2)

    # 再设为 0.0（闭合）
    action_close = list(xyzrpy) + [0.0]
    print("发送 action gripper=0.0 闭合夹爪（ARX5 范围 0-5）...")
    client.execute_actions([action_close], duration=0)
    time.sleep(2)

    state_after = client.get_state()
    print(f"\n执行后夹爪值: {state_after[6]}")
    print("测试结束。")


if __name__ == "__main__":
    main()
