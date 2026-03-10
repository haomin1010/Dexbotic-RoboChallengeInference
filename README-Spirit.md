# Spirit v1.5 真机部署（ARX 单臂 + 本地传感器 + 远程推理）

本文档目标：在 **不使用 Docker** 的前提下，复现 `spirit-v1.5/` 中的 Spirit v1.5 VLA 模型，并用与 DM0 相近的方式部署到同一套真机（ARX 机械臂，单臂）上。

部署由两部分组成：
- **服务端（GPU 服务器）**：`spirit-v1.5/server/`，通过 WebSocket 接收观测（图像+状态），调用 Spirit 模型推理，并返回动作序列。
- **客户端（机械臂本地机）**：`Dexbotic-RoboChallengeInference/arx5_client/`，采集相机与机械臂状态，通过 WebSocket 请求远程推理，并把返回动作执行到机械臂上。

---

## 1. 代码位置与关键文件

- DM0 参考（已可跑通）：`Dexbotic-RoboChallengeInference/dm0.md`
- Spirit 1.5 仓库：`spirit-v1.5/`
  - checkpoint：`spirit-v1.5/checkpoints/spirit-v1.5/`
  - mock 推理参考：`spirit-v1.5/mock_infer.py`
  - RoboChallenge 侧“中间层”逻辑（动作/状态/图像的具体处理）：`spirit-v1.5/robochallenge/`
- Spirit 推理服务端（本次新增）：`spirit-v1.5/server/run_ws_inference_server.py`
- Spirit 真机客户端（本次新增）：`Dexbotic-RoboChallengeInference/arx5_client/run_local_arx5_spirit.py`

---

## 2. 服务端（GPU 服务器）部署

### 2.1 环境要求（建议）

- OS：Linux
- Python：3.11+（`spirit-v1.5/pyproject.toml` 的 requires-python 为 `>=3.11`）
- GPU：建议至少 24GB 显存起步（实际取决于你的显存、batch、以及 Qwen3-VL backbone 的加载方式）
- 网络：客户端机器能访问服务端的 `host:port`

### 2.2 安装依赖

在服务器上进入 Spirit 仓库目录：

```bash
cd /path/to/spirit-v1.5
pip install -r requirements.txt
```

说明：
- Spirit 依赖较重（`torch/transformers/diffusers/...`），请根据你的 CUDA/驱动选择合适的安装方式。
- WebSocket 依赖已加入 `requirements.txt`（`websockets`）。

### 2.3 启动 Spirit WebSocket 推理服务

```bash
cd /path/to/spirit-v1.5
python server/run_ws_inference_server.py \
  --task_name open_the_drawer \
  --ckpt_path checkpoints/spirit-v1.5 \
  --host 0.0.0.0 \
  --port 8765
```

可选参数（常用）：
- `--used_chunk_size 60`：每次最多返回多少步动作（默认 60）
- `--run_id ws-run`：输出保存目录标识（默认写到 `spirit-v1.5/output/<task>/<run_id>/...`）

---

## 3. 客户端（机械臂本地机）部署

客户端复用 DM0 真机部署中已经可用的硬件与传感器代码（ARX + 相机），只替换“推理”部分为远程 WebSocket 调用。

### 3.1 环境要求

- OS：Linux（键盘监听与 CAN 示例均按 Linux 编写）
- Python：与 DM0 客户端相同环境即可
- 依赖：`Dexbotic-RoboChallengeInference/requirements.txt`（已加入 `websockets`）

安装：

```bash
cd /path/to/Dexbotic-RoboChallengeInference
pip install -r requirements.txt
```

### 3.2 机械臂 CAN 盒重新插拔后（参考 DM0）

```bash
sudo -S slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up
```

### 3.3 相机检查（RealSense）

```bash
python -m arx5_client.run_local_arx5_spirit --list_cameras --task_name open_the_drawer --server_url ws://127.0.0.1:8765
```

### 3.4 启动客户端（Spirit 远程推理 + 真机执行）

```bash
cd /path/to/Dexbotic-RoboChallengeInference
python -m arx5_client.run_local_arx5_spirit \
  --task_name open_the_drawer \
  --server_url ws://<GPU服务器IP>:8765 \
  --cameras side:<SIDE_SN> wrist:<WRIST_SN> front:<FRONT_SN>
```

说明：
- 客户端默认 `image_size` 为 `320x240`（与 `spirit-v1.5/mock_infer.py`、RoboChallenge 侧一致）。
- Spirit 的三路图像在 `spirit-v1.5/robochallenge/runner/task_info.py` 中有一层映射（ARX5 的 `cam_high/cam_left_wrist/cam_right_wrist` 与 `high/left_hand/right_hand` 的对应关系和 DM0 不同）。
  - 本客户端在 `Dexbotic-RoboChallengeInference/arx5_client/run_local_arx5_spirit.py` 中通过 `SPIRIT_IMAGE_TYPE_TO_CAMERA` 做了对应的本地相机重映射，以尽量对齐 Spirit 的输入约定。

---

## 4. 运行方式与顺序

- **服务端与客户端分别启动**，运行在不同机器、不同环境中。
- **启动顺序没有硬性要求**（与 DM0 一致）。

---

## 5. 键盘控制（与 DM0 一致）

|按键|功能|
|---|---|
|`space`|紧急停止（锁当前位置）|
|`h`|回 Home|
|`b`|进入拖动示教|
|`n`|退出示教并记录当前位置|
|`m`|回到记录位置|
|`r`|开始推理 + 执行|
|`q`|退出|

---

## 6. 常见问题排查

1) **连接不到服务端**
- 确认 `--server_url` 使用 `ws://` 或 `wss://`
- 确认服务器端口防火墙放行，且客户端能 ping 通服务器

2) **动作返回为空或报错**
- 先在服务器端用 `python mock_infer.py --ckpt_path checkpoints/spirit-v1.5 --task_name <task>` 做最小验证
- 检查服务端日志（模型加载是否成功、是否 OOM）

3) **图像/视角不对**
- 对照 `spirit-v1.5/robochallenge/runner/task_info.py` 的 ARX5 映射
- 调整客户端 `SPIRIT_IMAGE_TYPE_TO_CAMERA` 与 `--cameras` 的实际对应关系（side/wrist/front 的语义可能需要与数据集对齐）

