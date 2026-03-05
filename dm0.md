# DM0 真机测试

## 客户端

### 环境准备

```
conda activate dex-robo  
cd /home/user/workspace/lhm/dexbotic/Dexbotic-RoboChallengeInference
```

### 重新插拔机械臂 CAN 盒时执行

```
sudo -S slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up
```

### 检查相机标识

```
python -m arx5_client.run_local_arx5 --list_cameras
```

### 运行客户端

```
python -m arx5_client.run_local_arx5 \  
    --task_name place_shoes_on_rack \  
    --server_url ws://127.0.0.1:8765 \  
    --cameras side:152122075089 wrist:352122274400 front:254522071216
```

---

# 服务端

### 启动容器

```
sudo docker start dm0_1  
sudo docker exec -it dm0_1 bash
```

### 进入环境

```
conda activate dexbotic  
cd /dexbotic  
export PYTHONPATH=/dexbotic
```

### 启动推理服务

```
python Dexbotic-RoboChallengeInference/arx5_client/run_ws_inference_server.py \  
    --task_name place_shoes_on_rack \  
    --checkpoint checkpoints/dm0/DM0-table30_place_shoes_on_rack/ \  
    --host 0.0.0.0 \  
    --port 8765
```

---

# 运行方式

- **客户端与服务端分别启动**
    
- **启动顺序没有要求**
    

---

# 键盘控制

|按键|功能|
|---|---|
|`b`|进入类似示教模式|
|`n`|退出示教，并保存当前位置为存档|
|`m`|恢复存档位置|
|`r`|进入推理 + 执行|
|`space`|停止推理|