# 点追踪改进摘要与评测要点

- 追踪模式：`NN` / `RAFT` / `HYBRID`（光流+最近邻，`tracker_lambda` 可调） / `MULTISCALE`（多尺度特征）。
- 终止与步数：`stop_thresh_px` 控制提前停止；`max_steps` 限制迭代上限。
- 观测指标：界面右侧 `Distances (px)` 展示均值和逐点距离，用于同一步数对比收敛与稳定性。

## 快速评测流程

1) 设置：统一 `Seed` 与点对；`Max Steps = k`（如 50/100），`Stop Threshold (px)` 设 2–4；按需选择模式并在 HYBRID 下调 `tracker_lambda`（弱纹理 0.8–1.5，强纹理 0.2–0.6）。
2) 运行：点击 Start，跑到提前停或达 `Max Steps`。
3) 记录：读取 `Distances (px)` 的均值/逐点数据。
4) 对比：同一场景、同一步数下比较四种模式的距离表现。

## 调参与终止建议

- HYBRID：弱纹理提高 `tracker_lambda`，光流噪声大则降低。
- MULTISCALE：无需 `tracker_lambda`，适合弱纹理且光流不稳场景。
- `stop_thresh_px`：过小不收敛，过大过早停，常用 2–4。
- `max_steps`：防跑偏/长时间计算，通常 50–200。

## 相关文件

- UI 与参数：visualizer_drag_gradio.py
- 追踪核心：viz/renderer.py
- 光流封装：raft_tracker.py
