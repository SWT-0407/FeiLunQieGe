# 正式结果目录

当前运行会在这里生成以下本地结果：

- `feasible_centers_overview.png`：三维总览图；
- `feasible_centers.obj`：可由 MeshLab 直接打开的 OBJ，包含采样根部线（`l`）和可行中心点（`p`）；
- `feasible_centers_summary.json`：参数与每个目标点的可行性统计。

这些文件来自 `configs/default.json` 指定的外部案例和归一化参数。OBJ 中的点是候选砂轮中心，不是带厚度的砂轮实体；PNG/OBJ/JSON 均需结合输入、参数和人工检查解释。临时文件必须放入 `__cache__` 或 `__pycache__`。
