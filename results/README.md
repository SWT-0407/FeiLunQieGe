# 正式结果目录

每次运行会在 `run_YYYYMMDD_HHMMSS/` 下建立独立目录，不覆盖旧结果。每个运行目录包含：

- `feasible_centers_overview.png`：三维总览图；
- `local_feasibility_explanation.png`：单个代表性目标点的局部候选圆和可行弧解释图；
- `feasible_centers.obj`：可由 MeshLab 直接打开的 OBJ，包含采样根部线（`l`）和可行中心点（`p`）；
- `feasible_centers_summary.json`：参数与每个目标点的可行性统计。

这些文件来自 `configs/default.json` 指定的外部案例和归一化参数。OBJ 中的点是候选砂轮中心，不是带厚度的砂轮实体；OBJ 还包含外圈可行中心之间的 `l` 线段，方便 MeshLab 默认显示。PNG/OBJ/JSON 均需结合输入、参数和人工检查解释。临时文件必须放入 `__cache__` 或 `__pycache__`。
