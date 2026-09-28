# 正式结果目录

每次运行会在 `run_YYYYMMDD_HHMMSS/` 下建立独立目录，不覆盖旧结果。每个运行目录包含：

- `feasible_centers_overview.png`：三维总览图；
- `local_feasibility_explanation.png`：单个代表性目标点的局部候选圆和可行弧解释图；
- `workpiece_orientation_guide.png`：三维透视和 X-Z 正投影的工件视角对照图；
- `feasible_centers.obj`：可由 MeshLab 直接打开的 OBJ，包含采样根部线（`l`）和可行中心点（`p`）；
- `feasible_centers_summary.json`：参数与每个目标点的可行性统计。

另外，`feasible_centers_display.obj` 是带红色三角面标记的全局显示代理，`local_feasible_centers.obj` 是便于 Fit View 的局部放大显示；同目录的 `.mtl` 文件提供颜色。精确点线数据仍以 `feasible_centers.obj` 为准。

`interactive_demo.html` 是少量采样点的离线点击演示：点击左侧橙色根部采样点，右侧显示该点的局部放大法平面。它只用于交互验收，不替代精确 OBJ 或 JSON。

这些文件来自 `configs/default.json` 指定的外部案例和归一化参数。OBJ 中的点是候选砂轮中心，不是带厚度的砂轮实体；OBJ 还包含外圈可行中心之间的 `l` 线段，方便 MeshLab 默认显示。PNG/OBJ/JSON 均需结合输入、参数和人工检查解释。临时文件必须放入 `__cache__` 或 `__pycache__`。
