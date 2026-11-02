# FeiLunQieGe

砂轮切除铸件飞边的可行区域与路径规划研究。

本 README 是 **2026-10-31 的交接版**：记录已完成的几何闭环、法向/碰撞修复、circle 路径规划交付，以及下一轮需要注意的边界。仓库当前代码基线仍为 `e701368`；本轮及上一轮文档/代码更新可能尚未提交，以 `git status` 为准。

> 重要边界：当前结果仍是 **normalized units 下的几何演示与可行性筛选**，不是毫米级工艺结论、机床可达性结论或真实加工验收。路径规划只做节点级可行性，**未做边级扫掠碰撞审计**。

文档优先级：`AGENTS.md` 中的 `interactive_demo.html`、旧输出契约等条目属于历史阶段；当前实际交付以本 README 和最新运行结果为准，但 `AGENTS.md` 的文件安全、缓存和禁止删除规则始终有效。

---

## 1. 当前阶段一句话总结

已完成：

- `root_line.ply` 分支拆分、弧长采样和局部标架；
- P1 法向后端（法向聚类、shape-aware patch、jet fit、crease gate）；
- 有限厚度实心平端面圆柱 + 三角形连续窄相 + AABB 宽相；
- 接触斑/目标接触语义（按项目确认采用 C：允许 2R 范围内浅接触）；
- 对完全无解点的 **fallback axis（回退轴）**；
- circle 全量 415 点、72 角度的三 OBJ 历史交付；
- 全部 415 点至少 1 个可行角度，**完全无解点为 0**；
- **路径规划（2026-10-31）**：分层有向图 + 动态规划；先做“每点必选”的全采样点 baseline，再扩展为**覆盖约束**——每个选中点覆盖 root_line 上 `[s-r, s+r]` 的连续弧长，覆盖区间重叠即可跳过中间采样点；branch 内强制 `q` 符号连续，处理退化“往返”分支；输出两个 OBJ（可行中心域 + 路径）。

尚未完成：

- 连续角度区间：当前仍是 72 个离散角度；
- **边级扫掠碰撞审计**：当前只检查每个候选中心（节点），不检查相邻中心连线中间姿态；
- 机器人/机床可达性、姿态连续性、曲率/进给约束、分支间排序；
- 连续路径的严格无碰撞证明（需要连续扫掠体/保守可达体）。

---

## 2. 交付记录

### 2.1 历史三个 OBJ（2026-10-29）

目录：

```text
results/cases/circle/p1_full_delivery_20261029_233114_679352/
```

文件：

| 文件 | 内容 |
|---|---|
| `circle_rootline_circle_centers.obj` | 根线 + 415 个采样点圆心 |
| `circle_inspection_wheels.obj` | 红色不可行、绿色可行砂轮实体；已移除蓝色接触点标记 |
| `circle_feasible_center_domains.obj` | 每个点的候选中心：绿色通过当前判定，红色碰撞 |
| 对应 `.mtl` | 三个 OBJ 的材质 |
| `p1_full_summary.json` | 逐点可行角数、碰撞原因、normal confidence/source、fallback axis、参数 |
| `README.md` | 本次三 OBJ 的简要说明 |

坐标：

- OBJ：`source_obj`，可与原始 `circle/mesh_without_flash.obj` 直接叠加；
- 算法计算：normalized units；
- PNG（历史诊断）：normalized；本交付未生成 PNG。

### 2.2 当前结果

- 415 个 root 点；
- 每点 72 个角度，共 29,880 个候选；
- 可行 **26,919 / 29,880**；
- **完全无解点：0**；
- **57 个点仍有部分不可行角度**，但每个点至少 1 个可行角；
- 最小可行角数：1（点 402，branch 7，`two_side_bisector`，low confidence）；这是路径规划最紧的瓶颈点；
- fallback 点：仅点 400，从 0/72 变为 5/72；
- 低置信点：396–403 共 8 个（全部在 branch 7）。

碰撞原因计数：

| reason | count |
|---|---:|
| `contact_patch_only` | 26,434 |
| `target_contact_only` | 485 |
| `triangle_interior_collision` | 2,068 |
| `cylinder_cap_collision` | 893 |

因此当前绿色是：

> “允许根部/接触斑浅接触后的可行”，不是严格无碰撞。当前 C 语义下，严格 `safe`（无任何相交）为 0 是预期结果。

分支统计（可行角 / 总角）：

| branch | 可行/总 |
|---:|---:|
| 0 | 5,831 / 5,832 |
| 1 | 8,685 / 8,712 |
| 2 | 11,016 / 11,016 |
| 3 | 195 / 720 |
| 4 | 186 / 720 |
| 5 | 299 / 720 |
| 6 | 230 / 720 |
| 7 | 177 / 720 |
| 8 | 300 / 720 |

法向来源：

- `jet_fit`：405 点；
- `two_side_bisector`：5 点；
- `shape_patch_mean`：4 点；
- `fallback_axis`：1 点（400）；
- `normal_confidence`：high 407，low 8（396–403）。

### 2.3 路径规划两 OBJ（2026-10-31，覆盖约束版）

目录：

```text
results/cases/circle/p1_full_delivery_20261031_195757_314283/
```

文件：

| 文件 | 内容 |
|---|---|
| `circle_feasible_center_domains.obj` | 根线 + 415 点 × 72 候选中心；绿色可行、红色碰撞 |
| `circle_paths.obj` | 根线 + `+q` 蓝色路径 + `-q` 橙色路径 + 选中中心标记；只连选中点，允许跳过 |
| `circle_paths.mtl` | 路径材质 |
| `path_planning_summary.json` | 每 branch/侧节点、角度、中心、长度、覆盖区间、跳过点、同长解计数、参数 |
| `path_candidates.npz` | 415×72 候选数组（centers/feasible/angles/axis/tangent/radial/reason/source 坐标） |
| `p1_full_summary.json` | 旧字段保留，新增 `path_planning` 段 |
| `README.md` | 本次交付说明 |

覆盖约束结果：

- 17 条 `ok` 路径、1 条 `no_path`；
- `branch_07 +q`：`no_path`（396–403 在该侧无可行候选，覆盖区间不能跨过）；
- `branch_07 -q`：`ok`，1 个选中点覆盖全部 10 点，长度 0；
- long branches：
  - branch 0：6/81 选中，长度 0.12638；
  - branch 1：9/121 选中，长度 0.22649；
  - branch 2：11/153 选中，长度 0.30698；
- branch 3–8：每侧 1/10 选中，长度 0（短分支长度 < 2r，单姿态覆盖整段）；
- `max_axis_step_deg`：branch 0/1/2 分别约 4.80°/4.17°/4.49°；
- `max_angle_step_deg`：长分支约 140°，这是接触点在**端面圆周**上的方向 θ 变化，不是工具轴姿态；圆柱轴对称，实际轴向变化看 `max_axis_step_deg`；
- 同长多解：短分支每侧有 2 条同长解，按最低候选索引导出其中一条；
- 本轮未做边级碰撞审计。

自动校验：

- root_line 与 `point_source` 最大误差 `4.96e-12`；
- 17 条路径 OBJ 折线与 JSON `center_source` 最大误差 `4.66e-12`；
- `branch_07 +q` 无路径对象，OBJ 头部有 `NO PATH` 注释；
- MTL 含 `path_positive_blue`、`path_negative_orange`、`root_line_gray`。

> 覆盖模型是一维弧长抽象：选中点覆盖 `[s-r, s+r]`；相邻选中点覆盖区间重叠即可跳过中间点；若某点在该侧完全无可行候选，覆盖区间不能跨过它。不是砂轮实体与工件表面的精确切除仿真。
>
> 2026-11-01 可视化修订：`circle_paths.obj` 的相邻选中中心除 `l` 线外，额外写成细圆柱面（tube），点标记和线径加粗；规划节点/角度/覆盖/JSON 未改。可用 `python scripts/reexport_path_obj.py --run-dir <run> --overwrite` 从既有 JSON/NPZ 重新导出。
>
> 全采样点 baseline 保留在 `results/cases/circle/p1_full_delivery_20261031_153521_498629/`，作为对照，不是最终覆盖优化结果。

### 2.4 四 OBJ 路径可视化（2026-11-02）

目录：

```text
results/cases/circle/path_viz_4obj_20261102_125949/
```

文件：

| 文件 | 内容 |
|---|---|
| `circle_feasible_center_domains.obj` | 从 `p1_full_delivery_20261031_195757_314283/` 原样复制；绿色可行/红色碰撞中心域，未重算 |
| `circle_paths_dots.obj` | 只输出路径规划选得的 63 个中心点；亮黄色 `path_dot_yellow` |
| `circle_paths_lines.obj` | root_line + 选中点连线（`l` + tube）；`+q` 蓝、`-q` 橙；不重复点标记 |
| `circle_path_wheels.obj` | 每个 branch 只取一侧的选中点，画 32 个有限圆柱砂轮；`wheel_teal` |
| 对应 `.mtl` | 材质 |
| `README.md` | 文件说明、轮径/轮宽、侧向选择 |

结果与边界：

- 63 个选中点（两侧合计）；
- 17 条路径连线（单点短分支在 lines OBJ 中只有孤立顶点，无线段）；
- 32 个砂轮：branch 0/1/2 分别 6/9/11 个，branch 3–6、8 各 1 个，branch 7 因 `+q` 无解回退 `-q` 1 个；
- 每个 branch 只输出一侧砂轮，另一侧不输出；
- 坐标 `source_obj`；砂轮半径 `0.02` normalized、宽度 `0.004` normalized、圆柱 48 分段；
- 只是显示几何，未做边级扫掠碰撞审计。

生成命令：

```powershell
python scripts/export_path_visualizations.py --run-dir results/cases/circle/p1_full_delivery_20261031_195757_314283
```

---

## 3. 输入数据与坐标

### 3.1 数据位置

`configs/default.json` 的 `data_root` 指向：

```text
D:\0_School\0_Scientific_Research\01\flash_pipeline_12_models_20260924
```

circle 案例关键文件：

| 文件 | 用途 |
|---|---|
| `root_line.ply` | 飞边/工件根部交线，ASCII PLY，415 个顶点 |
| `mesh_without_flash.obj` | **唯一碰撞本体**；原始 full mesh，250,000 顶点 / 500,000 面 |
| `original_flash_cloud.obj` | 带颜色标签的点云；`0=body`、`1=flash`；用于法向符号/飞边方向 |
| `mesh_with_flash.obj` | 含飞边工件；**当前不参与碰撞**，因为与飞边相交是切飞边的必然条件 |
| `mesh_without_flash_simplified.obj` | 简化对照网格；当前不用于最终结果 |

### 3.2 root 点确认

已用精确点到三角形距离核对 415 个 root 点：

- 最大距离约 `2.78e-11 source units`；
- `distance > 1e-6` 的点：0 / 415；
- 因此 root 点确实在 `mesh_without_flash.obj` 表面上，误差为浮点量级。

局部位置：

- branch 4（365–374）：底部平面内部，source xy 半径约 0.449–0.459；
- branch 7（395–404）：外底边/凸边过渡，source xy 半径约 0.536；
- branch 4 局部以单一底面为主：示例点 365 的 440 个局部面中 439 个为底面，拟合/底面法向差约 `0.13°`，可视为平面；
- branch 7 局部同时含底面和侧壁：示例点 398 为 447 个底面 + 6 个侧面，patch coherence 约 `84°`；P0 的 branch 7 `angle12` 均值约 `8.0°`，其他光滑分支通常 `<1°`。因此 branch 7 是**凸边/圆角过渡**，不是平面、不是凹面。

### 3.3 坐标约定

- 计算坐标：normalized units：工件包围盒中心为原点、最长边为 1；
- 论文/OBJ 原始坐标：`source_obj`；
- 导出 OBJ 时：

```text
source = normalized * scale + center
```

circle 当前：

```text
scale ≈ 1.137405
center ≈ [-0.000316, 0.5687025, 0.001487]
```

- 根线与 OBJ 的 source 坐标对齐误差约 `4.96e-12`（历史审计）；导出后应继续用 `scripts/audit_meshlab_alignment.py` 复核。

---

## 4. 当前算法流程

### 4.1 root line 分支与采样

`split_and_resample(root_line, curve_points)`：

- 按 PLY edge 连通关系拆分分支；
- 每个分支按连接顺序/弧长重采样；
- 保留全局点编号、`branch_id`、分支内位置；
- circle：9 个分支，415 点：
  - B0 81，B1 121，B2 153，B3–B8 各 10。

点编号规则：

```text
point_index 从 0 开始，先分支、再分支内顺序；
branch_id 从 0 开始。
```

### 4.2 法向：P1 默认

`configs/default.json`：

```json
"normal_estimation": {
  "method": "p1"
}
```

`normal_p1.py` 包含：

- 全局绕向修复（共享边 BFS + signed volume）；
- 候选面 KDTree + 法向聚类；
- shape-aware patch（多尺度、平坦度/方向一致性、l0 式裁剪）；
- jet fit（PCA 切平面、加权二次多项式、条件数/凸包覆盖检查）；
- crease gate：`angle12 ≥ 10°`、`support_ratio ≥ 0.35`、`min_support ≥ 6`；
- source-specific low confidence；
- fallback axis（见 4.6）。

支持的实验方法：

| method | 说明 |
|---|---|
| `p1` | shape-aware patch + jet + crease gate（当前默认） |
| `zhong2021` | shape-aware patch / l0 裁剪（Zhong 2021 近似） |
| `zhang2015` | guidance patch + bilateral（Zhang 2015 近似） |
| `cazals2005` | 二次 jet 拟合（Cazals & Pouget 2005 近似） |
| `li2010` | Tukey 鲁棒 sharp-feature（Li 2010 近似） |
| `zhong+cazals` | 当前与 `p1` 同路径，不是独立方法 |
| `zhang+cazals` / `li+zhong` / `all` | 组合 |

当前 P1 全量与 baseline：

| 方法 | 可行/29,880 | 说明 |
|---|---:|---|
| baseline `surface_patch_continuity` | 26,929 | 旧生产路径 |
| P1 | 26,919（fallback 后） | 用户选定；branch7 / 特征导向 |

P1 不是全局可行率提升；选择 P1 的原因是 branch 7/凸边处的局部行为和可解释的低置信度/回退机制。

### 4.3 切平面与标架

对每个 root 点：

1. 曲线切线 `t`；
2. 表面法向 `n`（P1 或 baseline）；
3. 把 `t` 投影到 `n` 的切平面，保证 `t·n≈0`；
4. `q = n × t`；
5. 切平面 = 过 root 点、由 `t,q` 张成；
6. 砂轮轴 `a = n`。

中心公式：

```text
e(theta) = cos(theta) * t + sin(theta) * q
c = x + rho * e(theta) + side * (w/2) * a
```

当前：

- `rho = r`（三 OBJ/最终结果单层）；
- `side = +1`；
- 近端面与过 root 点的切平面共面；
- 中心到切平面的法向距离是 `w/2`。

**注意**：当前是“单点锚定 + patch 估计法向”，不是用多个 mesh 点拟合平面。branch 7 是凸边，局部法向不唯一；这是后续路径规划需要特别注意的地方。

### 4.4 候选中心

`generate_disk_candidates` / `_candidate_set`：

- 每个点 72 个角度（最终三 OBJ）；
- pipeline 默认 `candidate_angles=180`，`radial_samples=3`，`delta=0.001`；
- 最终三 OBJ 为了速度使用 `rho=r` 单层、72 角度；
- 因此三 OBJ 不是 pipeline 默认 180×3 的完整候选集。

### 4.5 碰撞检测

当前模型：

- 工件：`mesh_without_flash.obj`；
- 工具：半径 `r=0.02`、宽 `w=0.004` 的实心平端面圆柱；
- 宽相：三角形 AABB BVH；
- 窄相：默认 `analytic_closest_point_flat`，但**相交/穿透主判定仍是轴向裁剪 + 二维凸多边形距离**，解析部分只额外给接触法向；
- 三角形先按轴向裁剪到 `[-w/2,w/2]`，再投影到垂直轴的横截面，求裁剪多边形到轴线的最短距离；
- 半径 `≤R` 则相交；
- 封闭网格且无表面相交时，再用中心射线测试 `workpiece_contains_wheel`。

已修复：

- 径向穿透用裁剪多边形到轴线最近点，避免只取顶点导致低估；
- 接触斑最大距离加入边与圆柱侧面交点；
- 远端面接触不再按近端面接触斑豁免；
- `CollisionResult.penetration` 逐候选记录；
- 目标接触/接触斑接受时记录实际 penetration/patch distance/triangle count。

### 4.6 接触语义（已确认）

当前采用 C：

- 允许目标根部 **2R** 范围内的浅接触；
- `contact_patch_radius = 2r = 0.04`；
- `max_contact_penetration = 0.025r = 0.0005`（几何容差，不是物理允许嵌入）；
- `contact_normal_tolerance = 30°`；
- `allow_target_contact=true`；
- **不限制必须是目标三角面**；
- 不检查 `mesh_with_flash.obj`；
- 不增加 clearance，允许面相贴；
- `side=+1`。

全量结果中 `safe`（无相交）为 0；绿色全部是 `contact_patch_only`/`target_contact_only`。这是平端面 + 面接触 + C 语义的预期结果，不是“严格无碰撞”。

### 4.7 fallback axis（零可行回退）

`export_p1_full_delivery.py` 中：

- 先按 P1 主轴计算 72 角；
- 若某点 0 可行，尝试：
  - 当前法向；
  - 主/次法向簇；
  - 两侧角平分线；
  - 局部候选面法向簇；
  - 底部型面法向均值；
- 选择可行角最多的轴；
- 标记：

```text
normal_source = fallback_axis
normal_confidence = low
reason = zero_feasible_primary_axis
```

circle 中仅点 400 触发：

```text
axis = [-0.059940, -0.380319, -0.922911]
feasible = 5 / 72
```

这使全部 415 点至少 1 个可行角。

### 4.8 路径规划（2026-10-31）

实现位置：

```text
src/feilunqiege/planning/path_planning.py
src/feilunqiege/visualization/path_obj.py
scripts/export_p1_full_delivery.py
```

**问题（2026-10-31 第二版）**：对每个 branch，在 root_line 两侧各求一条路径；不要求经过每个采样点。每个选中点选一个可行候选中心，并覆盖 root_line 上 `[s-r, s+r]` 的连续弧长；相邻选中点覆盖区间重叠即可跳过中间采样点。目标是覆盖整条 branch 的前提下，中心折线总长度最小。

**覆盖模型**：

- 选中点 i 的覆盖区间为 `[s_i - r, s_i + r]`，`r` 为砂轮半径；
- 相邻选中点满足 `s_next - s_prev <= 2r` 时覆盖重叠；
- 起点/终点只需被某个选中点覆盖，不强制选首末点；
- 若某采样点在该侧完全没有可行候选（现有全域碰撞结果），覆盖区间不能跨过它，边界取在该点与相邻可覆盖点的中点；
- 这是一维弧长规划抽象，不是砂轮实体与工件表面的精确切除仿真。

**侧向定义**：

```text
q_i = normalize(a_i × t_i)
lateral_side = sign(e_ij · q_i)
```

- `+q` / `-q` 是路径左右侧；`side=+1` 仍是砂轮轴向侧，不是路径侧。
- `θ=0°/180°` 的中线候选不参与两侧路径。
- **branch 内强制 q 符号连续**：若 `q_i · q_{i-1} < 0`，翻转 `q_i`。这是为了处理 branch 7 式退化“往返”分支——root_line 沿同一条空间线去程/回程，切线翻转 180°，逐点独立取 `q` 会把同一物理侧向拆成 +q/-q 两段，导致两侧都无解。
- 检查 `path_candidates.npz` 发现 branch 3–8 的相邻切线在中间某点反向（最小 `t_i·t_{i-1}` 约 `-1`），说明这些 10 点短分支都存在往返/退化形态；因此 q 符号连续不是 branch 7 专用，而是所有短分支的必要处理。覆盖优化可能只选 1 个点；路径按选中点计算，**不去重**。
- 侧向一致性只要求**砂轮中心**在指定一侧；砂轮其他部分可以越过该侧（用户 2026-10-31 确认）。

**算法**：在分层 DAG 上增加“跳边”：

```text
若 coverage_left[j] <= coverage_right[i] + tol
且 coverage_right[j] > coverage_right[i] + tol
则允许 i -> j，跳过中间层
D[j] = min_i ( D[i] + || c_i - c_j || )
目标：coverage_right[j] >= branch_length - tol
```

复杂度约 `O(n · K · 72²)`，`K` 为 2r 覆盖窗口内的采样点数；在离散层上全局最优。实现还统计同长最优路径条数（上限 2）；若存在多组同长解，只导出确定性的最低候选索引路径，并在 JSON/README 记录 branch 名称。

**角度诊断**：

- `max_angle_step_deg` 是接触点在砂轮端面圆周上的方向 θ 的变化；圆柱轴对称，θ 不是工具轴姿态；
- `max_axis_step_deg` 是实际砂轮轴向 `a` 的夹角，是更有意义的姿态连续性指标。

**无解**：若覆盖约束无法满足（例如该侧完全没有可行候选、或覆盖区间被无可行点截断），整侧标记 `no_path`，写明原因；不伪造路径。

**本轮边界**：

- 只做节点级可行性；**未做边级扫掠碰撞审计**；
- 不改变 72 角、碰撞判定、接触语义、法向 P1；
- 路径长度是 normalized units 的中心折线长度，不是加工时间；
- 同一 branch 的 `+q`/`-q` 独立求解，不连接 branch 间空行程；
- 覆盖模型是一维弧长抽象，不是二维根部面覆盖。

**方法借鉴**：Zhang 2026 有向姿态图；Chen 2015 一维姿态族；Wu 2025 断开可行域；Schneyer 2023 覆盖规划；Lu 2021 路径长度目标。详见 `docs/path_planning_literature_review_20261030.md` 第 3.4/3.5 节。

---

## 5. 参数总表（当前默认）

### 5.1 工具与采样

| 参数 | 值 |
|---|---:|
| `tool_model` | `finite_cylinder` |
| `r` | 0.02 normalized |
| `w` | 0.004 normalized |
| `delta` | 0.001 normalized |
| `axis_mode` | `free_about_curve_tangent` |
| 最终三 OBJ 角度数 | 72 |
| pipeline `candidate_angles` | 180 |
| pipeline `radial_samples` | 3 |

最终三 OBJ 直接使用 `wheel.radius=0.02`，没有调用 `radius_selection` 扫描；半径扫描只属于常规 pipeline/批量逻辑。

### 5.2 法向 P1

| 参数 | 值 |
|---|---:|
| `candidate_faces` | 96 |
| `candidate_radius_ratio` | 1.0 |
| `min_patch_faces` | 10 |
| `min_cluster_support` | 4 |
| `normal_cluster_angle_degrees` | 15 |
| `min_crease_angle_degrees` | 10 |
| `min_side_support_ratio` | 0.35 |
| `min_side_support_faces` | 6 |
| `l0_trim_angle_degrees` | 25 |
| `fit_residual_ratio` | 0.02 |
| `max_fit_condition` | 1e6 |
| `min_hull_area_ratio` | 0.1 |
| `orientation_repair` | true |

### 5.3 碰撞

| 参数 | 值 |
|---|---:|
| `narrow_phase` | `analytic_closest_point_flat` |
| `clearance` | 0 |
| `allow_target_contact` | true |
| `contact_patch_enabled` | true |
| `contact_patch_radius_ratio` | 2.0 |
| `contact_patch_max_penetration_ratio` | 0.025 |
| `contact_patch_normal_tolerance_degrees` | 30 |
| `check_flash_mesh` | false |

---

## 6. 输出语义与文件读取

### 6.1 三个主 OBJ

- `circle_rootline_circle_centers.obj`
  - `l`/根线；`o rootline_circle_centers`；
  - 每个 root 点一个黄色八面体圆心标记（415 个）；
  - 圆心是候选中心的均值，不是每个可行角。
- `circle_inspection_wheels.obj`
  - 每个存在不可行角的点最多一个红色不可行圆柱 + 一个绿色可行圆柱；
  - 已移除蓝色接触点；
  - 材质：`feasible_green`、`collision_red`、`root_line_gray`。
- `circle_feasible_center_domains.obj`
  - 每点 72 个候选中心；
  - 绿色八面体 = 通过当前判定；红色 = 碰撞；
  - 以 `o feasible_centers_point_XXXX` / `collision_centers_point_XXXX` 分组；
  - 不含角度编号；角度需从生成顺序/参数重算或另导数组。

### 6.2 `p1_full_summary.json`

当前 `point_records` 每点包含：

```text
point_index
branch_id
feasible_angle_count
infeasible_angle_count
normal_confidence
normal_source
patch_valid
reason_counts
```

**注意**：历史三 OBJ 交付的 summary **没有逐角度可行角列表、没有每角度中心坐标**。2026-10-31 路径规划交付已新增 `path_candidates.npz` 和 `path_planning_summary.json`，不再依赖 OBJ 反解析。

---

## 7. 路径规划实现与后续

### 7.1 已实现（2026-10-31）

- 分层有向图 + 动态规划；
- 每点 72 个候选中心，节点 = `(point_index, feasible_angle, lateral_side)`；
- 覆盖约束：选中点覆盖 `[s-r, s+r]`，覆盖区间重叠即可跳过中间层；不再要求每层必选；
- 边权 = 中心三维欧氏距离；
- `+q` / `-q` 两侧独立求解；
- branch 内强制 `q` 符号连续，处理退化“往返”分支；
- 无解侧标 `no_path`，写明是覆盖不可满足还是缺失可行点；
- 同长多解只导出一条，记录 branch 名称和最优路径条数（上限 2）；
- 输出两个 OBJ：可行中心域 + 路径；`circle_paths.obj` 使用蓝/橙两色；
- 输出 `path_planning_summary.json` 和 `path_candidates.npz`。

### 7.2 路径规划数据（已导出）

`path_candidates.npz` 包含：

- `point_index`, `branch_id`, `arc_length`;
- `point_norm`, `point_source`;
- `centers_norm`, `centers_source`（415×72×3）；
- `feasible`（415×72）；
- `angles_deg`, `lateral_projection`（branch 内 q 符号连续）；
- `reason`, `tangent`, `axis`, `radial`, `radial_values`;
- 归一化 scale/center。

`path_planning_summary.json` 包含每 branch/侧的路径节点、角度、中心、长度、覆盖区间、覆盖间隙、跳过点、缺失可行点、最大角度/轴向跳变、同长解计数和参数。

### 7.3 当前算法与限制

- 主目标：覆盖整条 branch 的前提下，中心折线总长度最短（normalized units）；
- 覆盖模型是一维弧长抽象，不是砂轮实体切除仿真；
- 若某采样点在该侧完全无可行候选，覆盖区间不能跨过它；
- 不做角度硬上限；报告 `max_angle_step_deg`（端面圆周接触方向 θ）和 `max_axis_step_deg`（实际砂轮轴向 a）；
- **未做边级扫掠碰撞审计**：节点可行不等于相邻中心连线中间姿态可行；
- 侧向一致性只要求中心在指定一侧；砂轮其他部分可以越过；
- 不连接 branch 间空行程；
- 不包含机器人关节/机床可达性、曲率/进给约束。

### 7.4 后续工作

- 边级插值扫掠碰撞审计（报告模式或硬约束模式）；
- 连续角度区间（当前 72 离散）；
- 机器人关节空间/可达性；
- 路径平滑、曲率/进给约束；
- 分支间 TSP/排序；
- 真实机床/工艺参数与实验验证。

---

## 8. 已知限制与已确认决策

### 已确认

- C：允许目标根部 2R 范围浅接触；
- 接触斑不限制目标面；
- 0.0005 是几何容差，不是物理允许嵌入；
- 最终三 OBJ 不要求 `delta` 多层；
- 接受 72 离散角度；
- `side=+1`；
- 只检查 `mesh_without_flash.obj`；
- 不检查 `mesh_with_flash.obj`；
- 允许面相贴，不增加 clearance；
- 已人工确认 `mesh_without_flash.obj` 闭合、无自交、代表不可切入实体；
- 路径两侧定义为 `σ=sign(e·q)`，`θ=0/180` 中线候选排除；
- branch 7 `+q` 无解按方案 A：只输出 `-q`，`+q` 标 `no_path`；
- 同一 branch 多组同长解只导出一条，记录 branch 名称；
- 路径侧一致性只要求砂轮中心在指定一侧；
- 本轮路径规划**不做边级碰撞审计**；
- 本轮只跑 circle；
- 覆盖模型：每个选中点覆盖 root_line 弧长 `[s-r, s+r]`，相邻覆盖区间重叠即可跳过中间点；不强制选首末点；
- 若某采样点在该侧完全无可行候选，覆盖区间不能跨过它；
- 短 branch（长度 < 2r）可能只选 1 个点、路径长度 0。

### 仍需注意

- 72 角度是离散采样，不是连续可行区间；
- 三 OBJ 是 `rho=r` 单层，不是完整 `[r-delta,r]`；
- 路径规划只做节点级可行性，不证明边/扫掠安全；
- 纯长度目标可能选择角度变化较大的组合（例如完整半圆可行时中心折线可短于 root 长度），必须看 `max_angle_step_deg`；`max_angle_step_deg` 是端面圆周接触方向，不是工具轴；工具轴看 `max_axis_step_deg`；
- branch 7 是退化“往返”分支，`+q` 无解，`-q` 可覆盖；
- 覆盖模型是一维弧长抽象，不是砂轮实体与工件表面的精确切除仿真；
- 三 OBJ 交付的 72 角来自 `export_p1_full_delivery.py` 硬编码，不是 config 的 `sampling.candidate_angles=180`；
- `analytic_closest_point_flat` 的名称容易误解：主判定仍是裁剪法；
- 内部包含检查是单射线中心测试，未做多射线/BVH；
- P1 低置信点 396–403、fallback 400 不能当作高置信法向；
- P1 全量 26,919 略低于 baseline 26,929，P1 的选择是 branch 7/特征导向，不是全局可行率提升；
- 结果仍是 preliminary geometry run，不是加工验收。

---

## 9. 仓库结构

```text
FeiLunQieGe/
├─ configs/default.json
├─ docs/
│  ├─ collision_review_20261027.md
│  ├─ path_planning_literature_review_20261030.md
│  └─ path_planning_task_plan_20261030.md
├─ scripts/
│  ├─ run_pipeline.py
│  ├─ run_all_cases.py
│  ├─ run_leaf_stage3_visualization.py
│  ├─ export_p1_full_delivery.py
│  ├─ run_p0_normal_diagnostics.py
│  ├─ run_p1_method_comparison.py
│  ├─ visualize_p1_method_comparison.py
│  ├─ export_p1_three_obj_branch4_7.py
│  ├─ regenerate_p1_point_patches.py
│  └─ audit_*.py
├─ src/feilunqiege/
│  ├─ io/
│  ├─ geometry/
│  │  ├─ frame.py
│  │  └─ normal_p1.py
│  ├─ tooling/grinding_wheel.py
│  ├─ collision/
│  │  ├─ disk_exact.py
│  │  ├─ cylinder_exact.py
│  │  └─ checker.py
│  ├─ planning/
│  │  └─ path_planning.py
│  ├─ visualization/
│  │  └─ path_obj.py
│  └─ pipeline.py
├─ tests/
│  ├─ test_path_planning.py
│  └─ ...
└─ README.md
```

原始论文和外部数据仍在父目录：

```text
D:\0_School\0_Scientific_Research\01\
├─ thesis\
└─ flash_pipeline_12_models_20260924\
```

未复制进 Git 仓库。

---

## 10. 常用命令

### 10.1 路径规划两 OBJ 全量交付（P1，circle）

```powershell
python scripts/export_p1_full_delivery.py --case circle --workers 16 --three-obj-only
```

输出：新的 `results/cases/circle/p1_full_delivery_*`，包含：

- `circle_feasible_center_domains.obj`（绿色可行/红色碰撞中心域）；
- `circle_paths.obj`（`+q` 蓝色、`-q` 橙色路径；只连接选中点，允许跳过采样点；短分支可能只有中心标记、没有线段）；
- `path_planning_summary.json`；
- `path_candidates.npz`；
- `p1_full_summary.json`、`README.md`。

`--three-obj-only` 是历史参数名；本次它表示“只输出主交付 OBJ（两个），不输出逐点 patch OBJ”。圆心位置和砂轮姿态 OBJ 的导出调用已注释，代码保留。

注意：`configs/default.json` 的 `case_name` 当前仍是 `leaf`；circle 必须显式传 `--case circle`，否则会跑 leaf。三 OBJ/路径交付使用脚本内硬编码的 72 角，不是 `sampling.candidate_angles=180`。

### 10.2 P0 法向诊断

```powershell
python scripts/run_p0_normal_diagnostics.py --max-points 8 --skip-collision
python scripts/run_p0_normal_diagnostics.py
```

### 10.3 P1 方法对比

```powershell
python scripts/run_p1_method_comparison.py --normal-metrics-only
python scripts/run_p1_method_comparison.py
python scripts/visualize_p1_method_comparison.py
```

### 10.4 常规 pipeline / 批量

```powershell
python scripts/run_pipeline.py --config configs/default.json
python scripts/run_all_cases.py --config configs/default.json --continue-on-error
```

注意：常规 pipeline 默认 180 角度 × 3 径向层；三 OBJ 交付是 72 角度 × 单层，两者不能直接等同。

### 10.5 测试

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
```

当前：**53 passed**（2026-10-31，新增路径规划测试 9 项）。

---

## 11. Git 与依赖

- 仓库：`https://github.com/SWT-0407/FeiLunQieGe`
- 分支：`main`
- 当前代码基线：`e701368 Add zero-feasible fallback axes and remove contact markers from wheel OBJ`
- 前一条：`f93d15c Fix collision penetration and contact-patch distance, select P1`
- 本轮 README 更新尚未提交；提交前请先 `git status`。
- Python 依赖已安装在 `.venv`：numpy、scipy、matplotlib、pytest；具体版本以 `pip freeze` 为准。
- 结果目录、`.venv`、缓存和原始数据不应强制提交 Git；历史上传只包含代码/文档/配置/测试。

---

## 12. 交接清单（给下一轮）

已完成的路径规划交付请核对：

- [ ] `path_planning_summary.json` 中每 branch/侧的 `status`、`selected_point_count`、`skipped_point_indices`、`coverage_union_ok`、`length_norm`、`max_angle_step_deg`、`max_axis_step_deg`、`missing_feasible_point_indices`；
- [ ] `branch_07` `+q` 为 `no_path`（396–403 在该侧无可行候选，覆盖区间不能跨过），`-q` 可覆盖；
- [ ] 长分支只选 6–11 个中心；短分支每侧可能只选 1 个点、长度 0；
- [ ] 同长多解（短分支每侧 2 条）只导出一条，README/JSON 记录 branch 名称；
- [ ] `circle_paths.obj` 的 root_line 和路径中心是 `source_obj`，可与 `mesh_without_flash.obj` 叠加；
- [ ] `circle_feasible_center_domains.obj` 与历史交付的绿色/红色语义一致；
- [ ] 圆心位置、砂轮姿态两个 OBJ 的调用已注释，但函数和依赖保留；
- [ ] 72 角来自脚本硬编码，不是 config 的 180；
- [ ] 本轮**未做边级碰撞审计**，报告必须保留这一限制；
- [ ] 覆盖模型是一维弧长抽象，不是砂轮实体切除仿真。

下一轮若继续：

- [ ] 边级插值扫掠碰撞审计（先报告模式，后考虑硬约束）；
- [ ] 连续角度区间；
- [ ] 机器人关节/机床可达性；
- [ ] 分支间 TSP/排序；
- [ ] 12 案例推广需单独授权，不能默认批量。
