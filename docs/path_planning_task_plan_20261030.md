# 路径规划任务理解、数据核对与执行方案（2026-10-30）

**状态**：只读调研与方案设计；**尚未修改任何源码、配置、正式结果或原始数据**。  
**目标**：在现有 72 个离散候选角度的基础上，为 root_line 两侧各求一条“每个采样点选一个可行砂轮中心、相邻中心连线、总长度尽量短”的路径，并交付两个 OBJ。  
**前置文献调研**：见 `docs/path_planning_literature_review_20261030.md`。

---

## 1. 任务复述

### 1.1 输入与不变量

- 输入仍为 `root_line.ply`，按 PLY edge 连通关系拆分为多个 branch；当前 `circle` 案例为 415 个采样点、9 个 branch。
- 保留现有 **72 个离散候选角度**（`0°, 5°, ..., 355°`），不改成连续角度，不改变每点的候选中心生成公式。
- 保留现有碰撞筛选结果与接触语义（C：允许目标根部 2R 范围浅接触；仅检查 `mesh_without_flash.obj`；`side=+1`；`rho=r` 单层）。
- 每个 root 采样点已经有 72 个候选中心，其中绿色为通过当前判定、红色为碰撞。

### 1.2 路径定义（2026-10-31 已确认，branch 7 例外见 Q2）

对每一条 root_line branch：

1. 在 root_line 的**左侧**和**右侧**各选一条路径；
2. 每条路径在每个采样点 `x_i` 上选择一个**可行**候选中心 `c_{i,j}`；
3. 相邻采样点的中心用直线段连接；
4. 路径总长度为各段三维欧氏距离之和；
5. 目标是在所有满足条件的组合中，使该侧路径总长度最小；
6. 如果某一侧在某点没有可行候选，则该侧“无解”，必须显式标记，不能静默跳过或伪造。

### 1.3 交付物

1. **OBJ A（保留现有逻辑）**：砂轮中心点的离散可视化，绿色 = 可行、红色 = 碰撞，即当前 `circle_feasible_center_domains.obj`。
2. **OBJ B（新增）**：算法求解出的路径可视化，用两种明显不同的颜色区分 root_line 两侧，并保留 root_line 作为参照。
3. 代码中现有的另外两个 OBJ 交付逻辑：
   - `circle_rootline_circle_centers.obj`（圆心位置）；
   - `circle_inspection_wheels.obj`（砂轮姿态可视化）。
   
   按要求**保留代码、以注释形式停用调用**，并加显著文字标记“不要删除，以备后续复现”。函数定义本身保留，便于恢复。
4. 文献调研 Markdown：已完成，见 `docs/path_planning_literature_review_20261030.md`。
5. 建议同时输出 `path_planning_summary.json`（路径节点、角度、长度、无解 branch/侧、角度跳变等），便于复核；不新增 PNG，除非用户明确要求。

---

## 2. 当前仓库与数据事实（只读核对）

### 2.1 当前三 OBJ 交付

目录：

```text
results/cases/circle/p1_full_delivery_20261029_233114_679352/
```

- 415 个 root 点；
- 每点 72 角，共 29,880 个候选；
- 可行 26,919 / 29,880；
- 完全无解点 0；
- 57 个点存在部分不可行角；
- 点 402 只有 1 个可行角（branch 7，约 270.2°）；
- 点 400 使用 fallback axis，5/72 可行；
- 低置信点 396–403（branch 7）。

### 2.2 候选中心与“两侧”的定义

当前代码：

```text
e(θ) = cosθ · t + sinθ · q
q = normalize(a × t)
c = x + r · e(θ) + side · (w/2) · a
```

其中：

- `t` 是投影到切平面内的 root_line 切线；
- `a` 是工件表面法向（砂轮轴向）；
- `side = +1` 是**砂轮轴向侧**，决定有限圆柱中心沿 `a` 偏移 `w/2`；
- `q` 是切平面内垂直于 `t` 的横向方向；
- `e(θ)` 在切平面内绕 root 点旋转。

因此，**路径的“左右两侧”应定义为**：

```text
σ = sign(e(θ) · q)
```

- `σ = +1`：`+q` 侧；
- `σ = -1`：`-q` 侧；
- `e(θ) · q = 0`（θ = 0° 或 180°）时，中心沿切线方向，**不属于任何一侧**。

需要特别区分：代码中的 `side=+1` 是砂轮轴向侧，不是路径左右侧。建议在代码和文档中把路径侧改名为 `lateral_side` 或 `path_side`，避免混淆。

### 2.3 只读探索结果（基于现有 OBJ，不是正式结果）

为确认任务是否可行，我用现有 `circle_feasible_center_domains.obj` 恢复了每点 72 个中心，按上述 `σ` 分类，并做了分层最短路径探索。结果如下（normalized units）：

| branch | 点数 | +q 可行缺失点 | -q 可行缺失点 | +q 最短长度 | -q 最短长度 |
|---:|---:|---:|---:|---:|---:|
| 0 | 81 | 0 | 0 | 0.196895 | 0.196892 |
| 1 | 121 | 0 | 0 | 0.310969 | 0.310964 |
| 2 | 153 | 0 | 0 | 0.402485 | 0.402487 |
| 3 | 10 | 0 | 0 | 0.014618 | 0.014528 |
| 4 | 10 | 0 | 0 | 0.018377 | 0.018377 |
| 5 | 10 | 0 | 0 | 0.010910 | 0.012452 |
| 6 | 10 | 0 | 0 | 0.021052 | 0.021036 |
| 7 | 10 | **8（396–403）** | 0 | **无解** | 0.045918 |
| 8 | 10 | 0 | 0 | 0.008153 | 0.008152 |

关键事实：

1. **branch 7 的 +q 侧无解**：点 396–403 的全部可行角都落在 `-q` 侧（角度约 250°–290°）；点 395 和 404 两侧都有 72/72 可行，但中间 8 个点没有 +q 候选。因此严格按“两侧各一条”在 branch 7 上不可能同时成立。
2. 点 402 只有 1 个可行角（约 270.2°），是 branch 7 `-q` 路径的必经瓶颈。
3. 除 branch 7 外，其余 8 个 branch 的两侧都能找到路径。
4. 纯长度最优解的相邻角度跳变大多 ≤ 5°（72 角网格步长），但 branch 5 `-q` 出现一次 **20.52°** 跳变；如果加 10° 硬上限，branch 5 `-q` 会无解；加 20° 上限则仍可解，长度从 0.012452 变为 0.014203（约 +14%）。
5. 两侧路径长度几乎相等，这是因为中心都在 root 点半径 `r` 的圆上；最短路径倾向于贴着 `±q` 方向走，形成两条近似偏移曲线。

> 这些数值来自现有 OBJ 的只读恢复，仅用于判断可行性；正式结果必须在实现后重新计算并由人工复核。

### 2.4 关于“72 个角度”与配置中的 180（Codex 指出的差异）

Codex 指出：`configs/default.json` 中 `sampling.candidate_angles` 是 **180**，`radius_selection.scan_angles` 才是 **72**。核对结果：

- `configs/default.json:32`：`"candidate_angles": 180`；
- `configs/default.json:26`：`"scan_angles": 72`，它属于 `radius_selection`，只用于半径扫描；
- 常规 `pipeline.py` 读取 `sampling.candidate_angles`，即 180（`pipeline.py:142`、`pipeline.py:298`）；
- 但当前三 OBJ 交付脚本 `scripts/export_p1_full_delivery.py` **硬编码**了 72：
  - `scripts/export_p1_full_delivery.py:77`：`angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)`；
  - `scripts/export_p1_full_delivery.py:455`：同一行；
- `scripts/run_leaf_stage3_visualization.py` 的 `--angle-count` 默认也是 72（`run_leaf_stage3_visualization.py:112`）。

因此，当前 `p1_full_delivery_20261029_233114_679352` 的 72 角来自导出脚本硬编码，不是 `configs/default.json` 的 180。本次路径规划必须：

1. 沿用**同一个 72 角数组**，不读取 `sampling.candidate_angles=180`；
2. 在 JSON 中记录 `angle_count=72` 和 `angle_count_source="export_p1_full_delivery.py hardcoded 72"`；
3. 不把 72 角结果与常规 pipeline 的 180 角结果混在同一组统计中；
4. 如后续要让角度数可配置，新增显式 `--angle-count`（默认 72）并写回 JSON，但不能静默改变当前交付。

Codex 的另一个判断“节点可行不等于连接段安全”是正确的，见 3.6 与 Q6 的边级碰撞审计说明。

---

## 3. 推荐算法

### 3.1 图结构

对每个 branch 独立建图：

- 层：`i = 0..n-1`，对应 branch 内第 `i` 个 root 采样点；
- 节点：`(i, j, σ)`，其中第 `j` 个候选可行，且 `σ = sign(e_ij · q_i)` 等于目标侧；
- 边：只连接相邻层 `i` 与 `i+1` 的节点；
- 边权：

```text
w((i,j), (i+1,k)) = || c_{i,j} - c_{i+1,k} ||
```

- 目标：从第 0 层到第 `n-1` 层的最小总权路径。

### 3.2 动态规划

```text
D_0(j) = 0
D_{i+1}(k) = min_j [ D_i(j) + || c_{i,j} - c_{i+1,k} || ]
```

回溯得到每层选中的候选索引。复杂度 `O(n · 72²)`；`circle` 全量约 2.1M 次距离计算，远小于碰撞筛选本身。

因为图是分层 DAG，DP 是**全局最优**，不需要 Dijkstra/A*；如果后续加入跨层边、角度相关代价或边级碰撞硬约束，再考虑 Dijkstra 或更一般的图搜索。

### 3.3 无解处理

- 若某侧在某一层没有节点：该侧状态为 `no_path`，记录所有缺失点编号；
- 不跳过该点，不用另一侧候选补位，不用 fallback 伪造；
- 在 OBJ 头部、JSON 和 README 中显式写明；
- 另一侧仍正常输出。

### 3.4 诊断指标

每条路径至少输出：

- `branch_id`, `lateral_side`；
- `status`（`ok` / `no_path`）；
- `point_count`；
- `path_length_norm`, `path_length_source`；
- `max_segment_length_norm`；
- `max_angle_step_deg`, `mean_angle_step_deg`；
- 每节点：`point_index`, `candidate_index`, `angle_deg`, `center_norm`, `center_source`, `reason`；
- 若 `no_path`：`missing_point_indices`。

### 3.5 可选：角度跳变上限

用户当前要求“总长度最短”，因此**主结果不加角度上限**。但建议同时报告：

- 纯长度解的 `max_angle_step_deg`；
- 若加 20° 上限的对照结果（branch 5 `-q` 仍可解，长度略增）；
- 明确说明 10° 上限会使 branch 5 `-q` 无解。

是否把角度上限作为硬约束，需要用户确认。

### 3.6 可选：边级碰撞审计（“边级碰撞审计”的完整定义）

#### 3.6.1 它检查什么

当前 72 角筛选是**节点级**的：对每个 `(root 点 i, 角度 j)`，单独构造一个砂轮圆柱，检查它是否与工件三角网格相交。绿色表示这个**静止姿态**可行。

路径是 `c_{i,j} -> c_{i+1,k}` 的连线。砂轮从第一个中心移动到第二个中心时，姿态也从 `(a_i, e_i, t_i)` 连续变化到 `(a_{i+1}, e_{i+1}, t_{i+1})`。**两个端点都可行，不代表中间过程可行**：砂轮可能在两个采样点之间扫过工件的凸棱、薄壁或凹槽。边级碰撞审计就是检查这条“移动过程”上的中间姿态，而不是只检查两个端点。

它检查的是：

- 砂轮圆柱在边上的中间位置是否与工件本体相交；
- 中间姿态的接触斑语义是否仍然满足当前 C 规则；
- 路径是否在边上穿到 root_line 的另一侧（侧向一致性）；
- 若发生碰撞，记录发生在哪个 `t`、哪个三角形、穿透深度/原因。

它**不**检查：

- 机器人手臂、刀柄、夹具、机床；
- 真实伺服/关节插补轨迹；
- 飞边网格（当前仍只检查 `mesh_without_flash.obj`）；
- 材料去除/力/热；
- 连续扫掠体的严格数学证明（下面会说明它仍是采样）。

#### 3.6.2 具体怎么插值

对一条边 `(i,j) -> (i+1,k)`，令 `t ∈ [0,1]`。端点已有 `t_i, q_i, a_i, θ_i` 和 `t_{i+1}, q_{i+1}, a_{i+1}, θ_{i+1}`，其中 `q=a×t`、`e=cosθ·t+sinθ·q`。推荐插值：

```text
center(t)  = (1-t) c_i + t c_{i+1}
axis(t)    = normalize((1-t) a_i + t a_{i+1})
t_raw(t)   = (1-t) t_i + t t_{i+1}
tangent(t) = normalize(t_raw - dot(t_raw, axis(t)) * axis(t))
q(t)       = normalize(cross(axis(t), tangent(t)))
theta(t)   = theta_i + t * wrap(theta_{i+1} - theta_i)   # 最短角路径
radial(t)  = cos(theta(t)) * tangent(t) + sin(theta(t)) * q(t)
contact(t) = center(t) - r * radial(t) - side * (w/2) * axis(t)
```

最后一行是“由中心反算接触点”，保证圆柱中心、轴向、半径、宽度和接触点在同一姿态下自洽；若直接线性插值 `x_i,x_{i+1}`，中间接触点可能不在该圆柱的近端面上。

- `center` 线性插值；
- `axis` 线性插值后归一化；
- `tangent` 先线性插值，再投影到 `axis(t)` 的垂面并归一化，保证 `a·t≈0`；
- `q` 由 `a×t` 重建，保证与 `frame.radial_basis_2` 一致；
- `theta` 按最短角路径插值，避免 0°/360° 绕行；
- `radial` 用插值后的 `θ, t, q` 重建，保证 `a·e≈0`、`||e||=1`；
- 不要直接对 `e_i, e_{i+1}` 线性插值再归一化：当两端角度相差较大或接近中线时，直接插值可能偏离圆弧；
- `contact` 线性插值，或直接用 root_line 在该段的采样点；
- `radius` 仍为 `r`，`width` 仍为 `w`，`side=+1` 不变。

然后把每个中间姿态打包成 `CandidateSet`，调用现有的 `check_finite_cylinder_candidates_exact(...)`，与端点使用完全相同的窄相/宽相/接触斑参数。

#### 3.6.3 采样密度

- 默认每条边取 `N=8` 个内部采样（`t = 1/9, ..., 8/9`；端点已单独检查）；
- 对长边或角度变化大的边自适应加密：例如要求相邻采样点的中心位移 ≤ `r/4`，轴转角 ≤ 2°；
- 记录每条边实际使用的采样数，避免“看起来检查过”但密度不足；
- 这是一次**离散采样审计**，不是连续扫掠体的严格证明。若两个采样点之间有一个很薄的三角形恰好穿过，仍可能漏检。要严格化，需要连续扫掠体/保守包围体（Ezair & Elber 2018 的方向），本轮不做。

#### 3.6.4 输出

每条边记录：

```text
branch_id, lateral_side
edge_index, point_i, point_i+1
candidate_i, candidate_i+1
t_samples
collision_free: true/false
worst_reason: safe / contact_patch_only / target_contact_only / triangle_interior_collision / cylinder_cap_collision / ...
max_penetration
witness_t, witness_triangle
side_consistent: true/false
```

在 JSON 中写 `edge_audit` 段；在 OBJ 头部写摘要（例如 `# edge_audit: 3 unsafe edges`）。如果某条边不安全，可以在路径 OBJ 中用第三种颜色（例如黄色）标出，或在 JSON 中列出，避免误读。

#### 3.6.5 两种使用方式

- **报告模式（推荐本轮）**：先按节点级 DP 求最短路径，再审计每条选中边；不改变路径。若发现不安全边，JSON/OBJ 显式标记，等人工决定。
- **硬约束模式**：把审计不通过的边从图中删除，再重新 DP；若某侧因此无路，标记 `no_path`。这会更保守，但可能让某些侧（尤其 branch 5/7）无解，且计算量更大。

#### 3.6.6 为什么本轮建议“报告模式”

- 用户当前目标是“在 72 个离散点上使总长度最短”，节点级 DP 是严格对应这个目标的；
- 边级审计若作为硬约束，会改变图结构，可能使 branch 5 `-q` 或 branch 7 `-q` 无解，需要用户单独决定；
- 报告模式能先量化“最短路径有多少条边不安全”，再决定是否值得牺牲长度；
- 它符合项目规范“不能把离散采样通过直接称为精确无碰撞”的要求。

#### 3.6.7 成本

- 每条边 `N=8` 个采样；`circle` 两侧合计约 `2 × (415 - 9) = 812` 条边，约 6,500 次圆柱检查；
- 相比节点筛选的 29,880 次，边审计约增加 22% 的检查量；
- 若自适应加密到 `N=16`，约 13,000 次，仍可接受；
- 可用多进程按 branch 并行；结果写入 JSON，不影响主路径。

是否采用报告模式、硬约束模式，或本轮不做，需要用户确认（见 Q6）。

### 3.7 路径侧一致性检查（低成本）

对每条选中边，检查插值中心的横向投影 `e(t)·q(t)` 是否始终同号；如果某条边穿越 root_line 或换侧，显式标记。该检查不替代碰撞检查。

---

## 4. 可视化设计

### 4.1 OBJ A：可行中心域（保留）

沿用现有 `circle_feasible_center_domains.obj`：

- root_line 灰色；
- 每点 72 个候选中心；
- 绿色八面体 = 可行；
- 红色八面体 = 碰撞；
- 文件头 `coordinate_frame: source_obj`。

不改颜色、不改采样、不改碰撞判定。

### 4.2 OBJ B：路径可视化（新增）

建议文件名：`circle_paths.obj` + `circle_paths.mtl`。

内容：

- root_line：灰色（复用 `_write_root_line`）；
- `+q` 路径：蓝色折线（`l` 元素）+ 每个选中中心一个小蓝色八面体；
- `-q` 路径：橙色折线 + 小橙色八面体；
- 无解的 branch/侧：不画伪造路径；在 OBJ 头部写注释，例如：
  ```text
  # branch 7 lateral_side=+1: NO PATH (points 396-403 have no feasible +q candidate)
  ```
- 颜色与 OBJ A 的绿/红明显区分，避免混淆；
- 所有坐标 `source_obj`，可直接与 `mesh_without_flash.obj` 叠加。

可选：在路径起点/终点加稍大的标记，或在 JSON 中记录起止点编号。

### 4.3 代码中保留但停用的两个 OBJ

在 `scripts/export_p1_full_delivery.py` 的导出调用处：

```python
# ============================================================================
# 【保留，不要删除】以下两个 OBJ 交付逻辑按 2026-10-30 路径规划任务要求
# 暂停输出，但代码必须保留，以备后续复现：
#   1) 圆心位置 OBJ：circle_rootline_circle_centers.obj
#   2) 砂轮姿态 OBJ：circle_inspection_wheels.obj
# 恢复时取消下面调用的注释即可；不要删除函数定义。
# ============================================================================
# _export_circle_centers(...)
# _export_inspection_wheels(...)
```

函数 `_export_circle_centers`、`_export_inspection_wheels` 及其依赖保留，不删除。

---

## 5. 代码改动清单（待批准后执行）

| 文件 | 改动 | 说明 |
|---|---|---|
| `src/feilunqiege/planning/__init__.py` | 新增 | 路径规划包 |
| `src/feilunqiege/planning/path_planning.py` | 新增 | 侧向分类、分层 DP、无解记录、诊断指标；可选写出 `path_candidates.npz`（每点 72 个中心、可行标志、角度、轴向/切向/径向），便于复现 |
| `src/feilunqiege/planning/edge_audit.py` | 新增（若 Q6 选择报告/硬约束） | 边级插值姿态、采样、调用现有有限圆柱 checker；输出逐边审计记录 |
| `src/feilunqiege/visualization/path_obj.py` | 新增 | 路径 OBJ/MTL 导出；复用 root_line/八面体写函数或独立实现 |
| `scripts/export_p1_full_delivery.py` | 修改 | 调用路径规划；导出 OBJ A + OBJ B；停用圆心/砂轮 OBJ 调用（注释 + 标记）；扩展 JSON/README。保留 `--three-obj-only` 参数名以兼容旧命令，但帮助文字改为“只输出主交付 OBJ（本次为 2 个），不输出逐点 patch OBJ” |
| `tests/test_path_planning.py` | 新增 | 合成直线/圆弧、单侧无解、断开可行集、角度上限、DP 最优性 |
| `tests/test_edge_audit.py` | 新增（若 Q6） | 合成直边/转角边、中间姿态碰撞、插值正交性 |
| `README.md` | 修改 | 更新当前交付、路径算法、branch 7 例外、限制 |
| `docs/path_planning_literature_review_20261030.md` | 已新增 | 文献调研 |
| `docs/path_planning_task_plan_20261030.md` | 已新增 | 本文件 |

不改动：

- `root_line.ply`、`mesh_without_flash.obj`、`original_flash_cloud.obj`；
- 碰撞判定、法向 P1、候选中心生成、72 角网格；
- 现有正式结果目录；
- `side=+1`、`rho=r`、C 接触语义。

---

## 6. 测试与验收

### 6.1 单元测试

1. 合成直线 root_line，两侧都有连续可行角：DP 路径长度 ≈ 直线长度，节点数 = 采样点数。
2. 合成对称几何：两侧路径长度相等（数值容差内）。
3. 某层只有一侧可行：另一侧 `no_path`，缺失点编号正确。
4. 可行角不连续（中间有碰撞角）：DP 仍能选到可行角；若整层无可行角，`no_path`。
5. 贪心最近邻会失败、DP 能成功的构造案例：验证 DP 全局最优。
6. 角度上限：10° 上限使某些构造无解，20° 上限可解；结果与纯长度不同。
7. 侧向分类：θ=0°/180° 被排除；`+q`/`-q` 计数正确。
8. 无解时不伪造节点，不静默丢点。

### 6.2 circle 全量验收

1. 运行修改后的 `scripts/export_p1_full_delivery.py --case circle --workers 16 --three-obj-only`（或新参数）；
2. 输出目录必须是新的时间戳目录，不覆盖旧结果；
3. 检查两个 OBJ + MTL + JSON + README；
4. 校验 OBJ A 的 root_line 与输入 `root_line.ply` 的 `source_obj` 坐标一致；
5. 校验 OBJ B 的 root_line 和路径中心坐标一致；
6. 在 MeshLab 中与 `mesh_without_flash.obj` 叠加，人工确认：
   - 绿色/红色中心域与旧交付一致；
   - 蓝色/橙色路径位于 root_line 两侧；
   - branch 7 只显示 `-q` 路径，OBJ 头部有 `NO PATH` 说明；
   - 路径不出现明显穿越工件本体的段（若做了边级审计，查看审计结果）；
7. 检查 JSON：每 branch/侧的长度、节点数、最大角度跳变、无解点；
8. 运行相关 pytest；记录通过数。

### 6.3 明确不声称的内容

- 不声称路径已通过真实加工验证；
- 不声称 72 离散角等价于连续可行域；
- 不声称节点级可行等价于整条路径无碰撞（除非边级审计通过）；
- 不声称 normalized units 的路径长度是毫米/秒；
- 不声称 branch 7 的 `+q` 无解是“工件不可加工”，只说明当前 72 角/当前姿态模型下无可行节点。

---

## 7. 需要用户确认的事项

### 7.0 确认状态（2026-10-31，按回复内容对应）

| 编号 | 事项 | 用户回复 | 记录 |
|---|---|---|---|
| Q1 | 两侧定义；排除 θ=0°/180° | “是” | 采用 `σ=sign(e·q)`；`+q`/`-q` 保持数学符号，不强行称“左/右”；中线候选排除 |
| Q2 | branch 7 `+q` 无解 | “A” | branch 7 只输出 `-q`；`+q` 标 `no_path`，列出 396–403 |
| Q3 | 每个 root 点选一个中心 | “是” | 不跳过、不多选 |
| Q4 | 按 branch 独立 | “是” | 不连接 branch 间空行程 |
| Q5 | 目标函数/角度上限 | 未明确 | 暂按“纯长度 + 报告角度跳变，不加硬上限”执行；如不对请纠正 |
| Q6 | 边级碰撞审计 | “本轮先不做” | 本轮只做节点级可行性；未做边级审计；已列入后续工作 |
| Q7 | 两个 OBJ 形式 | “同意” | 蓝/橙路径 + 节点标记；无解侧不画 |
| Q8 | 输出 JSON | “输出” | 输出 `path_planning_summary.json`；可选 `path_candidates.npz` |
| Q9 | 保留旧 summary 字段 | 未明确 | 暂按“保留旧字段 + 新增 `path_planning` 段”执行 |
| Q10 | 只跑 circle | “是” | 本轮只跑 circle，不批量 |

> Q5、Q9 没有被单独点名；我按推荐默认继续，若与你的意图不同，请在下一轮指出。Q6 见下方解释与 3.6 节。

### Q1. “两侧”的定义

建议：`σ = sign(e(θ)·q)`，`q = normalize(a×t)`；`+q` 为左/右中的一侧，`-q` 为另一侧；`θ=0°/180°` 的中线候选排除。

请确认：
- 是否采用这个定义？
- 是否允许把 `θ=0°/180°` 的中线候选排除？（当前 72 角包含它们）
- 是否需要把 `+q`/`-q` 命名为“左/右”，还是保持数学符号？

### Q2. branch 7 的 `+q` 无解

数据事实：点 396–403 只有 `-q` 可行；点 395、404 两侧可行。严格“两侧各一条”在 branch 7 上不可能。

可选方案：
- **A（推荐）**：branch 7 只输出 `-q` 路径；`+q` 标记 `no_path`，列出缺失点；JSON/OBJ 明确说明。
- B：允许 branch 7 的 `+q` 路径在 396–403 借用 `-q` 候选（会变成“同一侧”路径，语义不纯）。
- C：把 branch 7 单独作为“单侧可加工段”，不纳入两侧路径统计。
- D：重新定义侧向（例如按飞边两侧的面法向而非 `q`），需要重新建模。

请确认采用哪一种。

### Q3. 每个 root 点是否必须选一个中心

建议：是；不允许跳过采样点，不允许一个点选多个中心。这样路径覆盖全部 root 点，长度定义清晰。

请确认。

### Q4. 是否按 branch 独立输出路径

建议：是。9 个 branch 是独立曲线，不把 branch 之间用空行程连接；若需要全局多分支顺序，属于后续 TSP/路由问题。

请确认：本轮是否只做 branch 内路径？

### Q5. 目标函数

建议：只用中心点三维欧氏距离之和，严格按“总长度最短”。同时报告相邻角度跳变；不把角度平滑项混入主目标。

请确认：
- 是否需要加角度跳变上限？若加，建议 20° 对照（10° 会使 branch 5 `-q` 无解）。
- 是否需要把姿态/轴向变化作为第二目标？

### Q6. 边级碰撞审计

**定义见 3.6**：检查砂轮从一个选中中心移动到下一个选中中心时，中间姿态是否与工件相交；不是只检查两个端点。

建议：主结果按节点级 DP；另做边级插值碰撞审计并写入 JSON，不静默改变主结果。若审计发现碰撞，标记该边，等待人工决定是否重规划。

请确认三选一：
- **报告模式（推荐）**：主路径不变；边审计结果写入 JSON/OBJ，不安全边标黄或列出；
- **硬约束模式**：删掉不安全边后重新 DP；可能使某些侧无解；
- **本轮不做**：只输出节点级路径，并在文档中明确“未做边级审计”。

### Q7. 两个 OBJ 的具体形式

建议：
- OBJ A：现有绿色/红色中心域，不变；
- OBJ B：root_line 灰色 + `+q` 蓝色折线/节点 + `-q` 橙色折线/节点；
- 无解侧不画伪造路径，只在头部/JSON 说明。

请确认颜色和是否需要在路径节点上放标记。

### Q8. 是否输出 `path_planning_summary.json`

建议：是。包含每 branch/侧的路径节点、角度、中心、长度、无解点、角度跳变、参数和坐标框架。

请确认。

### Q9. 是否保留 `p1_full_summary.json` 的现有字段

建议：保留现有字段，新增 `path_planning` 段；不删除旧字段，避免破坏历史对照。

请确认。

### Q10. 运行范围

建议：先只跑 `circle`，人工复核两个 OBJ 后再推广到 12 个案例。批量阶段需要单独授权。

请确认。

---

## 8. 建议的执行顺序（待确认后）

1. **实现纯函数 + 单元测试**：不接 OBJ、不跑全量；先验证 DP、侧向分类、无解处理。
2. **接入导出脚本**：OBJ A + OBJ B；停用圆心/砂轮 OBJ 调用（注释 + 标记）；扩展 JSON/README。
3. **circle 全量运行**：新时间戳目录；生成两个 OBJ + MTL + JSON。
4. **自动检查**：坐标对齐、节点数、路径长度、无解点、角度跳变；运行 pytest。
5. **人工 MeshLab 复核**：与 `mesh_without_flash.obj` 叠加；确认两侧路径、branch 7 例外、颜色区分。
6. **更新 README**：记录本次交付、算法、限制、未验证事项。
7. **停止，等待人工复核**；不自动推广到 12 个案例。

---

## 9. 当前不做的事

- 不删除任何文件；
- 不覆盖旧结果目录；
- 不修改原始 OBJ/PLY；
- 不把 72 角改成连续角；
- 不改变碰撞判定/接触语义/法向 P1；
- 不引入机器人运动学、力控、材料去除模型；
- 不把几何路径长度解释为加工时间；
- 不把 branch 7 的 `+q` 无解解释为工件不可加工。

---

## 10. 实现记录（2026-10-31）

### 10.1 已实现文件

- `src/feilunqiege/planning/__init__.py`
- `src/feilunqiege/planning/path_planning.py`：侧向分类、branch 内 q 符号连续、分层 DP、无解记录、同长解计数（上限 2）。
- `src/feilunqiege/visualization/path_obj.py`：root_line + 蓝/橙路径 + 节点标记的 OBJ/MTL。
- `scripts/export_p1_full_delivery.py`：接入路径规划；只导出两个主 OBJ；圆心/砂轮 OBJ 调用已注释并标“不要删除”；输出 `path_planning_summary.json`、`path_candidates.npz`。
- `tests/test_path_planning.py`：9 项测试，覆盖侧向、中线排除、DP 最优性、无解、同长多解、branch 标签、退化往返分支。

### 10.2 重要修复：branch 内 q 符号连续

首轮全量运行发现 branch 7 的 `+q` 和 `-q` 都 `no_path`，缺失点分别为 396–399 和 400–403。检查 `path_candidates.npz` 后发现：

- branch 7 的空间点 395–399 与 404–400 沿同一条线往返；点 399 和 400 坐标相同；
- 因此切线在 400 处翻转约 180°，逐点独立取 `q=normalize(a×t)` 会让同一物理侧向在 400 处翻到另一侧；
- 修复：在 branch 内比较相邻 `q`，若 `q_i·q_{i-1}<0` 则翻转 `q_i`，再计算 `lateral_side=sign(e·q)`；
- 修复后 branch 7 `+q` 仍无解（396–403 只有 -q 可行），`-q` 覆盖全部 10 点；符合 Q2 方案 A。
- 该修复也改变了 branch 3/5 的路径长度和角度跳变（从 85°/95° 降到 5°/20°），说明这些短 branch 也存在局部标架翻转。
- 进一步检查发现 **branch 3–8 全部是 10 点短分支，且相邻切线在中间某点反向**（最小 `t_i·t_{i-1}` 约 `-1`）；也就是说它们是“去程-回程”型退化曲线。q 符号连续不是 branch 7 专用，而是所有短分支的必要处理。路径按全部 10 个采样点计算，**不去重**；可视化中这些短分支会表现为沿同一空间线去回一次。

### 10.3 当前结果（circle，全采样点 baseline）

> 2026-10-31 第二轮已加入覆盖约束、允许跳过采样点；本节是“每点必选”的 baseline，最终结果见第 11 节和新运行目录。baseline 目录 `p1_full_delivery_20261031_153521_498629/` 保留作为对照。

- 415 点、9 branch、72 角；可行 26,919 / 29,880；
- 路径候选：17 条 `ok`、1 条 `no_path`；
- `branch_07 +q`：`no_path`，缺失 396–403；
- `branch_07 -q`：`ok`，长度约 0.040371（normalized），最大角度跳变 5°；
- `branch_00 +q` 有 2 条同长最优路径，按最低候选索引导出其中一条；
- 其余 branch 两侧均 `ok`；
- 路径长度范围：0.007168（branch 8）～0.353864（branch 2）；
- 本轮**未做边级碰撞审计**。

正式交付目录：

```text
results/cases/circle/p1_full_delivery_20261031_153521_498629/
```

同目录 `README.md`、`path_planning_summary.json` 和 `p1_full_summary.json` 的数值以实际运行为准。首轮分支 7 被拆成两段的中间运行目录 `p1_full_delivery_20261031_150957_690769/` 保留，不删除；它不代表最终结果。

### 10.4 测试与检查

- `python -m pytest -q`：**53 passed**；
- `python -m compileall -q src scripts tests`：通过；
- 路径 OBJ 的 root_line 与候选中心使用 `source_obj`；`path_candidates.npz` 同时保存 normalized/source 坐标；
- 自动校验：root_line 与 `point_source` 最大误差 `4.96e-12`；17 条 ok 路径的 OBJ 折线与 JSON `center_source` 最大误差 `5.02e-12`；`branch_07 +q` 无路径对象且头部有 `NO PATH` 注释；MTL 材质齐全；
- 未覆盖旧结果目录；首轮 branch 7 分离的中间运行目录保留，不删除。

### 10.5 仍需人工确认

- 在 MeshLab 中叠加 `circle_paths.obj` 与 `mesh_without_flash.obj`，确认蓝/橙路径在 root_line 两侧、branch 7 只有橙色路径；
- 确认 `circle_feasible_center_domains.obj` 与历史绿色/红色语义一致；
- 确认 branch 3/5 的路径形态和角度跳变可接受；
- 确认同长多解只导出一条符合预期；
- 确认“未做边级碰撞审计”的限制在报告中保留。

---

## 11. 覆盖约束路径规划（2026-10-31 第二轮）

### 11.1 用户确认

1. 覆盖范围按半径 `r` 的切割区域，A（端面圆盘投影/root 点落入圆盘）；
2. 连续弧长覆盖，A；
3. 严格重叠、数值容差，A；
4. 被覆盖点也要通过碰撞检查，A；沿用现有全域碰撞结果；
5. 目标只最小化选中中心折线长度，不罚选中点数量；
6. 不一定选首末点，只要覆盖 root_line 起止；
7. 两侧独立覆盖，A；
8. 本轮不做边级碰撞审计，A；交付时提醒；
9. 可视化沿用当前方式，不需要额外 baseline/标记；
10. 可以新增覆盖区间字段。

### 11.2 本次实现

- 新增 `plan_side_path_coverage`：
  - 每个选中点覆盖 `[s_i-r, s_i+r]`；
  - 相邻选中点覆盖区间重叠即可跳过中间点；
  - 起点/终点只需被覆盖，不强制选首末点；
  - 目标为覆盖整条 branch 的最短中心折线。
- 保守覆盖限制：若某采样点在该侧没有可行候选（现有全域碰撞结果），覆盖区间不能跨过它，边界取在它与相邻可覆盖点的中点。这样不会用半径 r 跨过整段碰撞点。
- 新增 `max_axis_step_deg`：实际砂轮轴向 `a` 的夹角；`max_angle_step_deg` 是轴对砂轮端面圆周接触方向 θ 的变化，圆柱轴对称，θ 不是工具轴姿态。
- `path_planning_summary.json` 新增：
  - `selected_point_count`, `skipped_point_indices`;
  - `coverage_start_norm`, `coverage_end_norm`, `coverage_gaps_norm`, `coverage_union_ok`;
  - `missing_feasible_point_indices`;
  - `max_axis_step_deg`。
- `p1_full_summary.json` 的 `path_planning` 段同步记录 `allow_skips`、`coverage_half_length`、`coverage_model`、`coverage_restriction`。

### 11.3 覆盖模型边界

- 这是**一维弧长规划抽象**：把砂轮在 root 点附近视为覆盖 ±r 的弧段；
- 不是有限圆柱与工件表面的精确切除仿真；
- 不是二维根部面覆盖；
- 被覆盖点只要求“在现有全域碰撞结果中该侧至少有一个可行候选”，没有重新评估“选中姿态在覆盖点”的碰撞；
- 本轮仍不做边级扫掠碰撞审计；
- 若 `branch` 长度小于 `2r`，可能只选 1 个点、路径长度为 0；这是覆盖模型的数学结果，需人工判断是否可接受。

### 11.4 论文标记

- Schneyer et al. 2023（RA-L）：覆盖规划/间距思想，✅；未用可变接触点/二维覆盖；
- Weingartshofer et al. 2023（RCIM）：把覆盖/容差表述为沿路径约束，◐；
- Lu et al. 2021（JMSE）：路径长度目标，◐；
- Zhang et al. 2026（Machines）：有向图/节点-边安全表述，◐；
- Abele et al. 2016（Procedia CIRP）：A*/TSP 仅作为后续分支排序参考，⚠️。

详见 `docs/path_planning_literature_review_20261030.md` 第 3.5 节。

### 11.5 最终结果（circle 覆盖约束版）

目录：

```text
results/cases/circle/p1_full_delivery_20261031_195757_314283/
```

结果：

| branch | `+q` | `-q` |
|---|---|---|
| 0 | 6/81 选中，0.12638，轴跳变 4.80° | 6/81，0.12638，4.80° |
| 1 | 9/121，0.22649，4.17° | 9/121，0.22649，4.17° |
| 2 | 11/153，0.30698，4.49° | 11/153，0.30699，4.49° |
| 3–6、8 | 1/10，0.0，0° | 1/10，0.0，0° |
| 7 | **no_path**（396–403 在该侧无可行候选） | 1/10，0.0，0° |

- 17 条 `ok`、1 条 `no_path`；
- 长分支 `max_angle_step_deg` 约 140°，是端面圆周接触方向 θ；实际 `max_axis_step_deg` 约 4.2–4.8°；
- 短分支每侧有 2 条同长最优路径，导出一条；
- 自动校验：root_line 最大误差 `4.96e-12`，17 条路径坐标最大误差 `4.66e-12`；
- 本轮未做边级碰撞审计。

### 11.6 仍需注意

- 覆盖模型是一维弧长抽象；不是砂轮实体切除仿真；
- 被覆盖点只要求在现有碰撞结果中该侧有可行候选，没有重新评估“选中姿态在覆盖点”的碰撞；
- 短 branch 只选 1 个点、长度 0；需人工判断是否可接受；
- 长分支的 `max_angle_step_deg` 约 140°，但 `max_axis_step_deg` 约 4–5°；报告中不能把 θ 当作工具轴姿态；
- 边级扫掠碰撞审计仍未做。
