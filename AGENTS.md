# FeiLunQieGe 仓库协作规范

本仓库用于验证砂轮切除铸件飞边的几何可行性算法。更完整的项目背景和数据说明位于上级项目目录的 `AGENTS.md`；本文件是克隆仓库内随代码同步的简要版本。

## 强制规则

- 禁止执行任何删除操作。需要淘汰的文件只能在人工确认后移动到 `__cache__`。
- 缓存、临时提取物和中间文件只放入 `__cache__` 或 `__pycache__`；不要把正式代码、结果和报告放入缓存。
- 不覆盖原始论文和模型数据，不把父目录的约 930 MB 原始 OBJ/PLY 材料复制进仓库。
- 当前主流程读取 ASCII `root_line.ply`、`mesh_without_flash.obj` 和带标签的 `original_flash_cloud.obj`；后者的红色飞边点用于对每个局部三角面法向按 `dot(n, F-O) >= 0` 定正负。候选中心位于局部切平面，有限圆柱主判定使用三角形 AABB 宽相与连续窄相。运行目录不覆盖旧结果；OBJ 是 MeshLab 可检查的可视化/审计材料，仍不能代替设备可达性或真实加工验收。
- 当前还生成离线 `interactive_demo.html`：默认导出全部离散采样点，点击根部线投影中的点后，右侧显示该点局部切平面内的可行半径区域；`interactive_demo.max_points > 0` 可限制为少量演示点。它是交互式显示层，不改变精确 OBJ、JSON 或碰撞结果。
- 候选角度和径向位置仍是离散采样；“全红/全绿”不等于连续角度范围或真实加工一定安全。当前固定姿态下使用三角形 AABB 宽相与连续有限圆柱窄相，能发现三角形内部穿入，但非封闭网格仍无法证明实体包含关系；所有结果仍是 preliminary geometry run。
- 算法计算坐标使用输入数据的归一化单位，文档中写作 `normalized units`，不能擅自标注为毫米。为了供 MeshLab 与原始 OBJ 叠加，`feasible_centers.obj`、`feasible_centers_display.obj` 和 `local_feasible_centers.obj` 必须在导出时逆变换为 `source_obj` 坐标，并在 OBJ 文件头和 JSON 摘要中明确标注；HTML 与 PNG 保持 normalized 计算坐标。
- 批量运行默认在内存中对每个工件及其 `root_line` 施加同一个包围盒最长边归一化变换；原始材料不改写，变换中心、尺度和源包围盒必须进入 JSON 摘要。

## 当前模型约定

- `root_line.ply` 暂作为飞边与工件本体根部交线 `l`；全部有效点参与分析。多个 edge 连通分支必须分别保留并记录。
- 在采样点 `x_i` 处，曲线切线为 `t_i`，工件表面法向为 `n_i`。砂轮中心偏移方向 `e_i` 在局部切平面内，砂轮轴向取 `n_i`；砂轮圆盘平面与切平面平行、与法平面垂直。`e_i` 不再强制与 `t_i` 垂直。
- 初版工具是无厚度圆盘，半径 `r`，候选半径范围为 `rho ∈ [r - delta, r]`。厚度圆盘/圆柱只保留扩展接口。
- `r` 不是全项目固定常数。批量运行可扫描多个归一化候选半径并按覆盖规则为每个案例选择主半径，也可通过 `radius_selection.case_overrides` 人工指定；扫描表、规则、最终值和源坐标等效值必须写入 JSON。
- 默认碰撞对象是 `mesh_without_flash.obj`，但实现前需用小案例人工核对。

## 代码和验证

- 代码应增加足够的中文注释，使读者能够看懂数据流、坐标系、参数、几何约束、连续碰撞判定、BVH/空间索引、容差和输出语义。复杂算法前应说明输入、几何假设、判定步骤和失败边界；注释不能掩盖未验证的假设，也不能逐行重复代码字面含义。
- 若连续碰撞、BVH 或网格质量检查确实需要第三方库，可以直接在本机安装；必须记录库名、版本、安装命令和用途，并更新依赖声明和验证记录，不得为“可能有用”而批量安装。
- 绘图必须使用非交互式后端保存 PNG，并保留全局总览、局部半径解释和工件视角对照；全局图不能被解释为等比例显示砂轮半径。
- 大网格的全局/视角 PNG 允许对三角面做确定性显示抽样（默认最多 20,000 面），但碰撞筛选不得因此减少其输入网格；摘要和文档需说明该显示限制。
- 先验证合成直线/圆弧，再加载小案例，最后再批量处理 12 组模型。当前最小闭环结果必须标记为 preliminary geometry run。
- 自动测试通过不等于真实加工验证通过；所有图像和碰撞结论都要保留参数、输入和人工检查边界。
- 提交前运行静态检查和占位测试，并确认没有意外加入大文件。

## 案例结果目录约定

- 后续 12 个案例分别对应 `results/cases/<case_name>/`：`1041432`、`226633`、`252632`、`439142`、`5head`、`5pipe`、`719790`、`804299`、`804301`、`circle`、`gear`、`leaf`。
- 每个案例的每次运行放在自己的 `run_YYYYMMDD_HHMMSS/` 目录中，不覆盖旧运行结果。
- 每个案例至少应有一个精确可行中心 OBJ 和一个交互 HTML；原始模型继续从父目录材料引用，不复制进仓库。
- 每次运行还可生成 `feasible_centers_display.obj` 和 `local_feasible_centers.obj` 两个 MeshLab 显示代理；它们用八面体/局部标记增强可见性，不能替代精确 `feasible_centers.obj`。
- 修改坐标变换或 OBJ 导出后，必须运行 `python scripts/audit_meshlab_alignment.py --config configs/default.json`；它逐例比较导出 OBJ 的 root_line 与原始 PLY。审计通过只说明叠加坐标一致，不能替代碰撞或工艺验收。
- `scripts/run_all_cases.py` 按顺序批量运行 12 个案例；当前 12 个案例均已生成一套本地结果。每个案例的网格在独立调用结束后释放，输出仍分别写入自己的 `run_YYYYMMDD_HHMMSS/` 目录。
- 批处理完成后在 `results/batches/batch_*/` 写入 JSON 和 CSV 汇总，集中记录各案例半径选择、可行率、无解点数和结果路径；批次目录同样不覆盖旧结果。
- 运行结果目录、OBJ、HTML、MTL 和 PNG 默认被 `.gitignore` 排除，不得因为批量完成就强制加入普通 Git 历史；GitHub 只同步代码、配置和文档，结果通过本机路径和 JSON 摘要审计。
- 点编号从 `0` 开始：先按 PLY edge 连通关系拆分分支，再按分支内连接顺序排列，最后按分支顺序拼接；输出同时记录 `branch_id` 和弧长位置。

## 后续实施边界（四阶段）

后续修改压缩为四个有明确验收门的阶段。这里的主碰撞视角是“工件三角面/工件实体是否进入砂轮实体”，而不是只遍历砂轮采样点；已有表面采样方法保留为 baseline 和快速预筛，不能作为连续无碰撞证明。

1. **姿态与输入契约冻结**：选取一个小案例和合成几何，核对 root line 分支、归一化坐标、工件网格封闭性，明确 `t/e/n`、切平面、法平面、砂轮圆盘平面和轴向。自动检查 `n·t≈0`、`n·e≈0`、`||e||≈1`；不检查 `e·t≈0`，因为当前模型允许 `e` 在局部切平面内转动。人工确认砂轮圆盘与切平面平行、与法平面垂直。
2. **无厚度圆盘连续碰撞基线**：实现三角形与圆盘的连续判定，加入 BVH/AABB broad phase；用三角形内部穿入、相切、顶点均在外部和闭合工件内部嵌入等合成案例验收，再运行一个真实小案例。输出连续可行角区间及边界复核结果。
3. **有限厚度圆柱连续碰撞**：扩展为半径 `r`、轴向宽度 `w` 的圆柱实体，连续检查侧面、端面、边缘和实体包含；记录 `epsilon`、碰撞原因、最小间隙和见证几何。人工确认安全姿态和至少两类碰撞姿态，并核对 OBJ/PNG/JSON 一致。
4. **单案例回归与批量推广**：完成新旧方法对照、坐标审计和输出契约检查后，才运行 12 个案例。批量完成后停止，等待 PNG、OBJ/MTL、JSON 和 alignment audit 的人工检查。

每阶段都要记录输入、参数、后端、连续/近似性质、运行时间和人工验收状态；自动测试通过不等于真实加工验收通过。
## Current acceptance gate

- New runs must export `pose_examples.obj` with adjacent `pose_examples.mtl`: feasible pose is high-lightness mint green (`feasible_pose`, RGB `0.60, 1.00, 0.80`), collision pose is high-lightness coral red (`collision_pose`, RGB `1.00, 0.78, 0.70`). This coral red must remain visually distinct from the deep-red feasible-center display markers. The OBJ vertex records and MTL must carry the same colors. The MTL sets matching `Ka`/`Kd`/`Ke` plus `illum 0`, so its semantic colors are not darkened by directional lighting in MeshLab.
- Finite-cylinder collision filtering now uses a triangle AABB tree as broad phase and an axial-clipping/continuous radial-distance narrow phase for each fixed pose. The `finite_nearest_samples` setting is retained only for historical baseline comparisons; it no longer decides the primary feasible set.
- `226633` 的姿态示例绿色环固定使用候选索引 `135`：它与红色碰撞姿态位于同一目标点，并经过更高密度表面样本复核；该覆盖只影响展示姿态，不改变全体候选筛选。
- New runs must not generate `interactive_demo.html`; existing historical HTML outputs are retained and must not be deleted.
- New run directories contain only `feasible_centers_overview.png`, `feasible_centers_summary.json`, `pose_examples.obj`/`.mtl`, `feasible_centers_display.obj`/`.mtl`, and `local_feasible_centers.obj`/`.mtl`. Do not generate other PNG files or `feasible_centers.obj`.
- After the 12-case batch, stop for human inspection of PNG, JSON, OBJ/MTL coloring, and the source-coordinate alignment audit.

The earlier interactive-demo notes in this file describe historical runs only. They are superseded for the current stage: the pipeline no longer creates HTML files.
