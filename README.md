# FeiLunQieGe

砂轮切除铸件飞边的可行区域与路径规划研究。

本仓库现在包含一个可运行的最小几何闭环。它用于验证数据读取、曲线切线、无厚度砂轮候选中心生成、保守碰撞筛选和总览图输出；结果仍属于 preliminary geometry run，不是实际加工结论。远程仓库原先为空，本地已从 `https://github.com/SWT-0407/FeiLunQieGe` 克隆。

## 研究目标

给定飞边根部交线 `root_line.ply`、工件本体网格 `mesh_without_flash.obj`、砂轮半径 `r` 和径向容差 `delta`：

1. 把 `root_line.ply` 作为目标空间曲线 `l`，按 edge 连通关系拆分分支并按弧长采样；
2. 在每个采样点建立局部切线 `t_i`；
3. 生成满足砂轮半径方向 `e_i` 与 `t_i` 垂直、砂轮圆周切线与 `t_i` 平行的候选砂轮中心；
4. 判断候选无厚度圆盘是否与工件本体发生碰撞；
5. 输出所有采样点对应的可行中心集合/可行圆弧总览图。

这里的“可行性”首先是几何算法可行性，不等同于真实机床加工可行性。

## 当前约定

- `root_line.ply` 暂按飞边与工件本体的根部交线处理，而不是把它当作完整飞边实体。所有有效顶点都应参与分析；多个连通分支应分别保留，不能静默丢弃。
- 坐标统一使用输入数据已经归一化后的单位，代码中标注为 `normalized units`。不在初版中换算成毫米；以后若获得比例，只允许整体等比放缩。
- 初版砂轮采用**无厚度圆盘**模型，以减少实现量并验证中心候选和碰撞筛选流程。厚度 `w`、有限圆柱和圆角砂轮只预留接口。
- `delta` 是归一化单位下的径向候选区间参数，不直接解释为毫米加工公差。
- `mesh_without_flash.obj` 是默认碰撞对象，但仍需用小案例人工核对其含义和坐标范围。

## 目录结构

```text
FeiLunQieGe/
├─ configs/
│  └─ default.json                  # 初版参数和外部数据目录配置
├─ results/
│  └─ cases/                         # 按 12 个工件案例分目录保存独立运行结果
├─ scripts/
│  └─ run_pipeline.py               # 最小闭环命令行入口
├─ src/feilunqiege/
│  ├─ io/                           # OBJ/PLY 和数据契约
│  ├─ geometry/                     # 根部线、切线和局部标架
│  ├─ tooling/                     # 无厚度圆盘及未来厚度砂轮模型
│  ├─ collision/                    # 工件碰撞检测和近似筛选
│  ├─ visualization/                # 可行中心总览图和 OBJ 导出
│  └─ pipeline.py                   # 流程编排
├─ tests/                           # 几何单元测试
├─ .gitignore
├─ AGENTS.md                        # Git 仓库内的简要协作规范
└─ pyproject.toml                   # Python 包元数据
```

原始论文和约 930 MB 的 `flash_pipeline_12_models_20260924` 数据仍位于本地项目父目录，没有复制进本仓库。`configs/default.json` 通过 `data_root` 引用外部目录，避免把原始材料误上传 GitHub。

## 目前如何使用

运行最小闭环：

```powershell
python scripts/run_pipeline.py --config configs/default.json
```

每次运行都会在 `results/cases/<case_name>/run_YYYYMMDD_HHMMSS/` 下创建一个新的结果目录，不覆盖旧运行结果。输出写入以下文件：

- `results/cases/<case_name>/run_.../feasible_centers_overview.png`：带工件表面、根部线、目标点和可行中心的三维总览图；
- `results/cases/<case_name>/run_.../local_feasibility_explanation.png`：自动放大的局部法平面图，显示 `r`、`delta`、不可行候选、可行中心和可行圆弧；
- `results/cases/<case_name>/run_.../workpiece_orientation_guide.png`：同一工件的三维透视与 X-Z 正投影对照图，用于解释 T 形外观差异；
- `results/cases/<case_name>/run_.../feasible_centers.obj`：精确结果 OBJ 点/线文件，MeshLab 可直接打开；其中 `l` 是采样根部线或离散可行弧，`p` 是按目标点分组的可行砂轮中心；
- `results/cases/<case_name>/run_.../feasible_centers_display.obj`：用于 MeshLab 放大的八面体显示代理；
- `results/cases/<case_name>/run_.../local_feasible_centers.obj`：单个代表性目标点的局部显示代理，区分可行、不可行候选和目标点；
- `results/cases/<case_name>/run_.../feasible_centers_summary.json`：参数、计数、角区间、输入和限制的审计摘要；
- `results/cases/<case_name>/run_.../interactive_demo.html`：全部离散采样点可点击的离线交互页面。

OBJ 文件使用 normalized units，不是砂轮实体网格，也不代表厚度碰撞结果。文件中除了 `p` 点以外，还会把外圈相邻可行中心写成 `l` 线段，因此 MeshLab 即使没有打开顶点显示，也能看到可行圆弧。若要查看密集的中心点，选中图层后在右侧渲染面板把 `Vert` 从 `None` 改成点显示并适当增大点尺寸。若只想查看工件本体，仍应单独打开外部的 `mesh_without_flash.obj`。当前碰撞筛选方法是工件表面顶点、三角形质心和边中点构成的离散采样近似，并用球形包络估计安全距离，可能误判可行；结果摘要会明确记录这一限制。

### 当前碰撞判定的阅读边界

交互页面中的红色候选表示“在当前离散半径/角度采样和近似表面检查下通过”，灰色叉号表示该候选未通过当前检查。某个采样点出现整圈红色，只能说明本次抽样的候选均未发现碰撞，不能证明连续角度范围内、精确三角面距离意义下或真实砂轮厚度模型下都安全。当前 `leaf` 演示中，点 165 的可行数为 `371/540`，点 217 为 `540/540`；这种差异可能来自局部几何，也可能暴露离散表面采样对大三角面内部的漏检风险。

因此，现阶段结果应作为几何算法演示和问题定位依据。若要形成加工安全结论，后续需要升级为精确点到三角形/网格距离或更密集、可收敛的表面采样，明确接触容差 `epsilon`，并重新人工抽查所有全红点。

## 如何阅读结果

全局 PNG 的图例含义是：

- `[1] workpiece body`：灰色半透明工件本体；
- `[2] flash root line`：蓝色根部交线，也就是输入 `root_line.ply`，不是整个 T 形飞边；
- `[3] feasible wheel centers`：橙色可行砂轮中心。由于 `r=0.02` 而工件跨度约为 40--58 normalized units，全局图中它们会贴近根部线，不能在全局比例下直接读出半径；
- `[4] target points`：黑色目标接触点。

局部 PNG 才用于读取单点的可行半径区域：黑点是目标点，圆心到候选点的距离是 `rho`；黑色实线为 `rho=r`，蓝色虚线为 `rho=r-delta`；橙色点/粗弧表示通过当前碰撞近似筛选的候选。右图只把外圈 `rho=r` 上的离散可行点连接起来，所以它是“可行圆弧”的可读表达，不是实体砂轮。

MeshLab 中直接打开 OBJ 时，先按 `Ctrl+Shift+A` 或工具栏的适合视图按钮将模型置中，再在右侧图层面板选中 `feasible_centers`。如果只看到线段，说明当前显示的是 `l` 线原语；这仍然能检查根部线和可行弧。要看 `p` 中心点，开启 `Vert` 点显示并调大点尺寸。由于文件同时包含根部线、可行中心和可行弧，MeshLab 适合检查位置关系，而局部 PNG 更适合检查 `r`、`delta` 和角区间。

后续精化前仍需要确认：

- 多分支 `root_line` 的输出组织方式；
- 工件本体网格是否是唯一不可碰撞对象；
- 砂轮轴向方向是否自由变化；
- `r`、`delta`、候选角采样数、曲线采样数和碰撞容差；
- “接触”是否视为碰撞，以及目标根部允许的接触规则。

## Git 约定

```powershell
git status
git add <changed-files>
git commit -m "说明本次变更"
git push origin main
```

禁止提交大体积原始模型、临时缓存、虚拟环境和未审查的科研结果。每次算法、依赖、输入或参数发生变化，都要重新检查受影响的图和数据，并在提交说明中记录。

## MeshLab 放大与可行中心显示

每次运行还会生成两个专门用于查看的 OBJ：`feasible_centers_display.obj` 把均匀抽样的可行中心绘制成红色八面体面片，`local_feasible_centers.obj` 只保留一个代表性目标点及其局部候选（红色为可行、灰色为不可行、黑色为目标点）。八面体是显示代理，不是砂轮实体；精确数据仍以 `feasible_centers.obj` 为准。

同时会生成 `interactive_demo.html`。当前配置导出全部离散采样点：点击左侧根部线投影中的橙色点，右侧会自动放大并显示该点的候选半径区域、可行/不可行中心和外半径可行弧。该页面可直接双击离线打开；当前点击位置对应离散采样点，不是连续曲线上的重新计算点。若将 `interactive_demo.max_points` 设置为正数，则可退回少量演示点模式；设置为 `0` 表示全部点。

### 采样点编号规则

点编号是程序内部的全局离散索引，从 `0` 开始。程序先根据 `root_line.ply` 的 edge 连通关系拆分分支，再在每个分支内沿连接顺序排列点；最终按分支编号依次拼接，因此先出现的分支点编号较小。每个点同时保留 `branch_id` 和分支内弧长位置。编号用于定位和复现实验，不代表工件上的物理编号或某个固定全局坐标轴方向。

建议先单独打开 `local_feasible_centers.obj`，再使用 MeshLab 的 Fit View；查看全局关系时打开 `feasible_centers_display.obj`。如果双指缩放后线段消失，先 Fit View，再调整 Near/Far clipping；底部的 `Clipping Near/Far` 是视图裁剪状态，不代表 OBJ 数据被删除。
