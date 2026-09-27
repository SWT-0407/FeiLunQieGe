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
├─ results/                         # 正式结果目录；保存本地运行结果，不默认提交生成图
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

每次运行都会在 `results/run_YYYYMMDD_HHMMSS/` 下创建一个新的结果目录，不覆盖旧运行结果。输出写入以下五个文件：

- `results/run_.../feasible_centers_overview.png`：带工件表面、根部线、目标点和可行中心的三维总览图；
- `results/run_.../local_feasibility_explanation.png`：自动放大的局部法平面图，显示 `r`、`delta`、不可行候选、可行中心和可行圆弧；
- `results/run_.../workpiece_orientation_guide.png`：同一工件的三维透视与 X-Z 正投影对照图，用于解释 T 形外观差异；
- `results/run_.../feasible_centers.obj`：MeshLab 可直接打开的 OBJ 点/线文件，其中 `l` 是采样根部线或离散可行弧，`p` 是按目标点分组的可行砂轮中心；
- `results/run_.../feasible_centers_summary.json`：参数、计数、角区间、输入和限制的审计摘要。

OBJ 文件使用 normalized units，不是砂轮实体网格，也不代表厚度碰撞结果。文件中除了 `p` 点以外，还会把外圈相邻可行中心写成 `l` 线段，因此 MeshLab 即使没有打开顶点显示，也能看到可行圆弧。若要查看密集的中心点，选中图层后在右侧渲染面板把 `Vert` 从 `None` 改成点显示并适当增大点尺寸。若只想查看工件本体，仍应单独打开外部的 `mesh_without_flash.obj`。当前碰撞筛选方法是工件表面采样点的近似球形包络，可能误判可行；结果摘要会明确记录这一限制。

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
