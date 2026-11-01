# 路径规划文献调研（2026-10-30）

**范围**：为“砂轮切除铸件飞边”的根部线路径规划寻找可借鉴的论文与算法。  
**当前任务边界**：保留现有 72 个离散候选角度，在每个 root 采样点上选择一个可行砂轮中心，在 root_line 两侧各连接成一条总长度尽量短的路径。  
**核对方式**：2026-10-30 通过 Crossref / OpenAlex / Semantic Scholar 核对题名、作者、年份、DOI 和部分公开摘要；2026-10-31 整合 Codex 补充的 4 篇论文（Zhang 2026、Yuan 2021、Lu 2021、Onstein 2023），并再次核对元数据。**未逐篇获取并精读全文**；下文标为“摘要未获取”的条目，其算法细节需在正式引用前用全文复核。  
**重要限制**：本项目仍是 normalized units 下的几何演示，不是毫米级工艺结论；文献中的机床运动学、力控、材料去除模型不能直接当作本项目已实现的能力。

---

## 0. 与本项目最接近的算法结论

当前问题可以写成一张**分层有向无环图**：

- 第 `i` 层对应第 `i` 个 root 采样点；
- 每个节点是 `(i, j)`：第 `i` 点第 `j` 个可行候选中心；
- 只允许相邻采样点之间连边；
- 边权是两中心的三维欧氏距离；
- 在每一层只选一个节点，求从第一层到最后一层的最短路径。

这个离散问题用**动态规划/分层最短路**即可得到全局最优，不需要启发式搜索。文献中最值得借鉴的不是“某个更复杂的图搜索器”，而是：

1. **如何把刀轴/砂轮姿态的可行域组织成沿路径的约束**（Chen 2015；Wu 2025；Wang 2020；Mi 2016；Yuan 2021）；
2. **如何保证整条路径而不只是离散采样点无碰撞**（Ezair & Elber 2018；Zhang 2026）；
3. **如何把路径长度、姿态平滑、过切/碰撞放在同一个优化框架里**（Weingartshofer 2023；Kim 2015；Lu 2021）；
4. **去毛刺/磨削领域如何从特征曲线生成轨迹**（Feng 2021；Schneyer 2023；Guo 2019；Onstein 2023）。

需要特别说明：Zhang 2026 的图结构和本项目最接近，但它用改进 A* 优化**机器人关节偏差 + 安全距离**；Lu 2021 用采样式运动规划优化**机器人各关节路径长度**；Yuan 2021 分两阶段做姿态优化和 cutter-contact 路径规划。它们都不能直接替代本项目“在 72 个离散中心上最小化三维折线长度”的 DP，但分别提供了图搜索、路径长度目标和“局部可行 + 沿路径连续可行”的佐证。

---

## 1. 主表：按贴合度排序

| 排名 | 论文 | 年份/期刊 | 贴合度 | 一句话可借鉴点 | 链接 |
|---:|---|---|:---:|---|---|
| 1 | Chen, Xu, Tang, *Collision-free tool orientation optimization in five-axis machining of bladed disk* | 2015, Journal of Computational Design and Engineering | ★★★★★ | 用参考平面把刀轴限制在一维族内，再沿整条路径全局优化平滑且无碰撞的姿态；与“每侧一条中心路径”结构最接近 | [DOI](https://doi.org/10.1016/j.jcde.2015.06.001) |
| 2 | Zhang, Xiao, Wan, Jiang, *Collision-Free Robot Pose Optimization Method Based on Improved Algorithms* | 2026, Machines | ★★★★★ | 把离散路径点上的候选姿态组织成有向图，筛可行姿态后用改进 A* 求全局姿态序列；图结构与本项目几乎同构，但其目标是关节偏差而不是中心路径长度 | [MDPI](https://www.mdpi.com/2075-1702/14/1/65) · [DOI](https://doi.org/10.3390/machines14010065) |
| 3 | Weingartshofer et al., *Optimization-based path planning framework for industrial manufacturing processes with complex continuous paths* | 2023, Robotics and Computer-Integrated Manufacturing | ★★★★★ | 把“沿给定制造路径的工艺自由度 + 容差 + 碰撞”统一成约束优化；我们的 `θ` 就是可优化的工艺自由度 | [DOI](https://doi.org/10.1016/j.rcim.2022.102516) |
| 4 | Ezair, Elber, *Automatic generation of globally assured collision free orientations for 5-axis ball-end tool-paths* | 2018, Computer-Aided Design | ★★★★☆ | 对整条 cutter-contact 曲线构造保守可达体，避免“逐点采样通过、点间碰撞”的漏洞 | [DOI](https://doi.org/10.1016/j.cad.2018.04.011) |
| 5 | Wu, Xu, Lin, Sun, *Safe Attitude Corridor: A Convex Optimization Method for Tool Orientation Smoothing in Five-Axis CNC Machining* | 2025, IEEE/ASME Transactions on Mechatronics | ★★★★☆ | 可行姿态域可能是凹的、断开的；先用凸走廊包住，再做 B 样条姿态平滑和 jerk 最小化 | [DOI](https://doi.org/10.1109/TMECH.2024.3510852) |
| 6 | Kim, Elber, Bartoň, Pottmann, *Precise gouging-free tool orientations for 5-axis CNC machining* | 2015, Computer-Aided Design | ★★★★☆ | 把“刀具不得切到相邻曲面”表达为姿态约束；与本项目“不得碰工件本体”同构 | [DOI](https://doi.org/10.1016/j.cad.2014.08.010) |
| 7 | Yuan, Mi, Jia, Lin, Shen, *Tool Orientation Optimization and Path Planning for 5-Axis Machining* | 2021, Journal of Systems Science and Complexity | ★★★★☆ | 两阶段：先求平滑、无碰撞、无过切的刀轴姿态，再规划 cutter-contact 点路径；可借鉴“局部可行 + 沿路径连续可行”的组织方式 | [DOI](https://doi.org/10.1007/s11424-020-9270-1) |
| 8 | Lu, Zhang, Tian, Han, Wang, *Tool Path Optimization for Robotic Surface Machining by Using Sampling-Based Motion Planning Algorithms* | 2021, Journal of Manufacturing Science and Engineering | ★★★★☆ | 在机器人关节空间采样有效状态，约束刀尖跟随指定曲线、限制刀轴方向并检查碰撞，目标为各关节路径长度之和 | [DOI](https://doi.org/10.1115/1.4047734) |
| 9 | Mi, Yuan, Ma, Shen, *Tool orientation optimization for 5-axis machining with C-space method* | 2016, The International Journal of Advanced Manufacturing Technology | ★★★★☆ | 在姿态 C-space 中构造可行域并优化姿态序列；可把 72 个角度理解为 C-space 的一维切片 | [DOI](https://doi.org/10.1007/s00170-016-8849-0) |
| 10 | Schneyer, Sachtler, Eiband, Nottensteiner, *Segmentation and Coverage Planning of Freeform Geometries for Robotic Surface Finishing* | 2023, IEEE Robotics and Automation Letters | ★★★★☆ | 把砂轮/磨盘上的接触点位置作为额外自由度，并按覆盖间距规划路径 | [DOI](https://doi.org/10.1109/LRA.2023.3293309) |
| 11 | Feng et al., *A novel feature-guided trajectory generation method based on point cloud for robotic grinding of freeform welds* | 2021, The International Journal of Advanced Manufacturing Technology | ★★★★☆ | 从点云提取三维特征曲线，再沿曲线生成磨削轨迹；与 root_line 的角色最接近 | [DOI](https://doi.org/10.1007/s00170-021-07095-2) |
| 12 | Wang, Lin, Shi, *Efficiently constructing collision-free regions of tool orientations for holder in five-axis machining of blisk* | 2020, Chinese Journal of Aeronautics | ★★★☆☆ | 构造刀轴/刀柄的无碰撞姿态区域；可用于把离散可行角整理成连续区间 | [DOI](https://doi.org/10.1016/j.cja.2020.05.032) |
| 13 | Guo et al., *A Robotic Deburring Methodology for Tool Path Planning and Process Parameter Control of a Five-Degree-of-Freedom Robot Manipulator* | 2019, Applied Sciences | ★★★☆☆ | 去毛刺的刀位/刀姿规划 + 分层去毛刺 + 工艺参数控制 | [DOI](https://doi.org/10.3390/app9102033) |
| 14 | Zhang et al., *A novel trajectory planning method for robotic deburring of automotive castings considering adaptive weights* | 2024, Robotics and Computer-Integrated Manufacturing | ★★★☆☆ | 汽车铸件去毛刺轨迹规划中的自适应权重思想，可用于后续加入平滑/工艺项 | [DOI](https://doi.org/10.1016/j.rcim.2023.102677) |
| 15 | Tafuro et al., *Towards intelligent robotic sole deburring: from burrs identification to path planning* | 2024, IEEE/RSJ IROS | ★★★☆☆ | “毛刺识别 → 路径规划”的完整流程组织；应用对象不同 | [DOI](https://doi.org/10.1109/IROS58592.2024.10802795) |
| 16 | Onstein, Bjerkeng, Martinsen, *Automated Tool Trajectory Generation for Robotized Deburring of Cast Parts Based on 3D Scans* | 2023, Procedia CIRP | ★★★☆☆ | 基于 CAD + 3D 扫描的铸件去毛刺轨迹生成，并在 KUKA KR 60 上验证；适合借鉴工程流程与验证方式 | [DOI](https://doi.org/10.1016/j.procir.2023.06.087) · [SINTEF](https://www.sintef.no/en/publications/publication/2222321/) |

### 补充背景（不作为核心算法依据）

| 论文 | 年份/期刊 | 贴合度 | 说明 | 链接 |
|---|---|:---:|---|---|
| Amersdorfer, Meurer, *Equidistant Tool Path and Cartesian Trajectory Planning for Robotic Machining of Curved Freeform Surfaces* | 2022, IEEE Transactions on Automation Science and Engineering | ★★★☆☆ | 弧长参数化与等距路径；可用于 root_line 采样和路径长度对比 | [DOI](https://doi.org/10.1109/TASE.2021.3117691) |
| Sun et al., *Path, feedrate and trajectory planning for free-form surface machining: A state-of-the-art review* | 2022, Chinese Journal of Aeronautics | ★★★☆☆ | 自由曲面路径/进给/轨迹规划综述，适合作为相关工作总览 | [DOI](https://doi.org/10.1016/j.cja.2021.06.011) |
| Abele et al., *Time Optimal Path Planning for Industrial Robots Using STL Data Files* | 2016, Procedia CIRP | ★★☆☆☆ | 用 STL 建图 + A* + TSP 做去毛刺路径；与“分层最短路径”不同，但可参考建图/排序 | [DOI](https://doi.org/10.1016/j.procir.2016.08.038) |
| Zhao et al., *Robotic Jigsaw: A Non-Holonomic Cutting Robot and Path Planning Algorithm* | 2021, IEEE/RSJ IROS | ★★☆☆☆ | 沿曲线切割的路径规划与覆盖/入口区域；不是砂轮姿态问题 | [DOI](https://doi.org/10.1109/IROS51168.2021.9636066) |
| Zhu et al., *Research on the Visual Guidance System of Zoning Casting Grinding Based on Feature Points* | 2022, Applied Sciences | ★★☆☆☆ | 视觉引导分区铸造打磨；应用 baseline，算法较简化 | [DOI](https://doi.org/10.3390/app12178771) |
| Lozano-Pérez, *Spatial Planning: A Configuration Space Approach* | 1983, IEEE Transactions on Computers | ★★★☆☆ | 配置空间方法的奠基性论文；可用于解释“把砂轮姿态映射为 C-space 中的可行点” | [DOI](https://doi.org/10.1109/TC.1983.1676196) |
| Li et al., *Cutter partition-based tool orientation optimization for gouge avoidance in five-axis machining* | 2017, The International Journal of Advanced Manufacturing Technology | ★★★★☆ | 刀具/曲面分区 + 刀轴优化以避免过切；与“砂轮不得切入相邻面”同构 | [DOI](https://doi.org/10.1007/s00170-017-1263-4) |

---

## 2. 逐篇核对与可借鉴思路

> 本节顺序保留初次整理顺序，**不代表第 1 节主表排名**；2.13–2.16 为 2026-10-31 根据 Codex 补充材料新增。主表排名与逐篇编号的对应关系以题名为准。

### 2.1 Weingartshofer et al. 2023（★★★★★）

**题名**：Optimization-based path planning framework for industrial manufacturing processes with complex continuous paths  
**作者**：Thomas Weingartshofer, Bernhard Bischof, Martin Meiringer, Christian Hartl-Nesic, Andreas Kugi  
**期刊**：Robotics and Computer-Integrated Manufacturing, 2023  
**DOI**：https://doi.org/10.1016/j.rcim.2022.102516

**主要内容**：针对机器人必须沿给定制造路径运动、但允许工艺容差和冗余自由度的场景，提出一个统一的路径规划框架。框架把工艺自由度、容差、硬/软约束和碰撞 avoidance 参数化，求出关节空间中的最优连续路径；给出目标函数和约束的解析梯度，并利用多核 CPU 加速。论文用拉丝工艺和喷涂仿真演示。

**算法要点**：
- 以“沿路径的工艺自由度”为优化变量；
- 目标函数和约束可按工艺参数化；
- 碰撞作为约束/惩罚；
- 解析梯度 + 多核并行。

**与当前任务的贴合点**：
- 我们的 `θ_i`（候选中心绕 root 点的角度）就是沿 root_line 的工艺自由度；每个点只允许在可行角集合内取值；
- 我们的“路径长度最短”可以看作该框架的一个简化目标；
- 框架允许容差和冗余自由度，适合后续从 72 离散角扩展到连续角区间。

**不能直接照搬的原因**：
- 论文优化的是机器人关节空间连续轨迹，包含运动学/动力学；我们当前只有几何中心点和 72 离散角，没有机器人模型；
- 论文的碰撞约束是通用机器人-环境碰撞，我们的碰撞是砂轮有限圆柱 vs 工件三角网格，语义更具体；
- 论文没有“左右两侧各一条路径”的分解。

**可落地做法**：把当前 DP 看作该框架的离散特例；先在离散层上求最短路径，后续若要连续化，把每层可行角区间作为约束，把路径长度 + 角速度/jerk 作为目标，用类似的分层优化。

---

### 2.2 Chen, Xu, Tang 2015（★★★★★）

**题名**：Collision-free tool orientation optimization in five-axis machining of bladed disk  
**作者**：Li Chen, Ke Xu, Kai Tang  
**期刊**：Journal of Computational Design and Engineering, 2015  
**DOI**：https://doi.org/10.1016/j.jcde.2015.06.001

**主要内容（公开摘要）**：整体叶盘形状复杂，刀姿选择不当会导致碰撞和机床振动。论文提出一种**基于参考平面**的算法，沿刀具路径生成一组平滑对齐的刀轴方向，保证整条路径上都不发生碰撞，并对整体平滑性做全局优化；在叶盘模型上做了仿真验证。

**算法要点**：
- 用参考平面约束刀轴方向，把二维/三维姿态问题降为一维角度族；
- 沿路径逐点构造可行方向，再全局优化相邻姿态的平滑性；
- 目标是“整条路径无碰撞 + 全局平滑”，不是逐点独立选择。

**与当前任务的贴合点**：
- 我们的“每一侧”本质上也是把中心偏移方向限制在切平面内的一个半圆（一维角度族），与参考平面降维思想高度一致；
- 论文的“沿路径全局优化”对应我们的分层 DP；
- 论文的“整条路径无碰撞”对应我们下一步应补的边级碰撞审计。

**不能直接照搬的原因**：
- 论文是五轴铣削刀轴，刀具是球头/环形刀，不是有限厚度砂轮；
- 论文的参考平面由叶盘通道几何构造，我们的参考方向由工件表面法向和 root 切线构造；
- 论文没有“左右两侧分别成路径”的显式要求。

**可落地做法**：把 `q = a × t` 看作参考平面内的横向基；每一侧的 `θ` 就是一维姿态族；用 DP 在离散角度上求全局最短，再用 DP 回溯结果检查相邻角度跳变。若后续要连续化，可借用其“参考平面 + 全局平滑”的表述。

---

### 2.3 Wu et al. 2025（★★★★☆）

**题名**：Safe Attitude Corridor: A Convex Optimization Method for Tool Orientation Smoothing in Five-Axis CNC Machining  
**作者**：Lei Wu, Jinting Xu, Jianquan Lin, Yuwen Sun  
**期刊**：IEEE/ASME Transactions on Mechatronics, 2025（早期访问 2024）  
**DOI**：https://doi.org/10.1109/TMECH.2024.3510852

**主要内容（公开摘要）**：自由曲面五轴加工中，无碰撞的可行姿态域常常是**凹的或断开的**；在这样不规则的区域里规划平滑姿态很困难。论文提出 Safe Attitude Corridor（SAC）：在可行域内构造凸多面体走廊，把不规则/非凸可行域转化为线性不等式约束；用五次 B 样条表示旋转轴运动，把运动学指标写成 B 样条形式，最终把姿态平滑写成最小化旋转轴 jerk 的二次规划。

**算法要点**：
- 可行姿态域 → 凸走廊（SAC）；
- B 样条姿态参数化；
- 二次规划，目标为 jerk 积分最小；
- 严格满足无碰撞约束。

**与当前任务的贴合点**：
- 我们的可行角集合在很多点上是**离散、可能不连续**的（例如 branch 7 只有 250°–290° 的窄带）；论文明确处理“可行域断开”的情况，提醒我们不能假设可行角是一个连续弧；
- 论文的“先保证无碰撞，再优化平滑”与我们的“先筛可行节点，再最短路径”顺序一致；
- 若后续要从 72 角扩展到连续区间，SAC 的凸走廊思路可用于把每点可行角区间转成约束。

**不能直接照搬的原因**：
- 论文优化的是旋转轴 jerk 和 B 样条控制点，我们的当前目标只是中心点路径长度；
- 论文的可行域是三维姿态空间，我们的每点可行集是一维角度（在固定轴向下）；
- 论文需要机床运动学模型。

**可落地做法**：当前不要引入 B 样条/QP；先把“可行角可能断开”作为事实写入输出。后续若用户要求平滑路径，可在 DP 后对选中角度序列做分段拟合，断点处显式标记，不要跨断点强行拟合。

---

### 2.4 Ezair & Elber 2018（★★★★☆）

**题名**：Automatic generation of globally assured collision free orientations for 5-axis ball-end tool-paths  
**作者**：Ben Ezair, Gershon Elber  
**期刊**：Computer-Aided Design, 2018  
**DOI**：https://doi.org/10.1016/j.cad.2018.04.011

**主要内容（公开摘要）**：给定球头刀、cutter-contact 曲线以及模型/障碍曲面，生成全局可访问的平滑五轴刀路。方法基于对有限曲面/方向元素的**保守可达性计算**，保证解的正确性，并在刀具配置空间中定义连续可达体；有了全局可达体，就可以规划可访问刀路并做全局优化。论文强调：保守计算可以覆盖整条 cutter-contact 曲线，避免逐点采样带来的缺陷。

**算法要点**：
- 保守可达性/配置空间；
- 连续可达体；
- 整条路径的可访问性，而不是逐点检查；
- 全局路径优化。

**与当前任务的贴合点**：
- 我们当前是 72 角离散、逐点碰撞；论文正好指出“逐点采样通过 ≠ 整条路径安全”；
- 我们的相邻点之间如果只检查端点，可能漏掉中间姿态碰撞；论文的“保守可达体”是边级检查的理论依据；
- 我们的两根路径可以看作两条 cutter-contact 曲线的中心轨迹。

**不能直接照搬的原因**：
- 论文是球头刀，不是有限厚度砂轮；
- 保守可达体计算复杂，当前 72 离散角 + 有限圆柱 checker 不适合直接实现完整保守体；
- 论文没有左右两侧路径的分解。

**可落地做法**：在路径生成后增加**边级碰撞审计**：对每条相邻节点连线，插值若干中间姿态，用现有有限圆柱 checker 检查；把结果写入 JSON，不静默通过。若用户要求“路径级无碰撞”，则把边审计作为硬约束重新建图，而不是只在端点筛。

---

### 2.5 Kim et al. 2015（★★★★☆）

**题名**：Precise gouging-free tool orientations for 5-axis CNC machining  
**作者**：Yong-Joon Kim, Gershon Elber, Michael Bartoň, Helmut Pottmann  
**期刊**：Computer-Aided Design, 2015  
**DOI**：https://doi.org/10.1016/j.cad.2014.08.010

**主要内容（摘要未获取，依据题名与公开元数据）**：研究五轴加工的“无过切（gouging-free）”刀轴方向，要求刀具在给定刀位处不要切到设计曲面/相邻曲面；通常通过局部几何分析构造允许的刀轴方向区域，再在路径上选择精确无过切的姿态。具体公式和实验需全文核对。

**算法要点（需全文确认）**：
- 过切/干涉的局部几何判定；
- 刀轴方向的可行区域；
- 沿路径的姿态选择。

**与当前任务的贴合点**：
- 我们的“砂轮不得碰工件本体”与“刀具不得过切相邻曲面”是同构约束；
- 在凸边/凹边处，可行角往往是一段窄区间；论文的可行方向区域思路可帮助解释 branch 7 的单侧可行现象；
- 可用于后续把 72 角离散可行集整理成“允许/禁止角区间”。

**不能直接照搬的原因**：
- 铣刀几何与砂轮差异大；
- 论文关注设计曲面过切，我们的碰撞对象是三角网格工件本体；
- 没有左右两侧路径概念。

**可落地做法**：当前不需要改碰撞判定；在报告中用“gouging-free”类比解释“砂轮端面不得切入相邻面”。若后续需要连续角区间，可参考其可行方向区域的边界构造。

---

### 2.6 Mi et al. 2016（★★★★☆）

**题名**：Tool orientation optimization for 5-axis machining with C-space method  
**作者**：Zhenpeng Mi, Chun-Ming Yuan, Xiaohui Ma, Li-Yong Shen  
**期刊**：The International Journal of Advanced Manufacturing Technology, 2016  
**DOI**：https://doi.org/10.1007/s00170-016-8849-0

**主要内容（摘要未获取，依据题名与公开元数据）**：用 C-space（配置空间）方法优化五轴加工刀轴方向。通常做法是把刀轴方向映射到二维/三维姿态空间，构造无碰撞可行域，再在可行域内选择平滑、可加工的刀轴序列。具体构造和优化目标需全文核对。

**算法要点（需全文确认）**：
- 刀轴方向的 C-space 表示；
- 可行域/碰撞区域构造；
- 姿态序列优化。

**与当前任务的贴合点**：
- 我们的 72 个角度就是 C-space 在固定轴向下的一维切片；每点的绿色/红色候选就是可行/禁止姿态；
- 论文的“在 C-space 中找可行序列”与我们的分层图搜索同构；
- 可用于向审稿人解释：当前不是“任意选点”，而是在配置空间可行域内做路径优化。

**不能直接照搬的原因**：
- 论文是通用五轴刀轴，不含砂轮厚度/端面接触语义；
- C-space 构造通常需要连续几何运算，当前是离散采样。

**可落地做法**：在方法部分把 `(i, θ)` 解释为离散 C-space 切片；保留 72 角，不强行连续化。后续若扩展，可把每点可行角区间作为 C-space 的可行子集。

---

### 2.7 Wang, Lin, Shi 2020（★★★☆☆）

**题名**：Efficiently constructing collision-free regions of tool orientations for holder in five-axis machining of blisk  
**作者**：Zhiwei Wang, Xiaojun Lin, Yaoyao Shi  
**期刊**：Chinese Journal of Aeronautics, 2020  
**DOI**：https://doi.org/10.1016/j.cja.2020.05.032

**主要内容（公开摘要）**：整体叶盘通道加工中，刀柄需要深入通道，刀柄半径随高度变化，传统的 line-visibility 方法不再适用。论文提出高效构造刀柄无碰撞刀轴方向区域的方法，用于刀轴规划。

**算法要点**：
- 变半径刀柄的碰撞区域构造；
- 无碰撞方向区域；
- 区域边界/可见性分析。

**与当前任务的贴合点**：
- 我们的有限厚度圆柱砂轮也有“半径 + 宽度 + 端面”，比无厚度圆盘更接近论文的“变半径工具”问题；
- 论文的“无碰撞方向区域”可以对应我们每点的可行角集合；
- 可用于解释为什么某些点只有窄角带（branch 7）。

**不能直接照搬的原因**：
- 论文针对刀柄/叶盘通道，几何对象不同；
- 论文关注方向区域构造，不是路径长度最短；
- 没有左右两侧路径。

**可落地做法**：当前保留 72 角离散；在输出中用“可行角区间/窄带”描述每点可行域。若后续要做连续区间，可参考其区域构造，而不是简单对 72 个布尔值插值。

---

### 2.8 Schneyer et al. 2023（★★★★☆）

**题名**：Segmentation and Coverage Planning of Freeform Geometries for Robotic Surface Finishing  
**作者**：Stefan Schneyer, Arne Sachtler, Thomas Eiband, Korbinian Nottensteiner  
**期刊**：IEEE Robotics and Automation Letters, 2023  
**DOI**：https://doi.org/10.1109/LRA.2023.3293309

**主要内容（公开摘要）**：打磨/抛光中，复杂几何的高曲率区域无法用“大而平的磨盘 + 简单策略”最优覆盖。论文把**磨盘上的可变接触点**作为额外自由度，提出自动分割方法（按统一加工策略划分区域）和覆盖路径规划方法（对接触区域建模，实现路径行间距的自适应调整），在 DLR SARA 机器人上仿真和实验验证，能减少未覆盖间隙和重叠。

**算法要点**：
- 磨盘接触点作为额外 DoF；
- 基于工具/工件精确几何的分割；
- 覆盖路径 + 自适应行距；
- 覆盖率和重叠优化。

**与当前任务的贴合点**：
- 我们的砂轮是有限半径圆盘，当前把接触点固定在 root 点、中心沿切平面偏移；论文提示“接触点可以在砂轮端面上移动”，这是后续扩展的重要自由度；
- 论文的“覆盖率/行距”对应我们的两条侧向路径：如果两条路径都贴着 root_line，覆盖宽度由砂轮半径决定；若要考虑飞边宽度，需要覆盖规划；
- 论文的分割思想可用于处理 root_line 多分支/不同局部几何。

**不能直接照搬的原因**：
- 论文是表面覆盖，不是沿一条 root_line 的两侧路径；
- 论文的接触点可变模型会改变我们的候选中心公式和碰撞判定；
- 论文没有“最短路径”目标，而是覆盖率/重叠。

**可落地做法**：当前任务先固定接触点、只做两侧中心路径；在报告中把“接触点可变”列为后续扩展，不要偷偷改变当前几何模型。

---

### 2.9 Feng et al. 2021（★★★★☆）

**题名**：A novel feature-guided trajectory generation method based on point cloud for robotic grinding of freeform welds  
**作者**：Hengjian Feng, Xukai Ren, Lufeng Li, Xiaoqiang Zhang, Huabin Chen, Ze Chai, Xiaoqi Chen  
**期刊**：The International Journal of Advanced Manufacturing Technology, 2021  
**DOI**：https://doi.org/10.1007/s00170-021-07095-2

**主要内容（摘要未获取，依据题名与公开元数据）**：面向自由曲面焊缝的机器人磨削，从点云中提取特征引导的轨迹。通常包括焊缝特征点/曲线提取、局部坐标系构造、沿特征曲线生成磨削轨迹和姿态。具体特征描述与轨迹生成公式需全文核对。

**算法要点（需全文确认）**：
- 点云特征提取；
- 三维特征曲线/焊缝中心线；
- 沿曲线的磨削轨迹与姿态；
- 机器人执行。

**与当前任务的贴合点**：
- 我们的 `root_line.ply` 就是一条三维特征曲线；论文的“从点云到特征曲线再到磨削轨迹”与我们的数据流最接近；
- 论文的局部标架/姿态生成可以对照我们的 `t, q, a` 标架；
- 可用于相关工作部分说明“沿特征曲线生成磨削路径”的合理性。

**不能直接照搬的原因**：
- 论文是焊缝磨削，飞边/根部交线的几何语义不同；
- 论文可能没有“左右两侧各一条路径”和有限厚度砂轮碰撞；
- 需要全文确认其姿态约束和碰撞处理。

**可落地做法**：当前直接使用 `root_line.ply`，不需要重新做特征提取；在文档中引用其“特征曲线 → 轨迹”思路即可。后续若要处理多分支/断线，可借鉴其曲线提取与排序。

---

### 2.10 Guo et al. 2019（★★★☆☆）

**题名**：A Robotic Deburring Methodology for Tool Path Planning and Process Parameter Control of a Five-Degree-of-Freedom Robot Manipulator  
**作者**：Wanjin Guo, Ruifeng Li, Yaguang Zhu, Tong Yang, Rui Qin, Zhixin Hu  
**期刊**：Applied Sciences, 2019  
**DOI**：https://doi.org/10.3390/app9102033

**主要内容（公开摘要）**：面向五自由度混合机器人，提出去毛刺刀位/刀姿规划和分层去毛刺规划，并用模糊控制做工艺参数控制；通过灵巧操作验证和两组去毛刺实验验证。

**算法要点**：
- 刀位/刀姿规划；
- 分层去毛刺（layered deburring）；
- 工艺参数模糊控制；
- 机器人可达性/灵巧性验证。

**与当前任务的贴合点**：
- 去毛刺领域直接相关；
- “分层去毛刺”提示：如果飞边厚度方向需要多层去除，路径可能不止一条平面路径；
- 刀姿规划与我们的中心/姿态选择对应。

**不能直接照搬的原因**：
- 论文是五自由度专用机器人，强调运动学可达性；
- 没有砂轮有限圆柱 vs 三角网格的精确碰撞；
- 没有两侧路径最短的离散优化。

**可落地做法**：当前只做单层几何路径；在报告中把“分层去除/工艺参数”列为后续工作，不把几何路径长度当作加工时间。

---

### 2.11 Zhang et al. 2024（★★★☆☆）

**题名**：A novel trajectory planning method for robotic deburring of automotive castings considering adaptive weights  
**作者**：Yu Zhang, Hongdi Liu, Weikang Cheng, Lin Hua, Dahu Zhu  
**期刊**：Robotics and Computer-Integrated Manufacturing, 2024  
**DOI**：https://doi.org/10.1016/j.rcim.2023.102677

**主要内容（摘要未获取，依据题名与公开元数据）**：面向汽车铸件机器人去毛刺的轨迹规划，考虑自适应权重。通常是在轨迹优化目标中平衡路径长度、姿态变化、工艺要求等多项，权重随局部几何/工艺状态调整。具体目标函数和权重策略需全文核对。

**算法要点（需全文确认）**：
- 多目标轨迹优化；
- 自适应权重；
- 铸件去毛刺应用。

**与当前任务的贴合点**：
- 我们当前只优化路径长度；论文提示后续可加入姿态变化/工艺项，并用权重平衡；
- 汽车铸件去毛刺的应用背景与铸件飞边接近。

**不能直接照搬的原因**：
- 需要全文确认其轨迹表示和约束；
- 可能不包含砂轮有限厚度碰撞；
- 没有两侧路径分解。

**可落地做法**：当前保持“纯长度”目标，避免引入未经验证的权重；在报告中把自适应权重列为后续扩展。

---

### 2.12 Tafuro et al. 2024（★★★☆☆）

**题名**：Towards intelligent robotic sole deburring: from burrs identification to path planning  
**作者**：Alessandra Tafuro, Luigi Cacciani, Andrea Maria Zanchettin, Paolo Rocco  
**会议**：2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)  
**DOI**：https://doi.org/10.1109/IROS58592.2024.10802795

**主要内容（摘要未获取，依据题名与公开元数据）**：面向鞋底去毛刺，从毛刺识别到路径规划的完整流程。通常包括毛刺检测/分割、路径生成、机器人执行。具体算法需全文核对。

**算法要点（需全文确认）**：
- 毛刺识别；
- 路径规划；
- 机器人去毛刺流程。

**与当前任务的贴合点**：
- “识别 → 路径规划”的流程组织与我们的“root_line 输入 → 可行中心 → 路径”一致；
- 去毛刺应用背景相关。

**不能直接照搬的原因**：
- 对象是鞋底，材料/工具/几何不同；
- 可能没有砂轮有限圆柱碰撞和 72 角可行域；
- 没有两侧路径。

**可落地做法**：作为应用流程引用，不作为核心算法依据。

---

### 2.13 Zhang et al. 2026（★★★★★，Codex 补充）

**题名**：Collision-Free Robot Pose Optimization Method Based on Improved Algorithms  
**作者**：Yongwei Zhang, Qiao Xiao, Lujun Wan, Bo Jiang  
**期刊**：Machines, 2026, 14(1), 65  
**DOI**：https://doi.org/10.3390/machines14010065  
**开放获取**：https://www.mdpi.com/2075-1702/14/1/65

**主要内容（公开摘要）**：针对船舶构件机器人磨削在复杂受限空间中易碰撞的问题，提出无碰撞姿态优化方法。先建立正/逆运动学，把沿路径的姿态组织成**有向图**；在关节限位和奇异性约束下筛出可行姿态；用有向包围盒和 GJK 碰撞检测构造无碰撞姿态集合；再用改进 A* 搜索姿态序列，启发式包含关节空间几何距离和安全距离惩罚，目标是关节偏差最小、相邻姿态平滑。仿真结果显示搜索时间降低 15.8%，最小安全距离提高近 5 倍，累计关节变化最多降低 92.5%。

**算法要点**：
- 沿路径的离散姿态 → 有向图；
- 关节限位/奇异性/碰撞三重筛选；
- 有向包围盒 + GJK 碰撞检测；
- 改进 A*（多源启发式 + 安全距离惩罚）；
- 目标为关节偏差/平滑，不是中心路径长度。

**与当前任务的贴合点**：
- **图结构几乎同构**：路径点分层、每层多个候选姿态、只连相邻层；
- 它也把“节点可行”与“边/转移安全”分开处理，支持边级碰撞审计；
- 安全距离惩罚可以对应我们后续的 clearance/接触语义；
- 如果未来加入机器人关节角，改进 A* 的启发式可以借鉴。

**不能直接照搬的原因**：
- 论文目标是机器人关节偏差和安全性，不是砂轮中心折线总长度；
- 它需要机器人运动学、关节限位和奇异性模型，本项目当前没有；
- 它用 A* 是因为图可能不是简单分层链（或需要启发式加速）；我们的图是严格分层 DAG，DP 更简单且精确；
- 研究对象是船舶构件磨削，不是铸件飞边/砂轮端面接触；
- 该文为 2026 年新作，公开引用数 0，实验以仿真为主；引用时应注明其验证范围，不宜把它当作成熟工业结论。

**可落地做法**：
- 当前用 DP，不用 A*；但保留其“有向图 + 可行姿态集合 + 边安全”的建模语言；
- 若后续加入边级碰撞硬约束，图不再保证每层全连通，可借鉴其“先筛边、再搜索”的流程；
- 若后续加入机器人关节空间，可把边权从中心距离改为关节距离 + 安全惩罚，并考虑 A*。

---

### 2.14 Yuan et al. 2021（★★★★☆，Codex 补充）

**题名**：Tool Orientation Optimization and Path Planning for 5-Axis Machining  
**作者**：Chunming Yuan, Zhenpeng Mi, Xiaohong Jia, Fengming Lin, Liyong Shen  
**期刊**：Journal of Systems Science and Complexity, 2021, 34(1), 83–106（在线 2020）  
**DOI**：https://doi.org/10.1007/s11424-020-9270-1

**主要内容（公开摘要未获取；依据题名与 Codex 提供的信息，需全文复核）**：论文分两阶段处理五轴加工：先求平滑、无碰撞、无过切的刀具姿态，再规划 cutter-contact 点路径。其核心是把“局部可行姿态”与“沿路径连续可行”结合起来，而不是逐点独立选姿态。

**算法要点（需全文确认）**：
- 第一阶段：刀轴姿态优化（平滑、无碰撞、无过切）；
- 第二阶段：cutter-contact 点路径规划；
- 两阶段之间可能存在迭代/一致性检查。

**与当前任务的贴合点**：
- 我们的任务也可以看成“姿态可行”与“路径连接”两件事；论文提供了两阶段分解的参考；
- 它强调沿路径连续可行，支持我们做边级审计；
- 对“每点可行角集合”与“相邻点角度变化”的处理有参考价值。

**不能直接照搬的原因**：
- 研究对象是五轴平底刀加工曲面，不是砂轮中心点串联；
- 没有直接优化“两侧离散中心折线长度”；
- 两阶段分解可能无法保证全局最短；我们的 DP 在离散层上是全局最优；
- 需要全文核对它的碰撞/过切模型。

**可落地做法**：
- 当前仍用 DP 联合选择角度和路径；不拆成两阶段，避免“先选姿态、后连路径”导致次优；
- 在报告中把两阶段方法作为对照思路：如果用户更重视姿态平滑而非长度，可以先筛姿态再做路径；
- 后续若要加入机器人关节约束，可借鉴其两阶段组织。

---

### 2.15 Lu et al. 2021（★★★★☆，Codex 补充）

**题名**：Tool Path Optimization for Robotic Surface Machining by Using Sampling-Based Motion Planning Algorithms  
**作者**：Lei Lu, Jiong Zhang, Xiaoqing Tian, Jiang Han, Hao Wang  
**期刊**：Journal of Manufacturing Science and Engineering, 2021, 143(1)（在线 2020）  
**DOI**：https://doi.org/10.1115/1.4047734

**主要内容（公开摘要）**：机器人表面加工中，刀尖需要严格跟随给定刀路曲线，刀轴方向限制在一定范围内，机器人有多余自由度。论文在机器人关节空间中使用采样式运动规划（基于 OMPL）随机探索配置空间，生成离散的有效机器人状态路径；约束包括刀尖跟随、刀轴方向限制和通用碰撞检查；优化目标是机器人各关节路径长度之和。用两个例子验证有效性和最优性。

**算法要点**：
- 配置空间 = 机器人关节空间；
- 操作空间约束 = 刀尖跟随曲线 + 刀轴方向范围；
- 采样式规划（RRT 类）+ 碰撞检查；
- 目标 = 各关节路径长度之和；
- 使用 OMPL。

**与当前任务的贴合点**：
- 它明确把“跟随指定曲线 + 刀轴方向限制 + 碰撞 + 路径长度”放在同一个问题里，与我们的约束结构一致；
- 它的目标也是路径长度（虽然是关节空间），支持我们当前“长度最短”的目标；
- 它提醒我们：刀轴方向应作为约束范围，而不是由法向唯一决定。

**不能直接照搬的原因**：
- 它使用采样式运动规划，结果是概率完备/渐近最优，不是离散全局最优；
- 它优化机器人关节路径，不是砂轮中心三维折线；
- 它需要机器人运动学模型和 OMPL；
- 计算量和随机性不适合当前 72 离散角、415 点的确定性交付。

**可落地做法**：
- 当前问题规模小、层数固定、每层最多 72 节点，**DP 比采样规划更合适**；
- 在报告中引用其问题表述，说明“路径长度 + 曲线约束 + 姿态范围 + 碰撞”是合理的建模；
- 若未来加入机器人关节空间，可把 DP 的边权换成关节距离，或在其不可行时用采样规划补洞。

---

### 2.16 Onstein et al. 2023（★★★☆☆，Codex 补充）

**题名**：Automated Tool Trajectory Generation for Robotized Deburring of Cast Parts Based on 3D Scans  
**作者**：Ingrid Fjordheim Onstein, Magnus Bjerkeng, Kristian Martinsen  
**期刊**：Procedia CIRP, 2023, 118, 507–512  
**DOI**：https://doi.org/10.1016/j.procir.2023.06.087  
**SINTEF 记录**：https://www.sintef.no/en/publications/publication/2222321/

**主要内容（公开摘要）**：手工去除铸件毛刺有健康、安全和环境问题；自动化需要机器人、传感器和算法。论文提出基于 CAD 模型和 3D 扫描的机器人去毛刺轨迹生成系统：把配准后的扫描与 CAD 模型融合成每个工件的三维模型，再结合铸造工艺的先验知识生成刀具轨迹；在多个铸件模型上测试，并在 KUKA KR 60 上仿真和实验。

**算法要点**：
- CAD + 3D 扫描配准；
- 生成工件个体化三维模型；
- 结合铸造工艺先验生成轨迹；
- 机器人仿真 + 实机验证。

**与当前任务的贴合点**：
- 铸件去毛刺场景直接相关；
- “扫描 → 个体化模型 → 轨迹”的工程流程与我们的 root_line/网格数据流有相似性；
- 其实机验证方式可作为后续实验设计的参考。

**不能直接照搬的原因**：
- 公开摘要未显示其采用“离散可行中心最短路径”目标；
- 可能更多是工艺规则/几何启发式，而不是图优化；
- 没有砂轮有限圆柱碰撞和两侧路径；
- 需要全文核对其轨迹生成细节。

**可落地做法**：
- 作为铸件去毛刺应用和验证流程的引用；
- 不作为当前路径优化器的直接算法来源；
- 若后续要做真实工件实验，可参考其扫描配准和 KUKA 验证流程。

---

## 3. 这些论文对当前算法选择的具体影响

### 3.1 应该直接采用的

1. **分层最短路/动态规划**：当前问题的离散结构决定了 DP 是精确且最简单的；不需要为了“像论文”而引入 A*、RRT 或强化学习。
2. **每侧一维角度族**：借鉴 Chen 2015 的参考平面降维，把 `θ` 限制在 `q` 的同一侧半圆内。
3. **断开的可行域要显式处理**：借鉴 Wu 2025，不能假设可行角是连续弧；DP 的每层节点可以是不连续的角度集合。
4. **整条路径的碰撞审计**：借鉴 Ezair & Elber 2018，节点可行不等于边可行；至少输出边级审计结果。
5. **长度目标 + 姿态诊断**：借鉴 Weingartshofer 2023 和 Zhang 2024，当前先做纯长度，但记录相邻角度跳变、最大段长，为后续多目标优化留接口。
6. **配置空间表述**：借鉴 Lozano-Pérez 1983，把每个 `(root 点, 角度)` 解释为配置空间中的离散可行点；这不改变算法，但能让“可行域/禁止域”的表述更严谨。
7. **过切/干涉分区**：借鉴 Li 2017，若后续要处理凸边/凹边附近的窄可行带，可按局部曲面类型分区，再分别优化刀轴；当前 branch 7 的单侧可行现象正说明“同一套侧向定义不一定适用于所有局部几何”。
8. **有向图 + 边安全**：借鉴 Zhang 2026，把“每层候选姿态 + 相邻层转移”显式建成有向图，并把节点可行与边/转移安全分开记录；但当前用 DP，不用 A*。
9. **“曲线约束 + 姿态范围 + 碰撞 + 长度目标”的问题表述**：借鉴 Lu 2021，确认我们的建模是合理的；但当前用确定性 DP，不用 OMPL 采样规划。
10. **两阶段姿态/路径分解作为对照**：借鉴 Yuan 2021，可在报告中说明“先姿态后路径”与“联合 DP”的差异；当前选联合 DP，因为离散层上可全局最优。

### 3.2 暂不采用的

- **B 样条/QP 姿态平滑**（Wu 2025）：当前没有机床运动学，且用户要求 72 离散角；可作为后续。
- **连续保守可达体**（Ezair 2018）：计算量大，当前离散 72 角 + 有限圆柱 checker 已足够做几何演示；可作为后续严格化方向。
- **可变接触点**（Schneyer 2023）：会改变当前候选中心模型，需用户单独确认。
- **机器人关节空间优化**（Weingartshofer 2023、Guo 2019、Zhang 2026、Lu 2021）：当前没有机器人模型；这些论文的关节空间目标不能直接替代中心路径长度。
- **C-space 连续构造**（Mi 2016、Wang 2020）：当前保留 72 离散角，不强行连续化。
- **A\* / RRT / OMPL 采样规划**（Zhang 2026、Lu 2021）：当前图是严格分层 DAG，DP 已是全局最优；引入 A* 只会增加复杂度，采样规划还会引入随机性。
- **两阶段“先姿态后路径”**（Yuan 2021）：当前联合 DP 能同时选角度和路径，避免两阶段次优；两阶段仅作为对照。

### 3.3 对“两侧”定义的启示

Chen 2015 的参考平面方法说明：把三维姿态问题降为一维角度族是合理且常见的。我们的 `q = a × t` 给出了切平面内的横向基，`sign(e·q)` 给出左右两侧；这与“参考平面内的角度族”一致。关键是要在文档中明确：

- `side=+1` 是**砂轮轴向侧**（有限圆柱中心沿 `a` 偏移 `w/2`），不是路径左右侧；
- 路径左右侧是 `σ = sign(e·q)`，`q` 在切平面内、垂直于 root_line 切线；
- `θ = 0°` 或 `180°` 时 `e` 沿切线，`e·q = 0`，不属于任何一侧；当前 72 角网格包含这些角度，应在两侧路径中排除并记录。

---

### 3.4 本次实现实际采用/借鉴标记（2026-10-31）

> 本节是给当前代码交付用的“可核对清单”。✅ = 已直接用于本次实现；◐ = 只用于问题表述/对照，未改变算法；⚠️ = 本次明确未做，作为限制或后续工作。

| 论文 | 本次标记 | 具体用在哪里 | 没有用什么 |
|---|:---:|---|---|
| **Zhang et al. 2026, Machines** | ✅ | 把每个 root 采样点的候选中心组织成有向图；节点可行与相邻层转移分开；输出中保留 branch/候选索引 | 没有用改进 A*；没有关节角/奇异性/安全距离；没有机器人运动学 |
| **Chen, Xu, Tang 2015, JCDE** | ✅ | 每一侧用 `σ=sign(e·q)` 限制为一维半圆角度族；沿 branch 做全局最短序列，而不是逐点独立选 | 没有用参考平面构造叶盘通道；没有球头刀几何；没有连续姿态平滑 |
| **Wu et al. 2025, TMech** | ✅ | 显式允许每层可行角集合断开；DP 不假设连续弧；`no_path` 显式记录 | 没有用凸走廊、B 样条、QP、jerk 最小化 |
| **Lu et al. 2021, JMSE** | ✅ | 目标函数采用“沿路径的累计长度”；报告最大段长/角度跳变 | 没有用机器人关节空间、OMPL/RRT 采样规划；没有概率完备性 |
| **Weingartshofer et al. 2023, RCIM** | ◐ | 把 `θ` 表述为沿 root_line 的工艺自由度；路径长度作为简化目标 | 没有用关节空间连续优化、解析梯度、容差约束 |
| **Ezair & Elber 2018, CAD** | ⚠️ | 用于说明“节点可行不等于整条路径无碰撞” | **本次未做边级扫掠碰撞审计**；没有保守可达体/连续扫掠证明 |
| **Kim et al. 2015, CAD** | ◐ | 用“无过切”类比“砂轮不得切入工件本体” | 没有改碰撞判定；没有过切方向区域构造 |
| **Yuan et al. 2021, JSSC** | ◐ | 作为“两阶段姿态/路径”的对照说明 | 没有采用两阶段分解；当前用联合 DP |
| **Mi et al. 2016, IJAMT；Wang et al. 2020, CJA** | ◐ | 用 C-space/可行姿态区域的术语解释 72 角切片 | 没有构造连续 C-space/连续角区间 |
| **Schneyer et al. 2023, RA-L** | ✅ | 覆盖规划思想：用覆盖区间/间距决定何时可以跳过中间点；本轮用半径 r 的一维覆盖区间实现 | 没有用可变接触点、二维覆盖区域或自适应行距；仍固定接触点在 root 点 |
| **Guo et al. 2019；Zhang et al. 2024；Tafuro et al. 2024；Onstein et al. 2023** | ◐ | 去毛刺/铸件应用背景、流程组织和验证方式参考 | 没有采用其工艺参数/力控/扫描配准算法 |
| **Abele et al. 2016；Zhao et al. 2021** | ⚠️ | 作为 A*/TSP 和多分支排序的后续参考 | 当前不连接 branch 间空行程；没有 TSP/入口区域规划 |
| **Lozano-Pérez 1983；Li et al. 2017** | ◐ | 配置空间/过切分区的术语参考 | 没有实现连续 C-space 或曲面分区优化 |

#### 3.4.1 代码/文档中的引用落点

- `src/feilunqiege/planning/path_planning.py` 模块 docstring：分层 DAG、DP、侧向定义。
- `src/feilunqiege/visualization/path_obj.py` 文件头：写出方法借鉴（Zhang 2026 / Chen 2015 / Wu 2025 / Lu 2021）。
- `scripts/export_p1_full_delivery.py` 的 `path_planning` JSON 段和运行目录 `README.md`：写出引用与限制。
- `docs/path_planning_task_plan_20261030.md`：记录确认事项、边级审计未做、branch 7 例外。

#### 3.4.2 交付时建议的表述

> 本次路径规划采用分层有向图上的动态规划：每个 root 采样点为一层，层内节点为通过当前有限圆柱碰撞筛选的 72 个离散候选中心，边权为相邻层中心的三维欧氏距离。图结构借鉴 Zhang et al. (2026) 的离散姿态有向图；每侧一维角度族借鉴 Chen et al. (2015) 的参考平面降维；断开可行域的处理借鉴 Wu et al. (2025)；路径长度目标借鉴 Lu et al. (2021)。本次只做节点级可行性，未做边级扫掠碰撞审计；因此结果是几何演示，不是整条路径无碰撞证明或加工验收。

---

### 3.5 覆盖约束路径规划（2026-10-31 第二轮）

用户新增要求：不必严格经过每个 root 采样点；如果相邻两个选中砂轮的覆盖区域在 root_line 上重叠，中间采样点可以跳过；目标是覆盖整条 root_line 的前提下中心折线总长最短。

#### 3.5.1 本次采用的覆盖模型

- 规划抽象：选中点 i 的覆盖区间为沿 branch 弧长的 `[s_i - r, s_i + r]`，`r` 为砂轮半径；
- 相邻选中点满足 `s_next - s_prev <= 2r` 时覆盖区间重叠；
- 起点/终点只需被某个选中点的覆盖区间覆盖，不强制选首末点；
- 若某采样点在该侧完全没有可行候选（现有全域碰撞检测结果），覆盖区间不允许跨过它；覆盖边界取在该点与相邻可覆盖点的中点。这是 Q4 A 的保守实现，避免用几何半径跨过整段碰撞点；
- 覆盖是连续的弧长区间，不是仅检查离散采样点。

#### 3.5.2 算法

在原来的分层 DAG 上增加“跳边”：

- 节点仍是 `(point_index, candidate_index, lateral_side)`；
- 若 `coverage_left[j] <= coverage_right[i] + tol` 且 `coverage_right[j] > coverage_right[i] + tol`，允许从 i 直接到 j，跳过中间层；
- 目标为 `coverage_right[j] >= branch_length - tol`；
- 边权仍为中心三维欧氏距离；
- 覆盖约束的 DP 仍是 DAG 上的最短路，可用动态规划求全局最优；复杂度约为 `O(n · K · 72²)`，`K` 为 2r 覆盖窗口内的采样点数。

#### 3.5.3 论文对应

| 论文 | 本次标记 | 用途 | 没有用什么 |
|---|:---:|---|---|
| **Schneyer et al. 2023, RA-L** | ✅ | 覆盖规划思想：用覆盖区间/间距决定何时可以跳过中间点；本轮实现一维弧长覆盖 | 没有可变接触点、二维覆盖区域、自适应行距 |
| **Weingartshofer et al. 2023, RCIM** | ◐ | 把覆盖/工艺容差表述为沿路径的约束 | 没有连续关节空间优化 |
| **Lu et al. 2021, JMSE** | ◐ | 仍然最小化中心折线长度 | 没有机器人关节空间/采样规划 |
| **Abele et al. 2016, Procedia CIRP** | ⚠️ | A*/TSP 是后续分支排序的参考 | 当前不是点集 TSP；覆盖 DP 是链上跳边 |
| **Zhang et al. 2026, Machines** | ◐ | 有向图/节点-边安全的分层表述仍适用 | 没有 A*、关节角、安全距离 |

#### 3.5.4 交付时建议的表述

> 第二版路径规划把“每个采样点必须选一个中心”放宽为“覆盖区间首尾相接”。每个选中点覆盖 root_line 上 `[s-r, s+r]` 的连续弧长；相邻选中点覆盖区间重叠即可跳过中间采样点。覆盖区间不能跨过该侧完全无可行候选的采样点（沿用现有全域碰撞结果）。在此约束下用动态规划求中心折线最短路径。该覆盖模型是一维规划抽象，不是砂轮实体与工件表面的精确接触/切除仿真；本轮仍未做边级扫掠碰撞审计。

---

## 4. 检索与核对说明

- Crossref：核对题名、作者、年份、期刊、DOI。
- OpenAlex：补充摘要、引用数、关键词。
- Semantic Scholar：补充部分摘要。
- 未使用 Sci-Hub 或非公开全文；未下载受版权保护的 PDF。
- 标注“摘要未获取”的条目，正式引用前需获取全文核对算法细节。
- 2026-10-31 整合 Codex 补充的 4 篇：Zhang 2026（有向图 + 改进 A*）、Yuan 2021（两阶段姿态/路径）、Lu 2021（采样式机器人路径优化）、Onstein 2023（铸件扫描 → 去毛刺轨迹）。其中 Zhang 2026 的图结构与当前问题最接近，但其目标函数是关节偏差，不是中心路径长度。
- Codex 指出的“配置 72 vs 180”已核对：`configs/default.json` 的 `sampling.candidate_angles=180`，当前三 OBJ 交付的 72 角来自 `scripts/export_p1_full_delivery.py` 硬编码；路径规划必须沿用同一 72 角数组。详见 `docs/path_planning_task_plan_20261030.md` 第 2.4 节。
- 本文件只做文献调研，不修改代码、配置或正式结果。

---

## 5. 建议的阅读顺序

1. Chen 2015（最接近的“沿路径一维姿态族 + 全局平滑”）
2. Zhang 2026（有向图结构 + 改进 A* + 边安全；结构与当前问题最接近）
3. Weingartshofer 2023（把工艺自由度/容差/碰撞统一优化）
4. Ezair & Elber 2018（路径级碰撞保证）
5. Wu 2025（断开的可行姿态域 + 平滑）
6. Kim 2015（无过切姿态约束）
7. Yuan 2021（两阶段姿态优化 + 路径规划，作为对照）
8. Lu 2021（曲线约束 + 姿态范围 + 碰撞 + 长度目标的采样式建模）
9. Schneyer 2023（砂轮接触点自由度 + 覆盖）
10. Feng 2021（特征曲线 → 磨削轨迹）
11. Onstein 2023（铸件扫描 → 去毛刺轨迹的工程流程）
12. 其余作为应用/背景引用
