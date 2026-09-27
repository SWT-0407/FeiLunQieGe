"""可行砂轮中心的全局图和局部解释图。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# 命令行批处理不需要 Qt 窗口。固定使用 Agg 可以规避 Windows 下 Qt 后端
# 在保存 PNG 时出现 ``OSError: [Errno 22] Invalid argument`` 的问题。
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_feasible_centers(
    sampled_curve: Any,
    decisions: Any,
    output_path: str | Path,
    *,
    radius: float,
    delta: float,
    workpiece_mesh: Any | None = None,
) -> Path:
    """绘制全部目标采样点对应的可行中心总览图。

    图中同时展示根部交线、工件顶点采样和通过近似筛选的候选中心。
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 将碰撞结果和坐标转换为绘图数组；绘图不改变判定结果。
    feasible = np.asarray(decisions.feasible, dtype=bool)
    points = np.asarray(sampled_curve.points)
    if decisions.candidates is None:
        raise ValueError("CollisionResult 缺少 candidates，无法绘图")
    centers = np.asarray(decisions.candidates.centers)
    # constrained layout 会在保存前自动为图例和坐标轴留白，不再依赖 tight bbox。
    fig = plt.figure(figsize=(13, 9), dpi=140, layout="constrained")
    axis = fig.add_subplot(111, projection="3d")

    # 使用半透明三角面显示工件轮廓；这比稀疏顶点更接近 MeshLab 中的 T 形工件。
    if workpiece_mesh is not None:
        _add_workpiece_surface(axis, workpiece_mesh)

    # 可行中心点可按上限抽样绘制，但 OBJ 导出会保留全部可行点。
    feasible_centers = centers[feasible]
    if len(feasible_centers):
        step = max(1, len(feasible_centers) // 20000)
        axis.scatter(
            feasible_centers[::step, 0],
            feasible_centers[::step, 1],
            feasible_centers[::step, 2],
            s=3,
            c="#D55E00",
            alpha=0.55,
            label="[3] feasible wheel centers",
        )

    # 最后覆盖绘制蓝色根部线，使半径很小时它不会被大量橙色候选点完全遮住。
    branch_ids = np.asarray(sampled_curve.branch_ids)
    _plot_root_branches(axis, points, branch_ids, linewidth=2.5)

    # 黑色点表示目标接触点；全局图抽样显示，避免把蓝色根部线涂黑。
    target_step = max(1, len(points) // 80)
    axis.scatter(
        points[::target_step, 0],
        points[::target_step, 1],
        points[::target_step, 2],
        s=10,
        c="black",
        alpha=0.85,
        label="[4] target points",
    )
    axis.set_xlabel("X (normalized units)")
    axis.set_ylabel("Y (normalized units)")
    axis.set_zlabel("Z (normalized units)")
    axis.set_title(f"Global inspection view: zero-thickness wheel  r={radius:g}, delta={delta:g}")
    axis.legend(loc="upper left", fontsize=8)
    values = [points, feasible_centers]
    if workpiece_mesh is not None:
        values.append(np.asarray(workpiece_mesh.vertices, dtype=float))
    _set_equal_axes(axis, np.vstack([value for value in values if len(value)]))
    axis.view_init(elev=22, azim=-62)

    # 在图内直接声明全局尺度下 r 很小，防止把贴近蓝线的红点误认成缺失。
    axis.text2D(
        0.02,
        0.02,
        "[1] Gray surface = workpiece body\nThe wheel radius is too small to read at this global scale; see the local view.",
        transform=axis.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#666666", "alpha": 0.9},
    )
    _save_figure(fig, output)
    return output


def plot_workpiece_orientation_guide(
    sampled_curve: Any,
    workpiece_mesh: Any,
    output_path: str | Path,
) -> Path:
    """用同一工件的透视图和正视图解释“T 形看起来不同”的原因。"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(sampled_curve.points, dtype=float)
    branch_ids = np.asarray(sampled_curve.branch_ids, dtype=np.int64)
    vertices = np.asarray(workpiece_mesh.vertices, dtype=float)

    # 两个面板使用完全相同的网格和根部线，只改变相机角度。
    fig = plt.figure(figsize=(13, 6), dpi=150, layout="constrained")
    perspective = fig.add_subplot(121, projection="3d")
    front = fig.add_subplot(122, projection="3d")
    for axis in (perspective, front):
        _add_workpiece_surface(axis, workpiece_mesh)
        _plot_root_branches(axis, points, branch_ids, linewidth=2.4)
        _set_equal_axes(axis, vertices)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.set_proj_type("ortho")

    perspective.view_init(elev=22, azim=-62)
    perspective.set_title("[A] 3D perspective used by the overview")
    perspective.legend(loc="upper left", fontsize=8)

    # elev=0, azim=0 沿 Y 方向观察，工件本体呈现熟悉的 T 形投影。
    front.view_init(elev=0, azim=0)
    front.set_title("[B] Front projection: the same workpiece appears T-shaped")
    front.legend(loc="upper left", fontsize=8)
    fig.suptitle("Same mesh, different camera direction (mesh_without_flash.obj)")
    _save_figure(fig, output)
    return output


def plot_local_feasibility_explanation(
    sampled_curve: Any,
    decisions: Any,
    output_path: str | Path,
    *,
    radius: float,
    delta: float,
) -> tuple[Path, int]:
    """绘制一个代表性目标点的局部法平面，以解释半径、余量和可行圆弧。

    图中的局部横纵轴不是工件的全局 X/Y，而是垂直于根部线切线 ``t_i``
    的二维平面。算法会优先选择既有可行候选也有不可行候选的点，从而
    同时显示两种判定；若不存在这种点，则选择可行率最低的目标点。
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = decisions.candidates
    if candidates is None:
        raise ValueError("CollisionResult 缺少 candidates，无法绘制局部解释图")

    # 统计每个目标点的候选总数和可行数，选择最接近 50% 可行率的部分可行点。
    feasible = np.asarray(decisions.feasible, dtype=bool)
    point_indices = np.asarray(candidates.point_indices, dtype=np.int64)
    point_count = len(sampled_curve.points)
    total_counts = np.bincount(point_indices, minlength=point_count)
    feasible_counts = np.bincount(point_indices[feasible], minlength=point_count)
    partial = np.flatnonzero((feasible_counts > 0) & (feasible_counts < total_counts))
    if len(partial):
        ratios = feasible_counts[partial] / total_counts[partial]
        point_index = int(partial[np.argmin(np.abs(ratios - 0.5))])
    else:
        ratios = np.divide(feasible_counts, total_counts, out=np.zeros_like(feasible_counts, dtype=float), where=total_counts > 0)
        point_index = int(np.argmin(ratios))

    # 在局部法平面中，候选点坐标可以直接由 rho、theta 写成极坐标。
    mask = point_indices == point_index
    local_feasible = feasible[mask]
    angles = np.asarray(candidates.angles, dtype=float)[mask]
    radial_values = np.asarray(candidates.radial_values, dtype=float)[mask]
    local_x = radial_values * np.cos(angles)
    local_y = radial_values * np.sin(angles)

    # 左图显示全部离散候选，右图只连接最外层 rho=r 上的连续可行角段。
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=150, layout="constrained")
    axes[0].scatter(local_x[~local_feasible], local_y[~local_feasible], s=14, c="#999999", marker="x", label="rejected candidate")
    axes[0].scatter(local_x[local_feasible], local_y[local_feasible], s=12, c="#D55E00", label="feasible wheel center")
    axes[0].scatter([0.0], [0.0], s=45, c="black", zorder=4, label="target/contact point")
    _draw_radius_band(axes[0], radius, delta)
    axes[0].annotate(
        "r",
        xy=(radius, 0.0),
        xytext=(radius * 0.48, radius * 0.10),
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=12,
    )
    axes[0].annotate(
        "delta",
        xy=(radius - delta, -radius * 0.14),
        xytext=(radius, -radius * 0.28),
        arrowprops={"arrowstyle": "<->", "color": "#0072B2"},
        color="#0072B2",
        fontsize=11,
    )
    axes[0].set_title("[A] Candidate centers in the plane perpendicular to tangent $t_i$")
    axes[0].legend(loc="upper right", fontsize=8)

    # 只取外圈候选并按角度排序，把相邻可行点连接成用户所说的“可行圆弧”。
    outer = mask & np.isclose(np.asarray(candidates.radial_values, dtype=float), radius)
    outer_angles = np.asarray(candidates.angles, dtype=float)[outer]
    outer_feasible = feasible[outer]
    order = np.argsort(outer_angles)
    outer_angles = outer_angles[order]
    outer_feasible = outer_feasible[order]
    _draw_radius_band(axes[1], radius, delta)
    _plot_boolean_arcs(axes[1], outer_angles, outer_feasible, radius)
    axes[1].scatter([0.0], [0.0], s=45, c="black", zorder=4, label="target/contact point")
    axes[1].set_title("[B] Feasible arc on the outer radius $rho=r$")
    axes[1].legend(loc="upper right", fontsize=8)

    # 两个局部图统一比例，避免圆被拉伸成椭圆。
    limit = radius * 1.32
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_xlabel("local radial basis 1 (normalized units)")
        axis.set_ylabel("local radial basis 2 (normalized units)")
        axis.grid(True, linewidth=0.5, alpha=0.35)
    fig.suptitle(
        f"Local explanation at target point {point_index}: r={radius:g}, delta={delta:g}, "
        f"feasible={int(feasible_counts[point_index])}/{int(total_counts[point_index])}"
    )
    _save_figure(fig, output)
    return output, point_index


def _add_workpiece_surface(axis: Any, workpiece_mesh: Any) -> None:
    """以半透明三角面绘制工件，并给图例添加一个同色代理图形。"""

    vertices = np.asarray(workpiece_mesh.vertices, dtype=float)
    faces = np.asarray(workpiece_mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    surface = Poly3DCollection(triangles, facecolor="#B8BDC5", edgecolor="none", alpha=0.30)
    axis.add_collection3d(surface)
    axis.scatter([], [], [], s=30, c="#B8BDC5", marker="s", label="[1] workpiece body")


def _plot_root_branches(axis: Any, points: np.ndarray, branch_ids: np.ndarray, *, linewidth: float) -> None:
    """逐分支绘制蓝色根部线，并只为第一条分支创建一个图例项。"""

    unique_branches = np.unique(branch_ids)
    for branch_id in unique_branches:
        mask = branch_ids == branch_id
        label = "[2] flash root line" if int(branch_id) == int(unique_branches[0]) else None
        axis.plot(points[mask, 0], points[mask, 1], points[mask, 2], color="#0072B2", linewidth=linewidth, label=label)


def _draw_radius_band(axis: Any, radius: float, delta: float) -> None:
    """画出 rho=r 和 rho=r-delta 两条边界圆，辅助理解径向容差带。"""

    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    axis.plot(radius * np.cos(theta), radius * np.sin(theta), color="black", linewidth=1.0, label="outer radius r")
    inner = radius - delta
    axis.plot(inner * np.cos(theta), inner * np.sin(theta), color="#0072B2", linestyle="--", linewidth=1.0, label="inner radius r-delta")


def _plot_boolean_arcs(axis: Any, angles: np.ndarray, feasible: np.ndarray, radius: float) -> None:
    """将外圈离散可行布尔值画成连续红弧，不跨越不可行角度。"""

    if not len(angles):
        return
    step = 2.0 * np.pi / len(angles)
    extended_angles = np.concatenate((angles, [angles[0] + 2.0 * np.pi]))
    extended_feasible = np.concatenate((feasible, [feasible[0]]))
    label_used = False
    for index in range(len(angles)):
        if extended_feasible[index] and extended_feasible[index + 1] and extended_angles[index + 1] - extended_angles[index] <= step * 1.1:
            arc_angles = extended_angles[index : index + 2]
            axis.plot(
                radius * np.cos(arc_angles),
                radius * np.sin(arc_angles),
                color="#D55E00",
                linewidth=4.0,
                label="feasible center arc" if not label_used else None,
            )
            label_used = True


def _save_figure(fig: Any, output: Path) -> None:
    """使用非交互后端保存 PNG，并确保发生异常时也关闭图形对象。"""

    try:
        fig.savefig(output, facecolor="white")
    finally:
        plt.close(fig)


def _set_equal_axes(axis: Any, values: np.ndarray) -> None:
    """让三维图三个坐标轴使用相同尺度。"""

    low = values.min(axis=0)
    high = values.max(axis=0)
    center = (low + high) / 2.0
    half = max(float((high - low).max()) / 2.0, 1e-9)
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)
