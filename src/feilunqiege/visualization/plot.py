"""可行砂轮中心总览图接口占位。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


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
    fig = plt.figure(figsize=(13, 9), dpi=140)
    axis = fig.add_subplot(111, projection="3d")
    # 工件只抽样显示背景点，避免 100 MB 级网格把总览图和内存占满。
    if workpiece_mesh is not None:
        vertices = np.asarray(workpiece_mesh.vertices)
        step = max(1, len(vertices) // 12000)
        axis.scatter(vertices[::step, 0], vertices[::step, 1], vertices[::step, 2], s=1, c="lightgray", alpha=0.18, label="workpiece samples")
    # 按 branch_id 单独画线，确保多个根部线分支之间不产生假连接。
    branch_ids = np.asarray(sampled_curve.branch_ids)
    for branch_id in np.unique(branch_ids):
        mask = branch_ids == branch_id
        axis.plot(points[mask, 0], points[mask, 1], points[mask, 2], linewidth=1.2, label=f"root branch {int(branch_id)}")
    # 可行中心点可按上限抽样绘制，但 OBJ 导出会保留全部可行点。
    feasible_centers = centers[feasible]
    if len(feasible_centers):
        step = max(1, len(feasible_centers) // 20000)
        axis.scatter(feasible_centers[::step, 0], feasible_centers[::step, 1], feasible_centers[::step, 2], s=2, c="crimson", alpha=0.35, label="feasible centers")
    # 黑色点表示每个候选集合对应的目标接触点，不是工件三角面。
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=4, c="black", alpha=0.8, label="target points")
    axis.set_xlabel("X (normalized units)")
    axis.set_ylabel("Y (normalized units)")
    axis.set_zlabel("Z (normalized units)")
    axis.set_title(f"Zero-thickness wheel feasible centers  r={radius:g}, delta={delta:g}")
    axis.legend(loc="upper left", fontsize=8)
    _set_equal_axes(axis, np.vstack((points, feasible_centers)) if len(feasible_centers) else points)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _set_equal_axes(axis: Any, values: np.ndarray) -> None:
    """让三维图三个坐标轴使用相同尺度。"""

    low = values.min(axis=0)
    high = values.max(axis=0)
    center = (low + high) / 2.0
    half = max(float((high - low).max()) / 2.0, 1e-9)
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)
