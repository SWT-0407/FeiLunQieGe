"""无厚度砂轮圆盘的候选中心接口占位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DiskCandidate:
    """单个候选圆盘的几何描述。"""

    contact_point: Any
    center: Any
    radius_direction: Any
    tangent_direction: Any
    axis_direction: Any
    radius: float


@dataclass(frozen=True)
class CandidateSet:
    """批量候选数组，避免为每个候选创建 Python 对象。"""

    contact_points: Any
    centers: Any
    radius_directions: Any
    tangent_directions: Any
    axis_directions: Any
    point_indices: Any
    radial_values: Any
    angles: Any
    radius: float


def generate_disk_candidates(
    points: Any,
    frames: Any,
    radius: float,
    delta: float,
    angle_count: int,
    radial_samples: int,
) -> Any:
    """生成候选无厚度圆盘中心和姿态。

    当前候选半径区间包含两个端点，便于对 `delta` 敏感性进行可重复采样。
    砂轮圆盘平面由 `tangent_direction` 与 `radius_direction` 张成，
    轴向为 `tangent_direction × radius_direction`。
    """

    # 先校验半径区间，保证 rho 不会变成负数且确实落在 [r-delta, r]。
    if radius <= 0 or delta < 0 or delta >= radius:
        raise ValueError("必须满足 radius > 0 且 0 <= delta < radius")

    # 角度和径向采样都是离散近似；角度不重复采样 2π 端点。
    if angle_count < 4 or radial_samples < 1:
        raise ValueError("候选角度至少需要 4 个，径向采样数至少为 1")
    angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
    radial_values = np.linspace(radius - delta, radius, radial_samples)

    # 以下列表按批次收集数组，最后一次性拼接，避免为每个候选创建 Python 对象。
    all_points = []
    all_centers = []
    all_radials = []
    all_tangents = []
    all_axes = []
    all_indices = []
    all_rhos = []
    all_angles = []
    for point_index, (point, frame) in enumerate(zip(points, frames, strict=True)):
        # 用局部法平面的两个基向量参数化半径方向，保证其与切线正交。
        e = (
            np.cos(angles)[:, None] * np.asarray(frame.radial_basis_1)
            + np.sin(angles)[:, None] * np.asarray(frame.radial_basis_2)
        )
        e /= np.linalg.norm(e, axis=1)[:, None]
        t = np.repeat(np.asarray(frame.tangent)[None, :], angle_count, axis=0)

        # t×e 给出圆盘轴向；这里只表达姿态，不生成有厚度的实体网格。
        axis = np.cross(t, e)
        axis /= np.linalg.norm(axis, axis=1)[:, None]
        for rho in radial_values:
            # 每个 rho 都保留同一组角度，便于后续按角度统计可行弧区间。
            all_points.append(np.repeat(np.asarray(point)[None, :], angle_count, axis=0))
            all_centers.append(np.asarray(point)[None, :] + rho * e)
            all_radials.append(e)
            all_tangents.append(t)
            all_axes.append(axis)
            all_indices.append(np.full(angle_count, point_index, dtype=np.int64))
            all_rhos.append(np.full(angle_count, rho, dtype=float))
            all_angles.append(angles)
    candidate_set = CandidateSet(
        contact_points=np.vstack(all_points),
        centers=np.vstack(all_centers),
        radius_directions=np.vstack(all_radials),
        tangent_directions=np.vstack(all_tangents),
        axis_directions=np.vstack(all_axes),
        point_indices=np.concatenate(all_indices),
        radial_values=np.concatenate(all_rhos),
        angles=np.concatenate(all_angles),
        radius=float(radius),
    )
    # 生成后再次检查 e·t=0，避免局部标架或数值误差破坏工艺几何约束。
    orthogonality = np.abs(np.einsum("ij,ij->i", candidate_set.radius_directions, candidate_set.tangent_directions))
    if float(orthogonality.max(initial=0.0)) > 1e-10:
        raise ValueError("候选砂轮半径方向未满足 e·t=0")
    return candidate_set
