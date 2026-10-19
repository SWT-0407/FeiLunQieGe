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
    """生成切平面内的候选无厚度圆盘中心和姿态。

    当前候选半径区间包含两个端点，便于对 `delta` 敏感性进行可重复采样。
    ``frame.radial_basis_1/2`` 现在是切平面基向量；候选中心在切平面内
    绕接触点转动，砂轮轴向固定为工件表面法向，圆盘平面因此与切平面平行。
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
        # 用局部切平面的两个基向量参数化中心偏移方向；这使中心轨迹圆
        # 与切平面平行，而不是落在法平面内。
        e = (
            np.cos(angles)[:, None] * np.asarray(frame.radial_basis_1)
            + np.sin(angles)[:, None] * np.asarray(frame.radial_basis_2)
        )
        e /= np.linalg.norm(e, axis=1)[:, None]
        t = np.repeat(np.asarray(frame.tangent)[None, :], angle_count, axis=0)

        # 轴向使用表面法向；这里只表达姿态，不生成有厚度的实体网格。
        axis_vector = np.asarray(frame.axis if frame.axis is not None else frame.surface_normal, dtype=float)
        if axis_vector.shape != (3,) or np.linalg.norm(axis_vector) <= 1e-12:
            raise ValueError("切平面候选必须提供有效的表面法向作为砂轮轴向")
        axis_vector /= np.linalg.norm(axis_vector)
        axis = np.repeat(axis_vector[None, :], angle_count, axis=0)
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
    # 生成后再次检查完整的局部姿态，避免后续绘图或碰撞模块误用错误方向。
    # e 与 t 同属切平面，a 是表面法向；因此圆盘平面与切平面平行。
    radius_tangent_dot = np.abs(
        np.einsum("ij,ij->i", candidate_set.radius_directions, candidate_set.tangent_directions)
    )
    axis_tangent_dot = np.abs(
        np.einsum("ij,ij->i", candidate_set.axis_directions, candidate_set.tangent_directions)
    )
    axis_radius_dot = np.abs(
        np.einsum("ij,ij->i", candidate_set.axis_directions, candidate_set.radius_directions)
    )
    if (
        float(axis_tangent_dot.max(initial=0.0)) > 1e-10
        or float(axis_radius_dot.max(initial=0.0)) > 1e-10
    ):
        raise ValueError("候选砂轮姿态未满足 a·t=0、a·e=0（圆盘平面平行切平面）")
    return candidate_set


def generate_tangent_cylinder_candidates(
    contact: Any,
    frame: Any,
    radius: float,
    width: float,
    angles: Any,
    point_index: int,
    *,
    side: float = 1.0,
    axis_reference: Any | None = None,
) -> CandidateSet:
    """生成圆周切线与 root line 平行的有限厚度砂轮姿态。

    阶段二的旧圆盘生成器继续保留。阶段三使用本函数时，接触点到中心的
    径向方向 ``e`` 必须满足 ``e·t=0``；随后由 ``t×e`` 得到砂轮轴向，
    并沿该有符号轴向偏移 ``h/2``，保证靠近工件的端面通过接触点。
    """

    if radius <= 0.0 or width <= 0.0:
        raise ValueError("radius and width must be positive")
    if abs(float(side)) != 1.0:
        raise ValueError("side must be +1 or -1")
    angle_array = np.asarray(angles, dtype=float)
    if angle_array.ndim != 1 or len(angle_array) == 0:
        raise ValueError("angles must be a non-empty one-dimensional array")

    contact_array = np.asarray(contact, dtype=float)
    if contact_array.shape != (3,):
        raise ValueError("contact must have shape (3,)")
    tangent = np.asarray(frame.tangent, dtype=float)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        raise ValueError("frame tangent cannot be normalized")
    tangent = tangent / tangent_norm

    reference = axis_reference
    if reference is None:
        reference = frame.axis if frame.axis is not None else frame.surface_normal
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (3,):
        raise ValueError("axis reference must have shape (3,)")
    # 现有表面法向只用于确定角度零点。去掉它在 t 上的分量后，候选径向
    # 平面严格落在 t 的正交补空间内。
    reference = reference - float(np.dot(reference, tangent)) * tangent
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm <= 1e-12:
        raise ValueError("axis reference is parallel to the root-line tangent")
    basis_1 = reference / reference_norm
    basis_2 = np.cross(tangent, basis_1)
    basis_2 /= max(float(np.linalg.norm(basis_2)), 1e-12)

    radial = np.cos(angle_array)[:, None] * basis_1[None, :] + np.sin(angle_array)[:, None] * basis_2[None, :]
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    # 每个径向方向都重新计算轴向，而不是复用固定表面法向；side 决定哪一
    # 个端面与目标切削平面重合。
    axes = np.cross(np.repeat(tangent[None, :], len(angle_array), axis=0), radial)
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    axes *= float(side)
    centers = contact_array[None, :] + float(radius) * radial + float(width) / 2.0 * axes

    return CandidateSet(
        contact_points=np.repeat(contact_array[None, :], len(angle_array), axis=0),
        centers=centers,
        radius_directions=radial,
        tangent_directions=np.repeat(tangent[None, :], len(angle_array), axis=0),
        axis_directions=axes,
        point_indices=np.full(len(angle_array), int(point_index), dtype=np.int64),
        radial_values=np.full(len(angle_array), float(radius), dtype=float),
        angles=angle_array,
        radius=float(radius),
    )
