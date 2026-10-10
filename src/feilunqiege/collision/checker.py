"""砂轮候选与工件本体的碰撞检测接口占位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CollisionDecision:
    """单个候选的可行性及审计信息。"""

    feasible: bool
    minimum_clearance: float | None
    reason: str


@dataclass(frozen=True)
class CollisionResult:
    """候选筛选结果。"""

    feasible: Any
    nearest_distance: Any
    clearance: Any
    method: str
    candidates: Any | None = None
    reason: Any | None = None
    witness_workpiece: Any | None = None
    witness_tool: Any | None = None
    witness_region: Any | None = None


@dataclass(frozen=True)
class CollisionIndex:
    """由工件网格表面采样点建立的快速查询索引。"""

    samples: Any
    tree: Any


def build_collision_index(workpiece_mesh: Any) -> CollisionIndex:
    """用顶点、三角形重心和三角形边中点建立表面点索引。"""

    # 将输入网格转换为连续浮点数组，后续只建立只读式查询索引。
    vertices = np.asarray(workpiece_mesh.vertices, dtype=float)
    faces = np.asarray(workpiece_mesh.faces, dtype=np.int64)
    triangles = vertices[faces]

    # 顶点、三角形重心和三条边中点共同构成表面采样点，提升近似包络覆盖率。
    centroids = triangles.mean(axis=1)
    midpoints = np.vstack(
        (
            (triangles[:, 0] + triangles[:, 1]) / 2.0,
            (triangles[:, 1] + triangles[:, 2]) / 2.0,
            (triangles[:, 2] + triangles[:, 0]) / 2.0,
        )
    )
    samples = np.vstack((vertices, centroids, midpoints))
    return CollisionIndex(samples=samples, tree=cKDTree(samples))


def check_candidates(
    candidates: Any,
    workpiece_mesh: Any,
    clearance: float,
    *,
    collision_index: CollisionIndex | None = None,
) -> CollisionResult:
    """筛选不与工件本体碰撞的候选。

    最小闭环使用“候选中心到工件表面采样点的最近距离 >= rho + clearance”
    作为近似球形包络筛选。它会忽略圆盘朝向和三角面内部的精确相交，
    可能误判可行；`method` 会在摘要中明确写出这个限制。
    """

    if clearance < 0:
        raise ValueError("碰撞 clearance 不能为负")
    # KDTree 只回答“候选中心到采样表面的最近距离”，不等价于精确三角形相交。
    index = collision_index if collision_index is not None else build_collision_index(workpiece_mesh)
    nearest_distance, _ = index.tree.query(np.asarray(candidates.centers), workers=-1)

    # 用候选实际 rho 作为球形保守包络半径，再减去它得到剩余安全间隙。
    effective_radius = np.asarray(candidates.radial_values, dtype=float)
    clearance_values = nearest_distance - effective_radius

    # 只有剩余间隙不小于用户给定容差的候选才标为可行。
    feasible = clearance_values >= float(clearance)
    return CollisionResult(
        feasible=feasible,
        nearest_distance=nearest_distance,
        clearance=clearance_values,
        method="approximate_surface_samples_spherical_envelope",
        candidates=candidates,
        reason=np.where(feasible, "safe", "wheel_outer_collision"),
        witness_workpiece=None,
        witness_tool=None,
        witness_region=np.where(feasible, "none", "sampled_center_envelope"),
    )


def check_finite_cylinder_candidates(
    candidates: Any,
    workpiece_mesh: Any,
    clearance: float,
    width: float,
    *,
    collision_index: CollisionIndex | None = None,
    nearest_samples: int = 12,
    chunk_size: int = 20000,
) -> CollisionResult:
    """使用有限宽度圆柱的局部姿态距离进行可解释的近似筛选。

    工件仍由顶点、三角形质心和边中点组成的离散表面索引表示。对每个
    候选中心查询若干近邻，再将近邻投影到候选砂轮的 ``(e, t, a)`` 标架，
    计算它到有限圆柱实体的有符号距离。该方法会考虑轴向 ``a`` 和宽度
    ``width``，但不替代精确三角面-圆柱布尔运算；见证点因此明确标记为
    ``sampled_surface_nearest``。
    """

    if clearance < 0:
        raise ValueError("碰撞 clearance 不能为负")
    if width <= 0:
        raise ValueError("有限厚度砂轮的 width 必须为正")
    if nearest_samples < 1 or chunk_size < 1:
        raise ValueError("nearest_samples 和 chunk_size 必须为正")
    index = collision_index if collision_index is not None else build_collision_index(workpiece_mesh)
    centers = np.asarray(candidates.centers, dtype=float)
    radials = np.asarray(candidates.radius_directions, dtype=float)
    tangents = np.asarray(candidates.tangent_directions, dtype=float)
    axes = np.asarray(candidates.axis_directions, dtype=float)
    radii = np.asarray(candidates.radial_values, dtype=float)
    sample_count = len(centers)
    signed_clearance = np.full(sample_count, np.inf, dtype=float)
    witness_workpiece = np.full((sample_count, 3), np.nan, dtype=float)
    witness_tool = np.full((sample_count, 3), np.nan, dtype=float)
    witness_region = np.full(sample_count, "unresolved", dtype=object)

    # 分块查询近邻，避免 50 万级候选一次性展开为过大的三维数组。
    half_width = float(width) / 2.0
    for start in range(0, sample_count, int(chunk_size)):
        stop = min(start + int(chunk_size), sample_count)
        block_centers = centers[start:stop]
        distances, nearest_indices = index.tree.query(block_centers, k=int(nearest_samples), workers=-1)
        if nearest_samples == 1:
            distances = distances[:, None]
            nearest_indices = nearest_indices[:, None]
        workpiece_points = index.samples[np.asarray(nearest_indices, dtype=np.int64)]
        delta = workpiece_points - block_centers[:, None, :]
        radial_coord = np.einsum("bkj,bj->bk", delta, radials[start:stop])
        tangent_coord = np.einsum("bkj,bj->bk", delta, tangents[start:stop])
        axial_coord = np.abs(np.einsum("bkj,bj->bk", delta, axes[start:stop]))
        radial_distance = np.hypot(radial_coord, tangent_coord)
        outside_radial = np.maximum(radial_distance - radii[start:stop, None], 0.0)
        outside_axial = np.maximum(axial_coord - half_width, 0.0)
        outside_distance = np.hypot(outside_radial, outside_axial)
        inside_penetration = np.minimum(
            radii[start:stop, None] - radial_distance,
            half_width - axial_coord,
        )
        signed_distance = np.where(
            (radial_distance <= radii[start:stop, None]) & (axial_coord <= half_width),
            -inside_penetration,
            outside_distance,
        )
        local_index = np.argmin(signed_distance, axis=1)
        rows = np.arange(stop - start)
        signed_clearance[start:stop] = signed_distance[rows, local_index]
        witness_workpiece[start:stop] = workpiece_points[rows, local_index]

        # 将工件见证点投影回圆柱表面，形成可在三维图中画出的砂轮见证点。
        nearest_radial = radial_coord[rows, local_index]
        nearest_tangent = tangent_coord[rows, local_index]
        nearest_axial = np.clip(
            np.einsum("bkj,bj->bk", delta, axes[start:stop])[rows, local_index],
            -half_width,
            half_width,
        )
        radial_norm = np.hypot(nearest_radial, nearest_tangent)
        safe_norm = np.where(radial_norm > 1e-12, radial_norm, 1.0)
        projected_radial = np.minimum(radial_norm, radii[start:stop]) / safe_norm
        projected_radial = projected_radial[:, None]
        tool_offset = (
            projected_radial * nearest_radial[:, None] * radials[start:stop]
            + projected_radial * nearest_tangent[:, None] * tangents[start:stop]
            + nearest_axial[:, None] * axes[start:stop]
        )
        # 上式在 radial_norm 非零时已归一化；零点使用轴向投影，避免 NaN。
        tool_offset = np.where(
            (radial_norm > 1e-12)[:, None],
            tool_offset,
            nearest_axial[:, None] * axes[start:stop],
        )
        witness_tool[start:stop] = block_centers + tool_offset

        local_radial = radial_distance[rows, local_index]
        local_axial = axial_coord[rows, local_index]
        inside = (local_radial <= radii[start:stop]) & (local_axial <= half_width)
        side = (local_axial > half_width) & (local_radial <= radii[start:stop])
        outer = (local_radial > radii[start:stop]) & (local_axial <= half_width)
        corner = (local_radial > radii[start:stop]) & (local_axial > half_width)
        radial_penetration = radii[start:stop] - local_radial
        axial_penetration = half_width - local_axial
        inside_region = np.where(
            axial_penetration <= radial_penetration,
            "wheel_side_collision",
            "wheel_outer_collision",
        )
        witness_region[start:stop] = np.where(
            inside,
            inside_region,
            np.where(
            side,
            "wheel_side_collision",
            np.where(outer, "wheel_outer_collision", np.where(corner, "wheel_corner_collision", "clearance")),
            ),
        )

    feasible = signed_clearance >= float(clearance)
    reason = np.where(feasible, "safe", witness_region)
    # 目标接触在 mesh_without_flash 中通常不是独立标签；保留显式状态供人工验收。
    reason = np.where((feasible) & (signed_clearance <= float(clearance) + 1e-12), "target_contact_only", reason)
    return CollisionResult(
        feasible=feasible,
        nearest_distance=signed_clearance + radii,
        clearance=signed_clearance,
        method="finite_cylinder_surface_samples",
        candidates=candidates,
        reason=reason,
        witness_workpiece=witness_workpiece,
        witness_tool=witness_tool,
        witness_region=witness_region,
    )
