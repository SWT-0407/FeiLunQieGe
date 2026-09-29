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
    )
