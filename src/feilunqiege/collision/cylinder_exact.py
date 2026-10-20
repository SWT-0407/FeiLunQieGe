"""有限厚度圆柱砂轮与工件三角网格的连续碰撞判定。

本模块把工件三角形看作连续的二维区域，而不是只检查顶点、质心或边中点。
对一个固定砂轮姿态，先用圆柱 AABB 做宽相筛选，再把三角形裁剪到圆柱的轴向
范围内，并求裁剪区域到砂轮轴线的最小距离。这样可以发现“三个顶点都在圆柱
外部、但三角形内部穿入圆柱”的情况。角度参数的搜索和连续区间提取仍由调用方
负责，本模块只负责固定姿态的几何判定。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from feilunqiege.collision.checker import CollisionResult
from feilunqiege.collision.disk_exact import AABBNode, _build_node, _is_closed_triangle_mesh


@dataclass(frozen=True)
class CylinderTriangleIndex:
    """工件三角形数组及可复用的 AABB 宽相索引。"""

    triangles: np.ndarray
    root: AABBNode
    leaf_size: int
    mesh_is_closed: bool


def build_cylinder_triangle_index(workpiece_mesh: Any, *, leaf_size: int = 32) -> CylinderTriangleIndex:
    """为工件三角形建立 AABB 树，并记录拓扑封闭性。"""

    if leaf_size < 1:
        raise ValueError("AABB leaf_size 必须为正")
    vertices = np.asarray(workpiece_mesh.vertices, dtype=float)
    faces = np.asarray(workpiece_mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    if len(triangles) == 0:
        raise ValueError("工件网格没有三角形")
    lowers = triangles.min(axis=1)
    uppers = triangles.max(axis=1)
    root = _build_node(np.arange(len(triangles), dtype=np.int64), lowers, uppers, int(leaf_size))
    return CylinderTriangleIndex(
        triangles=triangles,
        root=root,
        leaf_size=int(leaf_size),
        mesh_is_closed=_is_closed_triangle_mesh(faces),
    )


def move_candidates_to_lower_face(candidates: Any, width: float, side: float = 1.0) -> Any:
    """将圆盘候选中心沿砂轮轴向移动 ``width/2``，形成有限圆柱中心。

    原候选中心 ``x + rho*e`` 位于目标切平面内。移动后得到
    ``c = x + rho*e + side*(width/2)*a``。由于 ``e`` 与 ``a`` 正交，
    砂轮靠近目标的一侧端面仍与经过目标点的切平面共面。``side`` 用来选择
    砂轮位于切平面的正侧还是负侧，必须在输出摘要中记录。
    """

    if width <= 0 or abs(float(side)) != 1.0:
        raise ValueError("width 必须为正，side 必须为 +1 或 -1")
    axes = np.asarray(candidates.axis_directions, dtype=float)
    centers = np.asarray(candidates.centers, dtype=float) + float(side) * float(width) / 2.0 * axes
    return replace(candidates, centers=centers)


def cylinder_aabb(center: np.ndarray, axis: np.ndarray, radius: float, width: float, epsilon: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """返回任意轴向有限圆柱的保守 AABB。"""

    if radius < 0 or width < 0 or epsilon < 0:
        raise ValueError("radius、width 和 epsilon 不能为负")
    axis = _unit(axis)
    half = float(width) / 2.0
    # 第 j 个坐标方向的圆柱投影半径为 r*sqrt(1-a_j^2)，轴向贡献为 half*|a_j|。
    extents = (float(radius) * np.sqrt(np.maximum(0.0, 1.0 - axis * axis)) + half * np.abs(axis) + float(epsilon))
    center = np.asarray(center, dtype=float)
    return center - extents, center + extents


def query_cylinder_aabb(index: CylinderTriangleIndex, center: np.ndarray, axis: np.ndarray, radius: float, width: float, *, epsilon: float = 0.0) -> np.ndarray:
    """查询与圆柱 AABB 相交的三角形索引。"""

    lower, upper = cylinder_aabb(center, axis, radius, width, epsilon)
    hits: list[np.ndarray] = []
    stack = [index.root]
    while stack:
        node = stack.pop()
        if np.any(node.upper < lower) or np.any(node.lower > upper):
            continue
        if node.triangle_indices is not None:
            hits.append(node.triangle_indices)
        else:
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
    return np.concatenate(hits) if hits else np.empty(0, dtype=np.int64)


def triangle_intersects_cylinder(
    triangle: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    basis_1: np.ndarray,
    basis_2: np.ndarray,
    radius: float,
    width: float,
    *,
    inner_radius: float = 0.0,
    epsilon: float = 1e-10,
) -> tuple[bool, np.ndarray | None, str]:
    """连续判断一个三角形是否进入有限圆柱实体。

    三角形先按轴向坐标裁剪到 ``[-width/2,width/2]``，再在所得凸多边形上
    求局部径向坐标 ``(u,v)`` 到原点的最小距离。凸多边形的最近点只需检查
    顶点、边上投影以及原点是否落在多边形内，因此不会依赖表面采样。
    """

    triangle = np.asarray(triangle, dtype=float)
    center = np.asarray(center, dtype=float)
    axis = _unit(axis)
    # ``basis_1`` is kept only to choose a stable orientation for a true
    # orthonormal cross-section basis.  Candidate direction e and tangent t
    # need not be orthogonal, so treating their two dot products as Cartesian
    # coordinates would distort the radial distance.
    basis_1 = np.array(basis_1, dtype=float, copy=True)
    basis_1 -= float(np.dot(basis_1, axis)) * axis
    basis_1 = _unit(basis_1)
    basis_2 = np.cross(axis, basis_1)
    if radius < 0 or width < 0 or inner_radius < 0 or inner_radius > radius or epsilon < 0:
        raise ValueError("radius、width 和 epsilon 不能为负")
    half = float(width) / 2.0 + float(epsilon)
    clipped = _clip_polygon_axis(triangle, center, axis, half, epsilon)
    if len(clipped) == 0:
        return False, None, "separated_by_cylinder_caps"
    # For each point P, remove its axial component z*a first.  The remaining
    # vector r_p = (P-C) - ((P-C) dot a)*a lies in the wheel cross-section;
    # its Euclidean length is the physical distance to the wheel axis.
    local = np.column_stack(((clipped - center) @ basis_1, (clipped - center) @ basis_2))
    distance2, position = _closest_point_to_origin_in_convex_polygon(local)
    if distance2 > (float(radius) + float(epsilon)) ** 2:
        return False, None, "separated_by_cylinder_side"
    if inner_radius > 0.0:
        radial2_at_vertices = np.sum(local * local, axis=1)
        if float(np.max(radial2_at_vertices, initial=0.0)) < (float(inner_radius) - float(epsilon)) ** 2:
            return False, None, "inside_inner_contact_void"
        # The closest point can be in the inner void; use an outer vertex as witness.
        outer_index = int(np.argmax(radial2_at_vertices))
        position = (outer_index, local[outer_index])
    # position 返回的是多边形上的局部二维点；用重心坐标插值恢复三维见证点。
    witness = _recover_polygon_point(clipped, local, position[1])
    # Use the selected witness radius (which may have been moved out of the
    # inner contact void) when assigning a collision cause.
    radial = float(np.linalg.norm(position[1]))
    axial = abs(float(np.dot(witness - center, axis)))
    on_side = abs(radial - float(radius)) <= float(epsilon) * 2.0
    on_cap = abs(axial - float(width) / 2.0) <= float(epsilon) * 2.0
    if on_side and on_cap:
        kind = "cylinder_edge_collision"
    elif on_side:
        kind = "cylinder_side_collision"
    elif on_cap:
        kind = "cylinder_cap_collision"
    else:
        kind = "triangle_interior_collision"
    return True, witness, kind


def check_finite_cylinder_candidates_exact(
    candidates: Any,
    workpiece_mesh: Any,
    width: float,
    *,
    inner_radius: float = 0.0,
    clearance: float = 0.0,
    epsilon: float = 1e-10,
    allow_target_contact: bool = False,
    target_contact_tolerance: float = 1e-8,
    cylinder_index: CylinderTriangleIndex | None = None,
) -> CollisionResult:
    """用 AABB 宽相和三角形连续窄相筛选有限圆柱候选。

    ``clearance`` 采用保守膨胀：同时扩大半径和半宽。这样不会把需要安全
    间隙的姿态错误地判为可行，但在圆柱棱角处可能比精确欧氏距离更严格。
    """

    if width <= 0 or inner_radius < 0 or clearance < 0 or epsilon < 0:
        raise ValueError("width 必须为正，clearance 和 epsilon 不能为负")
    index = cylinder_index if cylinder_index is not None else build_cylinder_triangle_index(workpiece_mesh)
    centers = np.asarray(candidates.centers, dtype=float)
    axes = np.asarray(candidates.axis_directions, dtype=float)
    tangents = np.asarray(candidates.tangent_directions, dtype=float)
    radials = np.asarray(candidates.radius_directions, dtype=float)
    radii = np.asarray(candidates.radial_values, dtype=float)
    outer_radius = float(candidates.radius)
    if outer_radius <= 0.0:
        outer_radius = float(np.max(radii, initial=0.0))
    if inner_radius > outer_radius:
        raise ValueError("inner_radius cannot exceed the nominal outer radius")
    feasible = np.ones(len(centers), dtype=bool)
    witnesses = np.full((len(centers), 3), np.nan, dtype=float)
    tool_witnesses = np.full((len(centers), 3), np.nan, dtype=float)
    reasons = np.full(len(centers), "safe", dtype=object)
    triangle_counts = np.zeros(len(centers), dtype=np.int64)
    for candidate_index, (center, axis, tangent, radial, radius) in enumerate(zip(centers, axes, tangents, radials, radii, strict=True)):
        axis = _unit(axis)
        tangent = _unit(tangent)
        radial = _unit(radial)
        nearby = query_cylinder_aabb(index, center, axis, outer_radius + clearance, width + 2.0 * clearance, epsilon=epsilon)
        triangle_counts[candidate_index] = len(nearby)
        target_contact_seen = False
        contact = np.asarray(candidates.contact_points[candidate_index], dtype=float)
        for triangle_index in nearby:
            hit, witness, kind = triangle_intersects_cylinder(
                index.triangles[int(triangle_index)], center, axis, tangent, radial,
                outer_radius + clearance, width + 2.0 * clearance,
                inner_radius=max(0.0, inner_radius - clearance), epsilon=epsilon,
            )
            if not hit:
                continue
            if allow_target_contact and witness is not None and np.linalg.norm(witness - contact) <= target_contact_tolerance:
                target_contact_seen = True
                continue
            feasible[candidate_index] = False
            witnesses[candidate_index] = witness
            tool_witnesses[candidate_index] = _project_to_cylinder(witness, center, axis, radial, tangent, outer_radius, width)
            reasons[candidate_index] = kind
            break
        if feasible[candidate_index] and inner_radius <= epsilon and index.mesh_is_closed and _point_inside_closed_mesh_vectorized(center, index.triangles, epsilon):
            feasible[candidate_index] = False
            witnesses[candidate_index] = center
            tool_witnesses[candidate_index] = center
            reasons[candidate_index] = "workpiece_contains_wheel"
        elif feasible[candidate_index] and not index.mesh_is_closed:
            reasons[candidate_index] = "target_contact_only_surface_only_non_closed_mesh" if target_contact_seen else "safe_surface_only_non_closed_mesh"
        elif feasible[candidate_index] and target_contact_seen:
            reasons[candidate_index] = "target_contact_only"
    return CollisionResult(
        feasible=feasible,
        nearest_distance=np.where(feasible, np.inf, 0.0),
        clearance=np.where(feasible, np.inf, -float(clearance)),
        method="exact_triangle_finite_annular_cylinder_aabb" if inner_radius > epsilon else "exact_triangle_finite_cylinder_aabb",
        candidates=candidates,
        reason=reasons,
        witness_workpiece=witnesses,
        witness_tool=tool_witnesses,
        witness_region=triangle_counts,
    )


def _clip_polygon_axis(points: np.ndarray, center: np.ndarray, axis: np.ndarray, half: float, epsilon: float) -> np.ndarray:
    """按两个轴向半空间裁剪三角形，返回有序凸多边形顶点。"""

    polygon = [np.asarray(point, dtype=float) for point in points]
    for bound, keep_less in ((half, True), (-half, False)):
        if not polygon:
            break
        output: list[np.ndarray] = []
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            start_value = float(np.dot(start - center, axis))
            end_value = float(np.dot(end - center, axis))
            start_inside = start_value <= bound + epsilon if keep_less else start_value >= bound - epsilon
            end_inside = end_value <= bound + epsilon if keep_less else end_value >= bound - epsilon
            if start_inside != end_inside:
                denominator = end_value - start_value
                if abs(denominator) > 1e-18:
                    fraction = (bound - start_value) / denominator
                    output.append(start + fraction * (end - start))
            if end_inside:
                output.append(end)
        polygon = _unique_polygon_points(output, max(epsilon, 1e-12))
    return np.asarray(polygon, dtype=float)


def _closest_point_to_origin_in_convex_polygon(points: np.ndarray) -> tuple[float, tuple[int, np.ndarray]]:
    """求二维凸多边形到原点的最近点；返回距离平方和点来源。"""

    if len(points) == 0:
        raise ValueError("凸多边形不能为空")
    if _point_in_convex_polygon(np.zeros(2), points):
        return 0.0, (-1, np.zeros(2, dtype=float))
    best = (float("inf"), (0, points[0]))
    for index, point in enumerate(points):
        value = float(np.dot(point, point))
        if value < best[0]:
            best = (value, (index, point))
    for index, (start, end) in enumerate(zip(points, np.roll(points, -1, axis=0))):
        direction = end - start
        denominator = float(np.dot(direction, direction))
        if denominator <= 1e-24:
            continue
        fraction = np.clip(-float(np.dot(start, direction)) / denominator, 0.0, 1.0)
        point = start + fraction * direction
        value = float(np.dot(point, point))
        if value < best[0]:
            best = (value, (index, point))
    return best


def _recover_polygon_point(points: np.ndarray, local: np.ndarray, local_point: np.ndarray) -> np.ndarray:
    """将二维局部最近点恢复为原三维多边形上的点。"""

    if len(points) == 1:
        return points[0]
    for start_index, (a, b) in enumerate(zip(local, np.roll(local, -1, axis=0))):
        direction = b - a
        denominator = float(np.dot(direction, direction))
        if denominator <= 1e-24:
            continue
        fraction = float(np.dot(local_point - a, direction) / denominator)
        if -1e-8 <= fraction <= 1.0 + 1e-8 and np.linalg.norm(a + np.clip(fraction, 0.0, 1.0) * direction - local_point) <= 1e-7:
            return points[start_index] + np.clip(fraction, 0.0, 1.0) * (points[(start_index + 1) % len(points)] - points[start_index])
    if _point_in_convex_polygon(local_point, local):
        # 原点/内部最近点通常落在三角形平面内部；用最小残差的三角扇插值恢复它。
        base = points[0]
        for index in range(1, len(points) - 1):
            triangle_local = np.array([local[0], local[index], local[index + 1]])
            bary = _barycentric_2d(local_point, triangle_local)
            if bary is not None and np.all(bary >= -1e-8):
                return bary[0] * base + bary[1] * points[index] + bary[2] * points[index + 1]
    return points[int(np.argmin(np.linalg.norm(local - local_point[None, :], axis=1)))]


def _point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray, epsilon: float = 1e-9) -> bool:
    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = point[None, :] - polygon
    cross_values = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    return bool(np.all(cross_values >= -epsilon) or np.all(cross_values <= epsilon))


def _barycentric_2d(point: np.ndarray, triangle: np.ndarray) -> np.ndarray | None:
    matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) <= 1e-14:
        return None
    uv = np.linalg.solve(matrix, point - triangle[0])
    return np.array([1.0 - uv[0] - uv[1], uv[0], uv[1]])


def _unique_polygon_points(points: list[np.ndarray], epsilon: float) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - saved) <= epsilon for saved in unique):
            unique.append(point)
    return unique


def _project_to_cylinder(point: np.ndarray, center: np.ndarray, axis: np.ndarray, basis_1: np.ndarray, basis_2: np.ndarray, radius: float, width: float) -> np.ndarray:
    delta = np.asarray(point, dtype=float) - center
    axis = _unit(np.asarray(axis, dtype=float))
    axial = np.clip(float(np.dot(delta, axis)), -float(width) / 2.0, float(width) / 2.0)
    # Do not reconstruct the radial vector from e/t coordinates: e and t
    # are generally non-orthogonal in the current tangent-plane model.
    radial_vector = delta - float(np.dot(delta, axis)) * axis
    radial = float(np.linalg.norm(radial_vector))
    if radial <= 1e-12:
        return center + axial * axis
    return center + axial * axis + float(radius) * radial_vector / radial


def _point_inside_closed_mesh_vectorized(point: np.ndarray, triangles: np.ndarray, epsilon: float) -> bool:
    """对拓扑封闭网格执行一次向量化射线奇偶测试。"""

    origin = np.asarray(point, dtype=float) + float(epsilon) * 32.0 * _unit(np.array([0.743, 0.521, 0.419]))
    direction = _unit(np.array([0.743, 0.521, 0.419]))
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(direction[None, :], edge2)
    determinant = np.einsum("ij,ij->i", edge1, cross)
    valid = np.abs(determinant) > max(float(epsilon), 1e-12)
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    offset = origin[None, :] - triangles[:, 0]
    u = inverse * np.einsum("ij,ij->i", offset, cross)
    cross_offset = np.cross(offset, edge1)
    v = inverse * np.einsum("j,ij->i", direction, cross_offset)
    distance = inverse * np.einsum("ij,ij->i", edge2, cross_offset)
    hits = valid & (u >= -epsilon) & (u <= 1.0 + epsilon) & (v >= -epsilon) & (u + v <= 1.0 + epsilon) & (distance > epsilon)
    if not np.any(hits):
        return False
    unique = np.unique(np.round(distance[hits], decimals=10))
    return bool(len(unique) % 2 == 1)


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("方向向量不能为零")
    return vector / norm
