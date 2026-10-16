"""无厚度圆盘与三角网格的连续相交判定。

本模块的“连续”仅指：针对一个固定姿态的圆盘，直接判断三角形的连续
内部是否与圆盘相交，而不是把三角形替换成顶点、边中点或质心采样。
候选姿态的角度空间仍由调用方进行离散/自适应搜索，因此不能把本模块
单独解释成对所有连续角度的安全证明。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from feilunqiege.collision.checker import CollisionResult


@dataclass(frozen=True)
class AABBNode:
    """三角形 AABB 树节点；叶节点保存原网格三角形索引。"""

    lower: np.ndarray
    upper: np.ndarray
    left: "AABBNode | None" = None
    right: "AABBNode | None" = None
    triangle_indices: np.ndarray | None = None


@dataclass(frozen=True)
class DiskTriangleIndex:
    """固定工件网格的三角形数组及可复用 AABB 宽相索引。"""

    triangles: np.ndarray
    root: AABBNode
    leaf_size: int
    mesh_is_closed: bool


def build_disk_triangle_index(workpiece_mesh: Any, *, leaf_size: int = 32) -> DiskTriangleIndex:
    """建立三角形 AABB 树，并记录边连接意义下的网格封闭性。

    AABB 树只负责排除不可能接触圆盘的三角形；最终结论必须由后续的
    ``triangle_intersects_disk`` 窄相连续判定给出。
    """

    if leaf_size < 1:
        raise ValueError("AABB leaf_size 必须为正")
    vertices = np.asarray(workpiece_mesh.vertices, dtype=float)
    faces = np.asarray(workpiece_mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    if len(triangles) == 0:
        raise ValueError("工件网格没有三角形")
    triangle_lowers = triangles.min(axis=1)
    triangle_uppers = triangles.max(axis=1)
    root = _build_node(np.arange(len(triangles), dtype=np.int64), triangle_lowers, triangle_uppers, int(leaf_size))
    return DiskTriangleIndex(
        triangles=triangles,
        root=root,
        leaf_size=int(leaf_size),
        mesh_is_closed=_is_closed_triangle_mesh(faces),
    )


def _build_node(indices: np.ndarray, lowers: np.ndarray, uppers: np.ndarray, leaf_size: int) -> AABBNode:
    """递归按三角形包围盒中心的最长轴中位数构造二叉 AABB 树。"""

    lower = lowers[indices].min(axis=0)
    upper = uppers[indices].max(axis=0)
    if len(indices) <= leaf_size:
        return AABBNode(lower=lower, upper=upper, triangle_indices=indices)
    centers = (lowers[indices] + uppers[indices]) / 2.0
    axis = int(np.argmax(upper - lower))
    ordered = indices[np.argsort(centers[:, axis], kind="stable")]
    midpoint = len(ordered) // 2
    return AABBNode(
        lower=lower,
        upper=upper,
        left=_build_node(ordered[:midpoint], lowers, uppers, leaf_size),
        right=_build_node(ordered[midpoint:], lowers, uppers, leaf_size),
    )


def _is_closed_triangle_mesh(faces: np.ndarray) -> bool:
    """检查每条无向边是否恰好被两个三角形使用。

    这是对封闭性的拓扑检查，不证明法向一致、无自交或实体包含判定可靠。
    """

    directed = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(directed, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def query_disk_aabb(index: DiskTriangleIndex, center: np.ndarray, radius: float, *, epsilon: float = 0.0) -> np.ndarray:
    """返回与圆盘外接球 AABB 重叠的三角形索引。

    无厚度圆盘被包含在半径 ``radius`` 的球内，因此该盒是安全的宽相包络，
    只会保留额外三角形，不会因 AABB 剪枝而漏掉真实相交。
    """

    if radius < 0 or epsilon < 0:
        raise ValueError("radius 与 epsilon 不能为负")
    lower = np.asarray(center, dtype=float) - (float(radius) + float(epsilon))
    upper = np.asarray(center, dtype=float) + (float(radius) + float(epsilon))
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


def triangle_intersects_disk(
    triangle: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    *,
    epsilon: float = 1e-10,
) -> tuple[bool, np.ndarray | None, str]:
    """连续判断一个三角形与零厚度圆盘是否相交或相切。

    算法先求三角形与圆盘所在平面的连续交集：非共面时为点或线段，共面时
    检查三角形顶点、圆心和三条边到圆心的最近距离。交集到圆心的距离不超过
    ``radius + epsilon`` 即判为碰撞。返回工件侧见证点和交集类别。
    """

    triangle = np.asarray(triangle, dtype=float)
    center = np.asarray(center, dtype=float)
    normal = _unit(normal)
    if radius < 0 or epsilon < 0:
        raise ValueError("radius 与 epsilon 不能为负")
    signed = (triangle - center) @ normal
    if np.all(signed > epsilon) or np.all(signed < -epsilon):
        return False, None, "separated_by_disk_plane"

    if np.all(np.abs(signed) <= epsilon):
        point = _coplanar_triangle_disk_witness(triangle, center, normal, radius, epsilon)
        return (point is not None), point, "coplanar_triangle" if point is not None else "coplanar_separated"

    # 交集由位于平面上的顶点和跨越平面的三角形边组成。去重后，它最多为一条线段。
    intersections: list[np.ndarray] = []
    for vertex, distance in zip(triangle, signed, strict=True):
        if abs(float(distance)) <= epsilon:
            intersections.append(vertex)
    for start, end, first, second in _triangle_edges_with_signed_distances(triangle, signed):
        if first * second < -(epsilon * epsilon):
            fraction = first / (first - second)
            intersections.append(start + fraction * (end - start))
    unique = _unique_points(intersections, epsilon)
    if not unique:
        return False, None, "plane_touch_without_triangle_intersection"
    if len(unique) == 1:
        point = unique[0]
        return (np.linalg.norm(point - center) <= radius + epsilon), point, "point_plane_intersection"
    first, second = _farthest_pair(unique)
    point = _closest_point_on_segment(center, first, second)
    return (np.linalg.norm(point - center) <= radius + epsilon), point, "segment_plane_intersection"


def check_disk_candidates_exact(
    candidates: Any,
    workpiece_mesh: Any,
    clearance: float,
    *,
    disk_index: DiskTriangleIndex | None = None,
    epsilon: float = 1e-10,
    allow_target_contact: bool = False,
    target_contact_tolerance: float = 1e-8,
) -> CollisionResult:
    """对候选无厚度圆盘执行 AABB 宽相与三角形级连续窄相判定。

    ``clearance`` 通过扩大圆盘半径实现；当值为零时，接触本身默认视为碰撞。
    若 ``allow_target_contact`` 为真，仅允许见证点在目标接触点
    ``target_contact_tolerance`` 内的零测度接触，函数仍会检查同姿态的其余
    三角形，避免把接触三角形附近的实际穿入一并豁免。本函数只判断工件表面
    与圆盘相交。若网格拓扑封闭，还会用圆心射线测试检查圆盘中心是否处于
    工件内部；非封闭网格会明确标为无法完成实体包含检查。
    """

    if clearance < 0 or epsilon < 0 or target_contact_tolerance < 0:
        raise ValueError("clearance、epsilon 与 target_contact_tolerance 不能为负")
    index = disk_index if disk_index is not None else build_disk_triangle_index(workpiece_mesh)
    centers = np.asarray(candidates.centers, dtype=float)
    normals = np.asarray(candidates.axis_directions, dtype=float)
    radii = np.asarray(candidates.radial_values, dtype=float) + float(clearance)
    count = len(centers)
    feasible = np.ones(count, dtype=bool)
    witnesses = np.full((count, 3), np.nan, dtype=float)
    witness_tool = np.full((count, 3), np.nan, dtype=float)
    reason = np.full(count, "safe", dtype=object)
    candidate_triangle_counts = np.zeros(count, dtype=np.int64)

    for candidate_index, (center, normal, radius) in enumerate(zip(centers, normals, radii, strict=True)):
        nearby_indices = query_disk_aabb(index, center, float(radius), epsilon=epsilon)
        candidate_triangle_counts[candidate_index] = len(nearby_indices)
        target_only_contact_seen = False
        contact_point = np.asarray(candidates.contact_points[candidate_index], dtype=float)
        for triangle_index in nearby_indices:
            intersects, witness, kind = triangle_intersects_disk(
                index.triangles[triangle_index], center, normal, float(radius), epsilon=epsilon
            )
            if intersects:
                if allow_target_contact and witness is not None and np.linalg.norm(witness - contact_point) <= target_contact_tolerance:
                    target_only_contact_seen = True
                    continue
                feasible[candidate_index] = False
                witnesses[candidate_index] = witness
                witness_tool[candidate_index] = witness
                reason[candidate_index] = f"triangle_disk_{kind}"
                break
        # 表面不相交不能排除圆盘整体落在封闭实体内部，因此补做圆心包含检查。
        if feasible[candidate_index] and index.mesh_is_closed and _point_inside_closed_mesh(center, index.triangles, epsilon):
            feasible[candidate_index] = False
            witnesses[candidate_index] = center
            witness_tool[candidate_index] = center
            reason[candidate_index] = "disk_center_inside_closed_workpiece"
        elif feasible[candidate_index] and not index.mesh_is_closed:
            reason[candidate_index] = (
                "target_contact_only_surface_only_non_closed_mesh"
                if target_only_contact_seen
                else "safe_surface_only_non_closed_mesh"
            )
        elif feasible[candidate_index] and target_only_contact_seen:
            reason[candidate_index] = "target_contact_only"

    # 无厚度圆盘不定义与工件表面的完整有符号距离；这里明确仅记录表面相交状态。
    clearance_values = np.where(feasible, np.inf, -float(clearance))
    return CollisionResult(
        feasible=feasible,
        nearest_distance=np.where(feasible, np.inf, 0.0),
        clearance=clearance_values,
        method="exact_triangle_disk_aabb",
        candidates=candidates,
        reason=reason,
        witness_workpiece=witnesses,
        witness_tool=witness_tool,
        witness_region=candidate_triangle_counts,
    )


def _triangle_edges_with_signed_distances(triangle: np.ndarray, signed: np.ndarray):
    """按三角形闭环边返回端点和其到圆盘平面的有符号距离。"""

    for first, second in ((0, 1), (1, 2), (2, 0)):
        yield triangle[first], triangle[second], float(signed[first]), float(signed[second])


def _coplanar_triangle_disk_witness(
    triangle: np.ndarray, center: np.ndarray, normal: np.ndarray, radius: float, epsilon: float
) -> np.ndarray | None:
    """求共面三角形与圆盘的见证点，覆盖顶点、边和圆心位于三角形内三种情形。"""

    for vertex in triangle:
        if np.linalg.norm(vertex - center) <= radius + epsilon:
            return vertex
    if _point_in_triangle_3d(center, triangle, normal, epsilon):
        return center
    closest = min((_closest_point_on_segment(center, start, end) for start, end, _, _ in _triangle_edges_with_signed_distances(triangle, np.zeros(3))), key=lambda point: np.linalg.norm(point - center))
    return closest if np.linalg.norm(closest - center) <= radius + epsilon else None


def _point_in_triangle_3d(point: np.ndarray, triangle: np.ndarray, normal: np.ndarray, epsilon: float) -> bool:
    """以同向边叉积测试共面点是否位于三角形内或边界上。"""

    signs = []
    for start, end, _, _ in _triangle_edges_with_signed_distances(triangle, np.zeros(3)):
        signs.append(float(np.dot(np.cross(end - start, point - start), normal)))
    return bool(min(signs) >= -epsilon or max(signs) <= epsilon)


def _point_inside_closed_mesh(point: np.ndarray, triangles: np.ndarray, epsilon: float) -> bool:
    """对拓扑封闭网格做确定性非轴向射线奇偶包含测试。

    该检查处理“圆盘没有触碰表面却完全在工件内部”的必要情形；对自交网格
    或非流形网格不构成可靠实体分类，因此调用方必须同时检查网格质量标记。
    """

    direction = _unit(np.array([0.743, 0.521, 0.419], dtype=float))
    origins = point + epsilon * 32.0 * direction
    hits: list[float] = []
    for triangle in triangles:
        distance = _ray_triangle_distance(origins, direction, triangle, epsilon)
        if distance is not None:
            hits.append(distance)
    if not hits:
        return False
    unique_hits = np.unique(np.round(np.asarray(hits), decimals=10))
    return bool(len(unique_hits) % 2 == 1)


def _ray_triangle_distance(origin: np.ndarray, direction: np.ndarray, triangle: np.ndarray, epsilon: float) -> float | None:
    """Moller-Trumbore 射线与三角形相交距离；近平行与射线起点后方均忽略。"""

    first_edge = triangle[1] - triangle[0]
    second_edge = triangle[2] - triangle[0]
    cross = np.cross(direction, second_edge)
    determinant = float(np.dot(first_edge, cross))
    if abs(determinant) <= epsilon:
        return None
    inverse = 1.0 / determinant
    offset = origin - triangle[0]
    u = inverse * float(np.dot(offset, cross))
    if u < -epsilon or u > 1.0 + epsilon:
        return None
    cross_offset = np.cross(offset, first_edge)
    v = inverse * float(np.dot(direction, cross_offset))
    if v < -epsilon or u + v > 1.0 + epsilon:
        return None
    distance = inverse * float(np.dot(second_edge, cross_offset))
    return distance if distance > epsilon else None


def _closest_point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """返回线段上到给定点最近的连续点。"""

    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-24:
        return start
    fraction = np.clip(float(np.dot(point - start, direction)) / denominator, 0.0, 1.0)
    return start + fraction * direction


def _unique_points(points: list[np.ndarray], epsilon: float) -> list[np.ndarray]:
    """按容差去除三角形平面交集中的重复顶点。"""

    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - saved) <= epsilon for saved in unique):
            unique.append(point)
    return unique


def _farthest_pair(points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """从平面交集点中选出最远对，表示三角形切平面的交线段。"""

    best_pair = (points[0], points[1])
    best_distance = -1.0
    for first_index, first in enumerate(points):
        for second in points[first_index + 1 :]:
            distance = float(np.dot(first - second, first - second))
            if distance > best_distance:
                best_distance = distance
                best_pair = (first, second)
    return best_pair


def _unit(vector: np.ndarray) -> np.ndarray:
    """归一化三维向量。"""

    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("圆盘法向不能为零")
    return vector / norm
