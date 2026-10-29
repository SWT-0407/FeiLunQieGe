"""有限厚度圆柱砂轮与工件三角网格的连续碰撞判定。

本模块把工件三角形看作连续的二维区域，而不是只检查顶点、质心或边中点。
对一个固定砂轮姿态，先用圆柱 AABB 做宽相筛选，再把三角形裁剪到圆柱的轴向
范围内，并求裁剪区域到砂轮轴线的最小距离。这样可以发现“三个顶点都在圆柱
外部、但三角形内部穿入圆柱”的情况。角度参数的搜索和连续区间提取仍由调用方
负责，本模块只负责固定姿态的几何判定。
"""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class CylinderContactState:
    """一次三角形-圆柱相交的接触斑诊断。

    ``penetration`` 是三角形进入实心圆柱的最大深度上界；
    ``axial_penetration`` 是越过近端面的深度；
    ``radial_penetration`` 是越过外圆柱面的深度；
    ``patch_max_distance`` 是相交区域到目标接触点的最大距离；
    ``normal_alignment`` 是三角形法向与砂轮轴线的 |cos|。
    """

    hit: bool
    witness: np.ndarray | None
    kind: str
    penetration: float
    axial_penetration: float
    radial_penetration: float
    patch_max_distance: float
    normal_alignment: float
    contact_normal: np.ndarray | None = None
    near_cap: bool | None = None


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


def triangle_cylinder_contact_state(
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
    contact_point: np.ndarray | None = None,
) -> CylinderContactState:
    """计算三角形进入圆柱的深度和接触斑信息。

    与 ``triangle_intersects_cylinder`` 使用同一套裁剪和连续距离判定，
    但额外返回：

    - ``penetration``：进入实心圆柱的深度上界；
    - ``axial_penetration``：越过近端面（z=-w/2）的最大距离；
    - ``radial_penetration``：越过外圆柱面的最大距离；
    - ``patch_max_distance``：相交区域到目标点的最大距离；
    - ``normal_alignment``：三角形法向与砂轮轴线的 |cos|。

    这些量用于“接触斑模型”：只有浅穿透、法向接近砂轮端面、且位于
    目标点附近 patch 内的相交才允许作为接触；其余仍是碰撞。
    """

    triangle = np.asarray(triangle, dtype=float)
    center = np.asarray(center, dtype=float)
    axis = _unit(axis)
    basis_1 = np.array(basis_1, dtype=float, copy=True)
    basis_1 -= float(np.dot(basis_1, axis)) * axis
    basis_1 = _unit(basis_1)
    basis_2 = np.cross(axis, basis_1)
    if radius < 0 or width < 0 or inner_radius < 0 or inner_radius > radius or epsilon < 0:
        raise ValueError("radius、width 和 epsilon 不能为负")

    half = float(width) / 2.0 + float(epsilon)
    clipped = _clip_polygon_axis(triangle, center, axis, half, epsilon)
    if len(clipped) == 0:
        return CylinderContactState(False, None, "separated_by_cylinder_caps", 0.0, 0.0, 0.0, 0.0, 0.0)

    local = np.column_stack(((clipped - center) @ basis_1, (clipped - center) @ basis_2))
    distance2, position = _closest_point_to_origin_in_convex_polygon(local)
    if distance2 > (float(radius) + float(epsilon)) ** 2:
        return CylinderContactState(False, None, "separated_by_cylinder_side", 0.0, 0.0, 0.0, 0.0, 0.0)
    if inner_radius > 0.0:
        radial2_at_vertices = np.sum(local * local, axis=1)
        if float(np.max(radial2_at_vertices, initial=0.0)) < (float(inner_radius) - float(epsilon)) ** 2:
            return CylinderContactState(False, None, "inside_inner_contact_void", 0.0, 0.0, 0.0, 0.0, 0.0)
        outer_index = int(np.argmax(radial2_at_vertices))
        position = (outer_index, local[outer_index])

    witness = _recover_polygon_point(clipped, local, position[1])
    delta = clipped - center
    axial_coords = delta @ axis
    radial_vectors = delta - axial_coords[:, None] * axis[None, :]
    radial_coords = np.linalg.norm(radial_vectors, axis=1)

    # 径向穿透必须用“裁剪多边形到轴线的最近距离”，不能用顶点上的
    # R-rho 最大值；三角形内部/边穿过轴线时，顶点法会低估真实穿透。
    radial_min = float(np.sqrt(max(float(distance2), 0.0)))
    radial_penetration = max(float(radius) - radial_min, 0.0)
    # 近端面穿透（z=-w/2）用线性轴向坐标的最大值；它是精确的。
    axial_penetration = float(np.max(axial_coords + float(width) / 2.0, initial=0.0))
    witness_delta = witness - center
    witness_axial = float(np.dot(witness_delta, axis))
    witness_radial = float(np.linalg.norm(witness_delta - witness_axial * axis))
    witness_penetration = min(
        float(radius) - witness_radial,
        float(width) / 2.0 - abs(witness_axial),
    )
    if axial_penetration > 0.0 and radial_penetration > 0.0:
        penetration = min(axial_penetration, radial_penetration)
    else:
        penetration = max(0.0, witness_penetration)

    # 接触斑距离：不仅要看 inside 顶点，还要看裁剪多边形边与圆柱侧面的
    # 交点；否则“边穿过圆柱、顶点都在外面”的相交区域会被低估。
    patch_points: list[np.ndarray] = []
    half_width = float(width) / 2.0
    for point, radial, axial in zip(clipped, radial_coords, axial_coords, strict=True):
        if radial <= float(radius) + float(epsilon) and abs(float(axial)) <= half_width + float(epsilon):
            patch_points.append(point)
    for start_index, (start, end) in enumerate(zip(local, np.roll(local, -1, axis=0))):
        direction = end - start
        a = float(np.dot(direction, direction))
        if a <= 1e-24:
            continue
        b = 2.0 * float(np.dot(start, direction))
        c = float(np.dot(start, start)) - float(radius) ** 2
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            continue
        root = math.sqrt(max(discriminant, 0.0))
        for fraction in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
            if -1e-9 <= fraction <= 1.0 + 1e-9:
                point3 = clipped[start_index] + float(np.clip(fraction, 0.0, 1.0)) * (
                    clipped[(start_index + 1) % len(clipped)] - clipped[start_index]
                )
                axial = float(np.dot(point3 - center, axis))
                if abs(axial) <= half_width + float(epsilon):
                    patch_points.append(point3)
    if contact_point is not None:
        contact_point = np.asarray(contact_point, dtype=float)
        if patch_points:
            patch_max_distance = float(
                np.max(np.linalg.norm(np.asarray(patch_points) - contact_point[None, :], axis=1))
            )
        else:
            patch_max_distance = float(np.linalg.norm(witness - contact_point))
    else:
        patch_max_distance = 0.0

    triangle_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal_length = float(np.linalg.norm(triangle_normal))
    normal_alignment = (
        abs(float(np.dot(triangle_normal / normal_length, axis))) if normal_length > 1e-12 else 0.0
    )

    radial = witness_radial
    axial = abs(witness_axial)
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

    near_cap = None
    if kind in ("cylinder_cap_collision", "cylinder_edge_collision"):
        near_cap = bool(witness_axial <= 0.0)

    return CylinderContactState(
        True,
        witness,
        kind,
        float(penetration),
        float(axial_penetration),
        float(radial_penetration),
        float(patch_max_distance),
        float(normal_alignment),
        near_cap=near_cap,
    )


def _closest_point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """返回线段上到给定点最近的点。"""

    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-24:
        return np.asarray(start, dtype=float).copy()
    fraction = float(np.clip(np.dot(point - start, direction) / denominator, 0.0, 1.0))
    return np.asarray(start, dtype=float) + fraction * direction


def _closest_point_on_triangle(point: np.ndarray, triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回三角形上到给定点最近的点及其重心坐标。"""

    a, b, c = np.asarray(triangle, dtype=float)
    point = np.asarray(point, dtype=float)
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy(), np.array([1.0, 0.0, 0.0])
    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b.copy(), np.array([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        fraction = d1 / (d1 - d3)
        return a + fraction * ab, np.array([1.0 - fraction, fraction, 0.0])
    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c.copy(), np.array([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        fraction = d2 / (d2 - d6)
        return a + fraction * ac, np.array([1.0 - fraction, 0.0, fraction])
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        fraction = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + fraction * (c - b), np.array([0.0, 1.0 - fraction, fraction])
    denominator = va + vb + vc
    if abs(denominator) <= 1e-24:
        # 退化三角形：在三条边上分别求最近点。
        candidates = [
            _closest_point_on_segment(point, a, b),
            _closest_point_on_segment(point, b, c),
            _closest_point_on_segment(point, c, a),
        ]
        distances = [float(np.linalg.norm(point - candidate)) for candidate in candidates]
        best = int(np.argmin(distances))
        return candidates[best], np.array([1.0, 0.0, 0.0])
    v = vb / denominator
    w = vc / denominator
    return a + ab * v + ac * w, np.array([1.0 - v - w, v, w])


def _closest_points_between_segments(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回两条线段之间的最近点对（Ericson, Real-Time Collision Detection）。"""

    p1 = np.asarray(first_start, dtype=float)
    q1 = np.asarray(first_end, dtype=float)
    p2 = np.asarray(second_start, dtype=float)
    q2 = np.asarray(second_end, dtype=float)
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    tiny = 1e-24
    if a <= tiny and e <= tiny:
        return p1.copy(), p2.copy()
    if a <= tiny:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= tiny:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denominator = a * e - b * b
            s = float(np.clip((b * f - c * e) / denominator, 0.0, 1.0)) if abs(denominator) > tiny else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    return p1 + s * d1, p2 + t * d2


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray, normal: np.ndarray, epsilon: float) -> bool:
    """判断点是否在三角形内部或边界上（要求点近似位于三角形平面内）。"""

    a, b, c = np.asarray(triangle, dtype=float)
    point = np.asarray(point, dtype=float)
    signs = [
        float(np.dot(np.cross(b - a, point - a), normal)),
        float(np.dot(np.cross(c - b, point - b), normal)),
        float(np.dot(np.cross(a - c, point - c), normal)),
    ]
    return bool(min(signs) >= -epsilon or max(signs) <= epsilon)


def _closest_points_triangle_segment(
    triangle: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """精确计算三角形与线段之间的最近点对和距离。

    分类覆盖：线段穿过三角形内部、线段端点到三角形、三角形顶点到线段、
    三角形边到线段。对凸三角形和凸线段，这四类覆盖所有最近点配置。
    """

    triangle = np.asarray(triangle, dtype=float)
    segment_start = np.asarray(segment_start, dtype=float)
    segment_end = np.asarray(segment_end, dtype=float)
    a, b, c = triangle

    # 1. 线段与三角形平面相交且交点位于三角形内部：距离为 0。
    normal = np.cross(b - a, c - a)
    normal_length = float(np.linalg.norm(normal))
    if normal_length > 1e-14:
        unit_normal = normal / normal_length
        start_distance = float(np.dot(unit_normal, segment_start - a))
        end_distance = float(np.dot(unit_normal, segment_end - a))
        denominator = start_distance - end_distance
        if abs(denominator) > 1e-14:
            fraction = start_distance / denominator
            if 0.0 <= fraction <= 1.0:
                intersection = segment_start + fraction * (segment_end - segment_start)
                if _point_in_triangle(intersection, triangle, unit_normal, 1e-10):
                    return intersection.copy(), intersection.copy(), 0.0

    best_distance = float("inf")
    best_triangle_point = a.copy()
    best_segment_point = segment_start.copy()

    def update(triangle_point: np.ndarray, segment_point: np.ndarray) -> None:
        nonlocal best_distance, best_triangle_point, best_segment_point
        distance = float(np.linalg.norm(triangle_point - segment_point))
        if distance < best_distance:
            best_distance = distance
            best_triangle_point = np.asarray(triangle_point, dtype=float).copy()
            best_segment_point = np.asarray(segment_point, dtype=float).copy()

    # 2. 线段端点到三角形。
    for endpoint in (segment_start, segment_end):
        triangle_point, _ = _closest_point_on_triangle(endpoint, triangle)
        update(triangle_point, endpoint)

    # 3. 三角形顶点到线段。
    for vertex in triangle:
        segment_point = _closest_point_on_segment(vertex, segment_start, segment_end)
        update(vertex, segment_point)

    # 4. 三角形三条边与线段之间的最近点对。
    for first, second in ((a, b), (b, c), (c, a)):
        triangle_point, segment_point = _closest_points_between_segments(
            first, second, segment_start, segment_end
        )
        update(triangle_point, segment_point)

    return best_triangle_point, best_segment_point, float(best_distance)


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """点到线段的欧氏距离。"""

    return float(np.linalg.norm(np.asarray(point, dtype=float) - _closest_point_on_segment(point, start, end)))


def _flat_cylinder_penetration(
    point: np.ndarray, center: np.ndarray, axis: np.ndarray, radius: float, half_width: float
) -> float:
    """点进入平端面实心圆柱的深度；外部点返回 0。"""

    delta = np.asarray(point, dtype=float) - center
    axial = float(np.dot(delta, axis))
    radial = float(np.linalg.norm(delta - axial * axis))
    if radial > radius or abs(axial) > half_width:
        return 0.0
    return max(0.0, min(float(radius) - radial, float(half_width) - abs(axial)))


def triangle_cylinder_contact_state_analytic(
    triangle: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    width: float,
    *,
    contact_point: np.ndarray | None = None,
    epsilon: float = 1e-10,
) -> CylinderContactState:
    """解析法：三角形与平端面实心圆柱的接触状态。

    砂轮是平端面实心圆柱，不是胶囊/球端帽。处理步骤：

    1. 用精确轴向裁剪判断三角形是否进入平端面圆柱；
    2. 用三角形上见证点到轴线段的最近点构造解析接触法向；
    3. 穿透深度使用平端面定义 ``min(R-rho, h/2-|z|)``。

    不再包含“轴线 + 球形端帽”的胶囊模型。胶囊模型会把共面端面接触
    误算成 ``R-D`` 的大穿透，从而错误拒绝平端面砂轮的贴面姿态。
    """

    triangle = np.asarray(triangle, dtype=float)
    center = np.asarray(center, dtype=float)
    axis = _unit(axis)
    if radius < 0 or width < 0 or epsilon < 0:
        raise ValueError("radius、width 和 epsilon 不能为负")

    half = float(width) / 2.0
    segment_start = center - half * axis
    segment_end = center + half * axis

    triangle_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal_length = float(np.linalg.norm(triangle_normal))
    normal_alignment = (
        abs(float(np.dot(triangle_normal / normal_length, axis))) if normal_length > 1e-12 else 0.0
    )

    # 平端面圆柱的精确相交判定和穿透深度由裁剪法给出；最近点只用来
    # 构造解析接触法向。这样既保持平端面语义，又避免把共面端面接触
    # 误算成 R-D 的大穿透。
    reference = np.array([1.0, 0.0, 0.0]) if abs(float(axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    clipped = triangle_cylinder_contact_state(
        triangle, center, axis, reference, np.cross(axis, reference),
        radius, width, inner_radius=0.0, epsilon=epsilon, contact_point=contact_point,
    )
    if not clipped.hit:
        return CylinderContactState(False, None, clipped.kind, 0.0, 0.0, 0.0, 0.0, 0.0)
    triangle_point = (
        np.asarray(clipped.witness, dtype=float)
        if clipped.witness is not None
        else triangle.mean(axis=0)
    )
    axis_point = _closest_point_on_segment(triangle_point, segment_start, segment_end)
    distance = float(np.linalg.norm(triangle_point - axis_point))
    kind = clipped.kind
    penetration = clipped.penetration
    axial_penetration = clipped.axial_penetration
    radial_penetration = clipped.radial_penetration
    patch_max_distance = clipped.patch_max_distance
    normal_alignment = clipped.normal_alignment
    closest_normal = (
        (triangle_point - axis_point) / distance
        if distance > 1e-12
        else triangle_normal / max(normal_length, 1e-12)
    )
    if kind in ("cylinder_cap_collision", "cylinder_edge_collision"):
        cap_sign = 1.0 if float(np.dot(triangle_point - center, axis)) >= 0.0 else -1.0
        cap_normal = cap_sign * axis
        if kind == "cylinder_edge_collision":
            combined = cap_normal + closest_normal
            combined_length = float(np.linalg.norm(combined))
            contact_normal = combined / combined_length if combined_length > 1e-12 else cap_normal
        else:
            contact_normal = cap_normal
    else:
        contact_normal = closest_normal
    return CylinderContactState(
        True,
        triangle_point,
        kind,
        float(penetration),
        float(axial_penetration),
        float(radial_penetration),
        float(patch_max_distance),
        float(normal_alignment),
        contact_normal,
        clipped.near_cap,
    )


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
    contact_patch_enabled: bool = False,
    contact_patch_radius: float | None = None,
    max_contact_penetration: float = 0.0,
    contact_normal_tolerance_degrees: float = 30.0,
    narrow_phase: str = "analytic_closest_point_flat",
    cylinder_index: CylinderTriangleIndex | None = None,
) -> CollisionResult:
    """用 AABB 宽相和三角形连续窄相筛选有限圆柱候选。

    ``clearance`` 采用保守膨胀：同时扩大半径和半宽。这样不会把需要安全
    间隙的姿态错误地判为可行，但在圆柱棱角处可能比精确欧氏距离更严格。

    ``contact_patch_enabled`` 启用接触斑模型：如果相交三角形同时满足
    “浅穿透 <= max_contact_penetration”、“相交区域距目标点 <=
    contact_patch_radius”和“三角形法向与砂轮轴线的夹角 <=
    contact_normal_tolerance_degrees”，则把它视为允许的接触斑，而不是碰撞。
    其余相交仍然判为碰撞。
    """

    if width <= 0 or inner_radius < 0 or clearance < 0 or epsilon < 0:
        raise ValueError("width 必须为正，clearance 和 epsilon 不能为负")
    if narrow_phase not in ("analytic_closest_point_flat", "clipped_polygon"):
        raise ValueError(f"不支持的 narrow_phase: {narrow_phase}")
    if contact_patch_enabled:
        if max_contact_penetration < 0.0:
            raise ValueError("max_contact_penetration 不能为负")
        if contact_patch_radius is not None and contact_patch_radius <= 0.0:
            raise ValueError("contact_patch_radius 必须为正")
        if not 0.0 <= contact_normal_tolerance_degrees <= 90.0:
            raise ValueError("contact_normal_tolerance_degrees 必须位于 [0, 90]")
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
    if contact_patch_enabled and contact_patch_radius is None:
        # 默认 patch 覆盖整个砂轮端面 footprint：目标点在端面边缘，
        # 端面最远点距目标点约 2r。
        contact_patch_radius = 2.0 * outer_radius
    contact_patch_radius_value = float(contact_patch_radius) if contact_patch_radius is not None else 0.0
    cos_normal_tolerance = float(np.cos(np.deg2rad(float(contact_normal_tolerance_degrees))))

    feasible = np.ones(len(centers), dtype=bool)
    witnesses = np.full((len(centers), 3), np.nan, dtype=float)
    tool_witnesses = np.full((len(centers), 3), np.nan, dtype=float)
    reasons = np.full(len(centers), "safe", dtype=object)
    triangle_counts = np.zeros(len(centers), dtype=np.int64)
    contact_max_penetrations = np.zeros(len(centers), dtype=float)
    contact_patch_distances = np.zeros(len(centers), dtype=float)
    contact_triangle_counts = np.zeros(len(centers), dtype=np.int64)
    penetrations = np.zeros(len(centers), dtype=float)
    for candidate_index, (center, axis, tangent, radial, radius) in enumerate(zip(centers, axes, tangents, radials, radii, strict=True)):
        axis = _unit(axis)
        tangent = _unit(tangent)
        radial = _unit(radial)
        nearby = query_cylinder_aabb(index, center, axis, outer_radius + clearance, width + 2.0 * clearance, epsilon=epsilon)
        triangle_counts[candidate_index] = len(nearby)
        target_contact_seen = False
        contact_patch_seen = False
        contact = np.asarray(candidates.contact_points[candidate_index], dtype=float)
        for triangle_index in nearby:
            if inner_radius > epsilon or narrow_phase == "clipped_polygon":
                # 环形模型仍使用裁剪法；实心圆柱可按需选择解析最近点法。
                state = triangle_cylinder_contact_state(
                    index.triangles[int(triangle_index)], center, axis, tangent, radial,
                    outer_radius + clearance, width + 2.0 * clearance,
                    inner_radius=max(0.0, inner_radius - clearance), epsilon=epsilon,
                    contact_point=contact,
                )
            else:
                state = triangle_cylinder_contact_state_analytic(
                    index.triangles[int(triangle_index)], center, axis,
                    outer_radius + clearance, width + 2.0 * clearance,
                    contact_point=contact, epsilon=epsilon,
                )
            if not state.hit:
                continue
            witness = state.witness
            penetrations[candidate_index] = max(
                penetrations[candidate_index], float(state.penetration)
            )
            # 只允许靠近目标点的近端面/侧面接触；远端面接触不是目标接触。
            near_cap_ok = state.near_cap is not False
            if allow_target_contact and witness is not None and np.linalg.norm(witness - contact) <= target_contact_tolerance:
                # 不能只看一个 witness 点就跳过整片三角形。目标接触豁免
                # 必须同时约束整个相交状态的穿透深度和目标点附近的 patch
                # 范围；否则三角形其余部分可能有明显穿透仍被判为可行。
                target_depth_ok = float(state.penetration) <= float(max_contact_penetration) + epsilon
                target_patch_ok = (
                    not contact_patch_enabled
                    or float(state.patch_max_distance) <= contact_patch_radius_value + epsilon
                )
                if near_cap_ok and target_depth_ok and target_patch_ok:
                    target_contact_seen = True
                    contact_max_penetrations[candidate_index] = max(
                        contact_max_penetrations[candidate_index], float(state.penetration)
                    )
                    contact_patch_distances[candidate_index] = max(
                        contact_patch_distances[candidate_index], float(state.patch_max_distance)
                    )
                    contact_triangle_counts[candidate_index] += 1
                    continue
            if contact_patch_enabled:
                # 接触斑允许条件：浅穿透 + 位于目标 patch 内 + 法向接近
                # 砂轮端面法向。三个条件必须同时满足。
                normal_ok = state.normal_alignment >= cos_normal_tolerance - 1e-12
                patch_ok = state.patch_max_distance <= contact_patch_radius_value + epsilon
                depth_ok = state.penetration <= float(max_contact_penetration) + epsilon
                if near_cap_ok and normal_ok and patch_ok and depth_ok:
                    contact_patch_seen = True
                    contact_max_penetrations[candidate_index] = max(
                        contact_max_penetrations[candidate_index], float(state.penetration)
                    )
                    contact_patch_distances[candidate_index] = max(
                        contact_patch_distances[candidate_index], float(state.patch_max_distance)
                    )
                    contact_triangle_counts[candidate_index] += 1
                    continue
            feasible[candidate_index] = False
            witnesses[candidate_index] = witness
            tool_witnesses[candidate_index] = _project_to_cylinder(witness, center, axis, radial, tangent, outer_radius, width)
            reasons[candidate_index] = state.kind
            break
        if feasible[candidate_index] and inner_radius <= epsilon and index.mesh_is_closed and _point_inside_closed_mesh_vectorized(center, index.triangles, epsilon):
            feasible[candidate_index] = False
            witnesses[candidate_index] = center
            tool_witnesses[candidate_index] = center
            reasons[candidate_index] = "workpiece_contains_wheel"
        elif feasible[candidate_index] and not index.mesh_is_closed:
            if contact_patch_seen:
                reasons[candidate_index] = "contact_patch_only_surface_only_non_closed_mesh"
            elif target_contact_seen:
                reasons[candidate_index] = "target_contact_only_surface_only_non_closed_mesh"
            else:
                reasons[candidate_index] = "safe_surface_only_non_closed_mesh"
        elif feasible[candidate_index] and contact_patch_seen:
            reasons[candidate_index] = "contact_patch_only"
        elif feasible[candidate_index] and target_contact_seen:
            reasons[candidate_index] = "target_contact_only"
    method = "exact_triangle_finite_annular_cylinder_aabb" if inner_radius > epsilon else "exact_triangle_finite_cylinder_aabb"
    if contact_patch_enabled:
        method += "_contact_patch"
    if inner_radius > epsilon or narrow_phase == "clipped_polygon":
        method += "_clipped_polygon"
    else:
        method += "_analytic_closest_point_flat"
    return CollisionResult(
        feasible=feasible,
        nearest_distance=np.where(feasible, np.inf, 0.0),
        clearance=np.where(feasible, np.inf, -float(clearance)),
        method=method,
        candidates=candidates,
        reason=reasons,
        witness_workpiece=witnesses,
        witness_tool=tool_witnesses,
        witness_region=triangle_counts,
        contact_max_penetration=contact_max_penetrations,
        contact_patch_max_distance=contact_patch_distances,
        contact_triangle_count=contact_triangle_counts,
        penetration=penetrations,
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
