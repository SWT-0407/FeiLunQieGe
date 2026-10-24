"""曲线切线和砂轮局部标架接口占位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LocalFrame:
    """采样点处的局部单位标架。"""

    tangent: Any
    radial_basis_1: Any
    radial_basis_2: Any
    axis: Any | None = None
    surface_normal: Any | None = None


def build_frames(
    points: Any,
    tangents: Any,
    surface_normals: Any | None = None,
    *,
    normal_is_reliable: bool = False,
    tangent_normal_threshold: float = 0.2,
) -> list[LocalFrame]:
    """建立切平面标架。

    ``radial_basis_1`` 和 ``radial_basis_2`` 这两个历史字段名继续保留，
    但它们现在表示切平面内的两个基向量，而不是法平面基向量。候选中心
    因而在切平面内形成圆轨迹，砂轮轴向取工件表面法向，砂轮圆盘平面与
    切平面平行、与法平面垂直。

    未传入表面法向时使用一个稳定的合成法向，仅用于旧单元测试；正式流程
    必须通过 ``estimate_surface_normals`` 从工件三角网格提供法向。

    ``normal_is_reliable=True`` 时，法向是可信的输入，标架**始终先固定法向**，
    再把曲线切线投影到该法向的切平面内，保证 ``t·n=0``。历史实现只在
    ``|t·n|`` 超过阈值时才这样做；短分支/端点切线可能带有法向分量但未超过
    阈值，导致正确的表面法向被旋转数度（circle 点 383 约 6.2°）。现在不再
    依赖该阈值判断切线可信度。``tangent_normal_threshold`` 只保留给
    ``normal_is_reliable=False`` 的历史路径。
    """

    if not 0.0 <= float(tangent_normal_threshold) < 1.0:
        raise ValueError("tangent_normal_threshold 必须位于 [0, 1)")
    point_array = np.asarray(points, dtype=float)
    tangent_array = np.asarray(tangents, dtype=float)
    if len(point_array) != len(tangent_array):
        raise ValueError("points 与 tangents 的数量必须一致")
    if surface_normals is None:
        normal_array = None
    else:
        normal_array = np.asarray(surface_normals, dtype=float)
        if normal_array.shape != tangent_array.shape:
            raise ValueError("surface_normals 必须与 tangents 具有相同形状")
    frames: list[LocalFrame] = []
    # 每个采样点分别建立局部切平面，避免曲线弯折时使用全局固定平面。
    for index, tangent in enumerate(tangent_array):
        norm = np.linalg.norm(tangent)
        if norm <= 1e-12:
            raise ValueError("存在无法归一化的曲线切线")
        t = tangent / norm
        if normal_array is None:
            # 兼容只提供曲线的旧测试：构造一个不平行于 t 的合成表面法向。
            references = np.eye(3)
            normal = references[int(np.argmin(np.abs(references @ t)))]
            normal = normal - np.dot(normal, t) * t
            normal_norm = np.linalg.norm(normal)
            if normal_norm <= 1e-12:
                raise ValueError(f"第 {index} 个表面法向与曲线切线退化或平行")
            normal = normal / normal_norm
        elif normal_is_reliable:
            # 法向可信：始终保留法向，把切线投影到法向的切平面内。
            # 这样最终轴不会被短分支/端点的差分切线旋转。
            normal = normal_array[index]
            normal_norm = np.linalg.norm(normal)
            if normal_norm <= 1e-12:
                raise ValueError(f"第 {index} 个表面法向为零向量")
            normal = normal / normal_norm
            t_projected = t - float(np.dot(t, normal)) * normal
            t_norm = float(np.linalg.norm(t_projected))
            if t_norm <= 1e-8:
                # 切线几乎与法向平行：优先沿用相邻点的切平面方向。
                for candidate_tangent in (
                    frames[-1].tangent if frames else None,
                    tangent_array[index + 1] if index + 1 < len(tangent_array) else None,
                    tangent_array[index - 1] if index > 0 else None,
                ):
                    if candidate_tangent is None:
                        continue
                    candidate = np.asarray(candidate_tangent, dtype=float)
                    candidate_norm = float(np.linalg.norm(candidate))
                    if candidate_norm <= 1e-12:
                        continue
                    candidate = candidate / candidate_norm
                    candidate = candidate - float(np.dot(candidate, normal)) * normal
                    candidate_norm = float(np.linalg.norm(candidate))
                    if candidate_norm > 1e-8:
                        t_projected = candidate
                        t_norm = candidate_norm
                        break
            if t_norm <= 1e-8:
                # 最后退化处理：选择一个与法向不平行的固定参考方向。
                reference = np.array([1.0, 0.0, 0.0]) if abs(float(normal[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
                t_projected = np.cross(normal, reference)
                t_norm = float(np.linalg.norm(t_projected))
                if t_norm <= 1e-12:
                    raise ValueError(f"第 {index} 个局部标架退化")
            t = t_projected / t_norm
            # 保持与原始切线同向，避免分支内切向翻转。
            if float(np.dot(t, tangent)) < 0.0:
                t = -t
        else:
            # 历史顺序：固定切线，把法向投影掉切向分量。
            normal = normal_array[index]
            normal = normal - np.dot(normal, t) * t
            normal_norm = np.linalg.norm(normal)
            if normal_norm <= 1e-12:
                raise ValueError(f"第 {index} 个表面法向与曲线切线退化或平行")
            normal = normal / normal_norm
        # q 是切平面中垂直于 root_line 切线的方向；(t, q) 张成切平面。
        q = np.cross(normal, t)
        q_norm = float(np.linalg.norm(q))
        if q_norm <= 1e-12:
            raise ValueError(f"第 {index} 个局部标架退化")
        q = q / q_norm
        frames.append(LocalFrame(tangent=t, radial_basis_1=t, radial_basis_2=q, axis=normal, surface_normal=normal))
    return frames


# Estimate the angle bisector of the two local face-normal families around root_line.
# KD Tree narrows face candidates; it does not define the normal by itself.
def _estimate_surface_normals_previous(points: Any, mesh: Any, neighbor_count: int = 24) -> np.ndarray:
    """用工件网格中最近三角形的面法向估计根部采样点法向。

    这是几何姿态的输入估计，不是法向重建或曲面拟合。使用三角形质心
    KDTree 只加速最近面查询；面法向本身由原始三角形叉乘直接计算。
    """

    points = np.asarray(points, dtype=float)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    if int(neighbor_count) < 2:
        raise ValueError("Angle-bisector estimation requires at least 2 faces")
    if len(triangles) == 0:
        raise ValueError("工件网格没有三角形，无法估计表面法向")
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > 1e-12
    if not np.any(valid):
        raise ValueError("工件网格所有三角形均退化，无法估计表面法向")
    face_normals[valid] /= lengths[valid, None]
    centroids = triangles.mean(axis=1)
    mesh_center = vertices.mean(axis=0)
    face_tree = cKDTree(centroids[valid])
    query_count = min(int(neighbor_count), int(np.count_nonzero(valid)))
    if query_count < 2:
        raise ValueError("Mesh has too few valid faces for angle-bisector estimation")

    candidate_normals: list[list[np.ndarray]] = []
    for index, point in enumerate(points):
        _, nearest_indices = face_tree.query(point, k=query_count)
        nearest_indices = np.asarray(nearest_indices, dtype=np.int64).reshape(-1)
        local_normals = np.asarray(face_normals[valid][nearest_indices], dtype=float).reshape(-1, 3)
        reference = local_normals[0].reshape(3)

        # Select the two most different local face directions as side seeds.
        # The remaining faces are assigned to whichever seed they resemble more.
        similarity = local_normals @ local_normals.T
        seed_a, seed_b = np.unravel_index(int(np.argmin(similarity)), similarity.shape)
        group_a = (local_normals @ local_normals[seed_a]) >= (local_normals @ local_normals[seed_b])
        if np.all(group_a) or not np.any(group_a):
            group_a = np.zeros(query_count, dtype=bool)
            group_a[seed_a] = True
        group_b = ~group_a

        # A strongly imbalanced 1-vs-many neighborhood often means that the
        # minority face is the true adjacent side while the majority faces
        # belong to a nearby surface. In that case use mesh-center direction
        # only to correct the relative sign of the majority family. Balanced
        # neighborhoods are left untouched, preserving already accepted poses.
        if min(int(group_a.sum()), int(group_b.sum())) * 4 <= max(int(group_a.sum()), int(group_b.sum())):
            local_centroids = centroids[valid][nearest_indices]
            for group in (group_a, group_b):
                group_center = local_centroids[group].mean(axis=0)
                group_direction = local_normals[group].mean(axis=0)
                if float(np.dot(group_direction, group_center - mesh_center)) < 0.0:
                    local_normals[group] *= -1.0

        def mean_direction(values: np.ndarray) -> np.ndarray:
            result = values.mean(axis=0)
            length = np.linalg.norm(result)
            return result / length if length > 1e-12 else np.zeros(3, dtype=float)

        normal_a = mean_direction(local_normals[group_a])
        normal_b = mean_direction(local_normals[group_b])
        if np.any(normal_a) and np.any(normal_b):
            # Both internal and external angle bisectors are valid geometric
            # candidates; continuity along root_line decides which one to use.
            candidate_vectors = [normal_a + normal_b, normal_a - normal_b]
        else:
            candidate_vectors = [mean_direction(local_normals)]
        candidates_at_point = []
        for vector in candidate_vectors:
            vector = np.asarray(vector, dtype=float).reshape(-1)[:3]
            vector_length = np.linalg.norm(vector)
            if vector_length > 1e-12:
                candidates_at_point.append(vector / vector_length)
        if not candidates_at_point:
            candidates_at_point = [np.asarray(reference, dtype=float).reshape(3)]
        reference = np.asarray(reference, dtype=float).reshape(-1)[:3]
        # Remove the arbitrary global sign caused by OBJ face winding.
        for candidate_index, candidate in enumerate(candidates_at_point):
            if float(np.dot(candidate, reference)) < 0.0:
                candidates_at_point[candidate_index] = -candidate
        candidate_normals.append(candidates_at_point)

    # Dynamic programming selects the smoothest candidate sequence. The
    # absolute dot product treats n and -n as the same wheel plane.
    costs = np.full((len(points), 2), np.inf, dtype=float)
    choices = np.zeros((len(points), 2), dtype=np.int64)
    costs[0, : len(candidate_normals[0])] = 0.0
    for index in range(1, len(points)):
        for current_index, current in enumerate(candidate_normals[index]):
            previous_costs = []
            for previous_index, previous in enumerate(candidate_normals[index - 1]):
                transition = 1.0 - abs(float(np.dot(current, previous)))
                # Prefer the internal bisector by default; select the external
                # bisector only when continuity improves enough to overcome it.
                external_penalty = 0.2 if current_index == 1 else 0.0
                previous_costs.append(costs[index - 1, previous_index] + transition + external_penalty)
            best_previous = int(np.argmin(previous_costs))
            costs[index, current_index] = previous_costs[best_previous]
            choices[index, current_index] = best_previous
    selected = np.zeros(len(points), dtype=np.int64)
    selected[-1] = int(np.argmin(costs[-1, : len(candidate_normals[-1])]))
    for index in range(len(points) - 1, 0, -1):
        selected[index - 1] = choices[index, selected[index]]
    return np.asarray([candidate_normals[i][selected[i]] for i in range(len(points))], dtype=float)


def _point_triangle_distance(point: np.ndarray, triangle: np.ndarray) -> float:
    """Return the exact Euclidean distance from one point to one triangle."""

    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))
    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        projection = a + (d1 / (d1 - d3)) * ab
        return float(np.linalg.norm(point - projection))
    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        projection = a + (d2 / (d2 - d6)) * ac
        return float(np.linalg.norm(point - projection))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        projection = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
        return float(np.linalg.norm(point - projection))
    normal = np.cross(ab, ac)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1e-12:
        return min(
            _point_triangle_distance(point, np.array([a, b, b])),
            _point_triangle_distance(point, np.array([a, c, c])),
            _point_triangle_distance(point, np.array([b, c, c])),
        )
    return abs(float(np.dot(point - a, normal))) / normal_length


def _estimate_tangents_for_normal_projection(points: np.ndarray) -> np.ndarray:
    """用与曲线模块一致的差分切线投影最终法向。"""

    tangents = np.empty_like(points)
    if len(points) == 1:
        tangents[0] = np.array([1.0, 0.0, 0.0])
    else:
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        if len(points) > 2:
            tangents[1:-1] = points[2:] - points[:-2]
    lengths = np.linalg.norm(tangents, axis=1)
    tangents[lengths > 1e-12] /= lengths[lengths > 1e-12, None]
    tangents[lengths <= 1e-12] = np.array([1.0, 0.0, 0.0])
    return tangents


def _estimate_surface_normals_local(
    points: Any,
    mesh: Any,
    neighbor_count: int = 24,
    candidate_count: int | None = None,
    side_count: int = 8,
    *,
    coherence_angle_degrees: float = 25.0,
    offset_epsilon: float = 1e-5,
    weight_distance_floor: float = 1e-5,
    flash_points: Any | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, Any]]]:
    """估计 root_line 两侧面的法向及其内角平分法向。

    KD Tree 只负责快速取得候选三角形，不能直接决定法向。候选面先按
    三角形质心相对于 root_line 切线的横向偏移分成两侧；随后每侧只保留
    法向方向相互一致的局部面片，并用带距离下限的权重平均。这样可以避免
    邻近但属于第三个表面的三角形混入，也避免零距离面片产生数值上无限大
    的权重。横截面无法稳定分侧时，回退到按“最近且方向相异”的两个法向
    簇选择。最终法向固定为 ``normalize(n1 + n2)``，不再在外角平分线和内
    角平分线之间做连续性猜测。
    """

    points = np.asarray(points, dtype=float)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    if len(points) == 0:
        return np.empty((0, 3), dtype=float)
    if neighbor_count < 1 or side_count < 1:
        raise ValueError("neighbor_count and side_count must be positive")
    if coherence_angle_degrees <= 0.0 or coherence_angle_degrees >= 90.0:
        raise ValueError("coherence_angle_degrees 必须位于 (0, 90) 度")
    if offset_epsilon <= 0.0 or weight_distance_floor <= 0.0:
        raise ValueError("offset_epsilon 和 weight_distance_floor 必须为正数")
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > 1e-12
    if not np.any(valid):
        raise ValueError("Mesh has no valid triangles for normal estimation")
    face_normals[valid] /= lengths[valid, None]
    valid_indices = np.flatnonzero(valid)
    valid_triangles = triangles[valid]
    valid_centroids = valid_triangles.mean(axis=1)
    # ``flash_points`` carries the semantic label that an OBJ triangle mesh
    # lacks.  When it is present, every nearby face normal is oriented using
    # the closest labelled flash point F: choose the sign with n dot (F - O)
    # positive.  This is a local geometric rule, not a branch-wide smoothing
    # or a mesh-centre heuristic.
    flash = None if flash_points is None else np.asarray(flash_points, dtype=float)
    if flash is not None and (flash.ndim != 2 or flash.shape[1] != 3 or len(flash) == 0):
        raise ValueError("flash_points 必须是非空的 (n, 3) 飞边点数组")
    flash_tree = cKDTree(flash) if flash is not None else None
    mesh_center = vertices.mean(axis=0)
    tree = cKDTree(valid_centroids)
    broad_count = min(int(candidate_count or max(64, neighbor_count * 4)), len(valid_centroids))
    normal_cosine = float(np.cos(np.deg2rad(coherence_angle_degrees)))

    tangents = np.empty_like(points)
    if len(points) == 1:
        tangents[0] = np.array([1.0, 0.0, 0.0])
    else:
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        if len(points) > 2:
            tangents[1:-1] = points[2:] - points[:-2]
    tangent_lengths = np.linalg.norm(tangents, axis=1)
    tangents[tangent_lengths > 1e-12] /= tangent_lengths[tangent_lengths > 1e-12, None]
    tangents[tangent_lengths <= 1e-12] = np.array([1.0, 0.0, 0.0])

    # 下面两个局部函数只处理已经通过精确点-三角形距离筛选的候选面。
    # 面片网格可能是 triangle soup，不能假设相邻三角形共享顶点，因此
    # 用几何偏移和法向一致性完成分组，而不依赖拓扑邻接。
    def coherent_side(
        candidate_positions: np.ndarray,
        distances: np.ndarray,
        local_face_normals: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        if len(candidate_positions) == 0:
            return None
        best_mask: np.ndarray | None = None
        best_seed = 0
        best_score: tuple[int, float] | None = None
        for seed_position in range(len(candidate_positions)):
            seed = local_face_normals[seed_position]
            # abs(dot) 允许输入网格中少量三角形绕向相反，后续统一到 seed。
            mask = np.abs(local_face_normals @ seed) >= normal_cosine
            support = int(np.count_nonzero(mask))
            if support == 0:
                continue
            median_distance = float(np.median(distances[mask]))
            score = (support, -median_distance)
            if best_score is None or score > best_score:
                best_score = score
                best_mask = mask
                best_seed = seed_position
        if best_mask is None:
            return None
        selected_pool = np.flatnonzero(best_mask)
        selected_pool = selected_pool[np.argsort(distances[selected_pool])]
        selected = selected_pool[: int(side_count)]
        if len(selected) == 0:
            return None
        seed = local_face_normals[best_seed]
        selected_normals = local_face_normals[selected].copy()
        if flash_tree is None:
            signs = np.sign(selected_normals @ seed)
            signs[signs == 0.0] = 1.0
            selected_normals *= signs[:, None]
        # 距离下限至少取当前面片距离中位数，避免 d=0 时单个面片权重达到 1e10。
        distance_floor = max(float(weight_distance_floor), float(np.median(distances[selected])))
        weights = 1.0 / (distances[selected] + distance_floor)
        weighted = (selected_normals * weights[:, None]).sum(axis=0)
        length = float(np.linalg.norm(weighted))
        if length <= 1e-12:
            return None
        normal = weighted / length
        # 没有飞边标签的旧调用才使用网格中心作为回退符号规则。正式流程
        # 由 O->F 逐面确定正方向，不能再用这一全局启发式覆盖它。
        patch_center = valid_centroids[indices[candidate_positions[selected]]].mean(axis=0)
        if flash_tree is None and float(np.dot(normal, patch_center - mesh_center)) < 0.0:
            normal = -normal
            selected_normals *= -1.0
        detail = {
            "face_indices": valid_indices[indices[candidate_positions[selected]]].astype(int).tolist(),
            "distances": distances[selected].astype(float).tolist(),
            "weights": weights.astype(float).tolist(),
            "normal": normal.astype(float).tolist(),
            "selected_count": int(len(selected)),
            "coherent_support": int(np.count_nonzero(best_mask)),
            "coherence_angle_degrees": float(coherence_angle_degrees),
            "weight_distance_floor": float(distance_floor),
            "patch_center": patch_center.astype(float).tolist(),
        }
        return normal, detail

    candidate_normals: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for point, tangent in zip(points, tangents, strict=True):
        _, indices = tree.query(point, k=broad_count)
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        distances = np.asarray([_point_triangle_distance(point, valid_triangles[i]) for i in indices])
        order = np.argsort(distances)
        indices = indices[order]
        distances = distances[order]
        local_normals = face_normals[valid][indices].copy()
        flash_point = None
        flash_direction = None
        face_alignment_scores = None
        if flash_tree is not None:
            # O is the root-line sample and F is its nearest labelled flash
            # point.  The included angle alone is ambiguous because both n
            # and -n usually make an angle below 180 degrees with OF; the
            # smaller-angle condition is exactly n dot OF > 0.
            _, flash_index = flash_tree.query(point, k=1)
            flash_point = flash[int(flash_index)]
            flash_direction = flash_point - point
            direction_norm = float(np.linalg.norm(flash_direction))
            if direction_norm > 1e-12:
                flash_direction /= direction_norm
                face_alignment_scores = local_normals @ flash_direction
                local_normals[face_alignment_scores < 0.0] *= -1.0
            else:
                flash_direction = np.zeros(3, dtype=float)
                face_alignment_scores = np.zeros(len(local_normals), dtype=float)
        offsets = valid_centroids[indices] - point
        offsets -= np.einsum("ij,j->i", offsets, tangent)[:, None] * tangent[None, :]
        offset_lengths = np.linalg.norm(offsets, axis=1)
        usable = offset_lengths > float(offset_epsilon)
        side_axis = np.zeros(3, dtype=float)
        side_groups: list[np.ndarray] = []
        grouping_method = "projected_offset"
        if int(np.count_nonzero(usable)) >= 4:
            unit_offsets = offsets[usable] / offset_lengths[usable, None]
            similarity = unit_offsets @ unit_offsets.T
            seed_a, seed_b = np.unravel_index(int(np.argmin(similarity)), similarity.shape)
            side_axis = unit_offsets[seed_a] - unit_offsets[seed_b]
            side_axis_length = float(np.linalg.norm(side_axis))
            if side_axis_length > 1e-12:
                side_axis /= side_axis_length
                side_scores = unit_offsets @ side_axis
                usable_indices = np.flatnonzero(usable)
                group_a = usable_indices[side_scores >= 0.0]
                group_b = usable_indices[side_scores < 0.0]
                if len(group_a) >= 2 and len(group_b) >= 2:
                    side_groups = [group_a, group_b]
        if len(side_groups) != 2:
            # 退化时不强行按一个 PCA 轴切分。按距离顺序寻找两个方向相异
            # 的法向簇，可覆盖 root_line 局部质心重合或切线估计不稳定的情况。
            grouping_method = "nearest_distinct_normal_clusters"
            side_groups = []
            seed_a = 0
            seed_b = None
            for candidate_position in range(1, len(local_normals)):
                if abs(float(np.dot(local_normals[seed_a], local_normals[candidate_position]))) < normal_cosine:
                    seed_b = candidate_position
                    break
            if seed_b is not None:
                group_a = np.flatnonzero(np.abs(local_normals @ local_normals[seed_a]) >= normal_cosine)
                group_b = np.flatnonzero(np.abs(local_normals @ local_normals[seed_b]) >= normal_cosine)
                side_groups = [group_a, group_b]
                side_axis = np.zeros(3, dtype=float)

        side_vectors: list[np.ndarray] = []
        side_details: list[dict[str, Any]] = []
        for group in side_groups[:2]:
            result = coherent_side(group, distances[group], local_normals[group])
            if result is None:
                continue
            side_normal, detail = result
            side_vectors.append(side_normal)
            side_details.append(detail)

        if len(side_vectors) < 2:
            # 最后保留最近有效面法向，避免空结果；正常工件交线应在前面的
            # 两个簇中得到两个侧面，这个分支只表示输入几何不足以判断。
            side_vectors = [local_normals[0]]
            side_details = [
                {
                    "face_indices": [int(valid_indices[indices[0]])],
                    "distances": [float(distances[0])],
                    "weights": [1.0],
                    "normal": local_normals[0].astype(float).tolist(),
                    "selected_count": 1,
                    "coherent_support": 1,
                    "coherence_angle_degrees": float(coherence_angle_degrees),
                    "weight_distance_floor": float(weight_distance_floor),
                }
            ]
        if len(side_vectors) == 1:
            final_normal = side_vectors[0] / max(float(np.linalg.norm(side_vectors[0])), 1e-12)
        else:
            # 只取两侧面的内角平分线；外角平分线不再作为候选，避免在
            # 连续性优化中被错误选中而使砂轮嵌入工件。
            bisector = side_vectors[0] + side_vectors[1]
            if float(np.linalg.norm(bisector)) <= 1e-12:
                final_normal = side_vectors[0]
            else:
                final_normal = bisector / float(np.linalg.norm(bisector))
        candidate_normals.append(final_normal)
        diagnostics.append(
            {
                "tangent": tangent.astype(float).tolist(),
                "side_axis": side_axis.astype(float).tolist(),
                "group_sizes": [int(len(group)) for group in side_groups[:2]],
                "side_details": side_details,
                "grouping_method": grouping_method,
                "nearest_face_index": int(valid_indices[indices[0]]),
                "selected_normal": final_normal.astype(float).tolist(),
                "orientation_rule": "nearest_flash_OF_smaller_angle" if flash_tree is not None else "mesh_center_fallback",
                "flash_point": flash_point.astype(float).tolist() if flash_point is not None else None,
                "OF_unit": flash_direction.astype(float).tolist() if flash_direction is not None else None,
                "nearest_face_alignment_before_sign": float(face_alignment_scores[0]) if face_alignment_scores is not None else None,
            }
        )
    # 保留每个采样点由 n1+n2 得到的有向结果。法向平面虽然对 n 与 -n
    # 等价，但诊断输出需要能直接和工件的外侧方向比较，不能跨点任意翻转。
    result = np.asarray(candidate_normals, dtype=float)
    for index, item in enumerate(diagnostics):
        item["selected_normal"] = result[index].astype(float).tolist()
        item["normal_method"] = "coherent_two_side_internal_bisector"
    if return_diagnostics:
        return result, diagnostics
    return result


def _surface_patch_normals(
    points: Any,
    mesh: Any,
    *,
    neighbor_count: int = 32,
    distance_factor: float = 3.0,
    distance_floor: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, float, float]:
    """从 root 点附近的网格面片估计局部表面法向。

    与两侧角平分线不同，本函数不假设 root 点位于一条棱边上，而是先回答
    “离这个 root 点最近的网格表面是什么”。它只保留距离在
    ``max(d0 * distance_factor, distance_floor)`` 内的三角形，做距离加权
    平均；在光滑面上得到该面的法向，在真正的棱边上最近面片会同时包含
    两侧，``coherence`` 会很大，调用方据此退回两侧角平分线。

    返回：
    - ``normals``: 每个点的局部面片法向；
    - ``coherence_deg``: 被保留面片相对平均法向的最大夹角；
    - ``nearest_distance``: 最近三角形距离；
    - ``face_count``: 被保留的三角形数量；
    - ``mesh_is_closed``: 网格是否拓扑封闭；
    - ``outward_sign``: 由有向体积确定的全局外法向符号；
    - ``signed_volume``: 网格有向体积。
    """

    points = np.asarray(points, dtype=float)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    if len(points) == 0:
        return (
            np.empty((0, 3), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=np.int64),
            False,
            1.0,
            0.0,
        )
    if neighbor_count < 1:
        raise ValueError("surface_patch_neighbors 必须为正")
    if distance_factor <= 1.0 or distance_floor <= 0.0:
        raise ValueError("distance_factor 必须大于 1，distance_floor 必须为正")

    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > 1e-12
    if not np.any(valid):
        raise ValueError("Mesh has no valid triangles for surface-patch normals")
    face_normals[valid] /= lengths[valid, None]
    valid_indices = np.flatnonzero(valid)
    valid_triangles = triangles[valid]
    valid_centroids = valid_triangles.mean(axis=1)

    # 闭合网格的有向体积给出全局外法向符号；圆环等非凸形状不能用“网格中心”
    # 启发式，但有向体积仍能判断 winding 是朝外还是朝内。
    signed_volume = float(
        np.einsum(
            "ij,ij->i",
            valid_triangles[:, 0],
            np.cross(valid_triangles[:, 1], valid_triangles[:, 2]),
        ).sum()
        / 6.0
    )
    from feilunqiege.collision.disk_exact import _is_closed_triangle_mesh

    mesh_is_closed = _is_closed_triangle_mesh(faces)
    outward_sign = 1.0
    if mesh_is_closed and signed_volume < 0.0:
        outward_sign = -1.0
        face_normals *= -1.0

    tree = cKDTree(valid_centroids)
    query_count = min(int(neighbor_count), len(valid_centroids))
    normals = np.zeros((len(points), 3), dtype=float)
    coherence = np.zeros(len(points), dtype=float)
    nearest = np.zeros(len(points), dtype=float)
    counts = np.zeros(len(points), dtype=np.int64)

    for index, point in enumerate(points):
        _, neighbor_indices = tree.query(point, k=query_count)
        neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64).reshape(-1)
        distances = np.asarray(
            [_point_triangle_distance(point, valid_triangles[neighbor]) for neighbor in neighbor_indices],
            dtype=float,
        )
        order = np.argsort(distances)
        neighbor_indices = neighbor_indices[order]
        distances = distances[order]
        cutoff = max(float(distances[0]) * float(distance_factor), float(distance_floor))
        keep = distances <= cutoff
        if not np.any(keep):
            keep[0] = True
        kept_normals = face_normals[valid][neighbor_indices[keep]].copy()
        # 非闭合网格的局部 winding 可能不一致；以最近面为种子统一符号。
        if not mesh_is_closed:
            signs = np.sign(kept_normals @ kept_normals[0])
            signs[signs == 0.0] = 1.0
            kept_normals *= signs[:, None]
        weights = 1.0 / np.maximum(distances[keep], float(distance_floor))
        averaged = (kept_normals * weights[:, None]).sum(axis=0)
        averaged_norm = float(np.linalg.norm(averaged))
        if averaged_norm <= 1e-12:
            averaged = kept_normals[0].copy()
            averaged_norm = float(np.linalg.norm(averaged))
        averaged /= max(averaged_norm, 1e-12)
        cosines = np.clip(kept_normals @ averaged, -1.0, 1.0)
        coherence[index] = float(np.degrees(np.arccos(np.min(cosines))))
        normals[index] = averaged
        nearest[index] = float(distances[0])
        counts[index] = int(np.count_nonzero(keep))

    return normals, coherence, nearest, counts, mesh_is_closed, outward_sign, signed_volume


def _branch_runs_from_ids(branch_ids: Any, point_count: int) -> list[tuple[int, int]]:
    """把连续 branch_id 转成半开区间。"""

    if point_count == 0:
        return []
    branch_array = np.asarray(branch_ids, dtype=np.int64).reshape(-1)
    if len(branch_array) != point_count:
        raise ValueError("branch_ids length must match points")
    starts = [0]
    starts.extend((np.flatnonzero(branch_array[1:] != branch_array[:-1]) + 1).tolist())
    ends = starts[1:] + [point_count]
    return list(zip(starts, ends, strict=True))


def _refine_normals_with_surface_patch(
    points: Any,
    tangents: Any,
    mesh: Any,
    local_result: Any,
    local_diagnostics: list[dict[str, Any]],
    *,
    flash_points: Any | None = None,
    branch_ids: Any | None = None,
    surface_patch_neighbors: int = 32,
    surface_patch_distance_factor: float = 3.0,
    surface_patch_coherence_degrees: float = 25.0,
    max_bisector_deviation_degrees: float = 3.5,
    continuity_repair_degrees: float = 30.0,
    continuity_repair_iterations: int = 3,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """用局部表面片 + 分支连续性修复闪光点附近的法向。

    处理顺序：
    1. 计算最近网格面片法向及其一致性；
    2. 面片一致（光滑面）时直接用面片法向；不一致（真棱边）时保留原来的
       两侧内角平分线；
    3. 闭合网格用有向体积定外法向，不再依赖在“面上”几乎为零的 ``n·OF``；
    4. 沿同一分支做连续性修复，超过阈值的孤立跳变用相邻点替换。

    本函数不改变碰撞判定，只替换传入的法向场。
    """

    points_array = np.asarray(points, dtype=float)
    tangents_array = np.asarray(tangents, dtype=float)
    local_result = np.asarray(local_result, dtype=float)
    if len(points_array) == 0:
        return local_result, local_diagnostics
    if surface_patch_coherence_degrees <= 0.0 or surface_patch_coherence_degrees >= 90.0:
        raise ValueError("surface_patch_coherence_degrees 必须位于 (0, 90) 度")
    if max_bisector_deviation_degrees < 0.0 or max_bisector_deviation_degrees >= 90.0:
        raise ValueError("max_bisector_deviation_degrees 必须位于 [0, 90) 度")
    if continuity_repair_degrees <= 0.0 or continuity_repair_degrees >= 180.0:
        raise ValueError("continuity_repair_degrees 必须位于 (0, 180) 度")
    if continuity_repair_iterations < 0:
        raise ValueError("continuity_repair_iterations 不能为负")

    patch_normals, coherence, nearest, face_counts, mesh_is_closed, outward_sign, signed_volume = _surface_patch_normals(
        points_array,
        mesh,
        neighbor_count=surface_patch_neighbors,
        distance_factor=surface_patch_distance_factor,
    )

    flash_tree = None
    flash_array = None
    if flash_points is not None:
        flash_array = np.asarray(flash_points, dtype=float)
        if flash_array.ndim != 2 or flash_array.shape[1] != 3 or len(flash_array) == 0:
            raise ValueError("flash_points 必须是非空的 (n, 3) 飞边点数组")
        flash_tree = cKDTree(flash_array)

    coherence_threshold = float(surface_patch_coherence_degrees)
    result = np.zeros_like(local_result)
    normal_sources: list[str] = []
    patch_used = np.zeros(len(points_array), dtype=bool)
    bisector_deviations = np.zeros(len(points_array), dtype=float)

    for index, (point, tangent) in enumerate(zip(points_array, tangents_array, strict=True)):
        tangent = np.asarray(tangent, dtype=float)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / tangent_norm

        local_normal = local_result[index]
        local_norm = float(np.linalg.norm(local_normal))
        patch_normal = patch_normals[index]
        if local_norm <= 1e-12:
            bisector_deviation = 180.0
            local_normal = patch_normal
            local_norm = float(np.linalg.norm(local_normal))
        else:
            bisector_deviation = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            float(np.dot(local_normal / local_norm, patch_normal)),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
        bisector_deviations[index] = bisector_deviation
        # 两侧角平分线在光滑面上通常也是该面片的支撑法向；只有当它明显
        # 偏离局部表面片（端点/短分支/底面倒角等）时才改用表面片法向。
        # 这样既修复 90° 错误，又保留原方法中“轮面不会切入相邻面片”的
        # 支撑性。
        if float(coherence[index]) <= coherence_threshold and bisector_deviation > max_bisector_deviation_degrees:
            normal = patch_normal.copy()
            source = "surface_patch"
            patch_used[index] = True
            # 开放网格没有可靠的全局外法向；用飞边方向或局部结果定符号。
            # 闭合网格已由有向体积定好外法向，不能再让“面上几乎为零”的
            # n·OF 随机翻转它。
            if not mesh_is_closed:
                if flash_tree is not None:
                    _, flash_index = flash_tree.query(point, k=1)
                    of = flash_array[int(flash_index)] - point
                    if abs(float(np.dot(normal, of))) > 1e-3:
                        if float(np.dot(normal, of)) < 0.0:
                            normal = -normal
                    elif float(np.dot(normal, local_result[index])) < 0.0:
                        normal = -normal
                elif float(np.dot(normal, local_result[index])) < 0.0:
                    normal = -normal
        else:
            # 支撑性角平分线：光滑面上它与局部面片接近，保留它可以避免
            # 轮面切入相邻三角面片；真棱边也继续使用两侧内角平分线。
            normal = local_normal.copy()
            source = "two_side_bisector"
            if mesh_is_closed:
                # 闭合网格的 patch 法向是外法向参考；用它统一符号。
                if float(np.dot(normal, patch_normals[index])) < 0.0:
                    normal = -normal
            elif flash_tree is not None:
                _, flash_index = flash_tree.query(point, k=1)
                of = flash_array[int(flash_index)] - point
                if float(np.dot(normal, of)) < 0.0:
                    normal = -normal

        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            normal = patch_normals[index] if not patch_used[index] else local_result[index]
            normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            normal = np.cross(tangent, np.array([1.0, 0.0, 0.0]))
            normal_norm = float(np.linalg.norm(normal))
        result[index] = normal / max(normal_norm, 1e-12)
        normal_sources.append(source)

    # 分支级连续性修复：只处理超过阈值的孤立跳变；真棱边的整体平滑变化
    # 不会被当作跳变。被修复的点会在诊断中显式标记。
    branch_array = None
    if branch_ids is not None:
        branch_array = np.asarray(branch_ids, dtype=np.int64).reshape(-1)
        if len(branch_array) != len(points_array):
            raise ValueError("branch_ids length must match points")
        runs = _branch_runs_from_ids(branch_array, len(points_array))
    else:
        # 无显式分支标签时，用明显的空间间隙推断，避免跨分支平滑。
        if len(points_array) <= 1:
            runs = [(0, len(points_array))]
        else:
            gaps = np.linalg.norm(np.diff(points_array, axis=0), axis=1)
            positive = gaps[gaps > 1e-12]
            reference_gap = float(np.median(positive)) if len(positive) else 1.0
            boundaries = np.flatnonzero(gaps > max(reference_gap * 8.0, 1e-8)) + 1
            starts = [0] + boundaries.tolist()
            ends = boundaries.tolist() + [len(points_array)]
            runs = list(zip(starts, ends, strict=True))

    # 用最终法向修正切线：端点单边差分切线可能含有法向分量，不能反过来
    # 用法向去抵消它。这里先固定法向，再把原始切线投影到法向的切平面。
    corrected_tangents = np.zeros_like(tangents_array)
    for index in range(len(points_array)):
        tangent = tangents_array[index]
        tangent_norm = float(np.linalg.norm(tangent))
        tangent = tangent / tangent_norm if tangent_norm > 1e-12 else np.array([1.0, 0.0, 0.0])
        tangent = tangent - float(np.dot(tangent, result[index])) * result[index]
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            tangent = np.array([1.0, 0.0, 0.0])
            tangent = tangent - float(np.dot(tangent, result[index])) * result[index]
            tangent_norm = float(np.linalg.norm(tangent))
        corrected_tangents[index] = tangent / max(tangent_norm, 1e-12)

    repaired = np.zeros(len(points_array), dtype=bool)
    repair_angles = np.zeros(len(points_array), dtype=float)
    threshold = float(np.deg2rad(continuity_repair_degrees))
    neighbor_agreement_threshold = np.deg2rad(20.0)
    for start, end in runs:
        if end - start < 2:
            continue
        for _ in range(int(continuity_repair_iterations)):
            changed = False
            for index in range(start, end):
                if index == start:
                    neighbors = [index + 1]
                elif index == end - 1:
                    neighbors = [index - 1]
                else:
                    neighbors = [index - 1, index + 1]
                angles = [
                    float(np.arccos(np.clip(np.dot(result[index], result[neighbor]), -1.0, 1.0)))
                    for neighbor in neighbors
                ]
                if max(angles) <= threshold:
                    continue
                # 两个邻居本身不一致时，可能是真拐角，不做强行平滑。
                if len(neighbors) == 2:
                    agreement = float(
                        np.arccos(np.clip(np.dot(result[neighbors[0]], result[neighbors[1]]), -1.0, 1.0))
                    )
                    if agreement > neighbor_agreement_threshold:
                        continue
                replacement = result[neighbors].mean(axis=0)
                replacement = replacement - float(np.dot(replacement, corrected_tangents[index])) * corrected_tangents[index]
                replacement_norm = float(np.linalg.norm(replacement))
                if replacement_norm <= 1e-12:
                    continue
                replacement = replacement / replacement_norm
                # 只有替换后更接近局部表面片法向时才接受，避免用错误邻居
                # 覆盖本来正确的表面法向。
                current_score = float(np.dot(result[index], patch_normals[index]))
                replacement_score = float(np.dot(replacement, patch_normals[index]))
                if replacement_score + 1e-6 < current_score:
                    continue
                result[index] = replacement
                repaired[index] = True
                repair_angles[index] = max(angles)
                changed = True
            if not changed:
                break

    # 更新诊断，便于逐点审计。
    for index in range(len(points_array)):
        local_diagnostics[index]["selected_normal"] = result[index].tolist()
        local_diagnostics[index]["tangent_corrected"] = corrected_tangents[index].tolist()
        local_diagnostics[index]["normal_method"] = (
            "surface_patch_continuity" if normal_sources[index] == "surface_patch"
            else "two_side_bisector_continuity"
        )
        local_diagnostics[index]["normal_source"] = normal_sources[index]
        local_diagnostics[index]["surface_patch_normal"] = patch_normals[index].tolist()
        local_diagnostics[index]["surface_patch_coherence_deg"] = float(coherence[index])
        local_diagnostics[index]["surface_patch_face_count"] = int(face_counts[index])
        local_diagnostics[index]["surface_patch_nearest_distance"] = float(nearest[index])
        local_diagnostics[index]["bisector_deviation_deg"] = float(bisector_deviations[index])
        local_diagnostics[index]["outward_orientation"] = (
            "mesh_signed_volume" if mesh_is_closed else "flash_point_or_local"
        )
        local_diagnostics[index]["mesh_signed_volume"] = float(signed_volume)
        local_diagnostics[index]["continuity_repaired"] = bool(repaired[index])
        local_diagnostics[index]["continuity_repair_angle_deg"] = float(np.degrees(repair_angles[index]))

    return result, local_diagnostics


def estimate_surface_normals(
    points: Any,
    mesh: Any,
    neighbor_count: int = 24,
    candidate_count: int | None = None,
    side_count: int = 8,
    *,
    coherence_angle_degrees: float = 25.0,
    offset_epsilon: float = 1e-5,
    weight_distance_floor: float = 1e-5,
    continuity_jump_threshold_degrees: float = 20.0,
    continuity_normal_weight: float = 1.0,
    continuity_patch_weight: float = 0.15,
    continuity_distance_weight: float = 0.1,
    branch_ids: Any | None = None,
    side_direction_hints: Any | None = None,
    side_direction_weight: float = 2.0,
    flash_points: Any | None = None,
    normal_method: str = "two_face_angle_bisector",
    surface_patch_neighbors: int = 32,
    surface_patch_distance_factor: float = 3.0,
    surface_patch_coherence_degrees: float = 25.0,
    max_bisector_deviation_degrees: float = 3.5,
    continuity_repair_degrees: float = 30.0,
    continuity_repair_iterations: int = 3,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, Any]]]:
    """在局部两侧面法向基础上执行 root_line 分支级连续性修正。

    局部函数仍负责 KD Tree 粗筛、点到三角形真实距离、两侧面聚类和角平分
    法向。这里不重新发明局部法向，而是把每个点的两个侧面法向作为一个带
    标签的状态，沿同一条 root_line 用动态规划选择最连续的交换关系和符号。
    这样可以修复三角网格朝向翻转、两侧标签交换造成的孤立突变；无法由局部
    候选解释的异常会保留诊断标记，不静默伪装成高置信度结果。
    """
    points_array = np.asarray(points, dtype=float)
    if side_direction_hints is not None:
        side_hints = np.asarray(side_direction_hints, dtype=float)
        if side_hints.shape != points_array.shape:
            raise ValueError("side_direction_hints must have the same shape as points")
        if side_direction_weight < 0.0:
            raise ValueError("side_direction_weight must be non-negative")
    else:
        side_hints = None
    local_result, local_diagnostics = _estimate_surface_normals_local(
        points_array,
        mesh,
        neighbor_count=neighbor_count,
        candidate_count=candidate_count,
        side_count=side_count,
        coherence_angle_degrees=coherence_angle_degrees,
        offset_epsilon=offset_epsilon,
        weight_distance_floor=weight_distance_floor,
        flash_points=flash_points,
        return_diagnostics=True,
    )
    if len(points_array) == 0:
        return (local_result, local_diagnostics) if return_diagnostics else local_result

    if normal_method not in ("two_face_angle_bisector", "surface_patch_continuity"):
        raise ValueError(f"不支持的 normal_method: {normal_method}")

    if normal_method == "surface_patch_continuity":
        tangents = _estimate_tangents_for_normal_projection(points_array)
        result, diagnostics = _refine_normals_with_surface_patch(
            points_array,
            tangents,
            mesh,
            local_result,
            local_diagnostics,
            flash_points=flash_points,
            branch_ids=branch_ids,
            surface_patch_neighbors=surface_patch_neighbors,
            surface_patch_distance_factor=surface_patch_distance_factor,
            surface_patch_coherence_degrees=surface_patch_coherence_degrees,
            max_bisector_deviation_degrees=max_bisector_deviation_degrees,
            continuity_repair_degrees=continuity_repair_degrees,
            continuity_repair_iterations=continuity_repair_iterations,
        )
        return (result, diagnostics) if return_diagnostics else result

    if flash_points is not None:
        # The labelled O->F rule supplies the absolute normal direction at
        # every point.  Do not send it through the historical dynamic program:
        # that program can exchange/flip signs to improve cross-point
        # smoothness and would therefore override the requested local rule.
        tangents = _estimate_tangents_for_normal_projection(points_array)
        result = np.asarray(local_result, dtype=float).copy()
        flash = np.asarray(flash_points, dtype=float)
        flash_tree = cKDTree(flash)
        for index, (point, tangent) in enumerate(zip(points_array, tangents, strict=True)):
            normal = result[index] - float(np.dot(result[index], tangent)) * tangent
            if float(np.linalg.norm(normal)) <= 1e-12:
                normal = result[index]
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            _, flash_index = flash_tree.query(point, k=1)
            of = flash[int(flash_index)] - point
            if float(np.dot(normal, of)) < 0.0:
                normal = -normal
            result[index] = normal
            local_diagnostics[index]["selected_normal"] = normal.tolist()
            local_diagnostics[index]["normal_method"] = "two_side_internal_bisector_signed_by_nearest_flash_OF"
            local_diagnostics[index]["continuity_state"] = {"applied": False, "reason": "direct_local_flash_orientation"}
        return (result, local_diagnostics) if return_diagnostics else result

    if continuity_jump_threshold_degrees <= 0.0 or continuity_jump_threshold_degrees >= 90.0:
        raise ValueError("continuity_jump_threshold_degrees must be in (0, 90)")

    # 没有显式分支标签时，按明显的空间间隙拆分；正常重采样点不会触发该阈值。
    if branch_ids is None:
        if len(points_array) <= 1:
            inferred_branches = np.zeros(len(points_array), dtype=np.int64)
        else:
            gaps = np.linalg.norm(np.diff(points_array, axis=0), axis=1)
            positive = gaps[gaps > 1e-12]
            reference = float(np.median(positive)) if len(positive) else 1.0
            inferred_branches = np.zeros(len(points_array), dtype=np.int64)
            inferred_branches[1:] = np.cumsum(gaps > max(reference * 8.0, 1e-8))
        branch_array = inferred_branches
    else:
        branch_array = np.asarray(branch_ids, dtype=np.int64).reshape(-1)
        if len(branch_array) != len(points_array):
            raise ValueError("branch_ids length must match points")

    # 从局部诊断中取两侧法向。单侧退化点也保留，但会被标记为低置信度。
    states: list[list[dict[str, Any]]] = []
    for index, item in enumerate(local_diagnostics):
        tangent = np.asarray(item.get("tangent", [0.0, 0.0, 0.0]), dtype=float)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
        details = item.get("side_details", [])
        side_vectors = [np.asarray(detail.get("normal", [0.0, 0.0, 0.0]), dtype=float) for detail in details]
        side_vectors = [vector / max(float(np.linalg.norm(vector)), 1e-12) for vector in side_vectors if np.linalg.norm(vector) > 1e-12]
        if len(side_vectors) >= 2:
            pair = side_vectors[:2]
            pair_details = details[:2]
        else:
            fallback = np.asarray(local_result[index], dtype=float)
            fallback /= max(float(np.linalg.norm(fallback)), 1e-12)
            pair = [fallback, fallback]
            pair_details = details[:1] or [{"normal": fallback.tolist(), "face_indices": []}]

        # 八个状态分别表示侧面标签交换以及两侧法向独立的符号选择。
        # 两侧面可能来自不同网格片区，不能只把两个法向整体同时翻转；
        # 独立符号状态正是修复“某一侧三角形绕向相反”的关键。
        point_states: list[dict[str, Any]] = []
        for swapped in (False, True):
            ordered = pair[::-1] if swapped else pair
            ordered_details = pair_details[::-1] if swapped else pair_details
            for sign1 in (1.0, -1.0):
                for sign2 in (1.0, -1.0):
                    n1 = ordered[0] * sign1
                    n2 = ordered[1] * sign2
                    # Each side normal must first be projected into the root
                    # line normal plane. Otherwise the later frame projection
                    # changes n by an uncontrolled amount, especially when a
                    # side face normal has a large tangent component.
                    n1 -= float(np.dot(n1, tangent)) * tangent
                    n2 -= float(np.dot(n2, tangent)) * tangent
                    n1 /= max(float(np.linalg.norm(n1)), 1e-12)
                    n2 /= max(float(np.linalg.norm(n2)), 1e-12)
                    bisector = n1 + n2
                    bisector_norm = float(np.linalg.norm(bisector))
                    if bisector_norm <= 1e-12:
                        bisector = n1
                    bisector /= max(float(np.linalg.norm(bisector)), 1e-12)
                    hint_score = 0.0
                    if side_hints is not None:
                        hint = side_hints[index]
                        hint_norm = float(np.linalg.norm(hint))
                        if hint_norm > 1e-12:
                            hint_score = float(np.dot(bisector, hint / hint_norm))
                    # Oppositely signed side normals produce an unstable
                    # near-zero bisector and are not a valid internal crease
                    # normal. Keep the state for diagnostics but strongly
                    # penalize it during physical state selection.
                    if bisector_norm < 0.2:
                        hint_score -= 10.0
                    centers = []
                    means = []
                    for detail in ordered_details:
                        if "patch_center" in detail:
                            centers.append(np.asarray(detail["patch_center"], dtype=float))
                        distances = np.asarray(detail.get("distances", []), dtype=float)
                        if len(distances):
                            means.append(float(np.mean(distances)))
                    point_states.append({
                        "n1": n1, "n2": n2, "normal": bisector,
                        "side_details": ordered_details,
                        "grouping_method": item.get("grouping_method", "unknown"),
                        "patch_centers": centers,
                        "mean_distance": float(np.mean(means)) if means else 0.0,
                        "swapped": bool(swapped), "sign1": int(sign1), "sign2": int(sign2),
                        "hint_score": hint_score,
                    })
        states.append(point_states)

    def transition(previous: dict[str, Any], current: dict[str, Any]) -> float:
        """计算相邻状态代价，优先保证两侧法向和面片位置连续。"""
        normal_cost = (1.0 - float(np.clip(np.dot(previous["n1"], current["n1"]), -1.0, 1.0)))
        normal_cost += (1.0 - float(np.clip(np.dot(previous["n2"], current["n2"]), -1.0, 1.0)))
        patch_cost = 0.0
        previous_centers = previous.get("patch_centers", [])
        current_centers = current.get("patch_centers", [])
        for old_center, new_center in zip(previous_centers, current_centers, strict=False):
            patch_cost += float(np.linalg.norm(old_center - new_center))
        distance_cost = abs(float(previous.get("mean_distance", 0.0)) - float(current.get("mean_distance", 0.0)))
        hint_cost = -side_direction_weight * float(current.get("hint_score", 0.0))
        return continuity_normal_weight * normal_cost + continuity_patch_weight * patch_cost + continuity_distance_weight * distance_cost + hint_cost

    selected = [0] * len(points_array)
    low_confidence = np.zeros(len(points_array), dtype=bool)
    branch_starts = [0]
    branch_starts.extend(np.flatnonzero(branch_array[1:] != branch_array[:-1]) + 1)
    branch_starts.append(len(points_array))
    threshold = float(np.deg2rad(continuity_jump_threshold_degrees))

    for start, end in zip(branch_starts[:-1], branch_starts[1:], strict=True):
        if end <= start:
            continue
        state_count = len(states[start])
        costs = np.full((end - start, state_count), np.inf, dtype=float)
        parents = np.full((end - start, state_count), -1, dtype=np.int64)
        # 优先使用局部算法给出的原始标签和符号作为分支起点。
        for state_index in range(state_count):
            costs[0, state_index] = -side_direction_weight * float(states[start][state_index].get("hint_score", 0.0))
        for offset in range(1, end - start):
            for current_index, current_state in enumerate(states[start + offset]):
                previous_costs = costs[offset - 1]
                transition_costs = np.array([
                    previous_costs[previous_index] + transition(states[start + offset - 1][previous_index], current_state)
                    for previous_index in range(state_count)
                ])
                parents[offset, current_index] = int(np.argmin(transition_costs))
                costs[offset, current_index] = float(np.min(transition_costs))
        state_index = int(np.argmin(costs[-1]))
        for offset in range(end - start - 1, -1, -1):
            selected[start + offset] = state_index
            if offset > 0:
                state_index = int(parents[offset, state_index])

        # 第三层修复：若动态规划后仍有超过阈值的相邻跳变，尝试用前后邻居
        # 同时约束当前点；没有可接受候选时保留原结果并写入低置信度标记。
        for index in range(start + 1, end - 1):
            previous_state = states[index - 1][selected[index - 1]]
            current_state = states[index][selected[index]]
            next_state = states[index + 1][selected[index + 1]]
            angle_prev = float(np.arccos(np.clip(np.dot(previous_state["normal"], current_state["normal"]), -1.0, 1.0)))
            angle_next = float(np.arccos(np.clip(np.dot(current_state["normal"], next_state["normal"]), -1.0, 1.0)))
            if max(angle_prev, angle_next) <= threshold:
                continue
            candidate_costs = [transition(previous_state, candidate) + transition(candidate, next_state) for candidate in states[index]]
            replacement = int(np.argmin(candidate_costs))
            selected[index] = replacement
            repaired = states[index][replacement]
            repaired_prev = float(np.arccos(np.clip(np.dot(previous_state["normal"], repaired["normal"]), -1.0, 1.0)))
            repaired_next = float(np.arccos(np.clip(np.dot(repaired["normal"], next_state["normal"]), -1.0, 1.0)))
            low_confidence[index] = max(repaired_prev, repaired_next) > threshold

        # 若所有离散候选都来自错误面片簇，单纯交换/翻转仍可能留下孤立跳变。
        # 这里仅对超过阈值的异常点做局部几何平滑，并标记低置信度；正常点
        # 不会被改动，后续可根据诊断标记回到网格局部重新提取面片。
        # Do not replace a selected bisector with an unconstrained bridge
        # vector: that would make the reported n disagree with its signed n1/n2.
        # Large residual jumps remain explicitly marked for later review.
        for index in range(start + 1, end - 1):
            previous_normal = states[index - 1][selected[index - 1]]["normal"]
            current_normal = states[index][selected[index]]["normal"]
            next_normal = states[index + 1][selected[index + 1]]["normal"]
            before = float(np.arccos(np.clip(np.dot(previous_normal, current_normal), -1.0, 1.0)))
            after = float(np.arccos(np.clip(np.dot(current_normal, next_normal), -1.0, 1.0)))
            low_confidence[index] |= max(before, after) > threshold

    result = np.empty((len(points_array), 3), dtype=float)
    diagnostics: list[dict[str, Any]] = []
    for index, item in enumerate(local_diagnostics):
        state = states[index][selected[index]]
        # 砂轮法向必须位于 root_line 的法平面内；局部平滑后重新投影，
        # 防止桥接向量产生切向分量而使 build_frames 失去正交标架。
        tangent = np.asarray(item.get("tangent", [0.0, 0.0, 0.0]), dtype=float)
        normal = np.asarray(state["normal"], dtype=float)
        normal -= float(np.dot(normal, tangent)) * tangent
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= 1e-8:
            normal = np.asarray(local_result[index], dtype=float)
            normal -= float(np.dot(normal, tangent)) * tangent
            normal_length = float(np.linalg.norm(normal))
        if normal_length <= 1e-8:
            basis = np.array([1.0, 0.0, 0.0]) if abs(float(tangent[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
            normal = np.cross(tangent, basis)
            normal_length = float(np.linalg.norm(normal))
        normal /= max(normal_length, 1e-12)
        result[index] = normal
        diagnostic = dict(item)
        diagnostic["side_details"] = state["side_details"]
        diagnostic["selected_normal"] = result[index].tolist()
        diagnostic["selected_n1"] = np.asarray(state["n1"], dtype=float).tolist()
        diagnostic["selected_n2"] = np.asarray(state["n2"], dtype=float).tolist()
        diagnostic["normal_method"] = "coherent_two_side_internal_bisector_with_branch_continuity"
        diagnostic["continuity_state"] = {
            "swapped": state["swapped"], "sign1": state["sign1"], "sign2": state["sign2"],
            "candidate_state_count": len(states[index]),
            "low_confidence": bool(low_confidence[index]),
        }
        diagnostics.append(diagnostic)
    if return_diagnostics:
        return result, diagnostics
    return result
