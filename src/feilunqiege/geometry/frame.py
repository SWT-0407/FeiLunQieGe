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


def build_frames(points: Any, tangents: Any, surface_normals: Any | None = None) -> list[LocalFrame]:
    """建立切平面标架。

    ``radial_basis_1`` 和 ``radial_basis_2`` 这两个历史字段名继续保留，
    但它们现在表示切平面内的两个基向量，而不是法平面基向量。候选中心
    因而在切平面内形成圆轨迹，砂轮轴向取工件表面法向，砂轮圆盘平面与
    切平面平行、与法平面垂直。

    未传入表面法向时使用一个稳定的合成法向，仅用于旧单元测试；正式流程
    必须通过 ``estimate_surface_normals`` 从工件三角网格提供法向。
    """

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
        else:
            normal = normal_array[index]
        normal = normal - np.dot(normal, t) * t
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= 1e-12:
            raise ValueError(f"第 {index} 个表面法向与曲线切线退化或平行")
        normal = normal / normal_norm
        # q 是切平面中垂直于 root_line 切线的方向；(t, q) 张成切平面。
        q = np.cross(normal, t)
        q /= np.linalg.norm(q)
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
        # 输入网格的三角形绕向可能在局部不一致。用整体工件中心只做
        # 最终符号校正，不参与两侧面的分组，避免把空间邻近面混在一起。
        patch_center = valid_centroids[indices[candidate_positions[selected]]].mean(axis=0)
        if float(np.dot(normal, patch_center - mesh_center)) < 0.0:
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
        return_diagnostics=True,
    )
    if len(points_array) == 0:
        return (local_result, local_diagnostics) if return_diagnostics else local_result

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
