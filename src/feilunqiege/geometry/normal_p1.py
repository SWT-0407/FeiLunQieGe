"""P1 法向估计实验后端（只被显式选择时启用）。

本模块把 P1 建议拆成可组合的部件：

- 全局一致绕向修复（P1-6）；
- 法向聚类 + 主/次簇支持（P1-1/P1-7）；
- shape-aware patch 与 l0 式裁剪（借鉴 Zhong 2021）；
- guidance patch + bilateral normal filter（借鉴 Zhang 2015）；
- 带条件数/凸包覆盖检查的局部 jet 拟合（借鉴 Cazals & Pouget 2005）；
- 鲁棒 sharp-feature 聚类（借鉴 Li et al. 2010）；
- 统一的面积/距离/法向相似度权重和 source-specific confidence。

它不是对论文的逐字复现，而是可供本项目对比的实验实现。生产默认仍是
``surface_patch_continuity``；只有显式选择这里的 method 才会走新逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from feilunqiege.geometry.frame import _point_triangle_distance

P1_METHODS = {
    "p1",
    "zhong2021",
    "zhang2015",
    "cazals2005",
    "li2010",
    "zhong+cazals",
    "zhang+cazals",
    "li+zhong",
    "all",
}


@dataclass
class MeshNormalData:
    vertices: np.ndarray
    faces: np.ndarray
    triangles: np.ndarray
    valid_mask: np.ndarray
    valid_indices: np.ndarray
    valid_triangles: np.ndarray
    valid_centroids: np.ndarray
    valid_areas: np.ndarray
    raw_normals: np.ndarray
    normals: np.ndarray
    tree: cKDTree
    mesh_is_closed: bool
    signed_volume: float
    orientation_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalCluster:
    indices: np.ndarray
    mean: np.ndarray
    support: int
    area: float
    median_distance: float
    spread_deg: float


@dataclass
class PatchInfo:
    indices: np.ndarray
    distances: np.ndarray
    areas: np.ndarray
    primary: NormalCluster
    clusters: list[NormalCluster]
    secondary: NormalCluster | None
    angle12: float | None
    support_ratio: float | None
    min_support: int | None
    patch_scale: float
    patch_radius_current: float
    low_support: bool
    valid_patch: bool = True
    fallback_reason: str | None = None


@dataclass
class FitInfo:
    normal: np.ndarray | None
    residual: float
    threshold: float
    condition: float
    smin: float
    rank: int
    hull_area_ratio: float
    root_inside: bool
    valid: bool
    unique_vertices: int


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector, dtype=float)


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.degrees(
            np.arccos(np.clip(float(np.dot(_unit(first), _unit(second))), -1.0, 1.0))
        )
    )


def _mesh_is_closed(faces: np.ndarray) -> bool:
    try:
        from feilunqiege.collision.disk_exact import _is_closed_triangle_mesh

        return bool(_is_closed_triangle_mesh(faces))
    except Exception:
        return False


def _repair_orientation(
    faces: np.ndarray,
    triangles: np.ndarray,
    valid_mask: np.ndarray,
    raw_normals: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """用共享边 BFS 修复局部 winding 翻转，再用 signed volume 定全局朝向。

    约定：一致的相邻三角形对共享边应取相反方向。若两个相邻面取相同方向，
    则翻转其中一个。非流形边（>2 个面）只记录，不用于传播。
    """

    valid_indices = np.flatnonzero(valid_mask)
    edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index in valid_indices:
        face = faces[face_index]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = int(face[a]), int(face[b])
            key = (min(i, j), max(i, j))
            direction = 1 if i < j else -1
            edge_map.setdefault(key, []).append((int(face_index), direction))

    adjacency: dict[int, list[tuple[int, int, int]]] = {}
    nonmanifold_edges = 0
    for uses in edge_map.values():
        if len(uses) != 2:
            if len(uses) > 2:
                nonmanifold_edges += 1
            continue
        (first, dir_first), (second, dir_second) = uses
        adjacency.setdefault(first, []).append((second, dir_first, dir_second))
        adjacency.setdefault(second, []).append((first, dir_second, dir_first))

    visited: dict[int, bool] = {}
    flip: dict[int, bool] = {}
    components = 0
    conflicts = 0
    for seed in valid_indices:
        seed = int(seed)
        if seed in visited:
            continue
        components += 1
        visited[seed] = True
        flip[seed] = False
        queue = [seed]
        while queue:
            current = queue.pop()
            for neighbor, dir_current, dir_neighbor in adjacency.get(current, []):
                required = flip[current] if dir_current != dir_neighbor else not flip[current]
                if neighbor not in visited:
                    visited[neighbor] = True
                    flip[neighbor] = required
                    queue.append(neighbor)
                elif flip[neighbor] != required:
                    conflicts += 1

    oriented = raw_normals.copy()
    flipped_faces = [face for face, value in flip.items() if value]
    if flipped_faces:
        oriented[flipped_faces] *= -1.0

    # 用翻转后的三角形计算 signed volume，统一整体朝外。
    oriented_triangles = triangles.copy()
    for face in flipped_faces:
        oriented_triangles[face] = oriented_triangles[face][[0, 2, 1]]
    signed_volume = float(
        np.einsum(
            "ij,ij->i",
            oriented_triangles[:, 0],
            np.cross(oriented_triangles[:, 1], oriented_triangles[:, 2]),
        ).sum()
        / 6.0
    )
    global_flip = False
    if signed_volume < 0.0:
        oriented *= -1.0
        signed_volume = -signed_volume
        global_flip = True

    return oriented, {
        "orientation_components": components,
        "orientation_flipped_faces": len(flipped_faces),
        "orientation_conflicts": conflicts,
        "orientation_nonmanifold_edges": nonmanifold_edges,
        "orientation_global_flip": global_flip,
        "orientation_signed_volume": signed_volume,
    }


def prepare_mesh(mesh: Any, *, repair_orientation: bool = True) -> MeshNormalData:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    raw = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(raw, axis=1)
    valid_mask = lengths > 1e-12
    valid_indices = np.flatnonzero(valid_mask)
    valid_triangles = triangles[valid_mask]
    valid_centroids = valid_triangles.mean(axis=1)
    valid_areas = 0.5 * lengths[valid_mask]
    raw_normals = np.zeros((len(faces), 3), dtype=float)
    raw_normals[valid_mask] = raw[valid_mask] / lengths[valid_mask, None]
    if repair_orientation:
        normals, orientation_info = _repair_orientation(faces, triangles, valid_mask, raw_normals)
    else:
        normals = raw_normals.copy()
        orientation_info = {
            "orientation_components": 0,
            "orientation_flipped_faces": 0,
            "orientation_conflicts": 0,
            "orientation_nonmanifold_edges": 0,
            "orientation_global_flip": False,
            "orientation_signed_volume": float(
                np.einsum(
                    "ij,ij->i",
                    triangles[:, 0],
                    np.cross(triangles[:, 1], triangles[:, 2]),
                ).sum()
                / 6.0
            ),
        }
    return MeshNormalData(
        vertices=vertices,
        faces=faces,
        triangles=triangles,
        valid_mask=valid_mask,
        valid_indices=valid_indices,
        valid_triangles=valid_triangles,
        valid_centroids=valid_centroids,
        valid_areas=valid_areas,
        raw_normals=raw_normals,
        normals=normals,
        tree=cKDTree(valid_centroids),
        mesh_is_closed=_mesh_is_closed(faces),
        signed_volume=float(orientation_info.get("orientation_signed_volume", 0.0)),
        orientation_info=orientation_info,
    )


def query_candidates(
    point: np.ndarray,
    mesh: MeshNormalData,
    *,
    k: int,
    radius: float,
    min_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回候选面的原始面索引和精确点到三角形距离。"""

    query_count = min(max(int(k), int(min_faces)), len(mesh.valid_centroids))
    _, neighbor_indices = mesh.tree.query(point, k=query_count)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64).reshape(-1)
    distances = np.asarray(
        [_point_triangle_distance(point, mesh.valid_triangles[index]) for index in neighbor_indices],
        dtype=float,
    )
    order = np.argsort(distances)
    neighbor_indices = neighbor_indices[order]
    distances = distances[order]
    keep = distances <= float(radius)
    if int(np.count_nonzero(keep)) < int(min_faces):
        keep = np.zeros(len(distances), dtype=bool)
        keep[: min(int(min_faces), len(distances))] = True
    return mesh.valid_indices[neighbor_indices[keep]], distances[keep]


def cluster_normals(
    indices: np.ndarray,
    distances: np.ndarray,
    mesh: MeshNormalData,
    *,
    angle_deg: float,
    min_support: int,
) -> list[NormalCluster]:
    """按有向法向夹角做贪心聚类；输入法向已统一绕向。"""

    if len(indices) == 0:
        return []
    normals = mesh.normals[indices]
    areas = mesh.valid_areas[np.searchsorted(mesh.valid_indices, indices)]
    cosine = float(np.cos(np.deg2rad(angle_deg)))
    order = np.argsort(distances)
    clusters: list[dict[str, Any]] = []
    for position in order:
        normal = normals[position]
        best_index = None
        best_dot = -2.0
        for cluster_index, cluster in enumerate(clusters):
            dot = float(np.dot(normal, cluster["mean"]))
            if dot >= cosine and dot > best_dot:
                best_index = cluster_index
                best_dot = dot
        if best_index is None:
            clusters.append(
                {
                    "indices": [int(position)],
                    "normal_sum": normal.copy(),
                    "mean": normal.copy(),
                }
            )
        else:
            cluster = clusters[best_index]
            cluster["indices"].append(int(position))
            cluster["normal_sum"] = cluster["normal_sum"] + normal
            cluster["mean"] = _unit(cluster["normal_sum"])
    # 合并仍然相近的簇
    merged = True
    while merged:
        merged = False
        for first in range(len(clusters)):
            for second in range(first + 1, len(clusters)):
                if float(np.dot(clusters[first]["mean"], clusters[second]["mean"])) >= cosine:
                    clusters[first]["indices"].extend(clusters[second]["indices"])
                    clusters[first]["normal_sum"] += clusters[second]["normal_sum"]
                    clusters[first]["mean"] = _unit(clusters[first]["normal_sum"])
                    del clusters[second]
                    merged = True
                    break
            if merged:
                break

    result: list[NormalCluster] = []
    for cluster in clusters:
        positions = np.asarray(cluster["indices"], dtype=np.int64)
        mean = _unit(cluster["normal_sum"])
        cluster_normals = normals[positions]
        spread = float(
            np.degrees(np.arccos(np.clip(np.min(cluster_normals @ mean), -1.0, 1.0)))
        )
        result.append(
            NormalCluster(
                indices=indices[positions],
                mean=mean,
                support=int(len(positions)),
                area=float(np.sum(areas[positions])),
                median_distance=float(np.median(distances[positions])),
                spread_deg=spread,
            )
        )
    result.sort(key=lambda cluster: cluster.support, reverse=True)
    return [cluster for cluster in result if cluster.support >= int(min_support)] or result[:1]


def _primary_cluster(clusters: list[NormalCluster], nearest_index: int) -> NormalCluster:
    for cluster in clusters:
        if nearest_index in set(int(index) for index in cluster.indices):
            return cluster
    return clusters[0]


def _patch_stats(
    point: np.ndarray,
    indices: np.ndarray,
    distances: np.ndarray,
    mesh: MeshNormalData,
    clusters: list[NormalCluster],
) -> PatchInfo:
    normals = mesh.normals[indices]
    areas = mesh.valid_areas[np.searchsorted(mesh.valid_indices, indices)]
    weights = areas / np.maximum(distances, 1e-6)
    weights = weights / max(float(weights.sum()), 1e-18)
    mean = _unit((normals * weights[:, None]).sum(axis=0))
    primary = _primary_cluster(clusters, int(indices[0]))
    secondary = None
    angle12 = None
    support_ratio = None
    min_support = None
    if len(clusters) >= 2 and clusters[1].support >= 2:
        secondary = clusters[1]
        angle12 = _angle_deg(primary.mean, secondary.mean)
        support_ratio = float(min(primary.support, secondary.support) / max(primary.support, secondary.support))
        min_support = int(min(primary.support, secondary.support))
    unique_vertices = np.unique(mesh.faces[indices].reshape(-1))
    patch_vertices = mesh.vertices[unique_vertices]
    patch_distances = np.linalg.norm(patch_vertices - point[None, :], axis=1)
    patch_scale = float(np.quantile(patch_distances, 0.90)) if len(patch_distances) else 0.0
    patch_radius_current = float(np.max(distances)) if len(distances) else 0.0
    return PatchInfo(
        indices=indices,
        distances=distances,
        areas=areas,
        primary=primary,
        clusters=clusters,
        secondary=secondary,
        angle12=angle12,
        support_ratio=support_ratio,
        min_support=min_support,
        patch_scale=patch_scale,
        patch_radius_current=patch_radius_current,
        low_support=bool(patch_scale <= 1e-12 or len(indices) < 4),
    )


def shape_aware_patch(
    point: np.ndarray,
    candidates: np.ndarray,
    distances: np.ndarray,
    mesh: MeshNormalData,
    *,
    cluster_angle_deg: float,
    min_support: int,
    min_patch_support: int,
    trim_angle_deg: float,
) -> PatchInfo:
    """Zhong 2021 风格：多尺度 patch + 平坦度/方向一致性 + l0 式裁剪。"""

    order = np.argsort(distances)
    best: PatchInfo | None = None
    best_score = -1.0
    for scale in (12, 24, 48, 96):
        count = min(int(scale), len(candidates))
        if count < 4:
            continue
        selected = order[:count]
        selected_indices = candidates[selected]
        selected_distances = distances[selected]
        clusters = cluster_normals(
            selected_indices,
            selected_distances,
            mesh,
            angle_deg=cluster_angle_deg,
            min_support=2,
        )
        if not clusters:
            continue
        primary = _primary_cluster(clusters, int(selected_indices[0]))
        # l0 式裁剪：去掉与主簇均值夹角过大的面，保留主表面。
        keep = np.abs(
            np.degrees(
                np.arccos(
                    np.clip(mesh.normals[selected_indices] @ primary.mean, -1.0, 1.0)
                )
            )
        ) <= trim_angle_deg
        if int(np.count_nonzero(keep)) >= max(4, min_support):
            selected_indices = selected_indices[keep]
            selected_distances = selected_distances[keep]
            clusters = cluster_normals(
                selected_indices,
                selected_distances,
                mesh,
                angle_deg=cluster_angle_deg,
                min_support=2,
            )
            if clusters:
                primary = _primary_cluster(clusters, int(selected_indices[0]))
        patch = _patch_stats(point, selected_indices, selected_distances, mesh, clusters)
        if patch.primary.support < int(min_patch_support):
            continue
        # 支持度、平坦度、与主方向的相似度；过大/跨面片 patch 减分。
        spread = patch.primary.spread_deg
        flatness = float(np.exp(-((spread / 20.0) ** 2)))
        secondary_penalty = 1.0
        if patch.secondary is not None and patch.angle12 is not None and patch.angle12 > 10.0:
            secondary_penalty = 0.5
        score = (
            float(patch.primary.support)
            * flatness
            * secondary_penalty
            * (1.0 - 0.5 * min(spread / 90.0, 1.0))
        )
        if score > best_score:
            best_score = score
            best = patch
    if best is None:
        # 不再把完整混合邻域直接当作一个 patch。优先用支持最大的单一
        # 法向簇重建 patch；若该簇仍不足 min_patch_support，则标记为无效
        # patch，后续禁止 jet 拟合，避免把两个不同表面混在一起。
        clusters = cluster_normals(candidates, distances, mesh, angle_deg=cluster_angle_deg, min_support=1)
        primary = max(clusters, key=lambda cluster: cluster.support) if clusters else None
        if primary is not None and primary.support >= int(min_patch_support):
            mask = np.isin(candidates, primary.indices)
            selected_indices = candidates[mask]
            selected_distances = distances[mask]
            clusters = cluster_normals(
                selected_indices,
                selected_distances,
                mesh,
                angle_deg=cluster_angle_deg,
                min_support=1,
            )
            best = _patch_stats(point, selected_indices, selected_distances, mesh, clusters)
        else:
            best = _patch_stats(point, candidates, distances, mesh, clusters)
            best.valid_patch = False
            best.fallback_reason = "no_cluster_meets_min_patch_support"
    return best


def guided_bilateral_normal(
    point: np.ndarray,
    candidates: np.ndarray,
    distances: np.ndarray,
    mesh: MeshNormalData,
    *,
    cluster_angle_deg: float,
    min_support: int,
    sigma_scale: float,
    sigma_angle_deg: float,
) -> tuple[np.ndarray, PatchInfo]:
    """Zhang 2015 风格：先选最一致的 guidance patch，再做双边法向滤波。"""

    clusters = cluster_normals(candidates, distances, mesh, angle_deg=cluster_angle_deg, min_support=min_support)
    primary = _primary_cluster(clusters, int(candidates[0]))
    guide = primary.mean
    normals = mesh.normals[candidates]
    areas = mesh.valid_areas[np.searchsorted(mesh.valid_indices, candidates)]
    for _ in range(2):
        angular = np.degrees(np.arccos(np.clip(normals @ guide, -1.0, 1.0)))
        spatial = np.exp(-(distances / max(sigma_scale, 1e-9)) ** 2)
        normal_weight = np.exp(-(angular / max(sigma_angle_deg, 1e-9)) ** 2)
        weights = areas * spatial * normal_weight / np.maximum(distances, 1e-6)
        weights /= max(float(weights.sum()), 1e-18)
        guide = _unit((normals * weights[:, None]).sum(axis=0))
    patch = _patch_stats(point, candidates, distances, mesh, clusters)
    return guide, patch


def jet_fit(
    point: np.ndarray,
    patch_indices: np.ndarray,
    distances: np.ndarray,
    mesh: MeshNormalData,
    *,
    fit_residual_ratio: float,
    max_condition: float,
    min_hull_area_ratio: float,
    reference_normal: np.ndarray,
) -> FitInfo:
    """Cazals & Pouget 风格：PCA 切平面 + 加权二次 jet 拟合。"""

    unique_vertices = np.unique(mesh.faces[patch_indices].reshape(-1))
    if len(unique_vertices) < 8:
        return FitInfo(None, float("nan"), 0.0, float("inf"), 0.0, 0, 0.0, False, False, len(unique_vertices))
    vertices = mesh.vertices[unique_vertices]
    centered = vertices - point[None, :]
    # 顶点权重取 incident patch faces 的 area/distance 和。
    vertex_weights = np.zeros(len(unique_vertices), dtype=float)
    vertex_lookup = {int(vertex): index for index, vertex in enumerate(unique_vertices)}
    for face_index, distance in zip(patch_indices, distances, strict=True):
        weight = float(mesh.valid_areas[np.searchsorted(mesh.valid_indices, face_index)]) / max(float(distance), 1e-6)
        for vertex in mesh.faces[face_index]:
            vertex_weights[vertex_lookup[int(vertex)]] += weight
    vertex_weights = np.maximum(vertex_weights, 1e-12)
    vertex_weights /= float(vertex_weights.sum())
    covariance = (centered * vertex_weights[:, None]).T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal_guess = eigenvectors[:, 0]
    normal_guess = _unit(normal_guess)
    if float(np.dot(normal_guess, reference_normal)) < 0.0:
        normal_guess = -normal_guess
    u_axis = eigenvectors[:, -1]
    u_axis = _unit(u_axis - float(np.dot(u_axis, normal_guess)) * normal_guess)
    if float(np.linalg.norm(u_axis)) <= 1e-12:
        reference = np.array([1.0, 0.0, 0.0]) if abs(float(normal_guess[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
        u_axis = _unit(np.cross(normal_guess, reference))
    v_axis = np.cross(normal_guess, u_axis)
    u = centered @ u_axis
    v = centered @ v_axis
    w = centered @ normal_guess
    design = np.column_stack([u * u, u * v, v * v, u, v, np.ones_like(u)])
    sqrt_weights = np.sqrt(vertex_weights)
    weighted_design = design * sqrt_weights[:, None]
    weighted_w = w * sqrt_weights
    coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_w, rcond=None)
    if int(rank) < 5:
        return FitInfo(None, float("nan"), 0.0, float("inf"), 0.0, int(rank), 0.0, False, False, len(unique_vertices))
    residual_values = (weighted_w - weighted_design @ coefficients) ** 2
    residual = float(np.sqrt(np.sum(residual_values)))
    candidate = -float(coefficients[3]) * u_axis - float(coefficients[4]) * v_axis + normal_guess
    candidate = _unit(candidate)
    if float(np.dot(candidate, normal_guess)) < 0.0:
        candidate = -candidate
    singular_values = np.linalg.svd(weighted_design, compute_uv=False)
    condition = float(singular_values[0] / max(singular_values[-1], 1e-30))
    smin = float(singular_values[-1])
    hull_area_ratio = 0.0
    root_inside = False
    uv = np.column_stack([u, v])
    if len(uv) >= 3:
        try:
            hull = ConvexHull(uv)
            hull_area_ratio = float(hull.volume / max(np.quantile(np.linalg.norm(centered, axis=1), 0.9) ** 2, 1e-18))
            root_inside = bool(
                np.all(hull.equations[:, :2] @ np.array([0.0, 0.0]) + hull.equations[:, 2] <= 1e-12)
            )
        except Exception:
            hull_area_ratio = 0.0
            root_inside = False
    patch_scale = float(np.quantile(np.linalg.norm(centered, axis=1), 0.90))
    threshold = max(float(fit_residual_ratio) * patch_scale, 1e-6)
    valid = bool(
        np.isfinite(residual)
        and residual <= threshold
        and condition <= max_condition
        and root_inside
        and hull_area_ratio >= min_hull_area_ratio
    )
    return FitInfo(candidate, residual, threshold, condition, smin, int(rank), hull_area_ratio, root_inside, valid, len(unique_vertices))


def robust_cluster_normal(cluster: NormalCluster, mesh: MeshNormalData) -> tuple[np.ndarray, float]:
    """Tukey 鲁棒加权；返回 (normal, 最大残差角)。"""

    normals = mesh.normals[cluster.indices]
    median = cluster.mean
    angles = np.degrees(np.arccos(np.clip(normals @ median, -1.0, 1.0)))
    cutoff = max(float(np.median(angles) * 2.5), 5.0)
    weights = np.where(angles < cutoff, (1.0 - (angles / cutoff) ** 2) ** 2, 0.0)
    if float(weights.sum()) <= 1e-18:
        weights = np.ones(len(normals), dtype=float)
    weights /= float(weights.sum())
    return _unit((normals * weights[:, None]).sum(axis=0)), float(np.max(angles))


def _confidence_from_patch(
    patch: PatchInfo,
    *,
    feature_patch: PatchInfo | None,
    fit: FitInfo | None,
    source: str,
    min_crease_angle: float,
    min_support_ratio: float,
    min_support_faces: int,
    trim_angle: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not patch.valid_patch:
        reasons.append("invalid_patch")
    if patch.primary.support < min_support_faces:
        reasons.append("primary_support_too_small")
    if patch.patch_scale <= 1e-9:
        reasons.append("degenerate_patch_scale")
    feature = feature_patch if feature_patch is not None else patch
    if feature.secondary is not None:
        if feature.secondary.support >= min_support_faces:
            if feature.angle12 is None or feature.angle12 < min_crease_angle:
                reasons.append("no_two_sided_support")
            if feature.support_ratio is None or feature.support_ratio < min_support_ratio:
                reasons.append("support_imbalance")
        elif feature.angle12 is not None and feature.angle12 >= min_crease_angle:
            # 小簇但方向明显不同：可能是侧壁/飞边，不能当作高置信光滑点。
            reasons.append("small_feature_cluster")
    if (
        feature.secondary is not None
        and feature.angle12 is not None
        and feature.angle12 >= min_crease_angle
        and feature.secondary.support >= min_support_faces
        and source not in {"two_side_bisector", "robust_two_side_bisector"}
    ):
        # 原始邻域里有足够支撑的第二表面，但最终输出没有用它（例如 shape/jet
        # 只拟合了主面）。这必须降置信度，不能把“忽略了一整块表面”当成高置信。
        reasons.append("secondary_surface_ignored")
    if source in {"shape_patch_mean", "primary_cluster_mean"} and patch.primary.spread_deg > trim_angle:
        reasons.append("high_primary_spread")
    if source == "jet_fit" and (fit is None or not fit.valid):
        reasons.append("invalid_jet_fit")
    return len(reasons) == 0, reasons


def _method_uses_jet(method: str) -> bool:
    return method in {
        "p1",
        "cazals2005",
        "zhong+cazals",
        "zhang+cazals",
        "li+zhong",
        "all",
    }


def _method_uses_shape(method: str) -> bool:
    return method in {"p1", "zhong2021", "zhong+cazals", "li+zhong", "all"}


def _method_uses_guided(method: str) -> bool:
    return method in {"zhang2015", "zhang+cazals", "all"}


def _method_uses_robust(method: str) -> bool:
    return method in {"li2010", "li+zhong", "all"}


def _method_uses_crease(method: str) -> bool:
    # Cazals jet fitting is a smooth-surface method; it should not silently
    # synthesize a crease bisector. The other methods explicitly handle features.
    return method != "cazals2005"


def estimate_normals_experimental(
    points: Any,
    mesh: Any,
    *,
    method: str,
    config: dict[str, Any] | None = None,
    return_diagnostics: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """P1/文献实验法向估计。

    method:
      - p1: orientation + shape-aware patch + jet + crease gate
      - zhong2021: shape-aware patch（l0 裁剪）
      - zhang2015: guidance patch + bilateral
      - cazals2005: primary patch + jet
      - li2010: robust sharp-feature clusters
      - zhong+cazals / zhang+cazals / li+zhong / all
    """

    config = dict(config or {})
    points = np.asarray(points, dtype=float)
    repair = bool(config.get("orientation_repair", True))
    mesh_data = prepare_mesh(mesh, repair_orientation=repair)
    k = int(config.get("candidate_faces", 96))
    radius = float(config.get("candidate_radius", 0.02))
    min_faces = int(config.get("min_patch_faces", 10))
    cluster_angle = float(config.get("normal_cluster_angle_degrees", 15.0))
    min_support = int(config.get("min_cluster_support", 4))
    min_crease_angle = float(config.get("min_crease_angle_degrees", 10.0))
    min_support_ratio = float(config.get("min_side_support_ratio", 0.35))
    min_support_faces = int(config.get("min_side_support_faces", 6))
    trim_angle = float(config.get("l0_trim_angle_degrees", 25.0))
    fit_residual_ratio = float(config.get("fit_residual_ratio", 0.02))
    max_condition = float(config.get("max_fit_condition", 1e6))
    min_hull = float(config.get("min_hull_area_ratio", 0.1))
    sigma_scale = float(config.get("guided_sigma_scale", 0.01))
    sigma_angle = float(config.get("guided_sigma_angle_degrees", 15.0))

    normals: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for point in points:
        candidates, distances = query_candidates(
            point, mesh_data, k=k, radius=radius, min_faces=min_faces
        )
        clusters = cluster_normals(
            candidates,
            distances,
            mesh_data,
            angle_deg=cluster_angle,
            min_support=min_support,
        )
        original_patch = _patch_stats(point, candidates, distances, mesh_data, clusters)
        patch = original_patch
        if _method_uses_shape(method):
            patch = shape_aware_patch(
                point,
                candidates,
                distances,
                mesh_data,
                cluster_angle_deg=cluster_angle,
                min_support=min_support,
                min_patch_support=min_faces,
                trim_angle_deg=trim_angle,
            )
        guided_normal = None
        if _method_uses_guided(method):
            guided_normal, guided_patch = guided_bilateral_normal(
                point,
                candidates,
                distances,
                mesh_data,
                cluster_angle_deg=cluster_angle,
                min_support=min_support,
                sigma_scale=sigma_scale,
                sigma_angle_deg=sigma_angle,
            )
            if patch.secondary is None:
                patch = guided_patch
        crease_valid = False
        if _method_uses_crease(method) and patch.secondary is not None:
            crease_valid = bool(
                patch.angle12 is not None
                and patch.angle12 >= min_crease_angle
                and patch.support_ratio is not None
                and patch.support_ratio >= min_support_ratio
                and patch.min_support is not None
                and patch.min_support >= min_support_faces
            )
        robust_primary = None
        robust_secondary = None
        if _method_uses_robust(method):
            robust_primary, _ = robust_cluster_normal(patch.primary, mesh_data)
            if patch.secondary is not None:
                robust_secondary, _ = robust_cluster_normal(patch.secondary, mesh_data)
        fit = None
        if _method_uses_jet(method) and patch.valid_patch:
            fit = jet_fit(
                point,
                patch.indices,
                patch.distances,
                mesh_data,
                fit_residual_ratio=fit_residual_ratio,
                max_condition=max_condition,
                min_hull_area_ratio=min_hull,
                reference_normal=patch.primary.mean,
            )
        if crease_valid and patch.secondary is not None:
            if robust_primary is not None and robust_secondary is not None:
                normal = _unit(robust_primary + robust_secondary)
                source = "robust_two_side_bisector"
            else:
                normal = _unit(patch.primary.mean + patch.secondary.mean)
                source = "two_side_bisector"
        elif fit is not None and fit.valid and fit.normal is not None:
            normal = fit.normal
            source = "jet_fit"
        elif guided_normal is not None:
            normal = guided_normal
            source = "guided_bilateral"
        elif robust_primary is not None:
            normal = robust_primary
            source = "robust_feature"
        else:
            normal = patch.primary.mean
            source = "shape_patch_mean" if _method_uses_shape(method) else "primary_cluster_mean"
        confidence_ok, confidence_reasons = _confidence_from_patch(
            patch,
            feature_patch=original_patch,
            fit=fit,
            source=source,
            min_crease_angle=min_crease_angle,
            min_support_ratio=min_support_ratio,
            min_support_faces=min_support_faces,
            trim_angle=trim_angle,
        )
        if float(np.linalg.norm(normal)) <= 1e-12:
            normal = np.array([0.0, 0.0, 1.0])
            confidence_reasons.append("degenerate_normal")
        normal = _unit(normal)
        normals.append(normal)
        diagnostics.append(
            {
                "normal": normal.tolist(),
                "normal_source": source,
                "normal_confidence": "high" if confidence_ok else "low",
                "low_confidence_reasons": confidence_reasons,
                "angle_n1_n2_deg": patch.angle12,
                "support1": int(patch.primary.support),
                "support2": int(patch.secondary.support) if patch.secondary is not None else None,
                "support_ratio": patch.support_ratio,
                "min_support": patch.min_support,
                "primary_normal": patch.primary.mean.tolist(),
                "secondary_normal": patch.secondary.mean.tolist() if patch.secondary is not None else None,
                "patch_face_indices": patch.indices.astype(int).tolist(),
                "primary_face_indices": patch.primary.indices.astype(int).tolist(),
                "secondary_face_indices": (
                    patch.secondary.indices.astype(int).tolist()
                    if patch.secondary is not None
                    else []
                ),
                "patch_valid": bool(patch.valid_patch),
                "patch_fallback_reason": patch.fallback_reason,
                "patch_average_normal": patch.primary.mean.tolist(),
                "patch_scale": patch.patch_scale,
                "patch_radius_current": patch.patch_radius_current,
                "fit": None
                if fit is None
                else {
                    "normal": fit.normal.tolist() if fit.normal is not None else None,
                    "residual": fit.residual,
                    "threshold": fit.threshold,
                    "condition": fit.condition,
                    "smin": fit.smin,
                    "rank": fit.rank,
                    "hull_area_ratio": fit.hull_area_ratio,
                    "root_inside": fit.root_inside,
                    "valid": fit.valid,
                    "unique_vertices": fit.unique_vertices,
                },
                "orientation": mesh_data.orientation_info,
            }
        )
    result = np.asarray(normals, dtype=float)
    return (result, diagnostics) if return_diagnostics else result
