"""P0 只读诊断：逐点输出法向、面片、拟合质量和三套候选轴的碰撞统计。

这个脚本**不修改生产算法**。它从当前配置和当前 full-mesh 运行参数出发，重算
每个 root 点的：

- 两侧法向 n1/n2、夹角、支持面数、support ratio；
- patch 半径（现定义）与几何 patch scale；
- 二次拟合设计矩阵的条件数、最小奇异值、2D 凸包面积/覆盖率；
- raw fit normal、patch average normal、最终 chosen normal；
- bisector / patch / fit / chosen 四套轴的 72 角度碰撞可行数和原因。

输出为 JSON + CSV + Markdown，不生成 PNG/OBJ。

用法（先小样本验证）：
    python scripts/run_p0_normal_diagnostics.py --max-points 8 --skip-collision
全量：
    python scripts/run_p0_normal_diagnostics.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from feilunqiege.collision.cylinder_exact import (  # noqa: E402
    build_cylinder_triangle_index,
    check_finite_cylinder_candidates_exact,
)
from feilunqiege.geometry.curve import split_and_resample  # noqa: E402
from feilunqiege.geometry.frame import (  # noqa: E402
    _point_triangle_distance,
    build_frames,
    estimate_surface_normals,
)
from feilunqiege.io.flash_cloud import load_labeled_flash_cloud  # noqa: E402
from feilunqiege.io.mesh import load_mesh  # noqa: E402
from feilunqiege.io.root_line import load_root_line  # noqa: E402
from feilunqiege.pipeline import _normalize_case_geometry  # noqa: E402
from run_leaf_stage3_visualization import _candidate_set  # noqa: E402


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector * 0.0
    return vector / norm


def angle_deg(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return None
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def vector_list(vector: np.ndarray | None) -> list[float] | None:
    if vector is None:
        return None
    array = np.asarray(vector, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return None
    return array.tolist()


class MeshData:
    """把 patch/拟合诊断需要的网格数组一次性准备好。"""

    def __init__(self, mesh: Any) -> None:
        self.vertices = np.asarray(mesh.vertices, dtype=float)
        self.faces = np.asarray(mesh.faces, dtype=np.int64)
        self.triangles = self.vertices[self.faces]
        raw = np.cross(
            self.triangles[:, 1] - self.triangles[:, 0],
            self.triangles[:, 2] - self.triangles[:, 0],
        )
        lengths = np.linalg.norm(raw, axis=1)
        self.valid_mask = lengths > 1e-12
        self.valid_indices = np.flatnonzero(self.valid_mask)
        self.valid_triangles = self.triangles[self.valid_mask]
        self.valid_centroids = self.valid_triangles.mean(axis=1)
        self.valid_areas = 0.5 * lengths[self.valid_mask]
        self.valid_face_normals = raw[self.valid_mask] / lengths[self.valid_mask, None]
        self.tree = cKDTree(self.valid_centroids)
        self.signed_volume = float(
            np.einsum(
                "ij,ij->i",
                self.valid_triangles[:, 0],
                np.cross(self.valid_triangles[:, 1], self.valid_triangles[:, 2]),
            ).sum()
            / 6.0
        )
        # 与生产代码一致：闭合网格的整体朝向由 signed volume 决定；局部
        # winding 翻转不在 P0 修复，只如实记录。
        try:
            from feilunqiege.collision.disk_exact import _is_closed_triangle_mesh

            self.mesh_is_closed = bool(_is_closed_triangle_mesh(self.faces))
        except Exception:
            self.mesh_is_closed = False
        self.outward_sign = -1.0 if (self.mesh_is_closed and self.signed_volume < 0.0) else 1.0


def patch_diagnostics(
    point: np.ndarray,
    mesh: MeshData,
    *,
    neighbor_count: int,
    distance_factor: float,
    distance_floor: float,
    min_patch_faces: int,
    max_patch_radius: float,
    fit_min_points: int,
    fit_residual_ratio: float,
) -> dict[str, Any]:
    """复现 `_surface_patch_normals` 的 patch 选择/拟合，并额外保留 raw fit。"""

    point = np.asarray(point, dtype=float)
    query_count = min(max(int(neighbor_count), int(min_patch_faces)), len(mesh.valid_centroids))
    _, neighbor_indices = mesh.tree.query(point, k=query_count)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64).reshape(-1)
    distances = np.asarray(
        [_point_triangle_distance(point, mesh.valid_triangles[index]) for index in neighbor_indices],
        dtype=float,
    )
    order = np.argsort(distances)
    neighbor_indices = neighbor_indices[order]
    distances = distances[order]
    nearest_distance = float(distances[0])

    initial_cutoff = max(nearest_distance * float(distance_factor), float(distance_floor))
    keep = distances <= min(initial_cutoff, float(max_patch_radius))
    if int(np.count_nonzero(keep)) < int(min_patch_faces):
        eligible = distances <= float(max_patch_radius)
        if int(np.count_nonzero(eligible)) >= int(min_patch_faces):
            keep = np.zeros(len(distances), dtype=bool)
            keep[: int(min_patch_faces)] = True
        else:
            keep = eligible
    if not np.any(keep):
        keep[0] = True

    patch_indices = mesh.valid_indices[neighbor_indices[keep]]
    kept_distances = distances[keep]
    kept_areas = mesh.valid_areas[neighbor_indices[keep]]
    raw_kept = mesh.valid_face_normals[neighbor_indices[keep]].copy()
    patch_radius_current = float(np.max(kept_distances))
    low_support = bool(
        int(np.count_nonzero(keep)) < int(min_patch_faces)
        or patch_radius_current > float(max_patch_radius) + 1e-12
    )

    area_weighted_sum = (raw_kept * kept_areas[:, None]).sum(axis=0)
    sum_norm = float(np.linalg.norm(area_weighted_sum))
    seed = raw_kept[0].copy() if sum_norm <= 1e-12 else area_weighted_sum / sum_norm
    if mesh.mesh_is_closed:
        seed *= mesh.outward_sign
    signs = np.where(raw_kept @ seed >= 0.0, 1.0, -1.0)
    sign_consistency = abs(float(np.mean(signs)))
    aligned = raw_kept * signs[:, None]
    weights = kept_areas / np.maximum(kept_distances, float(distance_floor))
    if float(weights.sum()) <= 1e-18:
        weights = np.ones(len(kept_distances), dtype=float)
    weights /= float(weights.sum())
    averaged = (aligned * weights[:, None]).sum(axis=0)
    if float(np.linalg.norm(averaged)) <= 1e-12:
        averaged = seed.copy()
    averaged = unit(averaged)
    coherence = float(
        np.degrees(np.arccos(np.clip(np.min(aligned @ averaged), -1.0, 1.0)))
    )

    patch_faces = mesh.faces[patch_indices]
    unique_vertices = np.unique(patch_faces.reshape(-1))
    patch_vertices = mesh.vertices[unique_vertices]
    patch_vertex_distances = np.linalg.norm(patch_vertices - point[None, :], axis=1)
    patch_scale_max = float(np.max(patch_vertex_distances))
    patch_scale_q90 = float(np.quantile(patch_vertex_distances, 0.90))
    bbox = patch_vertices.max(axis=0) - patch_vertices.min(axis=0)
    patch_bbox_diag = float(0.5 * np.linalg.norm(bbox))

    raw_fit_normal = averaged.copy()
    fit_residual = float("nan")
    design_condition = float("nan")
    design_smin = float("nan")
    design_smax = float("nan")
    design_rank = 0
    hull_area = 0.0
    hull_area_ratio = 0.0
    root_inside_hull: bool | None = None
    cov2_eigen_ratio = float("nan")
    fit_valid = False

    if len(unique_vertices) >= int(fit_min_points):
        centered = patch_vertices - point[None, :]
        tangent_vectors = centered - (centered @ averaged)[:, None] * averaged[None, :]
        if len(tangent_vectors) >= 2:
            covariance3 = tangent_vectors.T @ tangent_vectors
            eigenvalues3, eigenvectors3 = np.linalg.eigh(covariance3)
            u_axis = eigenvectors3[:, int(np.argmax(eigenvalues3))]
            u_axis = u_axis - float(np.dot(u_axis, averaged)) * averaged
            if float(np.linalg.norm(u_axis)) <= 1e-12:
                reference = np.array([1.0, 0.0, 0.0]) if abs(float(averaged[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
                u_axis = np.cross(averaged, reference)
            u_axis = unit(u_axis)
            if float(np.linalg.norm(u_axis)) > 1e-12:
                v_axis = np.cross(averaged, u_axis)
                u = centered @ u_axis
                v = centered @ v_axis
                w = centered @ averaged
                design = np.column_stack([u * u, u * v, v * v, u, v, np.ones_like(u)])
                if design.shape[0] >= 6:
                    singular_values = np.linalg.svd(design, compute_uv=False)
                    design_smax = float(singular_values[0])
                    design_smin = float(singular_values[-1])
                    design_condition = (
                        float(design_smax / max(design_smin, 1e-30))
                        if design_smin > 0.0
                        else float("inf")
                    )
                    design_rank = int(np.linalg.matrix_rank(design))
                coefficients, _, rank, _ = np.linalg.lstsq(design, w, rcond=None)
                design_rank = int(rank)
                if int(rank) >= 5:
                    residual = w - design @ coefficients
                    fit_residual = float(np.sqrt(np.mean(residual**2)))
                    candidate = -float(coefficients[3]) * u_axis - float(coefficients[4]) * v_axis + averaged
                    candidate = unit(candidate)
                    if float(np.dot(candidate, averaged)) < 0.0:
                        candidate = -candidate
                    raw_fit_normal = candidate
                    # 2D 投影覆盖率：u/v 的凸包面积相对于几何 patch scale。
                    uv = np.column_stack([u, v])
                    if len(uv) >= 3:
                        try:
                            hull = ConvexHull(uv)
                            hull_area = float(hull.volume)
                            hull_area_ratio = float(hull_area / max(patch_scale_q90**2, 1e-18))
                            # root 在 (0,0)；hull.equations 是 A x + B y + C <= 0。
                            root_inside_hull = bool(
                                np.all(hull.equations[:, :2] @ np.array([0.0, 0.0]) + hull.equations[:, 2] <= 1e-12)
                            )
                        except Exception:
                            hull_area = 0.0
                            hull_area_ratio = 0.0
                            root_inside_hull = False
                    covariance2 = np.cov(uv.T) if len(uv) > 2 else np.eye(2)
                    eigenvalues2 = np.linalg.eigvalsh(covariance2)
                    cov2_eigen_ratio = float(
                        eigenvalues2[0] / max(eigenvalues2[-1], 1e-30)
                    )
                    fit_valid = bool(
                        np.isfinite(fit_residual)
                        and int(rank) >= 5
                        and len(unique_vertices) >= int(fit_min_points)
                    )

    fit_threshold_current = max(
        float(fit_residual_ratio) * max(patch_radius_current, 1e-9), 1e-6
    )
    fit_threshold_geometric = max(
        float(fit_residual_ratio) * max(patch_scale_q90, 1e-9), 1e-6
    )
    fit_used = bool(
        np.isfinite(fit_residual)
        and fit_residual <= fit_threshold_current
        and len(unique_vertices) >= int(fit_min_points)
        and sign_consistency >= 0.6
    )
    chosen_normal = raw_fit_normal if fit_used else averaged

    return {
        "patch_face_indices": patch_indices.astype(int).tolist(),
        "patch_face_count": int(len(patch_indices)),
        "nearest_distance": nearest_distance,
        "patch_radius_current": patch_radius_current,
        "patch_scale_max": patch_scale_max,
        "patch_scale_q90": patch_scale_q90,
        "patch_bbox_diag": patch_bbox_diag,
        "patch_scale_ratio": (
            float(patch_scale_q90 / patch_radius_current)
            if patch_radius_current > 1e-12
            else None
        ),
        "patch_radius_zero": bool(patch_radius_current <= 1e-12),
        "low_support": low_support,
        "patch_average_normal": averaged.tolist(),
        "patch_coherence_deg": coherence,
        "sign_consistency": sign_consistency,
        "raw_fit_normal": raw_fit_normal.tolist() if fit_valid else None,
        "fit_residual": fit_residual if np.isfinite(fit_residual) else None,
        "fit_threshold_current": fit_threshold_current,
        "fit_threshold_geometric": fit_threshold_geometric,
        "fit_used": fit_used,
        "fit_valid": fit_valid,
        "chosen_normal": chosen_normal.tolist(),
        "design_condition": design_condition if np.isfinite(design_condition) else None,
        "design_smin": design_smin if np.isfinite(design_smin) else None,
        "design_smax": design_smax if np.isfinite(design_smax) else None,
        "design_rank": design_rank,
        "fit_unique_vertices": int(len(unique_vertices)),
        "hull_area": hull_area,
        "hull_area_ratio": hull_area_ratio,
        "root_inside_hull": root_inside_hull,
        "cov2_eigen_ratio": cov2_eigen_ratio if np.isfinite(cov2_eigen_ratio) else None,
    }


def build_axis_frame(point: np.ndarray, tangent: np.ndarray, normal: np.ndarray) -> Any:
    return build_frames(
        np.asarray([point], dtype=float),
        np.asarray([tangent], dtype=float),
        np.asarray([normal], dtype=float),
        normal_is_reliable=True,
    )[0]


def collision_summary(
    point: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    *,
    radius: float,
    width: float,
    angles: np.ndarray,
    point_index: int,
    workpiece: Any,
    cylinder_index: Any,
    collision_config: dict[str, Any],
    contact_patch: dict[str, Any],
) -> dict[str, Any]:
    normal = unit(normal)
    if float(np.linalg.norm(normal)) <= 1e-12:
        return {
            "feasible_angle_count": None,
            "angle_count": int(len(angles)),
            "feasible_angles_deg": [],
            "reason_counts": {},
            "max_contact_penetration": None,
        }
    frame = build_axis_frame(point, tangent, normal)
    candidate_set = _candidate_set(point, frame, radius, width, angles, point_index, 1.0)
    result = check_finite_cylinder_candidates_exact(
        candidate_set,
        workpiece,
        width,
        inner_radius=0.0,
        clearance=float(collision_config.get("clearance", 0.0)),
        allow_target_contact=bool(collision_config.get("allow_target_contact", True)),
        target_contact_tolerance=float(collision_config.get("target_contact_tolerance", 1e-4)),
        contact_patch_enabled=bool(contact_patch.get("enabled", True)),
        contact_patch_radius=float(contact_patch.get("radius", 2.0 * radius)),
        max_contact_penetration=float(contact_patch.get("max_penetration", 0.025 * radius)),
        contact_normal_tolerance_degrees=float(contact_patch.get("normal_tolerance_degrees", 30.0)),
        narrow_phase=str(collision_config.get("narrow_phase", "analytic_closest_point_flat")),
        cylinder_index=cylinder_index,
    )
    feasible = np.asarray(result.feasible, dtype=bool)
    reasons = Counter(str(reason) for reason in np.asarray(result.reason, dtype=object))
    return {
        "feasible_angle_count": int(np.count_nonzero(feasible)),
        "angle_count": int(len(angles)),
        "feasible_angles_deg": np.degrees(angles[feasible]).round(4).tolist(),
        "reason_counts": dict(sorted(reasons.items())),
        "max_contact_penetration": float(np.max(result.contact_max_penetration, initial=0.0)),
    }


def safe_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def percentile(values: list[float | None], q: float) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.percentile(finite, q)) if finite else None


def main() -> int:
    parser = argparse.ArgumentParser(description="P0 法向/面片/碰撞只读诊断")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--case", default="circle")
    parser.add_argument("--source-run", default=None, help="用于读取 angle_count/contact_patch 的 full-mesh 运行目录")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--points", default=None, help="逗号分隔的点编号，用于抽查；与 --max-points 二选一")
    parser.add_argument("--skip-collision", action="store_true")
    parser.add_argument("--angle-count", type=int, default=None)
    arguments = parser.parse_args()

    start_time = time.perf_counter()
    config_path = (REPO / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    case_root = (config_path.parent / config["data_root"]).resolve() / arguments.case
    results_root = REPO / "results" / "cases" / arguments.case

    if arguments.source_run is not None:
        source_run = Path(arguments.source_run).resolve()
    else:
        runs = sorted(
            (path for path in results_root.glob("stage3_visualization_*") if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        source_run = runs[0] if runs else None
    source_summary: dict[str, Any] = {}
    if source_run is not None and (source_run / "stage3_visualization_summary.json").exists():
        source_summary = json.loads(
            (source_run / "stage3_visualization_summary.json").read_text(encoding="utf-8")
        )

    print("loading mesh/root line...", flush=True)
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / "mesh_without_flash.obj")
    flash_cloud = load_labeled_flash_cloud(case_root / "original_flash_cloud.obj")
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    center = np.asarray(normalization.get("center", [0.0, 0.0, 0.0]), dtype=float)
    scale = float(normalization.get("scale", 1.0))
    flash_points = (flash_cloud.flash_points - center) / scale
    mesh = MeshData(workpiece)
    radius = float(source_summary.get("radius", config["wheel"]["radius"]))
    width = float(source_summary.get("width_h", config["wheel"]["width"]))
    angle_count = int(
        arguments.angle_count
        or source_summary.get("angle_count_per_point", 72)
    )
    angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
    collision_config = dict(config.get("collision", {}))
    if source_summary.get("narrow_phase"):
        collision_config["narrow_phase"] = source_summary["narrow_phase"]
    contact_patch = dict(source_summary.get("contact_patch", {}))
    if not contact_patch:
        contact_patch = {
            "enabled": bool(collision_config.get("contact_patch_enabled", True)),
            "radius": float(collision_config.get("contact_patch_radius_ratio", 2.0)) * radius,
            "max_penetration": float(collision_config.get("contact_patch_max_penetration_ratio", 0.025)) * radius,
            "normal_tolerance_degrees": float(collision_config.get("contact_patch_normal_tolerance_degrees", 30.0)),
        }

    normal_config = dict(config.get("normal_estimation", {}))
    normal_config.update(source_summary.get("normal_estimation", {}))
    print("estimating production normals...", flush=True)
    normals, diagnostics = estimate_surface_normals(
        sampled.points,
        workpiece,
        neighbor_count=int(normal_config.get("neighbor_count", 24)),
        candidate_count=int(normal_config.get("candidate_count", 96)),
        side_count=int(normal_config.get("side_count", 8)),
        coherence_angle_degrees=float(normal_config.get("coherence_angle_degrees", 25.0)),
        offset_epsilon=float(normal_config.get("offset_epsilon", 1e-5)),
        weight_distance_floor=float(normal_config.get("weight_distance_floor", 1e-5)),
        flash_points=flash_points,
        branch_ids=sampled.branch_ids,
        normal_method=str(normal_config.get("method", "surface_patch_continuity")),
        surface_patch_neighbors=int(normal_config.get("surface_patch_neighbors", 32)),
        surface_patch_distance_factor=float(normal_config.get("surface_patch_distance_factor", 3.0)),
        surface_patch_coherence_degrees=float(normal_config.get("surface_patch_coherence_degrees", 25.0)),
        max_bisector_deviation_degrees=float(normal_config.get("max_bisector_deviation_degrees", 3.5)),
        surface_patch_min_faces=int(normal_config.get("min_patch_faces", 10)),
        surface_patch_max_radius=float(normal_config.get("max_patch_radius", 0.02)),
        surface_patch_fit_min_points=int(normal_config.get("fit_min_points", 8)),
        surface_patch_fit_residual_ratio=float(normal_config.get("fit_residual_ratio", 0.02)),
        continuity_repair_degrees=float(normal_config.get("continuity_repair_degrees", 30.0)),
        continuity_repair_iterations=int(normal_config.get("continuity_repair_iterations", 3)),
        return_diagnostics=True,
    )
    frames = build_frames(
        sampled.points,
        sampled.tangents,
        normals,
        normal_is_reliable=True,
        tangent_normal_threshold=float(normal_config.get("tangent_normal_threshold", 0.2)),
    )

    if arguments.points:
        point_indices = [int(value) for value in arguments.points.split(",") if value.strip()]
        if any(index < 0 or index >= len(sampled.points) for index in point_indices):
            raise ValueError("--points 包含越界采样点")
    elif arguments.max_points is not None:
        point_indices = list(range(min(int(arguments.max_points), len(sampled.points))))
    else:
        point_indices = list(range(len(sampled.points)))

    cylinder_index = None
    if not arguments.skip_collision:
        print("building cylinder index...", flush=True)
        cylinder_index = build_cylinder_triangle_index(workpiece)

    rows: list[dict[str, Any]] = []
    replication_errors: list[dict[str, float]] = []
    for position, point_index in enumerate(point_indices, start=1):
        point = np.asarray(sampled.points[point_index], dtype=float)
        tangent = np.asarray(sampled.tangents[point_index], dtype=float)
        point_diag = diagnostics[point_index]

        patch = patch_diagnostics(
            point,
            mesh,
            neighbor_count=int(normal_config.get("surface_patch_neighbors", 32)),
            distance_factor=float(normal_config.get("surface_patch_distance_factor", 3.0)),
            distance_floor=float(normal_config.get("weight_distance_floor", 1e-5)),
            min_patch_faces=int(normal_config.get("min_patch_faces", 10)),
            max_patch_radius=float(normal_config.get("max_patch_radius", 0.02)),
            fit_min_points=int(normal_config.get("fit_min_points", 8)),
            fit_residual_ratio=float(normal_config.get("fit_residual_ratio", 0.02)),
        )

        # 复现核对：如果和生产诊断不一致，在报告里显式列出，不能静默混合。
        replication = {
            "patch_face_count": abs(float(patch["patch_face_count"]) - float(point_diag["surface_patch_face_count"])),
            "patch_radius": abs(float(patch["patch_radius_current"]) - float(point_diag["surface_patch_patch_radius"])),
            "patch_average": float(
                np.linalg.norm(
                    np.asarray(patch["patch_average_normal"])
                    - np.asarray(point_diag["surface_patch_average_normal"])
                )
            ),
            "patch_chosen": float(
                np.linalg.norm(
                    np.asarray(patch["chosen_normal"])
                    - np.asarray(point_diag["surface_patch_normal"])
                )
            ),
            "fit_used": float(bool(patch["fit_used"]) != bool(point_diag["surface_patch_fit_used"])),
        }
        if max(replication.values()) > 1e-6:
            replication_errors.append({"point_index": int(point_index), **replication})

        side_details = point_diag.get("side_details", [])
        n1 = None
        n2 = None
        support1 = None
        support2 = None
        if len(side_details) >= 1:
            n1 = np.asarray(side_details[0].get("normal", []), dtype=float)
            support1 = int(side_details[0].get("coherent_support", 0))
        if len(side_details) >= 2:
            n2 = np.asarray(side_details[1].get("normal", []), dtype=float)
            support2 = int(side_details[1].get("coherent_support", 0))
        if n1 is not None and np.linalg.norm(n1) <= 1e-12:
            n1 = None
        if n2 is not None and np.linalg.norm(n2) <= 1e-12:
            n2 = None
        angle12 = angle_deg(n1, n2)
        support_ratio = (
            float(min(support1, support2) / max(support1, support2))
            if support1 is not None and support2 is not None and max(support1, support2) > 0
            else None
        )
        min_support = min(support1, support2) if support1 is not None and support2 is not None else None

        bisector = None
        if n1 is not None and n2 is not None:
            bisector = unit(n1 + n2)
            if float(np.linalg.norm(bisector)) <= 1e-12:
                bisector = None
        patch_axis = np.asarray(patch["patch_average_normal"], dtype=float)
        fit_axis = (
            np.asarray(patch["raw_fit_normal"], dtype=float)
            if patch["raw_fit_normal"] is not None
            else None
        )
        chosen_axis = np.asarray(normals[point_index], dtype=float)

        axes = {
            "bisector": bisector,
            "patch": patch_axis,
            "fit": fit_axis,
            "chosen": chosen_axis,
        }
        axis_results: dict[str, Any] = {}
        if not arguments.skip_collision:
            for axis_name, axis_normal in axes.items():
                if axis_normal is None:
                    axis_results[axis_name] = {
                        "feasible_angle_count": None,
                        "angle_count": int(len(angles)),
                        "feasible_angles_deg": [],
                        "reason_counts": {},
                        "max_contact_penetration": None,
                    }
                else:
                    axis_results[axis_name] = collision_summary(
                        point,
                        tangent,
                        axis_normal,
                        radius=radius,
                        width=width,
                        angles=angles,
                        point_index=point_index,
                        workpiece=workpiece,
                        cylinder_index=cylinder_index,
                        collision_config=collision_config,
                        contact_patch=contact_patch,
                    )
        else:
            for axis_name in axes:
                axis_results[axis_name] = {
                    "feasible_angle_count": None,
                    "angle_count": int(len(angles)),
                    "feasible_angles_deg": [],
                    "reason_counts": {},
                    "max_contact_penetration": None,
                }

        axis_deltas = {
            "bisector_patch": angle_deg(bisector, patch_axis),
            "bisector_fit": angle_deg(bisector, fit_axis),
            "patch_fit": angle_deg(patch_axis, fit_axis),
            "chosen_patch": angle_deg(chosen_axis, patch_axis),
            "chosen_fit": angle_deg(chosen_axis, fit_axis),
        }
        row: dict[str, Any] = {
            "point_index": int(point_index),
            "branch_id": int(sampled.branch_ids[point_index]),
            "point_normalized": point.tolist(),
            "point_source": (point * scale + center).tolist(),
            "n1": vector_list(n1),
            "n2": vector_list(n2),
            "angle_n1_n2_deg": angle12,
            "support1": support1,
            "support2": support2,
            "support_ratio": support_ratio,
            "min_support": min_support,
            "coherence_deg": float(point_diag["surface_patch_coherence_deg"]),
            "bisector_deviation_deg": float(point_diag["bisector_deviation_deg"]),
            "normal_source": str(point_diag["normal_source"]),
            "low_confidence": bool(point_diag.get("low_confidence", False)),
            "low_confidence_reasons": list(point_diag.get("low_confidence_reasons", [])),
            "selected_normal": chosen_axis.tolist(),
            "patch": patch,
            "axes": {
                name: {
                    "normal": vector_list(axis_normal),
                    **axis_results[name],
                }
                for name, axis_normal in axes.items()
            },
            "axis_angles_deg": axis_deltas,
            "replication": replication,
        }
        rows.append(row)
        if position % 25 == 0 or position == len(point_indices):
            print(f"processed {position}/{len(point_indices)} points...", flush=True)

    # 汇总统计
    def point_value(row: dict[str, Any], key: str) -> Any:
        if key in row:
            return row[key]
        if key in row["patch"]:
            return row["patch"][key]
        return None

    angle12_values = [row["angle_n1_n2_deg"] for row in rows]
    support_ratio_values = [row["support_ratio"] for row in rows]
    condition_values = [row["patch"]["design_condition"] for row in rows]
    patch_scale_ratio_values = [row["patch"]["patch_scale_ratio"] for row in rows]

    def axis_count(row: dict[str, Any], name: str) -> int | None:
        value = row["axes"][name]["feasible_angle_count"]
        return None if value is None else int(value)

    selected_rows = [row for row in rows if row["normal_source"] == "two_side_bisector"]
    branch_summary: dict[str, Any] = {}
    for branch_id in sorted({row["branch_id"] for row in rows}):
        branch_rows = [row for row in rows if row["branch_id"] == branch_id]
        branch_summary[str(branch_id)] = {
            "point_count": len(branch_rows),
            "angle12_mean_deg": safe_mean([row["angle_n1_n2_deg"] for row in branch_rows]),
            "angle12_lt10_count": sum(
                1 for row in branch_rows if row["angle_n1_n2_deg"] is not None and row["angle_n1_n2_deg"] < 10.0
            ),
            "support_ratio_mean": safe_mean([row["support_ratio"] for row in branch_rows]),
            "support_ratio_lt035_count": sum(
                1 for row in branch_rows if row["support_ratio"] is not None and row["support_ratio"] < 0.35
            ),
            "condition_median": percentile([row["patch"]["design_condition"] for row in branch_rows], 50),
            "condition_gt1e6_count": sum(
                1 for row in branch_rows
                if row["patch"]["design_condition"] is not None and row["patch"]["design_condition"] > 1e6
            ),
            "fit_used_count": sum(1 for row in branch_rows if row["patch"]["fit_used"]),
            "fit_valid_count": sum(1 for row in branch_rows if row["patch"]["fit_valid"]),
            "axis_zero_counts": {
                name: sum(1 for row in branch_rows if axis_count(row, name) == 0)
                for name in ("bisector", "patch", "fit", "chosen")
            },
            "axis_mean_feasible": {
                name: safe_mean([axis_count(row, name) for row in branch_rows])
                for name in ("bisector", "patch", "fit", "chosen")
            },
        }

    all_zero = [
        row["point_index"] for row in rows
        if all(axis_count(row, name) == 0 for name in ("bisector", "patch", "fit", "chosen"))
    ]
    chosen_zero_patch_positive = [
        row["point_index"] for row in rows
        if axis_count(row, "chosen") == 0 and (axis_count(row, "patch") or 0) > 0
    ]
    chosen_zero_fit_positive = [
        row["point_index"] for row in rows
        if axis_count(row, "chosen") == 0 and (axis_count(row, "fit") or 0) > 0
    ]
    chosen_zero_bisector_positive = [
        row["point_index"] for row in rows
        if axis_count(row, "chosen") == 0 and (axis_count(row, "bisector") or 0) > 0
    ]
    patch_better = [
        row["point_index"] for row in rows
        if axis_count(row, "patch") is not None and axis_count(row, "chosen") is not None
        and axis_count(row, "patch") > axis_count(row, "chosen")
    ]
    fit_better = [
        row["point_index"] for row in rows
        if axis_count(row, "fit") is not None and axis_count(row, "chosen") is not None
        and axis_count(row, "fit") > axis_count(row, "chosen")
    ]
    bisector_better = [
        row["point_index"] for row in rows
        if axis_count(row, "bisector") is not None and axis_count(row, "chosen") is not None
        and axis_count(row, "bisector") > axis_count(row, "chosen")
    ]
    patch_worse = [
        row["point_index"] for row in rows
        if axis_count(row, "patch") is not None and axis_count(row, "chosen") is not None
        and axis_count(row, "patch") < axis_count(row, "chosen")
    ]

    summary = {
        "point_count": len(rows),
        "source_run": str(source_run) if source_run is not None else None,
        "parameters": {
            "radius": radius,
            "width": width,
            "angle_count": angle_count,
            "contact_patch": contact_patch,
            "normal_estimation": normal_config,
            "normalized_units": True,
        },
        "normal_source_counts": dict(Counter(row["normal_source"] for row in rows)),
        "low_confidence_count": sum(1 for row in rows if row["low_confidence"]),
        "angle12": {
            "median_deg": percentile(angle12_values, 50),
            "mean_deg": safe_mean(angle12_values),
            "lt5_count": sum(1 for value in angle12_values if value is not None and value < 5.0),
            "lt10_count": sum(1 for value in angle12_values if value is not None and value < 10.0),
            "gte10_count": sum(1 for value in angle12_values if value is not None and value >= 10.0),
            "gte10_points": [
                row["point_index"] for row in rows
                if row["angle_n1_n2_deg"] is not None and row["angle_n1_n2_deg"] >= 10.0
            ],
            "gte15_count": sum(1 for value in angle12_values if value is not None and value >= 15.0),
        },
        "support_ratio": {
            "median": percentile(support_ratio_values, 50),
            "lt025_count": sum(1 for value in support_ratio_values if value is not None and value < 0.25),
            "lt035_count": sum(1 for value in support_ratio_values if value is not None and value < 0.35),
            "lt050_count": sum(1 for value in support_ratio_values if value is not None and value < 0.50),
            "min_support_lt6_count": sum(1 for row in rows if row["min_support"] is not None and row["min_support"] < 6),
            "min_support_lt8_count": sum(1 for row in rows if row["min_support"] is not None and row["min_support"] < 8),
        },
        "patch_scale": {
            "patch_radius_median": percentile([row["patch"]["patch_radius_current"] for row in rows], 50),
            "patch_scale_q90_median": percentile([row["patch"]["patch_scale_q90"] for row in rows], 50),
            "ratio_median": percentile(patch_scale_ratio_values, 50),
            "ratio_gt3_count": sum(1 for value in patch_scale_ratio_values if value is not None and value > 3.0),
            "ratio_gt5_count": sum(1 for value in patch_scale_ratio_values if value is not None and value > 5.0),
            "ratio_max": max((value for value in patch_scale_ratio_values if value is not None), default=None),
            "zero_radius_count": sum(1 for row in rows if row["patch"]["patch_radius_zero"]),
        },
        "fitting": {
            "fit_valid_count": sum(1 for row in rows if row["patch"]["fit_valid"]),
            "fit_used_count": sum(1 for row in rows if row["patch"]["fit_used"]),
            "condition_median": percentile(condition_values, 50),
            "condition_gt1e5_count": sum(1 for value in condition_values if value is not None and value > 1e5),
            "condition_gt1e6_count": sum(1 for value in condition_values if value is not None and value > 1e6),
            "root_inside_hull_count": sum(1 for row in rows if row["patch"]["root_inside_hull"] is True),
            "root_outside_hull_count": sum(1 for row in rows if row["patch"]["root_inside_hull"] is False),
            "hull_area_ratio_median": percentile([row["patch"]["hull_area_ratio"] for row in rows], 50),
            "hull_area_ratio_lt01_count": sum(
                1 for row in rows if row["patch"]["hull_area_ratio"] < 0.1
            ),
            "cov2_eigen_ratio_median": percentile([row["patch"]["cov2_eigen_ratio"] for row in rows], 50),
            "cov2_eigen_ratio_lt002_count": sum(
                1 for row in rows
                if row["patch"]["cov2_eigen_ratio"] is not None and row["patch"]["cov2_eigen_ratio"] < 0.02
            ),
        },
        "axis_comparison": {
            "all_zero_points": all_zero,
            "chosen_zero_patch_positive": chosen_zero_patch_positive,
            "chosen_zero_fit_positive": chosen_zero_fit_positive,
            "chosen_zero_bisector_positive": chosen_zero_bisector_positive,
            "patch_better_count": len(patch_better),
            "fit_better_count": len(fit_better),
            "bisector_better_count": len(bisector_better),
            "patch_worse_count": len(patch_worse),
            "patch_better_points": patch_better,
            "fit_better_points": fit_better,
            "bisector_better_points": bisector_better,
        },
        "branch_summary": branch_summary,
        "replication_mismatch_count": len(replication_errors),
        "replication_mismatches": replication_errors[:20],
        "problem_flags": {
            "angle12_lt10": [row["point_index"] for row in rows if row["angle_n1_n2_deg"] is not None and row["angle_n1_n2_deg"] < 10.0],
            "support_ratio_lt035": [row["point_index"] for row in rows if row["support_ratio"] is not None and row["support_ratio"] < 0.35],
            "condition_gt1e6": [row["point_index"] for row in rows if row["patch"]["design_condition"] is not None and row["patch"]["design_condition"] > 1e6],
            "root_outside_hull": [row["point_index"] for row in rows if row["patch"]["root_inside_hull"] is False],
            "patch_scale_ratio_gt3": [
                row["point_index"] for row in rows
                if row["patch"]["patch_scale_ratio"] is not None and row["patch"]["patch_scale_ratio"] > 3.0
            ],
        },
    }

    # 按“patch/fit 相对 chosen 的改善”排序，便于人工抽查。
    def improvement(row: dict[str, Any]) -> int:
        chosen = axis_count(row, "chosen") or 0
        return max(
            (axis_count(row, "patch") or 0) - chosen,
            (axis_count(row, "fit") or 0) - chosen,
            (axis_count(row, "bisector") or 0) - chosen,
        )

    priority_rows = sorted(rows, key=improvement, reverse=True)[:30]
    summary["priority_points"] = [
        {
            "point_index": row["point_index"],
            "branch_id": row["branch_id"],
            "chosen": axis_count(row, "chosen"),
            "patch": axis_count(row, "patch"),
            "fit": axis_count(row, "fit"),
            "bisector": axis_count(row, "bisector"),
            "angle12": row["angle_n1_n2_deg"],
            "support_ratio": row["support_ratio"],
            "condition": row["patch"]["design_condition"],
            "fit_used": row["patch"]["fit_used"],
            "normal_source": row["normal_source"],
        }
        for row in priority_rows
    ]

    output_dir = (
        Path(arguments.output_dir).resolve()
        if arguments.output_dir is not None
        else results_root / datetime.now().strftime("p0_normal_diagnostics_%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "p0_normal_diagnostics.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "case": arguments.case,
                    "config": str(config_path),
                    "source_run": str(source_run) if source_run is not None else None,
                    "workpiece_mesh": "mesh_without_flash.obj",
                    "units": "normalized",
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "point_indices": point_indices,
                    "skip_collision": bool(arguments.skip_collision),
                },
                "summary": summary,
                "points": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "p0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # CSV：固定列；向量以分号分隔；reason_counts 用 JSON 字符串。
    fieldnames = [
        "point_index", "branch_id", "point_norm_x", "point_norm_y", "point_norm_z",
        "point_src_x", "point_src_y", "point_src_z",
        "normal_source", "low_confidence", "low_confidence_reasons",
        "n1", "n2", "angle_n1_n2_deg", "support1", "support2", "support_ratio", "min_support",
        "patch_face_count", "patch_radius_current", "patch_scale_max", "patch_scale_q90",
        "patch_bbox_diag", "patch_scale_ratio",
        "patch_average_normal", "raw_fit_normal", "patch_chosen_normal", "selected_normal",
        "fit_residual", "fit_threshold_current", "fit_threshold_geometric",
        "fit_used", "fit_valid", "design_condition", "design_smin", "design_smax",
        "design_rank", "fit_unique_vertices", "hull_area", "hull_area_ratio",
        "root_inside_hull", "cov2_eigen_ratio",
        "bisector_feasible", "patch_feasible", "fit_feasible", "chosen_feasible",
        "bisector_reasons", "patch_reasons", "fit_reasons", "chosen_reasons",
        "bisector_patch_deg", "bisector_fit_deg", "patch_fit_deg",
        "chosen_patch_deg", "chosen_fit_deg",
        "replication_patch_face_count", "replication_patch_radius",
        "replication_patch_average", "replication_patch_chosen", "replication_fit_used",
    ]
    with (output_dir / "p0_normal_diagnostics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            patch = row["patch"]
            writer.writerow(
                {
                    "point_index": row["point_index"],
                    "branch_id": row["branch_id"],
                    "point_norm_x": row["point_normalized"][0],
                    "point_norm_y": row["point_normalized"][1],
                    "point_norm_z": row["point_normalized"][2],
                    "point_src_x": row["point_source"][0],
                    "point_src_y": row["point_source"][1],
                    "point_src_z": row["point_source"][2],
                    "normal_source": row["normal_source"],
                    "low_confidence": row["low_confidence"],
                    "low_confidence_reasons": ";".join(row["low_confidence_reasons"]),
                    "n1": "" if row["n1"] is None else ";".join(f"{v:.9g}" for v in row["n1"]),
                    "n2": "" if row["n2"] is None else ";".join(f"{v:.9g}" for v in row["n2"]),
                    "angle_n1_n2_deg": "" if row["angle_n1_n2_deg"] is None else f"{row['angle_n1_n2_deg']:.9g}",
                    "support1": "" if row["support1"] is None else row["support1"],
                    "support2": "" if row["support2"] is None else row["support2"],
                    "support_ratio": "" if row["support_ratio"] is None else f"{row['support_ratio']:.9g}",
                    "min_support": "" if row["min_support"] is None else row["min_support"],
                    "patch_face_count": patch["patch_face_count"],
                    "patch_radius_current": f"{patch['patch_radius_current']:.9g}",
                    "patch_scale_max": f"{patch['patch_scale_max']:.9g}",
                    "patch_scale_q90": f"{patch['patch_scale_q90']:.9g}",
                    "patch_bbox_diag": f"{patch['patch_bbox_diag']:.9g}",
                    "patch_scale_ratio": "" if patch["patch_scale_ratio"] is None else f"{patch['patch_scale_ratio']:.9g}",
                    "patch_average_normal": ";".join(f"{v:.9g}" for v in patch["patch_average_normal"]),
                    "raw_fit_normal": "" if patch["raw_fit_normal"] is None else ";".join(f"{v:.9g}" for v in patch["raw_fit_normal"]),
                    "patch_chosen_normal": ";".join(f"{v:.9g}" for v in patch["chosen_normal"]),
                    "selected_normal": ";".join(f"{v:.9g}" for v in row["selected_normal"]),
                    "fit_residual": "" if patch["fit_residual"] is None else f"{patch['fit_residual']:.9g}",
                    "fit_threshold_current": f"{patch['fit_threshold_current']:.9g}",
                    "fit_threshold_geometric": f"{patch['fit_threshold_geometric']:.9g}",
                    "fit_used": patch["fit_used"],
                    "fit_valid": patch["fit_valid"],
                    "design_condition": "" if patch["design_condition"] is None else f"{patch['design_condition']:.9g}",
                    "design_smin": "" if patch["design_smin"] is None else f"{patch['design_smin']:.9g}",
                    "design_smax": "" if patch["design_smax"] is None else f"{patch['design_smax']:.9g}",
                    "design_rank": patch["design_rank"],
                    "fit_unique_vertices": patch["fit_unique_vertices"],
                    "hull_area": f"{patch['hull_area']:.9g}",
                    "hull_area_ratio": f"{patch['hull_area_ratio']:.9g}",
                    "root_inside_hull": "" if patch["root_inside_hull"] is None else patch["root_inside_hull"],
                    "cov2_eigen_ratio": "" if patch["cov2_eigen_ratio"] is None else f"{patch['cov2_eigen_ratio']:.9g}",
                    "bisector_feasible": "" if row["axes"]["bisector"]["feasible_angle_count"] is None else row["axes"]["bisector"]["feasible_angle_count"],
                    "patch_feasible": "" if row["axes"]["patch"]["feasible_angle_count"] is None else row["axes"]["patch"]["feasible_angle_count"],
                    "fit_feasible": "" if row["axes"]["fit"]["feasible_angle_count"] is None else row["axes"]["fit"]["feasible_angle_count"],
                    "chosen_feasible": "" if row["axes"]["chosen"]["feasible_angle_count"] is None else row["axes"]["chosen"]["feasible_angle_count"],
                    "bisector_reasons": json.dumps(row["axes"]["bisector"]["reason_counts"], ensure_ascii=False),
                    "patch_reasons": json.dumps(row["axes"]["patch"]["reason_counts"], ensure_ascii=False),
                    "fit_reasons": json.dumps(row["axes"]["fit"]["reason_counts"], ensure_ascii=False),
                    "chosen_reasons": json.dumps(row["axes"]["chosen"]["reason_counts"], ensure_ascii=False),
                    "bisector_patch_deg": "" if row["axis_angles_deg"]["bisector_patch"] is None else f"{row['axis_angles_deg']['bisector_patch']:.9g}",
                    "bisector_fit_deg": "" if row["axis_angles_deg"]["bisector_fit"] is None else f"{row['axis_angles_deg']['bisector_fit']:.9g}",
                    "patch_fit_deg": "" if row["axis_angles_deg"]["patch_fit"] is None else f"{row['axis_angles_deg']['patch_fit']:.9g}",
                    "chosen_patch_deg": "" if row["axis_angles_deg"]["chosen_patch"] is None else f"{row['axis_angles_deg']['chosen_patch']:.9g}",
                    "chosen_fit_deg": "" if row["axis_angles_deg"]["chosen_fit"] is None else f"{row['axis_angles_deg']['chosen_fit']:.9g}",
                    "replication_patch_face_count": f"{row['replication']['patch_face_count']:.9g}",
                    "replication_patch_radius": f"{row['replication']['patch_radius']:.9g}",
                    "replication_patch_average": f"{row['replication']['patch_average']:.9g}",
                    "replication_patch_chosen": f"{row['replication']['patch_chosen']:.9g}",
                    "replication_fit_used": f"{row['replication']['fit_used']:.9g}",
                }
            )

    # 优先点 CSV
    with (output_dir / "p0_priority_points.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        priority_fields = [
            "point_index", "branch_id", "chosen", "patch", "fit", "bisector",
            "angle12", "support_ratio", "condition", "fit_used", "normal_source",
        ]
        writer = csv.DictWriter(handle, fieldnames=priority_fields)
        writer.writeheader()
        writer.writerows(summary["priority_points"])

    # Markdown 报告
    def fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "—"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value)

    branch_rows_md = "\n".join(
        f"| {branch} | {item['point_count']} | {fmt(item['angle12_mean_deg'], 2)} | "
        f"{item['angle12_lt10_count']} | {fmt(item['support_ratio_mean'], 2)} | "
        f"{item['support_ratio_lt035_count']} | {fmt(item['condition_median'], 1)} | "
        f"{item['fit_used_count']} | {fmt(item['axis_mean_feasible']['chosen'], 1)} | "
        f"{fmt(item['axis_mean_feasible']['patch'], 1)} | {fmt(item['axis_mean_feasible']['fit'], 1)} |"
        for branch, item in branch_summary.items()
    )
    priority_md = "\n".join(
        f"| {item['point_index']} | {item['branch_id']} | {fmt(item['chosen'])} | {fmt(item['patch'])} | "
        f"{fmt(item['fit'])} | {fmt(item['bisector'])} | {fmt(item['angle12'], 2)} | "
        f"{fmt(item['support_ratio'], 3)} | {fmt(item['condition'], 1)} | {item['normal_source']} |"
        for item in summary["priority_points"][:20]
    )
    report = f"""# P0 法向/面片/碰撞诊断报告

本报告是**只读诊断**，没有修改生产算法、配置或正式结果。
运行目录：`{output_dir}`
源 full-mesh 运行：`{source_run}`
点数：{len(rows)}；角度数：{angle_count}；碰撞计算：{'未运行' if arguments.skip_collision else '已运行'}；单位：normalized。

## 1. 关键计数

- `normal_source`：{summary['normal_source_counts']}
- `low_confidence`：{summary['low_confidence_count']} 个点
- `angle(n1,n2)`：中位数 {fmt(summary['angle12']['median_deg'], 2)}°，`<5°` {summary['angle12']['lt5_count']} 个，`<10°` {summary['angle12']['lt10_count']} 个，`>=15°` {summary['angle12']['gte15_count']} 个
- support ratio：中位数 {fmt(summary['support_ratio']['median'], 3)}，`<0.25` {summary['support_ratio']['lt025_count']} 个，`<0.35` {summary['support_ratio']['lt035_count']} 个，`<0.50` {summary['support_ratio']['lt050_count']} 个
- min support：`<6` {summary['support_ratio']['min_support_lt6_count']} 个，`<8` {summary['support_ratio']['min_support_lt8_count']} 个
- patch scale / patch radius：中位数 {fmt(summary['patch_scale']['ratio_median'], 2)}，`>3` {summary['patch_scale']['ratio_gt3_count']} 个，`>5` {summary['patch_scale']['ratio_gt5_count']} 个，`patch_radius=0` {summary['patch_scale']['zero_radius_count']} 个
- fit：valid {summary['fitting']['fit_valid_count']} 个，used {summary['fitting']['fit_used_count']} 个
- fit condition：中位数 {fmt(summary['fitting']['condition_median'], 1)}，`>1e5` {summary['fitting']['condition_gt1e5_count']} 个，`>1e6` {summary['fitting']['condition_gt1e6_count']} 个
- root outside hull：{summary['fitting']['root_outside_hull_count']} 个
- 复现不一致：{summary['replication_mismatch_count']} 个（阈值 1e-6）

## 2. 分支部统计

| branch | points | mean angle12 | angle12<10 | mean support ratio | ratio<0.35 | median cond | fit_used | mean chosen feasible | mean patch feasible | mean fit feasible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{branch_rows_md}

## 3. 三/四套轴的碰撞对照

- 全零（四个轴都 0）：{summary['axis_comparison']['all_zero_points']}
- chosen 0 但 patch >0：{summary['axis_comparison']['chosen_zero_patch_positive']}
- chosen 0 但 fit >0：{summary['axis_comparison']['chosen_zero_fit_positive']}
- chosen 0 但 bisector >0：{summary['axis_comparison']['chosen_zero_bisector_positive']}
- patch 比 chosen 好：{summary['axis_comparison']['patch_better_count']} 个
- fit 比 chosen 好：{summary['axis_comparison']['fit_better_count']} 个
- bisector 比 chosen 好：{summary['axis_comparison']['bisector_better_count']} 个
- patch 比 chosen 差：{summary['axis_comparison']['patch_worse_count']} 个

### 优先抽查点（按 max(patch,fit,bisector) − chosen 排序）

| point | branch | chosen | patch | fit | bisector | angle12 | support ratio | condition | source |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{priority_md}

## 4. P0 结论（自动生成，需人工确认）

1. **`angle(n1,n2)` 的分布决定“折边分支”是否可信**：如果大量点 `<10°`，则 `coherence >25°` 不能单独作为 bisector 依据。
2. **support ratio / min support** 暴露“8/8 选面”背后的真实支持不平衡；低 support 点应标记 low confidence。
3. **`patch_scale / patch_radius`** 显示当前拟合阈值是否被“点到三角形距离”错误压缩。
4. **condition / root inside hull** 显示 raw fit 是否可审计、是否病态。
5. **四套轴碰撞对照** 回答“换法向是否能改善可行率”。如果 patch/fit 仍大量 0，则剩余问题在砂轮几何/中心模型，而不是单纯法向。
6. **复现不一致** 若 >0，先检查 patch 选择/`valid` 映射，不能把诊断数值直接用于修改。
7. **不要把 `max_bisector_deviation_degrees=3.5` 直接当硬门**：P0 中可能存在“deviation 大但 bisector 可行、deviation 小但 bisector 0 可行”的点；该参数与“是否正确”不单调。应先看 `angle12` 和 support ratio，再决定是否使用 bisector。

## 5. P0 之后需要做什么

### 你需要做的

1. 打开 `p0_summary.json` 和 `p0_normal_diagnostics.csv`，先确认三个**定义**：
   - `patch_scale` 用 `q90` 还是 `max`（报告同时给了两者）；
   - support ratio 低置信阈值用 0.25 还是 0.35；
   - fit condition 告警/拒绝阈值用 1e5 还是 1e6。
2. 抽查 `p0_priority_points.csv` 的前 10–20 个点（尤其 branch 7 的 395–404、branch 4 的 365）：
   - `angle12` 是否真的很小（说明不是折边）；
   - raw fit 是否与 patch/bisector 明显不同；
   - condition 是否病态；
   - patch/fit 是否能改善 feasible count。
3. 决定低置信点的处理策略：
   - 只标记、仍输出 best-effort 轴；
   - 还是标记为“法向不确定/无解”，等待人工复核；
   - 还是输出支撑锥/可行轴区间（更符合 c-space 思路）。
4. 确认是否把 P0 指标作为 P1 的验收基线（每个点修复前后对比）。

### 我需要做的（P1）

1. 用 P0 的分布确定 `min_crease_angle_degrees`、`min_side_support_ratio`、`min_side_support_faces`、`max_fit_condition`、`min_hull_coverage` 的初始值。
2. 修 `frame.py` 的 crease gate：`coherence` 只作为候选信号，不再单独决定 bisector。
3. 修 `patch_scale`、fit 条件数/覆盖、raw fit 诊断字段。
4. 修符号/定向（全局绕向或逐面内外测试），再重跑 P0 对照。
5. 如果 P0 显示 patch/fit 也大量 0，则 P1 之后单独开 P3 砂轮中心/姿态模型，不把法向修复当作零可行的充分解。

## 6. 文件

- `p0_normal_diagnostics.json`：逐点完整数据（含向量、面索引、角度、碰撞原因）
- `p0_normal_diagnostics.csv`：Excel 可读的扁平表
- `p0_summary.json`：汇总统计
- `p0_priority_points.csv`：优先抽查点
- `p0_report.md`：本报告
"""
    (output_dir / "p0_report.md").write_text(report, encoding="utf-8")

    print(f"output_dir={output_dir}")
    print(f"points={len(rows)} skip_collision={arguments.skip_collision}")
    print(f"summary={output_dir / 'p0_summary.json'}")
    print(f"report={output_dir / 'p0_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
