"""诊断 circle branch 4 / branch 5 的局部表面平滑度与阈值敏感性。

只读脚本，不修改碰撞算法、不修改已有结果。输出 JSON：
- 每个问题点在 full / simplified 网格上的局部平滑度指标；
- simplified 相对 full 的表面偏差；
- 同一组法向/姿态下，full 与 simplified 的碰撞可行数对比；
- 不同 contact_patch_max_penetration 下的可行数扫描。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

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
from run_leaf_stage3_visualization import _candidate_set  # noqa: E402


def _mesh_data(vertices: np.ndarray, faces: np.ndarray):
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(lengths[:, None], 1e-12)
    signed_volume = float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0)
    if signed_volume < 0:
        normals = -normals
    return tris, normals, tris.mean(axis=1), cKDTree(tris.mean(axis=1))


def _local_surface_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    point: np.ndarray,
    *,
    radius: float,
) -> dict[str, object]:
    """在局部半径内拟合平面和二次曲面，统计平滑度。"""

    tris, normals, centroids, tree = _mesh_data(vertices, faces)
    _, indices = tree.query(point, k=min(400, len(centroids)))
    indices = np.asarray(indices, dtype=np.int64)
    distances = np.linalg.norm(centroids[indices] - point[None, :], axis=1)
    indices = indices[distances <= radius]
    if len(indices) < 4:
        return {"triangle_count": int(len(indices)), "insufficient": True}
    unique_vertices = np.unique(faces[indices].reshape(-1))
    patch = vertices[unique_vertices]
    patch_normals = normals[indices]

    center = patch.mean(axis=0)
    centered = patch - center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    plane_normal = vh[-1]
    if float(np.dot(plane_normal, patch_normals.mean(axis=0))) < 0.0:
        plane_normal = -plane_normal
    plane_residual = centered @ plane_normal
    plane_rms = float(np.sqrt(np.mean(plane_residual**2)))
    plane_max = float(np.max(np.abs(plane_residual)))

    # 局部二次曲面：w = a u^2 + b uv + c v^2 + d u + e v + f。
    u_axis = vh[0]
    v_axis = np.cross(plane_normal, u_axis)
    v_axis /= max(float(np.linalg.norm(v_axis)), 1e-12)
    u = centered @ u_axis
    v = centered @ v_axis
    w = centered @ plane_normal
    design = np.column_stack([u * u, u * v, v * v, u, v, np.ones_like(u)])
    coefficients, _, _, _ = np.linalg.lstsq(design, w, rcond=None)
    quad_residual = w - design @ coefficients
    quad_rms = float(np.sqrt(np.mean(quad_residual**2)))
    quad_max = float(np.max(np.abs(quad_residual)))

    mean_normal = patch_normals.mean(axis=0)
    mean_normal /= max(float(np.linalg.norm(mean_normal)), 1e-12)
    cosines = np.clip(patch_normals @ mean_normal, -1.0, 1.0)
    normal_spread_deg = float(np.degrees(np.arccos(np.min(cosines))))

    # 相对局部切平面的高度；正值表示表面突出到切平面外侧。
    heights = (patch - point[None, :]) @ plane_normal
    return {
        "triangle_count": int(len(indices)),
        "vertex_count": int(len(unique_vertices)),
        "plane_normal": plane_normal.tolist(),
        "plane_rms": plane_rms,
        "plane_max": plane_max,
        "quadratic_rms": quad_rms,
        "quadratic_max": quad_max,
        "normal_spread_deg": normal_spread_deg,
        "height_min": float(np.min(heights)),
        "height_max": float(np.max(heights)),
        "height_rms": float(np.sqrt(np.mean(heights**2))),
    }


def _signed_distance(point: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> tuple[float, float]:
    tris, normals, centroids, tree = _mesh_data(vertices, faces)
    _, indices = tree.query(point, k=min(128, len(centroids)))
    indices = np.asarray(indices, dtype=np.int64)
    distances = np.asarray([_point_triangle_distance(point, tris[i]) for i in indices])
    best = int(np.argmin(distances))
    triangle_index = int(indices[best])
    # 使用最近三角形的有符号平面距离；对局部平滑表面足够。
    signed = float(np.dot(point - tris[triangle_index][0], normals[triangle_index]))
    return float(distances[best]), signed


def _surface_deviation(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    point: np.ndarray,
    *,
    radius: float,
) -> dict[str, float]:
    """source 局部顶点到 target 网格的最近距离；用于量化简化误差。"""

    tris_t, _, centroids_t, tree_t = _mesh_data(target_vertices, target_faces)
    mask = np.linalg.norm(source_vertices - point[None, :], axis=1) <= radius
    source_patch = source_vertices[mask]
    if len(source_patch) == 0:
        return {"vertex_count": 0, "median": 0.0, "max": 0.0}
    distances = []
    for vertex in source_patch:
        _, indices = tree_t.query(vertex, k=min(64, len(centroids_t)))
        indices = np.asarray(indices, dtype=np.int64)
        distances.append(min(_point_triangle_distance(vertex, tris_t[i]) for i in indices))
    distances = np.asarray(distances)
    return {
        "vertex_count": int(len(source_patch)),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="circle 局部表面与阈值诊断")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--case", default="circle")
    parser.add_argument("--points", default="367,368,369,370,371,372,383")
    parser.add_argument("--radius", type=float, default=0.04)
    parser.add_argument("--result-dir", default=None)
    arguments = parser.parse_args()

    config_path = (REPO_ROOT / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    case_root = (config_path.parent / config["data_root"]).resolve() / arguments.case
    results_root = REPO_ROOT / "results" / "cases" / arguments.case
    if arguments.result_dir is not None:
        result_dir = Path(arguments.result_dir).resolve()
    else:
        runs = sorted((path for path in results_root.glob("stage3_visualization_*") if path.is_dir()), reverse=True)
        result_dir = runs[0]
    summary = json.loads((result_dir / "stage3_visualization_summary.json").read_text(encoding="utf-8"))

    full = load_mesh(case_root / "mesh_without_flash.obj")
    simplified = load_mesh(case_root / "mesh_without_flash_simplified.obj")
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    flash_cloud = load_labeled_flash_cloud(case_root / "original_flash_cloud.obj")

    # 使用 simplified 的包围盒做与主流程一致的归一化。
    vertices_s = np.asarray(simplified.vertices, dtype=float)
    low, high = vertices_s.min(axis=0), vertices_s.max(axis=0)
    center = (low + high) / 2.0
    scale = float(np.max(high - low))
    vertices_s_norm = (vertices_s - center) / scale
    vertices_f_norm = (np.asarray(full.vertices, dtype=float) - center) / scale
    faces_s = np.asarray(simplified.faces, dtype=np.int64)
    faces_f = np.asarray(full.faces, dtype=np.int64)
    root_norm = (np.asarray(root_line.vertices, dtype=float) - center) / scale
    sampled = split_and_resample(
        type(root_line)(vertices=root_norm, edges=root_line.edges), int(config["sampling"]["curve_points"])
    )

    normal_config = {**config.get("normal_estimation", {}), **summary.get("normal_estimation", {})}
    normal_kwargs = {
        key: normal_config[key]
        for key in ("neighbor_count", "candidate_count", "side_count", "coherence_angle_degrees",
                    "offset_epsilon", "weight_distance_floor")
        if key in normal_config
    }
    flash_norm = (flash_cloud.flash_points - center) / scale
    normals = estimate_surface_normals(
        sampled.points,
        type(simplified)(vertices=vertices_s_norm, faces=faces_s),
        flash_points=flash_norm,
        normal_method=str(normal_config.get("method", "two_face_angle_bisector")),
        surface_patch_neighbors=int(normal_config.get("surface_patch_neighbors", 32)),
        surface_patch_distance_factor=float(normal_config.get("surface_patch_distance_factor", 3.0)),
        surface_patch_coherence_degrees=float(normal_config.get("surface_patch_coherence_degrees", 25.0)),
        max_bisector_deviation_degrees=float(normal_config.get("max_bisector_deviation_degrees", 10.0)),
        continuity_repair_degrees=float(normal_config.get("continuity_repair_degrees", 30.0)),
        continuity_repair_iterations=int(normal_config.get("continuity_repair_iterations", 3)),
        **normal_kwargs,
    )
    frames = build_frames(
        sampled.points,
        sampled.tangents,
        normals,
        normal_is_reliable=(str(normal_config.get("method", "")) == "surface_patch_continuity"),
        tangent_normal_threshold=float(normal_config.get("tangent_normal_threshold", 0.2)),
    )

    radius = float(summary["radius"])
    width = float(summary["width_h"])
    angle_count = int(summary.get("angle_count_per_point", 72))
    angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
    patch = summary.get("contact_patch", {})
    contact_patch_enabled = bool(patch.get("enabled", True))
    contact_patch_radius = float(patch.get("radius", 2.0 * radius))
    current_max_penetration = float(patch.get("max_penetration", 0.025 * radius))
    normal_tolerance = float(patch.get("normal_tolerance_degrees", 30.0))
    narrow_phase = str(summary.get("narrow_phase", "analytic_closest_point_flat"))
    collision_config = config.get("collision", {})
    allow_target_contact = bool(collision_config.get("allow_target_contact", True))
    target_contact_tolerance = float(collision_config.get("target_contact_tolerance", 1e-4))

    points = [int(value) for value in arguments.points.split(",") if value.strip()]
    point_reports = []
    for point_index in points:
        point = np.asarray(sampled.points[point_index], dtype=float)
        distance_s, signed_s = _signed_distance(point, vertices_s_norm, faces_s)
        distance_f, signed_f = _signed_distance(point, vertices_f_norm, faces_f)
        point_reports.append(
            {
                "point_index": int(point_index),
                "branch_id": int(sampled.branch_ids[point_index]),
                "point_normalized": point.tolist(),
                "simplified": {
                    "signed_distance": signed_s,
                    "nearest_distance": distance_s,
                    "metrics": _local_surface_metrics(vertices_s_norm, faces_s, point, radius=arguments.radius),
                },
                "full": {
                    "signed_distance": signed_f,
                    "nearest_distance": distance_f,
                    "metrics": _local_surface_metrics(vertices_f_norm, faces_f, point, radius=arguments.radius),
                },
                "simplified_vs_full_surface_distance": _surface_deviation(
                    vertices_s_norm, vertices_f_norm, faces_f, point, radius=arguments.radius
                ),
            }
        )

    # 同一组法向/姿态下，full 与 simplified 的碰撞可行数对比。
    index_s = build_cylinder_triangle_index(type(simplified)(vertices=vertices_s_norm, faces=faces_s))
    index_f = build_cylinder_triangle_index(type(full)(vertices=vertices_f_norm, faces=faces_f))
    collision_comparison = []
    for point_index in points:
        candidates = _candidate_set(
            np.asarray(sampled.points[point_index]), frames[point_index], radius, width, angles,
            point_index, 1.0,
        )
        results = {}
        for name, mesh, index in (
            ("simplified", type(simplified)(vertices=vertices_s_norm, faces=faces_s), index_s),
            ("full", type(full)(vertices=vertices_f_norm, faces=faces_f), index_f),
        ):
            result = check_finite_cylinder_candidates_exact(
                candidates, mesh, width,
                inner_radius=0.0,
                clearance=float(collision_config.get("clearance", 0.0)),
                allow_target_contact=allow_target_contact,
                target_contact_tolerance=target_contact_tolerance,
                contact_patch_enabled=contact_patch_enabled,
                contact_patch_radius=contact_patch_radius,
                max_contact_penetration=current_max_penetration,
                contact_normal_tolerance_degrees=normal_tolerance,
                narrow_phase=narrow_phase,
                cylinder_index=index,
            )
            feasible = int(np.count_nonzero(result.feasible))
            results[name] = {
                "feasible": feasible,
                "infeasible": int(len(result.feasible) - feasible),
                "reasons": {
                    str(key): int(value)
                    for key, value in zip(*np.unique(np.asarray(result.reason, dtype=object), return_counts=True), strict=True)
                },
            }
        collision_comparison.append({"point_index": int(point_index), **results})

    # 阈值扫描：只对 7 个问题点逐个阈值重算。
    thresholds = [
        current_max_penetration,
        6.27e-4,
        7.174e-4,
        1.0e-3,
        2.0e-3,
        3.470596787597974e-3,
        3.5e-3,
        4.0e-3,
        4.5e-3,
        5.0e-3,
    ]
    thresholds = sorted(set(round(value, 12) for value in thresholds))
    threshold_sweep = []
    for threshold in thresholds:
        per_point = []
        total = 0
        for point_index in points:
            candidates = _candidate_set(
                np.asarray(sampled.points[point_index]), frames[point_index], radius, width, angles,
                point_index, 1.0,
            )
            result = check_finite_cylinder_candidates_exact(
                candidates, type(simplified)(vertices=vertices_s_norm, faces=faces_s), width,
                inner_radius=0.0,
                clearance=float(collision_config.get("clearance", 0.0)),
                allow_target_contact=allow_target_contact,
                target_contact_tolerance=target_contact_tolerance,
                contact_patch_enabled=contact_patch_enabled,
                contact_patch_radius=contact_patch_radius,
                max_contact_penetration=float(threshold),
                contact_normal_tolerance_degrees=normal_tolerance,
                narrow_phase=narrow_phase,
                cylinder_index=index_s,
            )
            feasible = int(np.count_nonzero(result.feasible))
            total += feasible
            per_point.append({"point_index": int(point_index), "feasible": feasible, "candidate_count": len(result.feasible)})
        threshold_sweep.append(
            {
                "max_contact_penetration": float(threshold),
                "ratio_of_radius": float(threshold) / radius,
                "total_feasible": int(total),
                "total_candidates": int(len(points) * angle_count),
                "all_points_all_angles_feasible": bool(total == len(points) * angle_count),
                "per_point": per_point,
            }
        )

    report = {
        "case_name": arguments.case,
        "source_result_dir": str(result_dir),
        "units": "normalized",
        "analysis_radius": arguments.radius,
        "points": point_reports,
        "collision_comparison_same_frames": collision_comparison,
        "threshold_sweep": threshold_sweep,
        "threshold_notes": {
            "current": current_max_penetration,
            "current_ratio_of_radius": current_max_penetration / radius,
            "branch4_max_observed_penetration": 6.270022354969425e-4,
            "point383_first_rejecting_hit_max_penetration": 3.470596787597974e-3,
            "depth_only_threshold_for_all_72_angles": 4.000000100000029e-3,
            "depth_only_ratio_of_radius": 0.200000005,
            "remaining_non_depth_failures": "point 383 angles 180/185: workpiece_contains_wheel",
            "all_green_note": "No finite depth threshold can remove the two containment failures; see the threshold_exact report.",
        },
        "limitations": [
            "平滑度指标使用局部平面/二次曲面拟合，弯曲表面会把曲率计入残差。",
            "full vs simplified 碰撞对比使用 simplified 上估计的法向与姿态，只改变碰撞网格。",
            "阈值扫描只覆盖 72 个离散角度，不代表连续角度域。",
        ],
    }
    output_dir = results_root / datetime.now().strftime("local_surface_audit_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / f"{arguments.case}_local_surface_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={report_path}")
    for item in point_reports:
        s = item["simplified"]["metrics"]
        f = item["full"]["metrics"]
        print(
            f"point {item['point_index']:3d} br{item['branch_id']:2d} "
            f"signed_s={item['simplified']['signed_distance']:+.2e} signed_f={item['full']['signed_distance']:+.2e} "
            f"plane_rms_s={s['plane_rms']:.2e} plane_rms_f={f['plane_rms']:.2e} "
            f"normal_spread_s={s['normal_spread_deg']:.2f} normal_spread_f={f['normal_spread_deg']:.2f}"
        )
    print("collision comparison:")
    for item in collision_comparison:
        print(item)
    print("threshold sweep:")
    for item in threshold_sweep:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
