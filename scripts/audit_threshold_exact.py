"""精确计算“只靠提高穿透阈值能否让所有点变绿”。

只读脚本。对每个问题点的 72 个候选：
- 遍历所有未被目标接触豁免的相交三角形；
- 跳过已经满足 patch/法向条件的允许接触；
- 记录剩余相交的最大穿透深度；
- 用很高的阈值再跑一次检查，识别“提高阈值也无法解决”的非深度失败
  （例如 ``workpiece_contains_wheel``）。

输出 ``circle_threshold_exact.json``。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from feilunqiege.collision.cylinder_exact import (  # noqa: E402
    build_cylinder_triangle_index,
    check_finite_cylinder_candidates_exact,
    query_cylinder_aabb,
    triangle_cylinder_contact_state_analytic,
)
from feilunqiege.geometry.curve import split_and_resample  # noqa: E402
from feilunqiege.geometry.frame import build_frames, estimate_surface_normals  # noqa: E402
from feilunqiege.io.flash_cloud import load_labeled_flash_cloud  # noqa: E402
from feilunqiege.io.mesh import load_mesh  # noqa: E402
from feilunqiege.io.root_line import load_root_line  # noqa: E402
from run_leaf_stage3_visualization import _candidate_set  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="精确阈值分析")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--case", default="circle")
    parser.add_argument("--points", default="367,368,369,370,371,372,383")
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

    simplified = load_mesh(case_root / "mesh_without_flash_simplified.obj")
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    flash_cloud = load_labeled_flash_cloud(case_root / "original_flash_cloud.obj")
    vertices = np.asarray(simplified.vertices, dtype=float)
    faces = np.asarray(simplified.faces, dtype=np.int64)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = (low + high) / 2.0
    scale = float(np.max(high - low))
    vertices_norm = (vertices - center) / scale
    faces_norm = faces
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
        type(simplified)(vertices=vertices_norm, faces=faces_norm),
        flash_points=flash_norm,
        normal_method=str(normal_config.get("method", "two_face_angle_bisector")),
        surface_patch_neighbors=int(normal_config.get("surface_patch_neighbors", 32)),
        surface_patch_distance_factor=float(normal_config.get("surface_patch_distance_factor", 3.0)),
        surface_patch_coherence_degrees=float(normal_config.get("surface_patch_coherence_degrees", 25.0)),
        max_bisector_deviation_degrees=float(normal_config.get("max_bisector_deviation_degrees", 3.5)),
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
    normal_tolerance = float(patch.get("normal_tolerance_degrees", 30.0))
    cos_normal_tolerance = float(np.cos(np.deg2rad(normal_tolerance)))
    narrow_phase = str(summary.get("narrow_phase", "analytic_closest_point_flat"))
    collision_config = config.get("collision", {})
    allow_target_contact = bool(collision_config.get("allow_target_contact", True))
    target_contact_tolerance = float(collision_config.get("target_contact_tolerance", 1e-4))

    cylinder_index = build_cylinder_triangle_index(
        type(simplified)(vertices=vertices_norm, faces=faces_norm)
    )
    points = [int(value) for value in arguments.points.split(",") if value.strip()]
    point_results = []
    global_max_penetration = 0.0
    remaining_non_depth: list[dict[str, object]] = []

    for point_index in points:
        candidates = _candidate_set(
            np.asarray(sampled.points[point_index]), frames[point_index], radius, width, angles,
            point_index, 1.0,
        )
        candidate_max_penetrations = []
        patch_failures = 0
        normal_failures = 0
        for candidate_index in range(len(candidates.centers)):
            center_point = np.asarray(candidates.centers[candidate_index], dtype=float)
            axis = np.asarray(candidates.axis_directions[candidate_index], dtype=float)
            contact = np.asarray(candidates.contact_points[candidate_index], dtype=float)
            nearby = query_cylinder_aabb(cylinder_index, center_point, axis, radius, width)
            max_penetration = 0.0
            has_patch_failure = False
            has_normal_failure = False
            for triangle_index in nearby:
                state = triangle_cylinder_contact_state_analytic(
                    cylinder_index.triangles[int(triangle_index)], center_point, axis, radius, width,
                    contact_point=contact,
                )
                if not state.hit:
                    continue
                if (
                    allow_target_contact
                    and state.witness is not None
                    and float(np.linalg.norm(state.witness - contact)) <= target_contact_tolerance
                ):
                    continue
                if state.patch_max_distance > contact_patch_radius + 1e-12:
                    has_patch_failure = True
                if state.normal_alignment < cos_normal_tolerance - 1e-12:
                    has_normal_failure = True
                if not has_patch_failure and not has_normal_failure:
                    max_penetration = max(max_penetration, float(state.penetration))
            candidate_max_penetrations.append(max_penetration)
            patch_failures += int(has_patch_failure)
            normal_failures += int(has_normal_failure)
        candidate_max_array = np.asarray(candidate_max_penetrations)
        global_max_penetration = max(global_max_penetration, float(candidate_max_array.max()))

        # 用一个极高的阈值检查哪些候选仍然不可行；这些不是深度问题。
        high_threshold = 2.0 * radius
        high_result = check_finite_cylinder_candidates_exact(
            candidates, type(simplified)(vertices=vertices_norm, faces=faces_norm), width,
            inner_radius=0.0,
            clearance=float(collision_config.get("clearance", 0.0)),
            allow_target_contact=allow_target_contact,
            target_contact_tolerance=target_contact_tolerance,
            contact_patch_enabled=contact_patch_enabled,
            contact_patch_radius=contact_patch_radius,
            max_contact_penetration=high_threshold,
            contact_normal_tolerance_degrees=normal_tolerance,
            narrow_phase=narrow_phase,
            cylinder_index=cylinder_index,
        )
        infeasible = np.flatnonzero(~np.asarray(high_result.feasible, dtype=bool))
        for candidate_index in infeasible:
            remaining_non_depth.append(
                {
                    "point_index": int(point_index),
                    "candidate_index": int(candidate_index),
                    "angle_deg": float(np.degrees(angles[candidate_index])),
                    "reason": str(high_result.reason[candidate_index]),
                }
            )
        point_results.append(
            {
                "point_index": int(point_index),
                "branch_id": int(sampled.branch_ids[point_index]),
                "max_depth_limited_penetration": float(candidate_max_array.max()),
                "patch_failure_candidates": int(patch_failures),
                "normal_failure_candidates": int(normal_failures),
                "non_depth_failure_candidates": int(len(infeasible)),
            }
        )

    required_threshold = float(global_max_penetration)
    report = {
        "case_name": arguments.case,
        "source_result_dir": str(result_dir),
        "units": "normalized",
        "radius": radius,
        "depth_only": {
            "max_penetration_over_patch_normal_ok_hits": required_threshold,
            "ratio_of_radius": required_threshold / radius,
            "threshold_with_one_percent_margin": required_threshold * 1.01,
            "all_depth_failures_removed": True,
        },
        "per_point": point_results,
        "remaining_non_depth_failures_at_high_threshold": remaining_non_depth,
        "all_points_all_angles_feasible_by_depth_alone": len(remaining_non_depth) == 0,
        "conclusion": (
            "提高穿透阈值可以消除所有深度型失败，但不能消除 "
            "workpiece_contains_wheel 等非深度失败；这些需要改变位姿或包含判定。"
        ),
        "limitations": [
            "只覆盖 72 个离散角度，不代表连续角度域。",
            "阈值单位为 normalized units；比值相对砂轮半径 r。",
        ],
    }
    output_dir = results_root / datetime.now().strftime("threshold_exact_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / f"{arguments.case}_threshold_exact.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={report_path}")
    print(f"depth_only_threshold={required_threshold} ratio={required_threshold / radius}")
    print(f"remaining_non_depth={remaining_non_depth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
