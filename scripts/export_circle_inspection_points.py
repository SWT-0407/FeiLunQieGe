"""为 circle 原始网格结果导出“优先检查点”的砂轮姿态 OBJ。

只重算指定点，不重跑全部 415 个点；用于把 branch 7 的零可行点（398–401）
和其他检查点一起放进一个独立交付目录。

不修改已有结果，不删除文件。
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
)
from feilunqiege.geometry.curve import split_and_resample  # noqa: E402
from feilunqiege.geometry.frame import build_frames, estimate_surface_normals  # noqa: E402
from feilunqiege.io.flash_cloud import load_labeled_flash_cloud  # noqa: E402
from feilunqiege.io.mesh import load_mesh  # noqa: E402
from feilunqiege.io.root_line import load_root_line  # noqa: E402
from feilunqiege.pipeline import _normalize_case_geometry  # noqa: E402
from run_leaf_stage3_visualization import (  # noqa: E402
    PointAudit,
    _branch_runs,
    _candidate_set,
    _export_inspection_wheels,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 circle 优先检查点砂轮姿态")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--case", default="circle")
    parser.add_argument("--source-run", default=None, help="原始网格结果目录；默认最新 stage3_visualization_*")
    parser.add_argument(
        "--points",
        default="187,74,75,359,366,373,381,389,395,397,398,399,400,401,404,413",
    )
    parser.add_argument("--per-point", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args()

    config_path = (REPO_ROOT / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    case_root = (config_path.parent / config["data_root"]).resolve() / arguments.case
    results_root = REPO_ROOT / "results" / "cases" / arguments.case
    if arguments.source_run is not None:
        source_run = Path(arguments.source_run).resolve()
    else:
        runs = sorted(
            (path for path in results_root.glob("stage3_visualization_*") if path.is_dir()),
            reverse=True,
        )
        source_run = runs[0]
    summary = json.loads((source_run / "stage3_visualization_summary.json").read_text(encoding="utf-8"))

    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / "mesh_without_flash.obj")
    flash_cloud = load_labeled_flash_cloud(case_root / "original_flash_cloud.obj")
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    center_source = np.asarray(normalization.get("center", [0.0, 0.0, 0.0]), dtype=float)
    scale_source = float(normalization.get("scale", 1.0))
    flash_points = (flash_cloud.flash_points - center_source) / scale_source

    normal_config = {**config.get("normal_estimation", {}), **summary.get("normal_estimation", {})}
    normal_kwargs = {
        key: normal_config[key]
        for key in ("neighbor_count", "candidate_count", "side_count", "coherence_angle_degrees",
                    "offset_epsilon", "weight_distance_floor")
        if key in normal_config
    }
    normals = estimate_surface_normals(
        sampled.points,
        workpiece,
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
    collision_config = config.get("collision", {})
    cylinder_index = build_cylinder_triangle_index(workpiece)
    branches = _branch_runs(np.asarray(sampled.branch_ids, dtype=np.int64))
    points = [int(value) for value in arguments.points.split(",") if value.strip()]
    if any(index < 0 or index >= len(sampled.points) for index in points):
        raise ValueError("points 包含越界采样点")

    audits: list[PointAudit | None] = [None] * len(sampled.points)
    for point_index in points:
        candidates = _candidate_set(
            np.asarray(sampled.points[point_index]), frames[point_index], radius, width, angles,
            point_index, 1.0,
        )
        result = check_finite_cylinder_candidates_exact(
            candidates, workpiece, width,
            inner_radius=0.0,
            clearance=float(collision_config.get("clearance", 0.0)),
            allow_target_contact=bool(collision_config.get("allow_target_contact", True)),
            target_contact_tolerance=float(collision_config.get("target_contact_tolerance", 1e-4)),
            contact_patch_enabled=bool(patch.get("enabled", True)),
            contact_patch_radius=float(patch.get("radius", 2.0 * radius)),
            max_contact_penetration=float(patch.get("max_penetration", 0.025 * radius)),
            contact_normal_tolerance_degrees=float(patch.get("normal_tolerance_degrees", 30.0)),
            narrow_phase=str(summary.get("narrow_phase", "analytic_closest_point_flat")),
            cylinder_index=cylinder_index,
        )
        audits[point_index] = PointAudit(point_index, candidates, result)

    output_dir = (
        Path(arguments.output_dir).resolve()
        if arguments.output_dir is not None
        else results_root / datetime.now().strftime("inspection_priority_%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = output_dir / f"{arguments.case}_inspection_wheels_priority.obj"
    records = _export_inspection_wheels(
        obj_path, sampled, audits, branches, normalization, radius, width,
        int(config.get("visualization", {}).get("cylinder_segments", 64)),
        points,
        per_point=int(arguments.per_point),
    )
    metadata = {
        "case_name": arguments.case,
        "source_run": str(source_run),
        "workpiece_mesh": "mesh_without_flash.obj",
        "points": points,
        "per_point": int(arguments.per_point),
        "records": records,
        "note": "只重算指定检查点；其余点未计算。",
    }
    (output_dir / f"{arguments.case}_inspection_wheels_priority.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"obj={obj_path}")
    print(f"records={len(records)}")
    print(f"points={points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
