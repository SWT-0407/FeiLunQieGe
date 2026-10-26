"""P1 方法对比：baseline、P1 推荐法和四篇文献启发的组合法。

只对指定点做碰撞（默认 branch 4/7），避免全量 415×多方法×72 角的长批次。
法向指标可对全部 415 点另行运行 `--all-normal-metrics`。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from feilunqiege.collision.cylinder_exact import (  # noqa: E402
    build_cylinder_triangle_index,
    check_finite_cylinder_candidates_exact,
)
from feilunqiege.geometry.curve import split_and_resample  # noqa: E402
from feilunqiege.geometry.frame import build_frames, estimate_surface_normals  # noqa: E402
from feilunqiege.geometry.normal_p1 import estimate_normals_experimental  # noqa: E402
from feilunqiege.io.flash_cloud import load_labeled_flash_cloud  # noqa: E402
from feilunqiege.io.mesh import load_mesh  # noqa: E402
from feilunqiege.io.root_line import load_root_line  # noqa: E402
from feilunqiege.pipeline import _normalize_case_geometry  # noqa: E402
from run_leaf_stage3_visualization import _candidate_set  # noqa: E402


METHODS = [
    "baseline",
    "p1",
    "zhong2021",
    "zhang2015",
    "cazals2005",
    "li2010",
    "zhong+cazals",
    "zhang+cazals",
    "li+zhong",
    "all",
]


def angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))))


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 方法对比")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--case", default="circle")
    parser.add_argument("--points", default="365,395,396,397,398,399,400,401,402,403,404")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--normal-metrics-only", action="store_true")
    parser.add_argument("--all-points", action="store_true", help="对所有 root 点计算法向指标（不跑碰撞）")
    parser.add_argument("--max-points", type=int, default=None)
    arguments = parser.parse_args()

    config_path = (REPO / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    case_root = (config_path.parent / config["data_root"]).resolve() / arguments.case
    results_root = REPO / "results" / "cases" / arguments.case
    runs = sorted(results_root.glob("stage3_visualization_*"), key=lambda path: path.name, reverse=True)
    source_summary = {}
    if runs and (runs[0] / "stage3_visualization_summary.json").exists():
        source_summary = json.loads(
            (runs[0] / "stage3_visualization_summary.json").read_text(encoding="utf-8")
        )

    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / "mesh_without_flash.obj")
    flash_cloud = load_labeled_flash_cloud(case_root / "original_flash_cloud.obj")
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    center = np.asarray(normalization["center"], dtype=float)
    scale = float(normalization["scale"])
    flash_points = (flash_cloud.flash_points - center) / scale
    radius = float(source_summary.get("radius", config["wheel"]["radius"]))
    width = float(source_summary.get("width_h", config["wheel"]["width"]))
    angle_count = int(source_summary.get("angle_count_per_point", 72))
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

    point_indices = [int(value) for value in arguments.points.split(",") if value.strip()]
    if arguments.all_points:
        point_indices = list(range(len(sampled.points)))
    if arguments.max_points is not None:
        point_indices = point_indices[: int(arguments.max_points)]
    methods = [value for value in arguments.methods.split(",") if value.strip()]
    cylinder_index = None
    if not arguments.normal_metrics_only:
        cylinder_index = build_cylinder_triangle_index(workpiece)

    normal_config = dict(config.get("normal_estimation", {}))
    normal_config.update(source_summary.get("normal_estimation", {}))
    # baseline 只算一次；新方法各自准备网格数据。
    baseline_normals, baseline_diags = estimate_surface_normals(
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

    p1_config = {
        "candidate_faces": 96,
        "candidate_radius": float(source_summary.get("normal_estimation", {}).get("max_patch_radius", 0.02)),
        "min_patch_faces": 10,
        "normal_cluster_angle_degrees": 15.0,
        "min_cluster_support": 4,
        "min_crease_angle_degrees": 10.0,
        "min_side_support_ratio": 0.35,
        "min_side_support_faces": 6,
        "l0_trim_angle_degrees": 25.0,
        "fit_residual_ratio": float(normal_config.get("fit_residual_ratio", 0.02)),
        "max_fit_condition": 1e6,
        "min_hull_area_ratio": 0.1,
        "guided_sigma_scale": 0.008,
        "guided_sigma_angle_degrees": 15.0,
        "orientation_repair": True,
    }
    configured_p1 = dict(normal_config.get("p1", {}))
    if configured_p1:
        p1_config.update(configured_p1)
        # config 用 radius ratio，运行时换成 normalized 半径。
        if "candidate_radius_ratio" in p1_config:
            p1_config["candidate_radius"] = float(p1_config["candidate_radius_ratio"]) * radius

    output_dir = (
        Path(arguments.output_dir).resolve()
        if arguments.output_dir is not None
        else results_root / datetime.now().strftime("p1_method_comparison_%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for method in methods:
        if method == "baseline":
            normals = baseline_normals
            diags = baseline_diags
        else:
            normals, diags = estimate_normals_experimental(
                sampled.points, workpiece, method=method, config=p1_config
            )
        for point_index in point_indices:
            point = np.asarray(sampled.points[point_index], dtype=float)
            tangent = np.asarray(sampled.tangents[point_index], dtype=float)
            normal = np.asarray(normals[point_index], dtype=float)
            diag = diags[point_index]
            frame = build_frames(
                np.asarray([point]), np.asarray([tangent]), np.asarray([normal]), normal_is_reliable=True
            )[0]
            feasible_count = None
            reason_counts: dict[str, int] = {}
            feasible_angles: list[float] = []
            if not arguments.normal_metrics_only:
                candidates = _candidate_set(point, frame, radius, width, angles, point_index, 1.0)
                result = check_finite_cylinder_candidates_exact(
                    candidates,
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
                feasible_count = int(np.count_nonzero(result.feasible))
                reason_counts = dict(Counter(str(reason) for reason in np.asarray(result.reason, dtype=object)))
                feasible_angles = np.degrees(angles[np.asarray(result.feasible, dtype=bool)]).round(4).tolist()
            fit = diag.get("fit") if isinstance(diag, dict) else None
            records.append(
                {
                    "method": method,
                    "point_index": int(point_index),
                    "branch_id": int(sampled.branch_ids[point_index]),
                    "normal": normal.tolist(),
                    "normal_source": str(diag.get("normal_source", "surface_fit" if diag.get("surface_patch_fit_used") else "surface_patch_average")),
                    "normal_confidence": str(diag.get("normal_confidence", "low" if diag.get("low_confidence") else "high")),
                    "angle_n1_n2_deg": diag.get("angle_n1_n2_deg", diag.get("bisector_deviation_deg")),
                    "support_ratio": diag.get("support_ratio"),
                    "support1": diag.get("support1"),
                    "support2": diag.get("support2"),
                    "fit_valid": None if fit is None else bool(fit.get("valid")),
                    "fit_condition": None if fit is None else fit.get("condition"),
                    "feasible_count": feasible_count,
                    "angle_count": int(len(angles)),
                    "reason_counts": reason_counts,
                    "feasible_angles_deg": feasible_angles,
                }
            )
        print(f"method {method}: {len(point_indices)} points done", flush=True)

    # 汇总
    summary = {"point_indices": point_indices, "angle_count": angle_count, "methods": {}}
    for method in methods:
        rows = [record for record in records if record["method"] == method]
        branch7 = [row for row in rows if row["branch_id"] == 7]
        branch4 = [row for row in rows if row["branch_id"] == 4]
        summary["methods"][method] = {
            "point_count": len(rows),
            "feasible_total": sum(row["feasible_count"] or 0 for row in rows),
            "candidate_total": len(rows) * angle_count if not arguments.normal_metrics_only else None,
            "feasible_ratio": (
                sum(row["feasible_count"] or 0 for row in rows) / (len(rows) * angle_count)
                if not arguments.normal_metrics_only and rows
                else None
            ),
            "zero_point_count": sum(1 for row in rows if row["feasible_count"] == 0),
            "branch7_feasible_mean": float(np.mean([row["feasible_count"] or 0 for row in branch7])) if branch7 else None,
            "branch4_feasible_mean": float(np.mean([row["feasible_count"] or 0 for row in branch4])) if branch4 else None,
            "low_confidence_count": sum(1 for row in rows if row["normal_confidence"] == "low"),
            "source_counts": dict(Counter(row["normal_source"] for row in rows)),
        }
    (output_dir / "p1_method_comparison.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "case": arguments.case,
                    "points": point_indices,
                    "methods": methods,
                    "normalized_units": True,
                    "normal_metrics_only": bool(arguments.normal_metrics_only),
                    "p1_config": p1_config,
                    "radius": radius,
                    "width": width,
                    "contact_patch": contact_patch,
                },
                "summary": summary,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "p1_method_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "method", "point_index", "branch_id", "normal_source", "normal_confidence",
            "angle_n1_n2_deg", "support_ratio", "support1", "support2",
            "fit_valid", "fit_condition", "feasible_count", "angle_count",
            "feasible_angles_deg", "reason_counts", "normal",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "method": record["method"],
                    "point_index": record["point_index"],
                    "branch_id": record["branch_id"],
                    "normal_source": record["normal_source"],
                    "normal_confidence": record["normal_confidence"],
                    "angle_n1_n2_deg": record["angle_n1_n2_deg"],
                    "support_ratio": record["support_ratio"],
                    "support1": record["support1"],
                    "support2": record["support2"],
                    "fit_valid": record["fit_valid"],
                    "fit_condition": record["fit_condition"],
                    "feasible_count": record["feasible_count"],
                    "angle_count": record["angle_count"],
                    "feasible_angles_deg": json.dumps(record["feasible_angles_deg"]),
                    "reason_counts": json.dumps(record["reason_counts"], ensure_ascii=False),
                    "normal": ";".join(f"{value:.9g}" for value in record["normal"]),
                }
            )
    (output_dir / "p1_method_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
