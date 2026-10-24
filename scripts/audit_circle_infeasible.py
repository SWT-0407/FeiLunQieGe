"""审计 circle 结果中所有不可行候选，并输出可复现的 JSON 报告。

只读已有 ``stage3_visualization_*`` 结果目录的参数和可行/不可行统计，
重新计算不可行点，记录每个不可行候选的：
- 碰撞类型（端面/侧面/三角形内部/棱边）；
- 穿透深度、轴向/径向穿透；
- 接触斑距离；
- 三角形法向与砂轮轴的夹角；
- 具体违反哪个接触斑条件；
- 工件见证点。

不修改已有结果，不重新导出 OBJ。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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
from feilunqiege.pipeline import _normalize_case_geometry  # noqa: E402
from run_leaf_stage3_visualization import _branch_runs, _candidate_set  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="审计 circle 所有不可行候选")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--case", default="circle")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--max-examples-per-point", type=int, default=3)
    return parser


def _latest_result_dir(results_root: Path) -> Path:
    runs = sorted((path for path in results_root.glob("stage3_visualization_*") if path.is_dir()), reverse=True)
    if not runs:
        raise FileNotFoundError(f"未找到 stage3_visualization_*: {results_root}")
    return runs[0]


def _parse_collision_counts(obj_path: Path) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    group = None
    faces = 0
    for line in obj_path.open(encoding="ascii"):
        if line.startswith("o "):
            group = line.split()[1]
            faces = 0
        elif line.startswith("f "):
            faces += 1
            if group and group.startswith("collision_centers_point_"):
                counts[int(group.rsplit("_", 1)[1])] = faces // 8
    return dict(counts)


def _first_collision_state(
    candidates: object,
    candidate_index: int,
    workpiece: object,
    cylinder_index: object,
    *,
    radius: float,
    width: float,
    clearance: float,
    allow_target_contact: bool,
    target_contact_tolerance: float,
    contact_patch_enabled: bool,
    contact_patch_radius: float,
    max_contact_penetration: float,
    contact_normal_tolerance_degrees: float,
) -> dict[str, object] | None:
    """找到该候选第一个真正导致拒绝的相交三角形。

    与 ``check_finite_cylinder_candidates_exact`` 一致：被目标接触豁免的
    三角形跳过；满足接触斑条件的三角形也跳过（它们是允许的接触）；只有
    至少违反一个接触斑条件的三角形才是拒绝原因。
    """

    center = np.asarray(candidates.centers[candidate_index], dtype=float)
    axis = np.asarray(candidates.axis_directions[candidate_index], dtype=float)
    contact = np.asarray(candidates.contact_points[candidate_index], dtype=float)
    outer_radius = radius + clearance
    expanded_width = width + 2.0 * clearance
    cos_normal_tolerance = float(np.cos(np.deg2rad(contact_normal_tolerance_degrees)))
    nearby = query_cylinder_aabb(cylinder_index, center, axis, outer_radius, expanded_width)
    for triangle_index in nearby:
        state = triangle_cylinder_contact_state_analytic(
            cylinder_index.triangles[int(triangle_index)], center, axis, outer_radius, expanded_width,
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
        if contact_patch_enabled:
            depth_ok = float(state.penetration) <= max_contact_penetration + 1e-12
            patch_ok = float(state.patch_max_distance) <= contact_patch_radius + 1e-12
            normal_ok = float(state.normal_alignment) >= cos_normal_tolerance - 1e-12
            if depth_ok and patch_ok and normal_ok:
                # 这是允许的接触斑，继续检查其余三角形。
                continue
        return {
            "triangle_index": int(triangle_index),
            "kind": state.kind,
            "penetration": float(state.penetration),
            "axial_penetration": float(state.axial_penetration),
            "radial_penetration": float(state.radial_penetration),
            "patch_max_distance": float(state.patch_max_distance),
            "normal_alignment": float(state.normal_alignment),
            "witness": state.witness.tolist() if state.witness is not None else None,
            "contact_normal": state.contact_normal.tolist() if state.contact_normal is not None else None,
        }
    return None


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.max_examples_per_point < 0:
        raise ValueError("max-examples-per-point 不能为负")
    config_path = (REPO_ROOT / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    case_root = (config_path.parent / config["data_root"]).resolve() / arguments.case
    results_root = REPO_ROOT / "results" / "cases" / arguments.case
    result_dir = (
        Path(arguments.result_dir).resolve()
        if arguments.result_dir is not None
        else _latest_result_dir(results_root)
    )
    summary_path = result_dir / "stage3_visualization_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_obj = result_dir / f"{arguments.case}_feasible_center_domains.obj"
    if not source_obj.is_file():
        raise FileNotFoundError(source_obj)

    # 使用源结果中的参数，避免当前配置变化导致审计对象不一致。
    radius = float(summary["radius"])
    width = float(summary["width_h"])
    angle_count = int(summary.get("angle_count_per_point", 72))
    patch = summary.get("contact_patch", {})
    contact_patch_enabled = bool(patch.get("enabled", True))
    contact_patch_radius = float(patch.get("radius", 2.0 * radius))
    max_contact_penetration = float(patch.get("max_penetration", 0.025 * radius))
    contact_normal_tolerance_degrees = float(patch.get("normal_tolerance_degrees", 30.0))
    narrow_phase = str(summary.get("narrow_phase", "analytic_closest_point_flat"))
    collision_config = config.get("collision", {})
    allow_target_contact = bool(collision_config.get("allow_target_contact", True))
    target_contact_tolerance = float(collision_config.get("target_contact_tolerance", 1e-4))

    workpiece_path = Path(summary.get("workpiece_mesh", case_root / config["inputs"]["workpiece_mesh"]))
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(workpiece_path)
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
    cylinder_index = build_cylinder_triangle_index(workpiece)
    angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
    collision_counts = _parse_collision_counts(source_obj)
    branches = _branch_runs(np.asarray(sampled.branch_ids, dtype=np.int64))
    branch_by_point = {}
    for branch_id, start, end in branches:
        for point_index in range(start, end):
            branch_by_point[point_index] = int(branch_id)

    per_point: list[dict[str, object]] = []
    all_reasons = Counter()
    all_failed = Counter()
    all_penetrations: list[float] = []
    all_patch_distances: list[float] = []
    all_normal_alignments: list[float] = []
    total_infeasible = 0

    for point_index in sorted(index for index, count in collision_counts.items() if count > 0):
        candidates = _candidate_set(
            np.asarray(sampled.points[point_index]), frames[point_index], radius, width, angles,
            point_index, 1.0,
        )
        result = check_finite_cylinder_candidates_exact(
            candidates, workpiece, width,
            inner_radius=0.0,
            clearance=float(collision_config.get("clearance", 0.0)),
            allow_target_contact=allow_target_contact,
            target_contact_tolerance=target_contact_tolerance,
            contact_patch_enabled=contact_patch_enabled,
            contact_patch_radius=contact_patch_radius,
            max_contact_penetration=max_contact_penetration,
            contact_normal_tolerance_degrees=contact_normal_tolerance_degrees,
            narrow_phase=narrow_phase,
            cylinder_index=cylinder_index,
        )
        infeasible = np.flatnonzero(~np.asarray(result.feasible, dtype=bool))
        if len(infeasible) == 0:
            continue
        point_records: list[dict[str, object]] = []
        point_reasons = Counter()
        point_failed = Counter()
        point_penetrations: list[float] = []
        point_patch: list[float] = []
        point_normals: list[float] = []
        for candidate_index in infeasible:
            state = _first_collision_state(
                candidates, int(candidate_index), workpiece, cylinder_index,
                radius=radius, width=width,
                clearance=float(collision_config.get("clearance", 0.0)),
                allow_target_contact=allow_target_contact,
                target_contact_tolerance=target_contact_tolerance,
                contact_patch_enabled=contact_patch_enabled,
                contact_patch_radius=contact_patch_radius,
                max_contact_penetration=max_contact_penetration,
                contact_normal_tolerance_degrees=contact_normal_tolerance_degrees,
            )
            if state is None:
                state = {
                    "kind": str(result.reason[candidate_index]),
                    "penetration": float("nan"),
                    "axial_penetration": float("nan"),
                    "radial_penetration": float("nan"),
                    "patch_max_distance": float("nan"),
                    "normal_alignment": float("nan"),
                    "witness": None,
                    "contact_normal": None,
                    "triangle_index": None,
                }
            failed = []
            if not (float(state["penetration"]) <= max_contact_penetration + 1e-12):
                failed.append("depth")
            if not (float(state["patch_max_distance"]) <= contact_patch_radius + 1e-12):
                failed.append("patch_distance")
            if not (
                float(state["normal_alignment"])
                >= float(np.cos(np.deg2rad(contact_normal_tolerance_degrees))) - 1e-12
            ):
                failed.append("normal_alignment")
            point_reasons[str(state["kind"])] += 1
            point_failed.update(failed)
            all_reasons[str(state["kind"])] += 1
            all_failed.update(failed)
            if np.isfinite(state["penetration"]):
                point_penetrations.append(float(state["penetration"]))
                all_penetrations.append(float(state["penetration"]))
            if np.isfinite(state["patch_max_distance"]):
                point_patch.append(float(state["patch_max_distance"]))
                all_patch_distances.append(float(state["patch_max_distance"]))
            if np.isfinite(state["normal_alignment"]):
                point_normals.append(float(state["normal_alignment"]))
                all_normal_alignments.append(float(state["normal_alignment"]))
            point_records.append(
                {
                    "candidate_index": int(candidate_index),
                    "angle_deg": float(np.degrees(angles[candidate_index])),
                    **state,
                    "failed_conditions": failed,
                }
            )
        total_infeasible += len(point_records)
        per_point.append(
            {
                "point_index": int(point_index),
                "branch_id": int(branch_by_point.get(point_index, -1)),
                "infeasible_count": int(len(point_records)),
                "reasons": dict(point_reasons),
                "failed_conditions": dict(point_failed),
                "penetration": {
                    "min": float(np.min(point_penetrations)) if point_penetrations else None,
                    "median": float(np.median(point_penetrations)) if point_penetrations else None,
                    "max": float(np.max(point_penetrations)) if point_penetrations else None,
                },
                "patch_distance_max": float(np.max(point_patch)) if point_patch else None,
                "normal_alignment_min": float(np.min(point_normals)) if point_normals else None,
                "examples": point_records[: int(arguments.max_examples_per_point)],
            }
        )

    report = {
        "case_name": arguments.case,
        "source_result_dir": str(result_dir),
        "parameters": {
            "workpiece_mesh": str(workpiece_path),
            "radius": radius,
            "width": width,
            "angle_count_per_point": angle_count,
            "narrow_phase": narrow_phase,
            "contact_patch_enabled": contact_patch_enabled,
            "contact_patch_radius": contact_patch_radius,
            "max_contact_penetration": max_contact_penetration,
            "contact_normal_tolerance_degrees": contact_normal_tolerance_degrees,
            "allow_target_contact": allow_target_contact,
            "target_contact_tolerance": target_contact_tolerance,
        },
        "points_with_infeasible_candidates": [item["point_index"] for item in per_point],
        "total_infeasible_candidates": total_infeasible,
        "summary": {
            "by_reason": dict(all_reasons),
            "by_failed_condition": dict(all_failed),
            "penetration": {
                "min": float(np.min(all_penetrations)) if all_penetrations else None,
                "median": float(np.median(all_penetrations)) if all_penetrations else None,
                "max": float(np.max(all_penetrations)) if all_penetrations else None,
            },
            "patch_distance_max": float(np.max(all_patch_distances)) if all_patch_distances else None,
            "normal_alignment_min": float(np.min(all_normal_alignments)) if all_normal_alignments else None,
        },
        "per_point": per_point,
        "notes": [
            "depth = 穿透深度超过 contact_patch_max_penetration；",
            "patch_distance = 相交区域超出目标点附近的 contact_patch_radius；",
            "normal_alignment = 三角形法向与砂轮轴夹角超过 contact_patch_normal_tolerance_degrees；",
            "本报告只解释当前几何模型，不代表真实设备或工艺结论。",
        ],
    }
    output_dir = results_root / datetime.now().strftime("infeasible_audit_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / f"{arguments.case}_infeasible_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"points_with_infeasible={len(per_point)} total_infeasible={total_infeasible}")
    print(f"by_reason={dict(all_reasons)}")
    print(f"by_failed_condition={dict(all_failed)}")
    for item in per_point:
        print(
            f"point {item['point_index']:3d} br{item['branch_id']:2d} "
            f"n={item['infeasible_count']:3d} reasons={item['reasons']} "
            f"pen={item['penetration']} failed={item['failed_conditions']}"
        )
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
