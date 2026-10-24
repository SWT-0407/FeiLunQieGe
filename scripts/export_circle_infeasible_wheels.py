"""从 circle 结果中抽取不可行砂轮姿态并导出单个 OBJ。

选择规则：
- 读取最新一次 ``stage3_visualization_*`` 的可行中心 OBJ；
- 按 root_line 分支统计不可行候选；
- 在每个“存在不可行候选”的分支内，随机选一个目标点，再随机选一个
  不可行候选（固定随机种子，可复现）；
- 没有不可行候选的分支写入 JSON，不静默丢弃。

输出：
- ``circle_infeasible_wheel_examples.obj`` + ``.mtl``；
- ``circle_infeasible_wheel_examples.json`` 选择与原因摘要。

本脚本只做可视化，不改变碰撞判定，也不覆盖已有运行目录。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
    _branch_runs,
    _candidate_set,
    _to_source,
    _write_center_marker,
    _write_cylinder,
    _write_root_line,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出 circle 不可行砂轮姿态示例")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--case", default="circle")
    parser.add_argument("--workpiece-mesh", default="mesh_without_flash_simplified.obj")
    parser.add_argument("--result-dir", default=None, help="默认使用该案例最新 stage3_visualization_* 目录")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--max-per-branch", type=int, default=1)
    parser.add_argument("--cylinder-segments", type=int, default=64)
    return parser


def _latest_result_dir(case_root: Path) -> Path:
    runs = sorted((path for path in case_root.glob("stage3_visualization_*") if path.is_dir()), reverse=True)
    if not runs:
        raise FileNotFoundError(f"未找到 stage3_visualization_* 结果目录: {case_root}")
    return runs[0]


def _parse_collision_counts(obj_path: Path) -> dict[int, int]:
    """从已有可行中心 OBJ 中统计每个目标点的不可行候选数。"""

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


def _write_materials(path: Path) -> None:
    """无光照材质：红色不可行砂轮、灰色根部线、蓝色接触点、洋红见证点。"""

    path.write_text(
        "# circle infeasible wheel example materials\n"
        "newmtl collision_red\nKa 0.95 0.03 0.03\nKd 0.95 0.03 0.03\nKe 0.95 0.03 0.03\nillum 0\n"
        "newmtl root_line_gray\nKa 0.45 0.45 0.45\nKd 0.45 0.45 0.45\nillum 0\n"
        "newmtl contact_blue\nKa 0.03 0.20 0.95\nKd 0.03 0.20 0.95\nKe 0.03 0.20 0.95\nillum 0\n"
        "newmtl witness_magenta\nKa 0.90 0.05 0.90\nKd 0.90 0.05 0.90\nKe 0.90 0.05 0.90\nillum 0\n",
        encoding="ascii",
    )


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.max_per_branch < 1 or arguments.cylinder_segments < 8:
        raise ValueError("max-per-branch 至少为 1，cylinder-segments 至少为 8")
    config_path = (REPO_ROOT / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    case_root = (config_path.parent / config["data_root"]).resolve() / arguments.case
    results_root = REPO_ROOT / "results" / "cases" / arguments.case
    result_dir = (
        Path(arguments.result_dir).resolve()
        if arguments.result_dir is not None
        else _latest_result_dir(results_root)
    )
    source_obj = result_dir / f"{arguments.case}_feasible_center_domains.obj"
    if not source_obj.is_file():
        raise FileNotFoundError(f"未找到已有结果 OBJ: {source_obj}")
    source_summary_path = result_dir / "stage3_visualization_summary.json"
    source_summary = (
        json.loads(source_summary_path.read_text(encoding="utf-8"))
        if source_summary_path.is_file()
        else {}
    )
    # 必须与源结果使用相同的角度网格，否则抽到的候选无法与已有彩色中心对应。
    angle_count = int(source_summary.get("angle_count_per_point", config["sampling"]["candidate_angles"]))

    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / arguments.workpiece_mesh)
    flash_cloud = load_labeled_flash_cloud(case_root / "original_flash_cloud.obj")
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    normal_config = {**config.get("normal_estimation", {}), **source_summary.get("normal_estimation", {})}
    source_radius = float(source_summary.get("radius", config["wheel"]["radius"]))
    normal_kwargs = {
        key: normal_config[key]
        for key in ("neighbor_count", "candidate_count", "side_count", "coherence_angle_degrees",
                    "offset_epsilon", "weight_distance_floor")
        if key in normal_config
    }
    center_source = np.asarray(normalization.get("center", [0.0, 0.0, 0.0]), dtype=float)
    scale_source = float(normalization.get("scale", 1.0))
    flash_points = (flash_cloud.flash_points - center_source) / scale_source
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
        surface_patch_max_radius=float(normal_config.get("max_patch_radius_ratio", 1.0)) * source_radius,
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

    wheel = config["wheel"]
    radius = float(wheel["radius"])
    width = float(wheel["width"])
    delta = float(wheel.get("delta", 0.0))
    angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
    collision_config = config.get("collision", {})
    contact_patch_enabled = bool(collision_config.get("contact_patch_enabled", True))
    contact_patch_radius = float(collision_config.get("contact_patch_radius_ratio", 2.0)) * radius
    max_contact_penetration = float(collision_config.get("contact_patch_max_penetration_ratio", 0.025)) * radius
    contact_normal_tolerance_degrees = float(
        collision_config.get("contact_patch_normal_tolerance_degrees", 30.0)
    )
    narrow_phase = str(collision_config.get("narrow_phase", "analytic_closest_point_flat"))
    cylinder_index = build_cylinder_triangle_index(workpiece)

    collision_counts = _parse_collision_counts(source_obj)
    branches = _branch_runs(np.asarray(sampled.branch_ids, dtype=np.int64))
    rng = np.random.default_rng(int(arguments.seed))
    selected: list[dict[str, object]] = []
    branches_without_infeasible: list[int] = []

    for branch_id, start, end in branches:
        point_candidates = [index for index in range(start, end) if collision_counts.get(index, 0) > 0]
        if not point_candidates:
            branches_without_infeasible.append(int(branch_id))
            continue
        # 每个分支最多抽取 max_per_branch 个不同目标点，每个目标点随机一个不可行角度。
        chosen_points = rng.choice(point_candidates, size=min(int(arguments.max_per_branch), len(point_candidates)), replace=False)
        for point_index in np.atleast_1d(chosen_points).astype(int):
            candidates = _candidate_set(
                np.asarray(sampled.points[point_index]), frames[point_index], radius, width, angles,
                int(point_index), 1.0,
            )
            result = check_finite_cylinder_candidates_exact(
                candidates, workpiece, width,
                inner_radius=0.0,
                clearance=float(collision_config.get("clearance", 0.0)),
                allow_target_contact=bool(collision_config.get("allow_target_contact", True)),
                target_contact_tolerance=float(collision_config.get("target_contact_tolerance", 1e-4)),
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
            candidate_index = int(rng.choice(infeasible))
            witness = np.asarray(result.witness_workpiece[candidate_index], dtype=float)
            selected.append(
                {
                    "branch_id": int(branch_id),
                    "point_index": int(point_index),
                    "candidate_index": candidate_index,
                    "angle_rad": float(candidates.angles[candidate_index]),
                    "angle_deg": float(np.degrees(candidates.angles[candidate_index])),
                    "reason": str(result.reason[candidate_index]),
                    "collision_count_at_point": int(len(infeasible)),
                    "contact_point": np.asarray(candidates.contact_points[candidate_index], dtype=float).tolist(),
                    "wheel_center": np.asarray(candidates.centers[candidate_index], dtype=float).tolist(),
                    "witness_workpiece": witness.tolist() if np.all(np.isfinite(witness)) else None,
                }
            )

    if not selected:
        raise RuntimeError("没有找到任何不可行候选，无法导出示例")

    output_dir = results_root / datetime.now().strftime("infeasible_wheels_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    obj_path = output_dir / f"{arguments.case}_infeasible_wheel_examples.obj"
    mtl_path = obj_path.with_suffix(".mtl")
    summary_path = output_dir / f"{arguments.case}_infeasible_wheel_examples.json"
    _write_materials(mtl_path)

    marker_size = max(radius * scale_source * 0.35, 1e-8)
    with obj_path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# circle infeasible wheel examples\n")
        handle.write("# coordinate_frame: source_obj\n")
        handle.write(f"# source_result_dir: {result_dir}\n")
        handle.write(f"# selection_seed: {arguments.seed}\n")
        handle.write("# red = infeasible finite cylinder; blue = target/contact point; magenta = workpiece witness\n")
        handle.write(f"mtllib {mtl_path.name}\n")
        handle.write("o root_line\nusemtl root_line_gray\n")
        next_vertex = _write_root_line(handle, sampled, branches, normalization)
        for item in selected:
            branch_id = int(item["branch_id"])
            point_index = int(item["point_index"])
            candidate_index = int(item["candidate_index"])
            center = np.asarray(item["wheel_center"], dtype=float)
            contact = np.asarray(item["contact_point"], dtype=float)
            # 从候选集重新取姿态，确保 OBJ 中的圆柱方向与判定时完全一致。
            candidates = _candidate_set(
                np.asarray(sampled.points[point_index]), frames[point_index], radius, width, angles,
                point_index, 1.0,
            )
            radial = np.asarray(candidates.radius_directions[candidate_index], dtype=float)
            axis = np.asarray(candidates.axis_directions[candidate_index], dtype=float)
            handle.write(
                f"o infeasible_wheel_b{branch_id:02d}_p{point_index:04d}_a{item['angle_deg']:.1f}\n"
                "usemtl collision_red\n"
            )
            next_vertex = _write_cylinder(
                handle,
                _to_source(center, normalization),
                radial,
                axis,
                float(candidates.radial_values[candidate_index]) * scale_source,
                width * scale_source,
                next_vertex,
                int(arguments.cylinder_segments),
            )
            handle.write(
                f"o contact_b{branch_id:02d}_p{point_index:04d}\nusemtl contact_blue\n"
            )
            next_vertex = _write_center_marker(
                handle, _to_source(contact, normalization), marker_size, next_vertex
            )
            witness = item["witness_workpiece"]
            if witness is not None:
                handle.write(
                    f"o witness_b{branch_id:02d}_p{point_index:04d}\nusemtl witness_magenta\n"
                )
                next_vertex = _write_center_marker(
                    handle,
                    _to_source(np.asarray(witness, dtype=float), normalization),
                    marker_size * 0.7,
                    next_vertex,
                )

    summary = {
        "case_name": arguments.case,
        "source_result_dir": str(result_dir),
        "workpiece_mesh": str(case_root / arguments.workpiece_mesh),
        "selection_seed": int(arguments.seed),
        "angle_count_per_point": int(angle_count),
        "selection_rule": "one random infeasible candidate per root-line branch that has any infeasible candidate",
        "branches_with_infeasible_candidates": sorted({int(item["branch_id"]) for item in selected}),
        "branches_without_infeasible_candidates": branches_without_infeasible,
        "selected": selected,
        "outputs": {"obj": str(obj_path), "mtl": str(mtl_path)},
        "limitations": [
            "红色砂轮是当前碰撞判定下的不可行姿态，不代表真实设备运动学或工艺结论。",
            "只抽取了部分不可行姿态；不能据此统计不可行率。",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status=completed selected={len(selected)}")
    print(f"obj={obj_path}")
    print(f"mtl={mtl_path}")
    print(f"summary={summary_path}")
    print(f"branches_without_infeasible_candidates={branches_without_infeasible}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
