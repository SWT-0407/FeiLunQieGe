"""阶段三：``leaf`` 单根部点的有限厚度圆柱连续碰撞审计。

本脚本只运行一个已经通过阶段二验收的案例和一个根部采样点。砂轮下表面
通过 ``h/2`` 轴向偏置与目标切平面共面；碰撞主判定是工件三角形连续区域
是否进入有限圆柱，旧的表面采样圆柱只作为对照。输出包含：

* ``stage3_finite_cylinder.obj/.mtl``：绿色可行中心、红色碰撞中心，以及
  一组绿色/红色代表性有限圆柱砂轮；
* ``stage3_finite_cylinder.png``：局部角度状态图；
* ``stage3_finite_cylinder_summary.json``：参数、后端、计数、原因和限制。

原始 OBJ/PLY 只读，输出目录使用时间戳，不覆盖历史结果。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from feilunqiege.collision.checker import build_collision_index, check_finite_cylinder_candidates
from feilunqiege.collision.cylinder_exact import (
    build_cylinder_triangle_index,
    check_finite_cylinder_candidates_exact,
)
from feilunqiege.geometry.curve import split_and_resample
from feilunqiege.geometry.frame import build_frames, estimate_surface_normals
from feilunqiege.geometry.orientation import orient_axes_to_flash_side
from feilunqiege.io.flash_cloud import load_labeled_flash_cloud
from feilunqiege.io.mesh import load_mesh
from feilunqiege.io.root_line import load_root_line
from feilunqiege.pipeline import _normalize_case_geometry
from feilunqiege.tooling.grinding_wheel import CandidateSet, generate_tangent_cylinder_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 leaf 阶段三有限圆柱连续碰撞审计")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--point-index", type=int, default=482)
    parser.add_argument("--angle-count", type=int, default=72)
    parser.add_argument("--boundary-iterations", type=int, default=10)
    parser.add_argument("--width", type=float, default=None, help="砂轮轴向厚度 h；默认读取 wheel.width")
    parser.add_argument("--radius", type=float, default=None, help="砂轮半径；默认读取 wheel.radius")
    parser.add_argument("--side", type=float, default=1.0, choices=(-1.0, 1.0), help="下表面相对切平面的轴向侧向")
    parser.add_argument("--epsilon", type=float, default=1e-10)
    parser.add_argument("--target-contact-tolerance", type=float, default=1e-4)
    return parser


def _candidate_set(contact: np.ndarray, frame: object, radius: float, width: float, angles: np.ndarray, point_index: int, side: float, axis_override: np.ndarray | None = None) -> CandidateSet:
    """在切平面内生成圆周候选，再统一加入 ``h/2`` 轴向偏置。"""

    return generate_tangent_cylinder_candidates(
        contact, frame, radius, width, angles, point_index,
        side=side, axis_reference=axis_override,
    )


def _to_source(point: np.ndarray, normalization: dict[str, object]) -> np.ndarray:
    scale = float(normalization.get("scale", 1.0))
    center = np.asarray(normalization.get("center", [0.0, 0.0, 0.0]), dtype=float)
    return np.asarray(point, dtype=float) * scale + center


def _write_materials(path: Path) -> None:
    path.write_text(
        "# Stage-3 finite-cylinder continuous collision audit materials\n"
        "newmtl feasible_green\nKa 0.10 0.95 0.28\nKd 0.10 0.95 0.28\nKe 0.10 0.95 0.28\nillum 0\n"
        "newmtl collision_red\nKa 1.00 0.05 0.05\nKd 1.00 0.05 0.05\nKe 1.00 0.05 0.05\nillum 0\n"
        "newmtl target_blue\nKa 0.05 0.30 1.00\nKd 0.05 0.30 1.00\nKe 0.05 0.30 1.00\nillum 0\n"
        "newmtl witness_magenta\nKa 0.90 0.05 0.90\nKd 0.90 0.05 0.90\nKe 0.90 0.05 0.90\nillum 0\n",
        encoding="ascii",
    )


def _write_vertex(handle: object, point: np.ndarray) -> None:
    handle.write(f"v {point[0]:.12g} {point[1]:.12g} {point[2]:.12g}\n")


def _write_octahedron(handle: object, center: np.ndarray, size: float, first: int) -> int:
    """写一个与坐标轴对齐的显示标记；它只表示中心状态，不是砂轮实体。"""

    vertices = np.array([
        center + [size, 0, 0], center + [-size, 0, 0],
        center + [0, size, 0], center + [0, -size, 0],
        center + [0, 0, size], center + [0, 0, -size],
    ])
    for vertex in vertices:
        _write_vertex(handle, vertex)
    faces = ((0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5))
    for face in faces:
        handle.write("f " + " ".join(str(first + index) for index in face) + "\n")
    return first + 6


def _write_cylinder(handle: object, center: np.ndarray, radial: np.ndarray, axis: np.ndarray, radius: float, width: float, first: int, segments: int = 32) -> int:
    """写一个有限圆柱显示实体，法向方向由候选姿态确定。"""

    radial = radial / np.linalg.norm(radial)
    axis = axis / np.linalg.norm(axis)
    # The disk section must lie in the plane perpendicular to the wheel axis.
    # The curve tangent is not a valid second disk basis for every candidate:
    # the candidate radial direction can contain a tangent component.
    disk_basis = np.cross(axis, radial)
    disk_basis_norm = np.linalg.norm(disk_basis)
    if disk_basis_norm <= 1e-12:
        raise ValueError("砂轮显示几何退化：radial 与 axis 不能平行")
    disk_basis /= disk_basis_norm
    half = width / 2.0
    rings = []
    for axial in (-half, half):
        ring = [center + axial * axis + radius * (np.cos(angle) * radial + np.sin(angle) * disk_basis) for angle in np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)]
        rings.append(ring)
    for ring in rings:
        for point in ring:
            _write_vertex(handle, point)
    bottom_center = center - half * axis
    top_center = center + half * axis
    _write_vertex(handle, bottom_center)
    _write_vertex(handle, top_center)
    bottom_index = first + 2 * segments
    top_index = bottom_index + 1
    for index in range(segments):
        following = (index + 1) % segments
        b0, b1 = first + index, first + following
        t0, t1 = first + segments + index, first + segments + following
        handle.write(f"f {b0} {b1} {t1} {t0}\n")
        handle.write(f"f {bottom_index} {b1} {b0}\n")
        handle.write(f"f {top_index} {t0} {t1}\n")
    return top_index + 1


def _select_representatives(centers: np.ndarray, feasible_indices: np.ndarray, collision_indices: np.ndarray) -> dict[str, int | None]:
    """选择彼此尽量分离的绿色/红色代表砂轮，避免两个实体重叠造成混色。

    代表砂轮仍然使用真实候选中心和姿态，不做显示偏移；这里只改变从候选集合
    中选哪两个姿态用于 MeshLab 的示例实体。所有候选中心的颜色分类仍完整导出。
    """

    if len(feasible_indices) == 0:
        feasible_index = None
    else:
        feasible_index = int(feasible_indices[0])
    if len(collision_indices) == 0:
        collision_index = None
    elif feasible_index is None:
        collision_index = int(collision_indices[0])
    else:
        distances = np.linalg.norm(
            centers[collision_indices] - centers[feasible_index][None, :], axis=1
        )
        collision_index = int(collision_indices[int(np.argmax(distances))])
    return {"feasible": feasible_index, "collision": collision_index}


def _export_obj(path: Path, candidates: CandidateSet, exact: object, sampled: object, normalization: dict[str, object], width: float, point_index: int) -> dict[str, int | None]:
    """输出绿色/红色中心和两个代表性有限圆柱砂轮。"""

    material_path = path.with_suffix(".mtl")
    _write_materials(material_path)
    feasible = np.asarray(exact.feasible, dtype=bool)
    centers = np.asarray(candidates.centers, dtype=float)
    contacts = np.asarray(candidates.contact_points, dtype=float)
    feasible_indices = np.flatnonzero(feasible)
    collision_indices = np.flatnonzero(~feasible)
    representatives = _select_representatives(centers, feasible_indices, collision_indices)
    scale = float(normalization.get("scale", 1.0))
    marker_size = max(float(candidates.radius) * scale * 0.04, 1e-8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# stage 3 finite-cylinder continuous collision audit\n")
        handle.write("# coordinate_frame: source_obj\n")
        handle.write(f"# wheel_width_normalized: {width:.12g}\n")
        handle.write(f"# root_line_point_index: {point_index}\n")
        handle.write(f"mtllib {material_path.name}\n")
        next_vertex = 1
        handle.write("o target_contact\nusemtl target_blue\n")
        next_vertex = _write_octahedron(handle, _to_source(contacts[0], normalization), marker_size * 1.5, next_vertex)
        for label, indices, material in (("feasible_centers", feasible_indices, "feasible_green"), ("collision_centers", collision_indices, "collision_red")):
            handle.write(f"o {label}\nusemtl {material}\n")
            for index in indices:
                next_vertex = _write_octahedron(handle, _to_source(centers[index], normalization), marker_size, next_vertex)
        for label, index, material in (("feasible_wheel", representatives["feasible"], "feasible_green"), ("collision_wheel", representatives["collision"], "collision_red")):
            if index is None:
                continue
            handle.write(f"o {label}\nusemtl {material}\n")
            next_vertex = _write_cylinder(
                handle,
                _to_source(centers[index], normalization),
                candidates.radius_directions[index], candidates.axis_directions[index],
                float(candidates.radial_values[index]) * scale, width * scale, next_vertex,
            )
        witnesses = np.asarray(exact.witness_workpiece, dtype=float)
        finite = np.flatnonzero(np.all(np.isfinite(witnesses), axis=1))
        if len(finite):
            handle.write("o collision_witnesses\nusemtl witness_magenta\n")
            for index in finite:
                next_vertex = _write_octahedron(handle, _to_source(witnesses[index], normalization), marker_size * 0.65, next_vertex)
    return representatives


def _plot_angle(path: Path, candidates: CandidateSet, exact: object, point_index: int, radius: float, width: float, side: float) -> None:
    """输出用户明确要求的绿色/红色局部角度图。"""

    feasible = np.asarray(exact.feasible, dtype=bool)
    angles = np.asarray(candidates.angles, dtype=float)
    figure, axis = plt.subplots(figsize=(8.0, 7.5), dpi=170, layout="constrained")
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    axis.plot(radius * np.cos(theta), radius * np.sin(theta), color="#4D5966", linewidth=1.0, label="candidate center radius")
    axis.scatter(radius * np.cos(angles[feasible]), radius * np.sin(angles[feasible]), c="#13B52A", s=30, label="feasible center / wheel")
    axis.scatter(radius * np.cos(angles[~feasible]), radius * np.sin(angles[~feasible]), c="#E31A1C", s=30, label="collision center / wheel")
    axis.scatter(0.0, 0.0, c="#0758D6", s=42, label="root-line sample point")
    axis.set_aspect("equal")
    limit = radius * 1.3
    axis.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="local tangent-plane basis 1 (normalized units)", ylabel="local tangent-plane basis 2 (normalized units)")
    axis.set_title(f"Stage 3 finite cylinder audit: leaf point {point_index}\nr={radius:g}, h={width:g}, lower-face side={side:+g}")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend(loc="upper right", fontsize=8)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def _refine_boundaries(contact: np.ndarray, frame: object, radius: float, width: float, side: float, angles: np.ndarray, states: np.ndarray, point_index: int, workpiece: object, cylinder_index: object, clearance: float, epsilon: float, target_contact_tolerance: float, iterations: int, target_contact_inner_radius: float = 0.0, axis_override: np.ndarray | None = None) -> list[dict[str, float | bool]]:
    """只对初始安全/碰撞状态变化的角区间做二分。"""

    boundaries: list[dict[str, float | bool]] = []
    count = len(angles)
    step = 2.0 * np.pi / count
    for index in range(count):
        left = float(angles[index])
        right = float(angles[(index + 1) % count]) + (2.0 * np.pi if index == count - 1 else 0.0)
        left_state = bool(states[index])
        right_state = bool(states[(index + 1) % count])
        if left_state == right_state:
            continue
        for _ in range(iterations):
            middle = (left + right) / 2.0
            candidate = _candidate_set(contact, frame, radius, width, np.array([middle % (2.0 * np.pi)]), point_index, side, axis_override)
            middle_state = bool(check_finite_cylinder_candidates_exact(candidate, workpiece, width, inner_radius=0.0, target_contact_inner_radius=target_contact_inner_radius, clearance=clearance, epsilon=epsilon, target_contact_tolerance=target_contact_tolerance, cylinder_index=cylinder_index).feasible[0])
            if middle_state == left_state:
                left = middle
            else:
                right = middle
        boundaries.append({"left_angle_rad": left % (2.0 * np.pi), "right_angle_rad": right % (2.0 * np.pi), "interval_width_rad": right - left, "left_feasible": left_state, "right_feasible": right_state, "initial_interval_rad": step})
    return boundaries


def _runs_from_boolean_circle(angles: np.ndarray, states: np.ndarray) -> list[dict[str, float | int | bool]]:
    """把固定角度状态压缩为环形区间，明确其仍依赖初始角度网格。"""

    transitions = np.flatnonzero(states != np.roll(states, 1))
    if len(transitions) == 0:
        return [{"feasible": bool(states[0]), "start_angle_rad": 0.0, "end_angle_rad": float(2.0 * np.pi), "sample_count": int(len(states))}]
    start = int(transitions[0])
    ordered = np.concatenate((np.arange(start, len(states)), np.arange(0, start)))
    runs: list[dict[str, float | int | bool]] = []
    run_start = 0
    for position in range(1, len(states) + 1):
        if position == len(states) or states[ordered[position]] != states[ordered[run_start]]:
            first = int(ordered[run_start])
            last = int(ordered[position - 1])
            runs.append({"feasible": bool(states[first]), "start_angle_rad": float(angles[first]), "end_angle_rad": float(angles[last] + 2.0 * np.pi / len(states)), "sample_count": position - run_start})
            run_start = position
    return runs


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.angle_count < 8 or arguments.boundary_iterations < 0 or arguments.epsilon < 0 or arguments.target_contact_tolerance < 0:
        raise ValueError("angle-count 至少为 8，boundary-iterations、epsilon 和 target-contact-tolerance 不能为负")
    config_path = (REPO_ROOT / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("case_name") != "leaf":
        raise ValueError("阶段三单案例入口固定使用 leaf")
    data_root = (config_path.parent / config["data_root"]).resolve()
    case_root = data_root / "leaf"
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / config["inputs"]["workpiece_mesh"])
    flash_cloud = load_labeled_flash_cloud(case_root / "original_flash_cloud.obj")
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    if not 0 <= arguments.point_index < len(sampled.points):
        raise ValueError(f"point-index 必须在 [0, {len(sampled.points) - 1}] 内")
    normal_config = config.get("normal_estimation", {})
    normals = estimate_surface_normals(
        sampled.points, workpiece,
        neighbor_count=int(normal_config.get("neighbor_count", 24)),
        candidate_count=int(normal_config.get("candidate_count", 96)),
        side_count=int(normal_config.get("side_count", 8)),
        coherence_angle_degrees=float(normal_config.get("coherence_angle_degrees", 25.0)),
        offset_epsilon=float(normal_config.get("offset_epsilon", 1e-5)),
        weight_distance_floor=float(normal_config.get("weight_distance_floor", 1e-5)),
    )
    frames = build_frames(sampled.points, sampled.tangents, normals)
    width = float(arguments.width if arguments.width is not None else config["wheel"].get("width", 0.004))
    radius = float(arguments.radius if arguments.radius is not None else config["wheel"]["radius"])
    delta = float(config["wheel"].get("delta", 0.0))
    if not (0.0 <= delta < radius):
        raise ValueError("wheel.delta must satisfy 0 <= delta < radius")
    center_source = np.asarray(normalization.get("center", [0.0, 0.0, 0.0]), dtype=float)
    scale_source = float(normalization.get("scale", 1.0))
    oriented_axes, orientation_audit = orient_axes_to_flash_side(
        sampled.points, sampled.tangents,
        np.asarray([frame.axis if frame.axis is not None else frame.surface_normal for frame in frames]),
        (flash_cloud.flash_points - center_source) / scale_source,
        (flash_cloud.body_points - center_source) / scale_source,
        sampled.branch_ids,
    )
    contact = np.asarray(sampled.points[arguments.point_index], dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, arguments.angle_count, endpoint=False)
    candidates = _candidate_set(contact, frames[arguments.point_index], radius, width, angles, arguments.point_index, arguments.side, oriented_axes[arguments.point_index])
    cylinder_index = build_cylinder_triangle_index(workpiece)
    exact = check_finite_cylinder_candidates_exact(
        candidates, workpiece, width,
        inner_radius=0.0,
        target_contact_inner_radius=radius - delta,
        clearance=float(config["collision"].get("clearance", 0.0)),
        epsilon=arguments.epsilon,
        allow_target_contact=bool(config["collision"].get("allow_target_contact", False)),
        target_contact_tolerance=arguments.target_contact_tolerance,
        cylinder_index=cylinder_index,
    )
    exact_states = np.asarray(exact.feasible, dtype=bool)
    boundary_refinements = _refine_boundaries(
        contact, frames[arguments.point_index], radius, width, arguments.side, angles, exact_states,
        arguments.point_index, workpiece, cylinder_index, float(config["collision"].get("clearance", 0.0)),
        arguments.epsilon, arguments.target_contact_tolerance, arguments.boundary_iterations, radius - delta, oriented_axes[arguments.point_index],
    )
    baseline = check_finite_cylinder_candidates(
        candidates, workpiece, float(config["collision"].get("clearance", 0.0)), width,
        collision_index=build_collision_index(workpiece),
        nearest_samples=int(config["collision"].get("finite_nearest_samples", 2048)),
        chunk_size=int(config["collision"].get("finite_chunk_size", 20000)),
    )
    output_root = REPO_ROOT / "results" / "cases" / "leaf" / datetime.now().strftime("stage3_cylinder_%Y%m%d_%H%M%S_%f")
    output_root.mkdir(parents=True, exist_ok=False)
    obj_path = output_root / "stage3_finite_cylinder.obj"
    png_path = output_root / "stage3_finite_cylinder.png"
    summary_path = output_root / "stage3_finite_cylinder_summary.json"
    representatives = _export_obj(obj_path, candidates, exact, sampled, normalization, width, arguments.point_index)
    _plot_angle(png_path, candidates, exact, arguments.point_index, radius, width, arguments.side)
    feasible = exact_states
    reason_values, reason_counts = np.unique(np.asarray(exact.reason, dtype=object), return_counts=True)
    summary = {
        "case_name": "leaf",
        "stage": "stage_3_finite_cylinder_continuous_collision",
        "units": "normalized",
        "point_index": arguments.point_index,
        "contact_point": contact.tolist(),
        "radius": radius,
        "width_h": width,
        "delta": delta,
        "contact_band": [radius - delta, radius],
        "candidate_center_radius": radius,
        "lower_face_offset": {"distance": width / 2.0, "axis_side": arguments.side, "formula": "center = x + rho*e + side*(h/2)*a"},
        "angle_count": arguments.angle_count,
        "boundary_bisection_iterations": arguments.boundary_iterations,
        "epsilon": arguments.epsilon,
        "clearance": float(config["collision"].get("clearance", 0.0)),
        "collision_backend": "exact_triangle_finite_cylinder_with_target_contact_band_aabb",
        "collision_geometry": "full physical cylinder; target contact exemption restricted to [r-delta, r]",
        "orientation": {"method": "branch_median_flash_body_anchor", "source": str(flash_cloud.source_path), "point": orientation_audit[arguments.point_index]},
        "broad_phase": {"type": "triangle_aabb_tree", "leaf_size": cylinder_index.leaf_size},
        "narrow_phase": "triangle axial clipping plus continuous minimum radial distance",
        "mesh": {"triangle_count": int(len(workpiece.faces)), "edge_manifold_closed": cylinder_index.mesh_is_closed, "containment_check": "vectorized ray parity when edge_manifold_closed=true"},
        "exact": {"candidate_count": int(len(feasible)), "feasible_count": int(np.count_nonzero(feasible)), "collision_count": int(np.count_nonzero(~feasible)), "reason_counts": {str(key): int(value) for key, value in zip(reason_values, reason_counts, strict=True)}, "initial_angle_runs": _runs_from_boolean_circle(angles, feasible), "boundary_refinements": boundary_refinements, "local_triangle_count": {"min": int(np.min(exact.witness_region)), "max": int(np.max(exact.witness_region)), "mean": float(np.mean(exact.witness_region))}},
        "baseline_surface_samples": {"method": baseline.method, "different_from_exact_count": int(np.count_nonzero(np.asarray(baseline.feasible, dtype=bool) != feasible))},
        "representative_indices": representatives,
        "outputs": {"obj": str(obj_path.relative_to(REPO_ROOT)), "mtl": str(obj_path.with_suffix(".mtl").relative_to(REPO_ROOT)), "png": str(png_path.relative_to(REPO_ROOT)), "summary": str(summary_path.relative_to(REPO_ROOT))},
        "human_review_status": "waiting_for_manual_review",
        "limitations": ["固定候选角度的三角形/圆柱连续判定，不等于全角度连续域证明。", "非封闭网格不能完成砂轮整体位于工件内部的实体包含证明。", "OBJ 中有限圆柱是可视化实体，不能替代设备可达性和真实加工验收。"],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status=completed case=leaf stage=3 exact_feasible={summary['exact']['feasible_count']}/{summary['exact']['candidate_count']}")
    print(f"obj={obj_path}")
    print(f"png={png_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
