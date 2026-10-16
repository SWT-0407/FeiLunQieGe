"""阶段二：对 ``leaf`` 单个根部点运行无厚度圆盘连续碰撞审计。

本入口不改动主流程的有限厚度近似后端。它固定使用 ``leaf`` 和一个根部
采样点，输出：

* 基于 AABB 宽相和三角形连续窄相的无厚度圆盘结果；
* 与现有表面采样球形包络 baseline 的同角度对照；
* 状态变化角度的二分边界复核；
* 可供人工验收的局部角度图与 JSON 审计摘要。

连续判定只适用于每一个固定角度的三角形/圆盘相交；未被初始角采样命中的
窄角度碰撞区仍不能自动视为不存在，摘要会明确记录该边界。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from feilunqiege.collision.checker import build_collision_index, check_candidates
from feilunqiege.collision.disk_exact import build_disk_triangle_index, check_disk_candidates_exact
from feilunqiege.geometry.curve import split_and_resample
from feilunqiege.geometry.frame import build_frames, estimate_surface_normals
from feilunqiege.io.mesh import load_mesh
from feilunqiege.io.root_line import load_root_line
from feilunqiege.pipeline import _normalize_case_geometry
from feilunqiege.tooling.grinding_wheel import CandidateSet
from feilunqiege.visualization.export_obj import export_stage2_disk_audit_obj


def build_parser() -> argparse.ArgumentParser:
    """构造阶段二单案例参数。"""

    parser = argparse.ArgumentParser(description="运行 leaf 无厚度圆盘连续碰撞审计")
    parser.add_argument("--config", default="configs/default.json", help="配置文件路径")
    parser.add_argument("--point-index", type=int, default=482, help="固定 root_line 采样点")
    parser.add_argument("--initial-angles", type=int, default=72, help="初始等距角度数，至少为 8")
    parser.add_argument("--boundary-iterations", type=int, default=10, help="每个状态变化区间的二分次数")
    parser.add_argument(
        "--target-contact-tolerance",
        type=float,
        default=1e-4,
        help="允许 root_line 与工件网格配准误差的接触豁免半径（normalized units）",
    )
    return parser


def _candidate_set_for_angles(contact: np.ndarray, frame, radius: float, angles: np.ndarray, point_index: int) -> CandidateSet:
    """以指定连续角参数构造单根部点候选，避免把角度边界绑定到固定网格。"""

    tangent = np.asarray(frame.tangent, dtype=float)
    basis_1 = np.asarray(frame.radial_basis_1, dtype=float)
    basis_2 = np.asarray(frame.radial_basis_2, dtype=float)
    radial = np.cos(angles)[:, None] * basis_1[None, :] + np.sin(angles)[:, None] * basis_2[None, :]
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    tangents = np.repeat(tangent[None, :], len(angles), axis=0)
    axis = np.asarray(frame.axis if frame.axis is not None else frame.surface_normal, dtype=float)
    axis /= np.linalg.norm(axis)
    axes = np.repeat(axis[None, :], len(angles), axis=0)
    return CandidateSet(
        contact_points=np.repeat(contact[None, :], len(angles), axis=0),
        centers=contact[None, :] + float(radius) * radial,
        radius_directions=radial,
        tangent_directions=tangents,
        axis_directions=axes,
        point_indices=np.full(len(angles), point_index, dtype=np.int64),
        radial_values=np.full(len(angles), float(radius), dtype=float),
        angles=np.asarray(angles, dtype=float),
        radius=float(radius),
    )


def _refine_boundaries(
    contact: np.ndarray,
    frame,
    radius: float,
    initial_angles: np.ndarray,
    initial_feasible: np.ndarray,
    point_index: int,
    workpiece,
    disk_index,
    clearance: float,
    epsilon: float,
    allow_target_contact: bool,
    target_contact_tolerance: float,
    iterations: int,
) -> list[dict[str, float | bool]]:
    """只在相邻初始角状态不同处二分，定位可行/碰撞切换边界。

    这提升边界报告精度，但不会搜索整个角度连续域内从未被初始网格触及的
    狭窄状态岛，因此 JSON 将该限制与结果一起输出。
    """

    boundaries: list[dict[str, float | bool]] = []
    count = len(initial_angles)
    step = 2.0 * np.pi / count
    for index in range(count):
        left_angle = float(initial_angles[index])
        right_angle = float(initial_angles[(index + 1) % count])
        if index == count - 1:
            right_angle += 2.0 * np.pi
        left_state = bool(initial_feasible[index])
        right_state = bool(initial_feasible[(index + 1) % count])
        if left_state == right_state:
            continue
        for _ in range(iterations):
            midpoint = (left_angle + right_angle) / 2.0
            candidate = _candidate_set_for_angles(contact, frame, radius, np.array([midpoint % (2.0 * np.pi)]), point_index)
            middle_state = bool(
                check_disk_candidates_exact(
                    candidate,
                    workpiece,
                    clearance,
                    disk_index=disk_index,
                    epsilon=epsilon,
                    allow_target_contact=allow_target_contact,
                    target_contact_tolerance=target_contact_tolerance,
                ).feasible[0]
            )
            if middle_state == left_state:
                left_angle = midpoint
            else:
                right_angle = midpoint
        boundaries.append(
            {
                "left_angle_rad": left_angle % (2.0 * np.pi),
                "right_angle_rad": right_angle % (2.0 * np.pi),
                "interval_width_rad": right_angle - left_angle,
                "left_feasible": left_state,
                "right_feasible": right_state,
                "initial_interval_rad": step,
            }
        )
    return boundaries


def _runs_from_boolean_circle(angles: np.ndarray, values: np.ndarray) -> list[dict[str, float | int | bool]]:
    """把初始角度上的布尔状态压缩为环形连续段，用于人工查看而非全局证明。"""

    count = len(values)
    transitions = np.flatnonzero(values != np.roll(values, 1))
    if len(transitions) == 0:
        return [{"feasible": bool(values[0]), "start_angle_rad": 0.0, "end_angle_rad": float(2.0 * np.pi), "sample_count": count}]
    start = int(transitions[0])
    ordered = np.concatenate((np.arange(start, count), np.arange(0, start)))
    runs: list[dict[str, float | int | bool]] = []
    run_start = 0
    for position in range(1, count + 1):
        if position == count or values[ordered[position]] != values[ordered[run_start]]:
            first = int(ordered[run_start])
            last = int(ordered[position - 1])
            start_angle = float(angles[first])
            end_angle = float(angles[last] + 2.0 * np.pi / count)
            runs.append(
                {
                    "feasible": bool(values[first]),
                    "start_angle_rad": start_angle,
                    "end_angle_rad": end_angle,
                    "sample_count": position - run_start,
                }
            )
            run_start = position
    return runs


def _plot_local_angle_audit(output: Path, candidates: CandidateSet, exact, baseline, radius: float, point_index: int) -> None:
    """画出一个根部点的法平面候选圆：颜色编码连续判定，标记编码算法差异。"""

    local_x = radius * np.cos(candidates.angles)
    local_y = radius * np.sin(candidates.angles)
    exact_feasible = np.asarray(exact.feasible, dtype=bool)
    baseline_feasible = np.asarray(baseline.feasible, dtype=bool)
    mismatch = exact_feasible != baseline_feasible
    figure, axis = plt.subplots(figsize=(8.5, 8.0), dpi=170, layout="constrained")
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    axis.plot(radius * np.cos(theta), radius * np.sin(theta), color="#4D5966", linewidth=1.0, label="candidate radius r")
    axis.scatter(0.0, 0.0, c="black", s=42, label="root-line contact")
    axis.scatter(local_x[exact_feasible], local_y[exact_feasible], c="#009E73", s=30, label="exact triangle-disk clear")
    axis.scatter(local_x[~exact_feasible], local_y[~exact_feasible], c="#D55E00", marker="x", s=40, label="exact triangle-disk collision")
    if np.any(mismatch):
        axis.scatter(local_x[mismatch], local_y[mismatch], facecolors="none", edgecolors="#7B2CBF", marker="o", s=110, linewidths=1.7, label="baseline differs")
    axis.set_aspect("equal")
    limit = radius * 1.35
    axis.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="local radial basis 1 (normalized units)", ylabel="local radial basis 2 (normalized units)")
    axis.set_title(f"leaf stage-2 disk audit, root point {point_index}\nfixed-angle exact triangle/disk checks in the local tangent plane; purple rings mark baseline differences")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend(loc="upper right", fontsize=8)
    figure.savefig(output, dpi=170, facecolor="white")
    plt.close(figure)


def main() -> int:
    """运行 leaf 单点阶段二审计，并写入一个新的不可覆盖输出目录。"""

    arguments = build_parser().parse_args()
    if arguments.initial_angles < 8 or arguments.boundary_iterations < 0 or arguments.target_contact_tolerance < 0:
        raise ValueError("initial-angles 至少为 8，boundary-iterations 与 target-contact-tolerance 不能为负")
    config_path = (REPO_ROOT / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("case_name") != "leaf":
        raise ValueError("阶段二固定使用 leaf；请保持 configs/default.json 的 case_name 为 leaf")

    # 输入和归一化方式与主流程一致；原始 OBJ/PLY 始终只读。
    data_root = (config_path.parent / config["data_root"]).resolve()
    case_root = data_root / "leaf"
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / config["inputs"]["workpiece_mesh"])
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    if not 0 <= arguments.point_index < len(sampled.points):
        raise ValueError(f"point-index 必须在 [0, {len(sampled.points) - 1}] 内")
    surface_normals = estimate_surface_normals(sampled.points, workpiece)
    frames = build_frames(sampled.points, sampled.tangents, surface_normals)
    contact = np.asarray(sampled.points[arguments.point_index], dtype=float)
    frame = frames[arguments.point_index]
    radius = float(config["wheel"]["radius"])
    clearance = float(config["collision"]["clearance"])
    allow_target_contact = bool(config["collision"].get("allow_target_contact", False))
    epsilon = 1e-10
    angles = np.linspace(0.0, 2.0 * np.pi, arguments.initial_angles, endpoint=False)
    candidates = _candidate_set_for_angles(contact, frame, radius, angles, arguments.point_index)

    # 两类索引独立建立：KDTree 仅供旧 baseline 对照，AABB 树服务于新的连续窄相判定。
    start = time.perf_counter()
    sample_index = build_collision_index(workpiece)
    baseline = check_candidates(candidates, workpiece, clearance, collision_index=sample_index)
    baseline_seconds = time.perf_counter() - start
    start = time.perf_counter()
    disk_index = build_disk_triangle_index(workpiece)
    exact = check_disk_candidates_exact(
        candidates,
        workpiece,
        clearance,
        disk_index=disk_index,
        epsilon=epsilon,
        allow_target_contact=allow_target_contact,
        target_contact_tolerance=arguments.target_contact_tolerance,
    )
    exact_seconds = time.perf_counter() - start
    start = time.perf_counter()
    boundaries = _refine_boundaries(
        contact, frame, radius, angles, np.asarray(exact.feasible, dtype=bool), arguments.point_index, workpiece,
        disk_index, clearance, epsilon, allow_target_contact, arguments.target_contact_tolerance, arguments.boundary_iterations,
    )
    boundary_seconds = time.perf_counter() - start

    output_root = REPO_ROOT / "results" / "cases" / "leaf" / datetime.now().strftime("stage2_disk_%Y%m%d_%H%M%S_%f")
    output_root.mkdir(parents=True, exist_ok=False)
    image_path = output_root / "leaf_stage2_disk_angle_audit.png"
    summary_path = output_root / "leaf_stage2_disk_summary.json"
    display_path = output_root / "leaf_stage2_disk_audit.obj"
    _plot_local_angle_audit(image_path, candidates, exact, baseline, radius, arguments.point_index)
    export_stage2_disk_audit_obj(
        sampled,
        candidates,
        exact,
        baseline,
        display_path,
        point_index=arguments.point_index,
        coordinate_transform=normalization,
    )

    exact_feasible = np.asarray(exact.feasible, dtype=bool)
    baseline_feasible = np.asarray(baseline.feasible, dtype=bool)
    local_counts = np.asarray(exact.witness_region, dtype=np.int64)
    reason_values, reason_counts = np.unique(np.asarray(exact.reason, dtype=object), return_counts=True)
    summary = {
        "case_name": "leaf",
        "stage": "stage_2_zero_thickness_disk_continuous_collision",
        "units": "normalized",
        "point_index": arguments.point_index,
        "contact_point": contact.tolist(),
        "radius": radius,
        "clearance": clearance,
        "allow_target_contact": allow_target_contact,
        "target_contact_tolerance": arguments.target_contact_tolerance,
        "target_contact_policy": "only witnesses within tolerance of the root_line contact are exempt; all other triangle intersections remain collisions",
        "epsilon": epsilon,
        "initial_angle_count": arguments.initial_angles,
        "boundary_bisection_iterations": arguments.boundary_iterations,
        "collision_backend": "exact_triangle_disk_aabb",
        "broad_phase": {"type": "triangle_aabb_tree", "leaf_size": disk_index.leaf_size},
        "narrow_phase": "triangle-plane continuous intersection plus coplanar triangle/disk test",
        "mesh": {
            "vertex_count": int(len(workpiece.vertices)),
            "triangle_count": int(len(workpiece.faces)),
            "edge_manifold_closed": disk_index.mesh_is_closed,
            "containment_check": "center ray parity only when edge_manifold_closed=true",
        },
        "exact": {
            "candidate_count": int(len(exact_feasible)),
            "feasible_count": int(np.count_nonzero(exact_feasible)),
            "collision_count": int(np.count_nonzero(~exact_feasible)),
            "reason_counts": {str(reason): int(count) for reason, count in zip(reason_values, reason_counts, strict=True)},
            "initial_angle_runs": _runs_from_boolean_circle(angles, exact_feasible),
            "boundary_refinements": boundaries,
            "local_triangle_count": {"min": int(local_counts.min()), "max": int(local_counts.max()), "mean": float(local_counts.mean())},
            "seconds": exact_seconds,
        },
        "baseline_surface_samples": {
            "method": baseline.method,
            "feasible_count": int(np.count_nonzero(baseline_feasible)),
            "collision_count": int(np.count_nonzero(~baseline_feasible)),
            "different_from_exact_count": int(np.count_nonzero(exact_feasible != baseline_feasible)),
            "seconds": baseline_seconds,
        },
        "boundary_refinement_seconds": boundary_seconds,
        "normalization": normalization,
        "outputs": {
            "image": str(image_path.relative_to(REPO_ROOT)),
            "meshlab_overlay_obj": str(display_path.relative_to(REPO_ROOT)),
            "meshlab_overlay_mtl": str(display_path.with_suffix(".mtl").relative_to(REPO_ROOT)),
            "summary": str(summary_path.relative_to(REPO_ROOT)),
        },
        "limitations": [
            "连续结论针对固定候选角的三角形/圆盘相交，不是全角度连续域的全局安全证明。",
            "边界仅在初始角状态变化处二分；初始网格完全跳过的窄碰撞区仍需后续区间包络或更强搜索处理。",
            "实体包含检查仅在边连接封闭时启用；不检查自交、法向一致性或真实加工可达性。",
            "本阶段是无厚度圆盘，不代表有限厚度砂轮碰撞结果。",
        ],
        "human_review_status": "waiting_for_manual_review",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("status=completed case=leaf stage=2")
    print(f"exact_feasible={summary['exact']['feasible_count']}/{summary['exact']['candidate_count']}")
    print(f"baseline_difference={summary['baseline_surface_samples']['different_from_exact_count']}")
    print(f"image={image_path}")
    print(f"meshlab_overlay={display_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
