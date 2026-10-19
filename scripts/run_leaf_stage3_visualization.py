"""阶段三批量可视化：全部中心点，以及每条 root_line 的代表砂轮。

本脚本与 ``run_leaf_stage3_cylinder.py`` 使用相同的连续有限圆柱碰撞后端，
但可视化范围不同：

* 所有 root_line 采样点的全部候选中心都导出；可行中心为绿色，碰撞中心为红色。
* 每条 root_line 分支只选一个采样点；该点的可行砂轮整体为绿色，不可行砂轮整体为红色。
* 砂轮是完整有限圆柱对象。OBJ 必须用多边形面近似曲面，但不会按碰撞状态
  将同一个砂轮切成红绿扇区。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from feilunqiege.collision.cylinder_exact import (
    build_cylinder_triangle_index,
    check_finite_cylinder_candidates_exact,
    move_candidates_to_lower_face,
)
from feilunqiege.geometry.curve import split_and_resample
from feilunqiege.geometry.frame import build_frames, estimate_surface_normals
from feilunqiege.geometry.orientation import estimate_flash_body_directions, orient_axes_to_flash_side
from feilunqiege.io.flash_cloud import load_labeled_flash_cloud
from feilunqiege.io.mesh import load_mesh
from feilunqiege.io.root_line import load_root_line
from feilunqiege.pipeline import _normalize_case_geometry
from feilunqiege.tooling.grinding_wheel import CandidateSet


@dataclass(frozen=True)
class PointAudit:
    point_index: int
    candidates: CandidateSet
    result: object


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量导出阶段三中心点和代表砂轮可视化")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--angle-count", type=int, default=72)
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--side", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument("--epsilon", type=float, default=1e-10)
    parser.add_argument("--target-contact-tolerance", type=float, default=1e-4)
    parser.add_argument("--cylinder-segments", type=int, default=64)
    parser.add_argument("--png", action="store_true", help="显式请求额外生成 PNG；默认只输出 OBJ/MTL/JSON")
    return parser


def _candidate_set(contact: np.ndarray, frame: object, radius: float, width: float, angles: np.ndarray, point_index: int, side: float, axis_override: np.ndarray | None = None) -> CandidateSet:
    """在一个采样点生成候选姿态，并把中心移动到有限圆柱中心。"""

    tangent = np.asarray(frame.tangent, dtype=float)
    basis_1 = np.asarray(frame.radial_basis_1, dtype=float)
    basis_2 = np.asarray(frame.radial_basis_2, dtype=float)
    axis = np.asarray(axis_override if axis_override is not None else (frame.axis if frame.axis is not None else frame.surface_normal), dtype=float)
    axis /= np.linalg.norm(axis)
    radial = np.cos(angles)[:, None] * basis_1[None, :] + np.sin(angles)[:, None] * basis_2[None, :]
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    candidates = CandidateSet(
        contact_points=np.repeat(contact[None, :], len(angles), axis=0),
        centers=contact[None, :] + float(radius) * radial,
        radius_directions=radial,
        tangent_directions=np.repeat(tangent[None, :], len(angles), axis=0),
        axis_directions=np.repeat(axis[None, :], len(angles), axis=0),
        point_indices=np.full(len(angles), point_index, dtype=np.int64),
        radial_values=np.full(len(angles), float(radius), dtype=float),
        angles=np.asarray(angles, dtype=float),
        radius=float(radius),
    )
    return move_candidates_to_lower_face(candidates, width, side)


def _to_source(point: np.ndarray, normalization: dict[str, object]) -> np.ndarray:
    return np.asarray(point, dtype=float) * float(normalization.get("scale", 1.0)) + np.asarray(normalization.get("center", [0.0, 0.0, 0.0]), dtype=float)


def _write_materials(path: Path) -> None:
    path.write_text(
        "# Stage-3 batch visualization materials\n"
        "newmtl feasible_green\nKa 0.05 0.95 0.15\nKd 0.05 0.95 0.15\nKe 0.05 0.95 0.15\nillum 0\n"
        "newmtl collision_red\nKa 0.95 0.03 0.03\nKd 0.95 0.03 0.03\nKe 0.95 0.03 0.03\nillum 0\n"
        "newmtl root_line_gray\nKa 0.45 0.45 0.45\nKd 0.45 0.45 0.45\nillum 0\n"
        "newmtl contact_blue\nKa 0.03 0.20 0.95\nKd 0.03 0.20 0.95\nKe 0.03 0.20 0.95\nillum 0\n"
        "newmtl circle_center_yellow\nKa 1.0 0.85 0.0\nKd 1.0 0.85 0.0\nKe 1.0 0.85 0.0\nillum 0\n",
        encoding="ascii",
    )


def _write_vertex(handle: object, point: np.ndarray) -> None:
    handle.write(f"v {point[0]:.12g} {point[1]:.12g} {point[2]:.12g}\n")


def _write_center_marker(handle: object, center: np.ndarray, size: float, first: int) -> int:
    """用小八面体标记中心，颜色由调用方的 usemtl 决定。"""

    vertices = np.array([
        center + [size, 0.0, 0.0], center + [-size, 0.0, 0.0],
        center + [0.0, size, 0.0], center + [0.0, -size, 0.0],
        center + [0.0, 0.0, size], center + [0.0, 0.0, -size],
    ])
    for vertex in vertices:
        _write_vertex(handle, vertex)
    faces = ((0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5))
    for face in faces:
        handle.write("f " + " ".join(str(first + index) for index in face) + "\n")
    return first + 6


def _write_cylinder(handle: object, center: np.ndarray, radial: np.ndarray, axis: np.ndarray, radius: float, width: float, first: int, segments: int) -> int:
    """写出一个完整有限圆柱；整个对象只由调用方指定的一种材质着色。"""

    radial = radial / np.linalg.norm(radial)
    axis = axis / np.linalg.norm(axis)
    disk_basis = np.cross(axis, radial)
    disk_basis /= np.linalg.norm(disk_basis)
    half = width / 2.0
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    rings = []
    for axial in (-half, half):
        ring = [center + axial * axis + radius * (np.cos(angle) * radial + np.sin(angle) * disk_basis) for angle in theta]
        rings.append(ring)
    for ring in rings:
        for point in ring:
            _write_vertex(handle, point)
    bottom_index = first + 2 * segments
    top_index = bottom_index + 1
    _write_vertex(handle, center - half * axis)
    _write_vertex(handle, center + half * axis)
    for index in range(segments):
        following = (index + 1) % segments
        b0, b1 = first + index, first + following
        t0, t1 = first + segments + index, first + segments + following
        handle.write(f"f {b0} {b1} {t1} {t0}\n")
        handle.write(f"f {bottom_index} {b1} {b0}\n")
        handle.write(f"f {top_index} {t0} {t1}\n")
    return top_index + 1


def _branch_runs(branch_ids: np.ndarray) -> list[tuple[int, int, int]]:
    if len(branch_ids) == 0:
        return []
    starts = [0] + (np.flatnonzero(branch_ids[1:] != branch_ids[:-1]) + 1).tolist()
    ends = starts[1:] + [len(branch_ids)]
    return [(int(branch_ids[start]), start, end) for start, end in zip(starts, ends, strict=True)]


def _select_branch_point(indices: list[int], audits: list[PointAudit]) -> tuple[int, int | None, int | None]:
    """优先选择同时存在可行和碰撞姿态的采样点，否则选择分支中部。"""

    best = None
    for point_index in indices:
        audit = audits[point_index]
        feasible = np.flatnonzero(np.asarray(audit.result.feasible, dtype=bool))
        collision = np.flatnonzero(~np.asarray(audit.result.feasible, dtype=bool))
        score = (int(len(feasible) > 0 and len(collision) > 0), -abs(point_index - indices[len(indices) // 2]))
        if best is None or score > best[0]:
            best = (score, point_index, feasible, collision)
    assert best is not None
    _, point_index, feasible, collision = best
    feasible_index = int(feasible[0]) if len(feasible) else None
    if len(collision) and feasible_index is not None:
        centers = np.asarray(audits[point_index].candidates.centers)
        collision_index = int(collision[np.argmax(np.linalg.norm(centers[collision] - centers[feasible_index], axis=1))])
    else:
        collision_index = int(collision[0]) if len(collision) else None
    return point_index, feasible_index, collision_index


def _export_obj_legacy(path: Path, sampled: object, audits: list[PointAudit], branches: list[tuple[int, int, int]], branch_representatives: list[dict[str, int | None]], normalization: dict[str, object], radius: float, width: float, cylinder_segments: int) -> None:
    material_path = path.with_suffix(".mtl")
    _write_materials(material_path)
    scale = float(normalization.get("scale", 1.0))
    marker_size = max(radius * scale * 0.025, 1e-8)
    points = np.asarray(sampled.points, dtype=float)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# stage 3 batch visualization\n# coordinate_frame: source_obj\n")
        handle.write(f"# circle_center_marker_count: {len(audits)}\n")
        handle.write(f"# root_line_branch_count: {len(branches)}\n")
        handle.write("# wheel geometry omitted; OBJ contains root_line and center-point markers only\n")
        handle.write(f"mtllib {material_path.name}\n")
        next_vertex = 1
        handle.write("o root_line\nusemtl root_line_gray\n")
        root_ids = []
        for point in points:
            root_ids.append(next_vertex)
            _write_vertex(handle, _to_source(point, normalization))
            next_vertex += 1
        for _branch_id, start, end in branches:
            for index in range(start, end - 1):
                handle.write(f"l {root_ids[index]} {root_ids[index + 1]}\n")

        # 每个采样点的候选中心构成圆（或有限厚度模型中沿轴向平移后的圆）。
        # 对同一采样点的全部角度候选取均值，得到该圆的几何圆心；只输出
        # 黄色圆心标记，不输出任何砂轮实体或圆周候选点。
        handle.write("o rootline_circle_centers\nusemtl circle_center_yellow\n")
        for audit in audits:
            candidates = np.asarray(audit.candidates.centers, dtype=float)
            center = _to_source(candidates.mean(axis=0), normalization)
            next_vertex = _write_center_marker(handle, center, marker_size * 1.5, next_vertex)

        """Wheel geometry is intentionally omitted; center markers above are the only pose visualization."""
        """
        if False:
            for representative in branch_representatives:
            point_index = representative["point_index"]
            audit = audits[point_index]
            for local_index, material, label in ((representative["feasible_index"], "feasible_green", "feasible_wheel"), (representative["collision_index"], "collision_red", "collision_wheel")):
                if local_index is None:
                    continue
                candidate = audit.candidates
                handle.write(f"o {label}_branch_{representative['branch_id']:04d}_point_{point_index:04d}\nusemtl {material}\n")
                next_vertex = _write_cylinder(
                    handle,
                    _to_source(candidate.centers[local_index], normalization),
                    candidate.radius_directions[local_index], candidate.axis_directions[local_index],
                    float(candidate.radial_values[local_index]) * scale, width * scale, next_vertex, cylinder_segments,
                )
                handle.write(f"o contact_branch_{representative['branch_id']:04d}_point_{point_index:04d}\nusemtl contact_blue\n")
                _write_vertex(handle, _to_source(candidate.contact_points[local_index], normalization))
                handle.write(f"p {next_vertex}\n")
                next_vertex += 1

        # 为每个 root_line 采样点保留一个代表砂轮，便于在 MeshLab 中检查整条
        # 分支的姿态连续性。优先选择可行候选；若该点没有可行候选，则保留
        # 第一个碰撞候选并以红色标记，不能静默丢弃无解点。
            for audit in audits:
            feasible = np.flatnonzero(np.asarray(audit.result.feasible, dtype=bool))
            local_index = int(feasible[0]) if len(feasible) else 0
            material = "feasible_green" if len(feasible) else "collision_red"
            candidate = audit.candidates
            handle.write(f"o wheel_point_{audit.point_index:04d}\nusemtl {material}\n")
            next_vertex = _write_cylinder(
                handle,
                _to_source(candidate.centers[local_index], normalization),
                candidate.radius_directions[local_index], candidate.axis_directions[local_index],
                float(candidate.radial_values[local_index]) * scale, width * scale, next_vertex, cylinder_segments,
            )
        """


def _write_root_line(handle: object, sampled: object, branches: list[tuple[int, int, int]], normalization: dict[str, object], first: int = 1) -> int:
    """写出根部线并返回下一个可用 OBJ 顶点编号。"""
    next_vertex = first
    root_ids = []
    for point in np.asarray(sampled.points, dtype=float):
        root_ids.append(next_vertex)
        _write_vertex(handle, _to_source(point, normalization))
        next_vertex += 1
    for _branch_id, start, end in branches:
        for index in range(start, end - 1):
            handle.write(f"l {root_ids[index]} {root_ids[index + 1]}\n")
    return next_vertex


def _export_circle_centers(path: Path, sampled: object, audits: list[PointAudit], branches: list[tuple[int, int, int]], normalization: dict[str, object], radius: float) -> None:
    """导出每个采样点对应的黄色圆心，候选中心半径固定为 r。"""
    material_path = path.with_suffix(".mtl")
    _write_materials(material_path)
    scale = float(normalization.get("scale", 1.0))
    marker_size = max(radius * scale * 0.025, 1e-8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# stage 3 circle centers\n# coordinate_frame: source_obj\n")
        handle.write(f"# circle_center_marker_count: {len(audits)}\nmtllib {material_path.name}\n")
        next_vertex = _write_root_line(handle, sampled, branches, normalization)
        handle.write("o rootline_circle_centers\nusemtl circle_center_yellow\n")
        for audit in audits:
            center = _to_source(np.asarray(audit.candidates.centers).mean(axis=0), normalization)
            next_vertex = _write_center_marker(handle, center, marker_size * 1.5, next_vertex)


def _export_center_domains(path: Path, sampled: object, audits: list[PointAudit], branches: list[tuple[int, int, int]], normalization: dict[str, object], radius: float) -> None:
    """导出所有候选中心：绿色为可行，红色为碰撞。"""
    material_path = path.with_suffix(".mtl")
    _write_materials(material_path)
    scale = float(normalization.get("scale", 1.0))
    marker_size = max(radius * scale * 0.012, 1e-8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# stage 3 all candidate center domains\n# coordinate_frame: source_obj\n")
        handle.write(f"# candidate_center_radius: {radius:.12g}\nmtllib {material_path.name}\n")
        next_vertex = _write_root_line(handle, sampled, branches, normalization)
        for audit in audits:
            centers = np.asarray(audit.candidates.centers, dtype=float)
            feasible = np.asarray(audit.result.feasible, dtype=bool)
            for state, material, label in ((feasible, "feasible_green", "feasible_centers"), (~feasible, "collision_red", "collision_centers")):
                handle.write(f"o {label}_point_{audit.point_index:04d}\nusemtl {material}\n")
                for local_index in np.flatnonzero(state):
                    next_vertex = _write_center_marker(handle, _to_source(centers[int(local_index)], normalization), marker_size, next_vertex)


def _export_random_wheels(path: Path, sampled: object, audits: list[PointAudit], branches: list[tuple[int, int, int]], normalization: dict[str, object], radius: float, width: float, cylinder_segments: int, seed: int = 20261018, per_state: int = 2) -> list[dict[str, object]]:
    """每个分支随机抽取少量可行/碰撞姿态，输出完整标准圆柱。"""
    material_path = path.with_suffix(".mtl")
    _write_materials(material_path)
    rng = np.random.default_rng(seed)
    selected: list[tuple[int, int, bool]] = []
    for _branch_id, start, end in branches:
        for feasible_state in (True, False):
            pool = [point for point in range(start, end) if bool(np.any(audits[point].result.feasible) == feasible_state)]
            for point_index in np.atleast_1d(rng.choice(pool, size=min(per_state, len(pool)), replace=False)) if pool else []:
                flags = np.asarray(audits[int(point_index)].result.feasible, dtype=bool)
                local_pool = np.flatnonzero(flags if feasible_state else ~flags)
                if len(local_pool):
                    selected.append((int(point_index), int(rng.choice(local_pool)), feasible_state))
    records: list[dict[str, object]] = []
    scale = float(normalization.get("scale", 1.0))
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# stage 3 random complete wheel shapes\n# coordinate_frame: source_obj\n")
        handle.write(f"# random_seed: {seed}\nmtllib {material_path.name}\n")
        next_vertex = _write_root_line(handle, sampled, branches, normalization)
        for serial, (point_index, local_index, is_feasible) in enumerate(selected):
            candidate = audits[point_index].candidates
            handle.write(f"o wheel_{'feasible' if is_feasible else 'collision'}_{serial:03d}_point_{point_index:04d}\nusemtl {'feasible_green' if is_feasible else 'collision_red'}\n")
            next_vertex = _write_cylinder(handle, _to_source(candidate.centers[local_index], normalization), candidate.radius_directions[local_index], candidate.axis_directions[local_index], radius * scale, width * scale, next_vertex, cylinder_segments)
            records.append({"serial": serial, "point_index": point_index, "angle": float(candidate.angles[local_index]), "feasible": is_feasible, "reason": str(audits[point_index].result.reason[local_index])})
    return records


def _plot_centers(path: Path, sampled: object, audits: list[PointAudit]) -> None:
    points = np.asarray(sampled.points, dtype=float)
    branch_ids = np.asarray(sampled.branch_ids, dtype=np.int64)
    figure, axis = plt.subplots(figsize=(10, 8), dpi=160, layout="constrained")
    # 分支之间没有几何连接，必须分别绘制，避免总览图出现跨分支的假连线。
    for branch_id in np.unique(branch_ids):
        branch_points = points[branch_ids == branch_id]
        axis.plot(branch_points[:, 0], branch_points[:, 1], color="#555555", linewidth=0.8, alpha=0.7)
    for audit in audits:
        centers = np.asarray(audit.candidates.centers)
        feasible = np.asarray(audit.result.feasible, dtype=bool)
        axis.scatter(centers[feasible, 0], centers[feasible, 1], c="#10B832", s=2, alpha=0.55)
        axis.scatter(centers[~feasible, 0], centers[~feasible, 1], c="#E31A1C", s=3, alpha=0.75)
    axis.set_aspect("equal")
    axis.set_title("Stage 3 all root-line samples: feasible green / collision red")
    axis.set_xlabel("normalized x")
    axis.set_ylabel("normalized y")
    axis.grid(True, linewidth=0.4, alpha=0.3)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def _plot_leaf_projection(path: Path, workpiece: object, sampled: object) -> None:
    """绘制 leaf 工件投影，并标注每条 root_line 的首尾采样编号。"""
    vertices = np.asarray(workpiece.vertices, dtype=float)
    faces = np.asarray(workpiece.faces, dtype=np.int64)
    center = vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(vertices - center, full_matrices=False)
    basis = vh[:2]
    projected_vertices = (vertices - center) @ basis.T
    projected_root = (np.asarray(sampled.points, dtype=float) - center) @ basis.T
    stride = max(1, len(faces) // 18000)
    polygons = [projected_vertices[face] for face in faces[::stride]]
    figure, axis = plt.subplots(figsize=(12, 10), dpi=180, layout="constrained")
    axis.add_collection(PolyCollection(polygons, facecolor="#d6d9df", edgecolor="#9aa0aa", linewidth=0.08, alpha=0.78))
    branches = _branch_runs(np.asarray(sampled.branch_ids, dtype=np.int64))
    colors = plt.get_cmap("tab10")
    for branch_id, start, end in branches:
        branch = projected_root[start:end]
        color = colors(branch_id % 10)
        axis.plot(branch[:, 0], branch[:, 1], color=color, linewidth=1.8, zorder=4)
        axis.scatter([branch[0, 0], branch[-1, 0]], [branch[0, 1], branch[-1, 1]], color=["#1565c0", "#c62828"], s=28, zorder=5)
        axis.annotate(f"B{branch_id} start: {start}", branch[0], xytext=(5, 6), textcoords="offset points", fontsize=8, color="#1565c0")
        axis.annotate(f"B{branch_id} end: {end - 1}", branch[-1], xytext=(5, -14), textcoords="offset points", fontsize=8, color="#c62828")
    low = projected_vertices.min(axis=0)
    high = projected_vertices.max(axis=0)
    margin = max(float(np.max(high - low)) * 0.04, 1e-6)
    axis.set_xlim(low[0] - margin, high[0] + margin)
    axis.set_ylim(low[1] - margin, high[1] + margin)
    axis.set_aspect("equal")
    axis.set_title("Leaf workpiece projection with root_line branch endpoints")
    axis.set_xlabel("PCA projection axis 1 (normalized units)")
    axis.set_ylabel("PCA projection axis 2 (normalized units)")
    axis.grid(True, linewidth=0.4, alpha=0.35)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.angle_count < 8 or arguments.cylinder_segments < 16:
        raise ValueError("angle-count 至少为 8，cylinder-segments 至少为 16")
    config_path = (REPO_ROOT / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("case_name") != "leaf":
        raise ValueError("批量阶段三入口固定使用 leaf")
    case_root = (config_path.parent / config["data_root"]).resolve() / "leaf"
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / config["inputs"]["workpiece_mesh"])
    # 飞边网格用于结果叠加/来源记录；碰撞主体仍保持为 mesh_without_flash。
    flash_reference_mesh = load_mesh(case_root / "mesh_with_flash.obj")
    flash_cloud = load_labeled_flash_cloud(case_root / "original_flash_cloud.obj")
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    normal_config = config.get("normal_estimation", {})
    normal_kwargs = {key: normal_config[key] for key in ("neighbor_count", "candidate_count", "side_count", "coherence_angle_degrees", "offset_epsilon", "weight_distance_floor", "continuity_jump_threshold_degrees", "continuity_normal_weight", "continuity_patch_weight", "continuity_distance_weight") if key in normal_config}
    normal_kwargs["branch_ids"] = sampled.branch_ids
    width = float(arguments.width if arguments.width is not None else config["wheel"].get("width", 0.004))
    radius = float(arguments.radius if arguments.radius is not None else config["wheel"]["radius"])
    delta = float(config["wheel"].get("delta", 0.0))
    if not (0.0 <= delta < radius):
        raise ValueError("wheel.delta must satisfy 0 <= delta < radius")
    center_source = np.asarray(normalization.get("center", [0.0, 0.0, 0.0]), dtype=float)
    scale_source = float(normalization.get("scale", 1.0))
    flash_points = (flash_cloud.flash_points - center_source) / scale_source
    body_points = (flash_cloud.body_points - center_source) / scale_source
    side_hints, side_hint_diagnostics = estimate_flash_body_directions(
        sampled.points, sampled.tangents, flash_points, body_points,
        neighbor_count=int(normal_config.get("flash_neighbor_count", 32)),
    )
    normal_kwargs["side_direction_hints"] = side_hints
    normal_kwargs["side_direction_weight"] = float(normal_config.get("side_direction_weight", 2.0))
    normals, normal_diagnostics = estimate_surface_normals(sampled.points, workpiece, return_diagnostics=True, **normal_kwargs)
    frames = build_frames(sampled.points, sampled.tangents, normals)
    frame_axes = np.asarray([frame.axis if frame.axis is not None else frame.surface_normal for frame in frames], dtype=float)
    oriented_axes, orientation_audit = orient_axes_to_flash_side(
        sampled.points, sampled.tangents, frame_axes, flash_points, body_points, sampled.branch_ids,
        neighbor_count=int(normal_config.get("flash_neighbor_count", 32)),
        confidence_threshold=float(normal_config.get("flash_confidence_threshold", 0.08)),
    )
    # Axis orientation and both radial basis vectors must come from the same
    # final normal field; mixing an old basis with a flipped axis changes angle
    # labels and can make a correct pose appear rotated.
    frames = build_frames(sampled.points, sampled.tangents, oriented_axes)
    angles = np.linspace(0.0, 2.0 * np.pi, arguments.angle_count, endpoint=False)
    cylinder_index = build_cylinder_triangle_index(workpiece)
    audits: list[PointAudit] = []
    for point_index, (point, frame) in enumerate(zip(sampled.points, frames, strict=True)):
        candidates = _candidate_set(np.asarray(point), frame, radius, width, angles, point_index, arguments.side)
        result = check_finite_cylinder_candidates_exact(
            candidates, workpiece, width,
            inner_radius=radius - delta,
            clearance=float(config["collision"].get("clearance", 0.0)), epsilon=arguments.epsilon,
            allow_target_contact=bool(config["collision"].get("allow_target_contact", False)),
            target_contact_tolerance=arguments.target_contact_tolerance, cylinder_index=cylinder_index,
        )
        audits.append(PointAudit(point_index, candidates, result))
        if (point_index + 1) % 25 == 0:
            print(f"audited_points={point_index + 1}/{len(sampled.points)}", flush=True)

    branches = _branch_runs(np.asarray(sampled.branch_ids, dtype=np.int64))
    representatives = []
    for branch_id, start, end in branches:
        point_index, feasible_index, collision_index = _select_branch_point(list(range(start, end)), audits)
        representatives.append({"branch_id": branch_id, "point_index": point_index, "feasible_index": feasible_index, "collision_index": collision_index})

    output_root = REPO_ROOT / "results" / "cases" / "leaf" / datetime.now().strftime("stage3_visualization_%Y%m%d_%H%M%S_%f")
    output_root.mkdir(parents=True, exist_ok=False)
    centers_obj_path = output_root / "leaf_rootline_circle_centers.obj"
    domains_obj_path = output_root / "leaf_feasible_center_domains.obj"
    wheels_obj_path = output_root / "leaf_random_wheel_shapes.obj"
    png_path = output_root / "leaf_rootline_projection.png"
    summary_path = output_root / "stage3_visualization_summary.json"
    normal_diagnostics_path = output_root / "normal_continuity_diagnostics.json"
    _export_circle_centers(centers_obj_path, sampled, audits, branches, normalization, radius)
    _export_center_domains(domains_obj_path, sampled, audits, branches, normalization, radius)
    wheel_records = _export_random_wheels(wheels_obj_path, sampled, audits, branches, normalization, radius, width, arguments.cylinder_segments)
    _plot_leaf_projection(png_path, workpiece, sampled)
    feasible_count = sum(int(np.count_nonzero(a.result.feasible)) for a in audits)
    candidate_count = sum(len(a.result.feasible) for a in audits)
    normal_diagnostics_path.write_text(json.dumps({
        "point_count": len(normal_diagnostics),
        "branch_count": len(branches),
        "continuity_threshold_degrees": normal_config.get("continuity_jump_threshold_degrees", 20.0),
        "low_confidence_point_count": sum(bool(item.get("continuity_state", {}).get("low_confidence", False)) for item in normal_diagnostics),
        "points": normal_diagnostics,
        "flash_side_hint_diagnostics": side_hint_diagnostics,
        "flash_side_orientation": orientation_audit,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "case_name": "leaf", "stage": "stage_3_batch_visualization", "units": "normalized",
        "sampled_point_count": len(audits), "root_line_branch_count": len(branches),
        "angle_count_per_point": arguments.angle_count, "candidate_center_count": candidate_count,
        "feasible_center_count": feasible_count, "collision_center_count": candidate_count - feasible_count,
        "radius": radius, "delta": delta, "contact_band": [radius - delta, radius], "width_h": width, "lower_face_offset": width / 2.0,
        "collision_backend": "exact_triangle_finite_annular_cylinder_aabb",
        "candidate_center_radius": radius,
        "wheel_visualization": {"center_material": "circle_center_yellow", "feasible_material": "feasible_green", "collision_material": "collision_red", "random_seed": 20261018, "random_wheel_count": len(wheel_records)},
        "orientation": {"method": "branch_median_flash_body_anchor", "source": str(flash_cloud.source_path), "reference_mesh": str(case_root / "mesh_with_flash.obj"), "reference_mesh_vertex_count": int(len(flash_reference_mesh.vertices))},
        "branch_representatives": representatives,
        "random_wheels": wheel_records,
        "outputs": {"circle_centers_obj": str(centers_obj_path.relative_to(REPO_ROOT)), "center_domains_obj": str(domains_obj_path.relative_to(REPO_ROOT)), "random_wheels_obj": str(wheels_obj_path.relative_to(REPO_ROOT)), "png": str(png_path.relative_to(REPO_ROOT)), "normal_diagnostics": str(normal_diagnostics_path.relative_to(REPO_ROOT)), "summary": str(summary_path.relative_to(REPO_ROOT))},
        "human_review_status": "waiting_for_manual_review",
        "limitations": ["OBJ 使用多边形面表达标准圆柱曲面，但每个砂轮对象只有一种材质颜色。", "连续碰撞结果仍受当前网格封闭性、容差和固定角度候选集限制。"],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status=completed points={len(audits)} branches={len(branches)} centers={candidate_count} feasible={feasible_count}")
    print(f"circle_centers_obj={centers_obj_path}\ncenter_domains_obj={domains_obj_path}\nrandom_wheels_obj={wheels_obj_path}\nsummary={summary_path}")
    print(f"png={png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
