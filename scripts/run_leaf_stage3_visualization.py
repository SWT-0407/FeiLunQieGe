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
from matplotlib.lines import Line2D
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
    # 默认行为保持与原 leaf 入口一致；指定 --case 可对其它工件运行完全相同的算法。
    parser.add_argument("--case", default=None, help="工件案例名；默认读取配置中的 case_name")
    parser.add_argument(
        "--workpiece-mesh",
        default=None,
        help="工件本体 OBJ 文件名；默认读取配置中的 inputs.workpiece_mesh，可用简化网格替代",
    )
    parser.add_argument(
        "--normal-method",
        default=None,
        choices=("two_face_angle_bisector", "surface_patch_continuity"),
        help="法向估计方法；默认读取 normal_estimation.method",
    )
    parser.add_argument("--surface-patch-neighbors", type=int, default=None)
    parser.add_argument("--surface-patch-distance-factor", type=float, default=None)
    parser.add_argument("--surface-patch-coherence-degrees", type=float, default=None)
    parser.add_argument("--max-bisector-deviation-degrees", type=float, default=None)
    parser.add_argument("--tangent-normal-threshold", type=float, default=None)
    parser.add_argument("--continuity-repair-degrees", type=float, default=None)
    parser.add_argument(
        "--contact-patch",
        dest="contact_patch",
        action="store_true",
        default=None,
        help="启用接触斑模型；默认读取 collision.contact_patch_enabled",
    )
    parser.add_argument(
        "--no-contact-patch",
        dest="contact_patch",
        action="store_false",
        help="禁用接触斑模型，退回纯相交判定",
    )
    parser.add_argument("--contact-patch-radius-ratio", type=float, default=None)
    parser.add_argument("--contact-patch-max-penetration-ratio", type=float, default=None)
    parser.add_argument("--contact-patch-normal-tolerance-degrees", type=float, default=None)
    parser.add_argument(
        "--narrow-phase",
        default=None,
        choices=("analytic_closest_point_flat", "clipped_polygon"),
        help="窄相算法；默认读取 collision.narrow_phase",
    )
    parser.add_argument("--continuity-repair-iterations", type=int, default=None)
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
    """每个分支随机抽取少量可行/碰撞姿态，输出完整标准圆柱。

    可行状态从该分支的可行候选中抽取；碰撞状态从该分支的**不可行候选**
    中抽取。注意不能要求“该点所有角度都不可行”，否则只要点还有任意一个
    可行角，就会错误地跳过红色砂轮。
    """
    material_path = path.with_suffix(".mtl")
    _write_materials(material_path)
    rng = np.random.default_rng(seed)
    selected: list[tuple[int, int, bool]] = []
    for _branch_id, start, end in branches:
        for feasible_state in (True, False):
            # 收集该分支内所有 (目标点, 候选角) 对，再从中随机抽 per_state 个。
            pairs: list[tuple[int, int]] = []
            for point_index in range(start, end):
                flags = np.asarray(audits[point_index].result.feasible, dtype=bool)
                local_pool = np.flatnonzero(flags if feasible_state else ~flags)
                for local_index in local_pool:
                    pairs.append((point_index, int(local_index)))
            if not pairs:
                continue
            chosen = rng.choice(len(pairs), size=min(per_state, len(pairs)), replace=False)
            for pair_index in np.atleast_1d(chosen).astype(int):
                point_index, local_index = pairs[int(pair_index)]
                selected.append((int(point_index), int(local_index), feasible_state))
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


def _plot_leaf_projection(
    path: Path,
    workpiece: object,
    sampled: object,
    *,
    normalization: dict[str, object] | None = None,
    title_prefix: str = "Leaf",
) -> None:
    """绘制与工件 OBJ 同轴向的投影，并用侧栏区分分支首尾。

    投影不再使用 PCA 轴，而是选择工件包围盒中跨度最大的两个**原始坐标轴**
    （X/Y/Z 中选两个），并直接使用 ``source_obj`` 坐标。这样图中的轴线方向
    和数值与 MeshLab 中打开的工件 OBJ 一致。

    分支首尾不再逐点写文字，避免多条分支端点重合时互相遮挡。改为：
    - 每条分支线用独立颜色；
    - 起点用圆点、终点用方点；
    - 右侧栏列出每个 branch 的起止编号和点数；
    - 分支中点标一个 ``B0``、``B1`` 等短标签。
    """

    vertices = np.asarray(workpiece.vertices, dtype=float)
    faces = np.asarray(workpiece.faces, dtype=np.int64)
    root_points = np.asarray(sampled.points, dtype=float)
    if normalization is not None:
        scale = float(normalization["scale"])
        center = np.asarray(normalization["center"], dtype=float)
        vertices = vertices * scale + center
        root_points = root_points * scale + center
        units = "source_obj coordinates"
    else:
        units = "normalized units"

    # 选择包围盒跨度最大的两个原始坐标轴，保证投影方向与 OBJ 一致。
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    axes = np.sort(np.argsort(extent)[-2:])
    axis_names = ("X", "Y", "Z")
    projected_vertices = vertices[:, axes]
    projected_root = root_points[:, axes]
    stride = max(1, len(faces) // 18000)
    polygons = [projected_vertices[face] for face in faces[::stride]]

    figure, axis = plt.subplots(figsize=(13.5, 10), dpi=180)
    figure.subplots_adjust(left=0.07, right=0.75, top=0.92, bottom=0.08)
    axis.add_collection(
        PolyCollection(polygons, facecolor="#d6d9df", edgecolor="#9aa0aa", linewidth=0.08, alpha=0.78)
    )
    branches = _branch_runs(np.asarray(sampled.branch_ids, dtype=np.int64))
    colors = plt.get_cmap("tab10")

    for branch_id, start, end in branches:
        branch = projected_root[start:end]
        color = colors(branch_id % 10)
        axis.plot(branch[:, 0], branch[:, 1], color=color, linewidth=2.2, zorder=4)
        # 起点圆点、终点方点；不写文字，避免端点重合时互相覆盖。
        axis.scatter(
            branch[0, 0], branch[0, 1], marker="o", s=34, color=color,
            edgecolors="white", linewidths=0.7, zorder=6,
        )
        axis.scatter(
            branch[-1, 0], branch[-1, 1], marker="s", s=34, color=color,
            edgecolors="white", linewidths=0.7, zorder=6,
        )
        # 分支中点标 B0/B1...；垂直分支方向偏移，并在分支间做轻微错位。
        middle = (start + end - 1) // 2
        previous = projected_root[max(start, middle - 1)]
        following = projected_root[min(end - 1, middle + 1)]
        direction = following - previous
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1e-12:
            normal = np.array([-direction[1], direction[0]]) / direction_norm
        else:
            normal = np.array([0.0, 1.0])
        offset = 10.0 + 3.0 * (branch_id % 3)
        axis.annotate(
            f"B{branch_id}",
            projected_root[middle],
            xytext=(normal[0] * offset, normal[1] * offset),
            textcoords="offset points",
            fontsize=8,
            color=color,
            fontweight="bold",
            zorder=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8},
        )

    low = projected_vertices.min(axis=0)
    high = projected_vertices.max(axis=0)
    margin = max(float(np.max(high - low)) * 0.04, 1e-6)
    axis.set_xlim(low[0] - margin, high[0] + margin)
    axis.set_ylim(low[1] - margin, high[1] + margin)
    axis.set_aspect("equal")
    axis.set_title(
        f"{title_prefix} workpiece projection ({axis_names[axes[0]]}-{axis_names[axes[1]]}, {units})\n"
        "circle = branch start, square = branch end; coordinates match the case OBJ"
    )
    axis.set_xlabel(f"{axis_names[axes[0]]} ({units})")
    axis.set_ylabel(f"{axis_names[axes[1]]} ({units})")
    axis.grid(True, linewidth=0.4, alpha=0.35)

    # 右侧栏：分支起止编号表。端点重合时也能可靠区分每条分支。
    figure.text(0.77, 0.90, "branch   start   end   count", fontsize=9, fontweight="bold", family="monospace")
    for row, (branch_id, start, end) in enumerate(branches):
        y = 0.865 - row * 0.028
        figure.text(0.77, y, f"B{branch_id:<2d}", fontsize=8, color=colors(branch_id % 10), family="monospace", fontweight="bold")
        figure.text(0.805, y, f"{start:>5d}   {end - 1:>5d}   {end - start:>4d}", fontsize=8, family="monospace")
    figure.text(0.77, 0.865 - len(branches) * 0.028 - 0.02, "markers: o start, s end", fontsize=8, color="#444444")
    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markersize=7, label="branch start"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#555555", markersize=7, label="branch end"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.77, 0.02),
        fontsize=8,
        frameon=False,
    )
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.angle_count < 8 or arguments.cylinder_segments < 16:
        raise ValueError("angle-count 至少为 8，cylinder-segments 至少为 16")
    config_path = (REPO_ROOT / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # 默认仍为 leaf；通过命令行可对 circle 等其它案例运行完全相同的算法。
    case_name = str(arguments.case or config.get("case_name", "leaf"))
    case_root = (config_path.parent / config["data_root"]).resolve() / case_name
    if not case_root.is_dir():
        raise FileNotFoundError(f"案例目录不存在: {case_root}")
    workpiece_name = str(arguments.workpiece_mesh or config["inputs"]["workpiece_mesh"])
    workpiece_path = case_root / workpiece_name
    if not workpiece_path.is_file():
        raise FileNotFoundError(f"工件网格不存在: {workpiece_path}")
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(workpiece_path)
    # 飞边网格只用于来源记录，不参与碰撞；优先使用简化版本以避免无谓的大网格解析。
    reference_path = None
    for candidate_name in ("mesh_with_flash_simplified.obj", "mesh_with_flash.obj"):
        candidate_path = case_root / candidate_name
        if candidate_path.is_file():
            reference_path = candidate_path
            break
    flash_reference_vertices = int(len(load_mesh(reference_path).vertices)) if reference_path is not None else None
    flash_cloud = load_labeled_flash_cloud(case_root / "original_flash_cloud.obj")
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    normal_config = config.get("normal_estimation", {})
    normal_kwargs = {key: normal_config[key] for key in ("neighbor_count", "candidate_count", "side_count", "coherence_angle_degrees", "offset_epsilon", "weight_distance_floor") if key in normal_config}
    normal_method = str(arguments.normal_method or normal_config.get("method", "two_face_angle_bisector"))
    normal_kwargs.update(
        {
            "normal_method": normal_method,
            "surface_patch_neighbors": int(
                arguments.surface_patch_neighbors
                if arguments.surface_patch_neighbors is not None
                else normal_config.get("surface_patch_neighbors", 32)
            ),
            "surface_patch_distance_factor": float(
                arguments.surface_patch_distance_factor
                if arguments.surface_patch_distance_factor is not None
                else normal_config.get("surface_patch_distance_factor", 3.0)
            ),
            "surface_patch_coherence_degrees": float(
                arguments.surface_patch_coherence_degrees
                if arguments.surface_patch_coherence_degrees is not None
                else normal_config.get("surface_patch_coherence_degrees", 25.0)
            ),
            "max_bisector_deviation_degrees": float(
                arguments.max_bisector_deviation_degrees
                if arguments.max_bisector_deviation_degrees is not None
                else normal_config.get("max_bisector_deviation_degrees", 10.0)
            ),
            "continuity_repair_degrees": float(
                arguments.continuity_repair_degrees
                if arguments.continuity_repair_degrees is not None
                else normal_config.get("continuity_repair_degrees", 30.0)
            ),
            "continuity_repair_iterations": int(
                arguments.continuity_repair_iterations
                if arguments.continuity_repair_iterations is not None
                else normal_config.get("continuity_repair_iterations", 3)
            ),
        }
    )
    tangent_normal_threshold = float(
        arguments.tangent_normal_threshold
        if arguments.tangent_normal_threshold is not None
        else normal_config.get("tangent_normal_threshold", 0.2)
    )
    width = float(arguments.width if arguments.width is not None else config["wheel"].get("width", 0.004))
    radius = float(arguments.radius if arguments.radius is not None else config["wheel"]["radius"])
    delta = float(config["wheel"].get("delta", 0.0))
    if not (0.0 <= delta < radius):
        raise ValueError("wheel.delta must satisfy 0 <= delta < radius")
    center_source = np.asarray(normalization.get("center", [0.0, 0.0, 0.0]), dtype=float)
    scale_source = float(normalization.get("scale", 1.0))
    flash_points = (flash_cloud.flash_points - center_source) / scale_source
    body_points = (flash_cloud.body_points - center_source) / scale_source
    # ``original_flash_cloud.obj`` is the authoritative source of F because
    # mesh_with_flash.obj has no face-level flash/body label.  The estimator
    # signs each nearby triangle normal toward its nearest F with n dot OF > 0;
    # it intentionally bypasses the former branch dynamic-programming flip.
    normals, normal_diagnostics = estimate_surface_normals(
        sampled.points,
        workpiece,
        flash_points=flash_points,
        return_diagnostics=True,
        **normal_kwargs,
    )
    frames = build_frames(
        sampled.points,
        sampled.tangents,
        normals,
        normal_is_reliable=(normal_method == "surface_patch_continuity"),
        tangent_normal_threshold=tangent_normal_threshold,
    )
    angles = np.linspace(0.0, 2.0 * np.pi, arguments.angle_count, endpoint=False)
    cylinder_index = build_cylinder_triangle_index(workpiece)
    collision_config = config.get("collision", {})
    contact_patch_enabled = (
        bool(arguments.contact_patch)
        if arguments.contact_patch is not None
        else bool(collision_config.get("contact_patch_enabled", True))
    )
    contact_patch_radius_ratio = float(
        arguments.contact_patch_radius_ratio
        if arguments.contact_patch_radius_ratio is not None
        else collision_config.get("contact_patch_radius_ratio", 2.0)
    )
    contact_patch_max_penetration_ratio = float(
        arguments.contact_patch_max_penetration_ratio
        if arguments.contact_patch_max_penetration_ratio is not None
        else collision_config.get("contact_patch_max_penetration_ratio", 0.025)
    )
    contact_normal_tolerance_degrees = float(
        arguments.contact_patch_normal_tolerance_degrees
        if arguments.contact_patch_normal_tolerance_degrees is not None
        else collision_config.get("contact_patch_normal_tolerance_degrees", 30.0)
    )
    contact_patch_radius = contact_patch_radius_ratio * radius
    max_contact_penetration = contact_patch_max_penetration_ratio * radius
    narrow_phase = str(
        arguments.narrow_phase
        if arguments.narrow_phase is not None
        else collision_config.get("narrow_phase", "analytic_closest_point_flat")
    )
    audits: list[PointAudit] = []
    for point_index, (point, frame) in enumerate(zip(sampled.points, frames, strict=True)):
        candidates = _candidate_set(np.asarray(point), frame, radius, width, angles, point_index, arguments.side)
        result = check_finite_cylinder_candidates_exact(
            candidates, workpiece, width,
            inner_radius=0.0,
            clearance=float(collision_config.get("clearance", 0.0)), epsilon=arguments.epsilon,
            allow_target_contact=bool(collision_config.get("allow_target_contact", False)),
            target_contact_tolerance=arguments.target_contact_tolerance,
            contact_patch_enabled=contact_patch_enabled,
            contact_patch_radius=contact_patch_radius,
            max_contact_penetration=max_contact_penetration,
            contact_normal_tolerance_degrees=contact_normal_tolerance_degrees,
            narrow_phase=narrow_phase,
            cylinder_index=cylinder_index,
        )
        audits.append(PointAudit(point_index, candidates, result))
        if (point_index + 1) % 25 == 0:
            print(f"audited_points={point_index + 1}/{len(sampled.points)}", flush=True)

    branches = _branch_runs(np.asarray(sampled.branch_ids, dtype=np.int64))
    representatives = []
    for branch_id, start, end in branches:
        point_index, feasible_index, collision_index = _select_branch_point(list(range(start, end)), audits)
        representatives.append({"branch_id": branch_id, "point_index": point_index, "feasible_index": feasible_index, "collision_index": collision_index})

    output_root = REPO_ROOT / "results" / "cases" / case_name / datetime.now().strftime("stage3_visualization_%Y%m%d_%H%M%S_%f")
    output_root.mkdir(parents=True, exist_ok=False)
    centers_obj_path = output_root / f"{case_name}_rootline_circle_centers.obj"
    domains_obj_path = output_root / f"{case_name}_feasible_center_domains.obj"
    wheels_obj_path = output_root / f"{case_name}_random_wheel_shapes.obj"
    png_path = output_root / f"{case_name}_rootline_projection.png"
    summary_path = output_root / "stage3_visualization_summary.json"
    normal_diagnostics_path = output_root / "normal_continuity_diagnostics.json"
    _export_circle_centers(centers_obj_path, sampled, audits, branches, normalization, radius)
    _export_center_domains(domains_obj_path, sampled, audits, branches, normalization, radius)
    wheel_records = _export_random_wheels(wheels_obj_path, sampled, audits, branches, normalization, radius, width, arguments.cylinder_segments)
    # 按项目规范，PNG 只在显式传入 --png 时生成；OBJ/MTL/JSON 始终输出。
    if arguments.png:
        _plot_leaf_projection(
            png_path, workpiece, sampled,
            normalization=normalization,
            title_prefix=case_name,
        )
    feasible_count = sum(int(np.count_nonzero(a.result.feasible)) for a in audits)
    candidate_count = sum(len(a.result.feasible) for a in audits)
    normal_diagnostics_path.write_text(json.dumps({
        "point_count": len(normal_diagnostics),
        "branch_count": len(branches),
        "continuity_threshold_degrees": normal_config.get("continuity_jump_threshold_degrees", 20.0),
        "low_confidence_point_count": sum(bool(item.get("continuity_state", {}).get("low_confidence", False)) for item in normal_diagnostics),
        "points": normal_diagnostics,
        "orientation_rule": "per_face_choose_normal_with_positive_dot_OF",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "case_name": case_name, "stage": "stage_3_batch_visualization", "units": "normalized",
        "sampled_point_count": len(audits), "root_line_branch_count": len(branches),
        "angle_count_per_point": arguments.angle_count, "candidate_center_count": candidate_count,
        "feasible_center_count": feasible_count, "collision_center_count": candidate_count - feasible_count,
        "radius": radius, "delta": delta, "contact_band": [radius - delta, radius], "width_h": width, "lower_face_offset": width / 2.0,
        "collision_backend": audits[0].result.method if audits else "unknown",
        "collision_body": "solid_cylinder",
        "narrow_phase": narrow_phase,
        "broad_phase": {"type": "aabb_bvh", "leaf_size": cylinder_index.leaf_size},
        "contact_patch": {
            "enabled": contact_patch_enabled,
            "radius": contact_patch_radius,
            "radius_ratio": contact_patch_radius_ratio,
            "max_penetration": max_contact_penetration,
            "max_penetration_ratio": contact_patch_max_penetration_ratio,
            "normal_tolerance_degrees": contact_normal_tolerance_degrees,
        },
        "contact_patch_only_count": int(
            sum(
                1
                for audit in audits
                for reason in np.asarray(audit.result.reason, dtype=object)
                if str(reason).startswith("contact_patch_only")
            )
        ),
        "collision_reason_counts": {
            str(key): int(value)
            for key, value in zip(
                *np.unique(
                    np.concatenate([np.asarray(audit.result.reason, dtype=object) for audit in audits]),
                    return_counts=True,
                ),
                strict=True,
            )
        },
        "candidate_center_radius": radius,
        "normal_estimation": {
            "method": normal_method,
            "surface_patch_neighbors": normal_kwargs["surface_patch_neighbors"],
            "surface_patch_distance_factor": normal_kwargs["surface_patch_distance_factor"],
            "surface_patch_coherence_degrees": normal_kwargs["surface_patch_coherence_degrees"],
            "max_bisector_deviation_degrees": normal_kwargs["max_bisector_deviation_degrees"],
            "tangent_normal_threshold": tangent_normal_threshold,
            "continuity_repair_degrees": normal_kwargs["continuity_repair_degrees"],
            "continuity_repair_iterations": normal_kwargs["continuity_repair_iterations"],
        },
        "workpiece_mesh": str(workpiece_path),
        "workpiece_triangle_count": int(len(workpiece.faces)),
        "wheel_visualization": {"center_material": "circle_center_yellow", "feasible_material": "feasible_green", "collision_material": "collision_red", "random_seed": 20261018, "random_wheel_count": len(wheel_records)},
        "orientation": {"method": "nearest_labelled_flash_point_per_face", "formula": "choose sign(n) so dot(n, F-O) >= 0", "source": str(flash_cloud.source_path), "reference_mesh": str(reference_path) if reference_path is not None else None, "reference_mesh_vertex_count": flash_reference_vertices, "flash_point_count": int(len(flash_points)), "body_point_count": int(len(body_points))},
        "branch_representatives": representatives,
        "random_wheels": wheel_records,
        "outputs": {"circle_centers_obj": str(centers_obj_path.relative_to(REPO_ROOT)), "center_domains_obj": str(domains_obj_path.relative_to(REPO_ROOT)), "random_wheels_obj": str(wheels_obj_path.relative_to(REPO_ROOT)), "normal_diagnostics": str(normal_diagnostics_path.relative_to(REPO_ROOT)), "summary": str(summary_path.relative_to(REPO_ROOT))},
        "human_review_status": "waiting_for_manual_review",
        "limitations": ["OBJ 使用多边形面表达标准圆柱曲面，但每个砂轮对象只有一种材质颜色。", "连续碰撞结果仍受当前网格封闭性、容差和固定角度候选集限制。"],
    }
    if arguments.png:
        summary["outputs"]["png"] = str(png_path.relative_to(REPO_ROOT))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status=completed case={case_name} points={len(audits)} branches={len(branches)} centers={candidate_count} feasible={feasible_count}")
    print(f"circle_centers_obj={centers_obj_path}\ncenter_domains_obj={domains_obj_path}\nrandom_wheels_obj={wheels_obj_path}\nsummary={summary_path}")
    if arguments.png:
        print(f"png={png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
