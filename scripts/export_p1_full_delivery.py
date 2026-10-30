"""P1 全量三 OBJ + 每个不可行点的局部面片/法向箭头 OBJ。

输出：
- circle_rootline_circle_centers.obj
- circle_feasible_center_domains.obj
- circle_inspection_wheels.obj
- <point_index>.obj（每个存在不可行角的点一个；面片分两侧着色，法向用箭头几何）
- point_patches.mtl、p1_full_summary.json、README.md

碰撞使用多进程；每个 worker 独立加载原始 full mesh，避免跨进程共享大对象。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from feilunqiege.collision.cylinder_exact import (  # noqa: E402
    build_cylinder_triangle_index,
    check_finite_cylinder_candidates_exact,
)
from feilunqiege.geometry.curve import split_and_resample  # noqa: E402
from feilunqiege.geometry.frame import build_frames  # noqa: E402
from feilunqiege.geometry.normal_p1 import (  # noqa: E402
    cluster_normals,
    estimate_normals_experimental,
    prepare_mesh,
    query_candidates,
)
from feilunqiege.io.mesh import load_mesh  # noqa: E402
from feilunqiege.io.root_line import load_root_line  # noqa: E402
from feilunqiege.pipeline import _normalize_case_geometry  # noqa: E402
from feilunqiege.tooling.grinding_wheel import CandidateSet  # noqa: E402
from run_leaf_stage3_visualization import (  # noqa: E402
    PointAudit,
    _branch_runs,
    _candidate_set,
    _export_center_domains,
    _export_circle_centers,
    _export_inspection_wheels,
)


# ----------------------------- multiprocessing worker -----------------------------
_WORKER: dict[str, object] = {}


def _worker_init(
    config_path: str,
    case_name: str,
    center: list[float],
    scale: float,
    points: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
) -> None:
    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    case_root = (config_file.parent / config["data_root"]).resolve() / case_name
    raw_mesh = load_mesh(case_root / "mesh_without_flash.obj")
    vertices = (np.asarray(raw_mesh.vertices, dtype=float) - np.asarray(center, dtype=float)) / float(scale)
    workpiece = SimpleNamespace(vertices=vertices, faces=np.asarray(raw_mesh.faces, dtype=np.int64))
    radius = float(config["wheel"]["radius"])
    width = float(config["wheel"]["width"])
    angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    frames = build_frames(points, tangents, normals, normal_is_reliable=True)
    collision_config = dict(config.get("collision", {}))
    contact_patch_radius = float(collision_config.get("contact_patch_radius_ratio", 2.0)) * radius
    max_contact_penetration = float(collision_config.get("contact_patch_max_penetration_ratio", 0.025)) * radius
    _WORKER.update(
        {
            "config": config,
            "points": points,
            "tangents": tangents,
            "frames": frames,
            "workpiece": workpiece,
            "radius": radius,
            "width": width,
            "angles": angles,
            "collision_config": collision_config,
            "contact_patch_radius": contact_patch_radius,
            "max_contact_penetration": max_contact_penetration,
            "cylinder_index": build_cylinder_triangle_index(workpiece),
        }
    )


def _worker_point(point_index: int) -> dict[str, object]:
    point = np.asarray(_WORKER["points"][point_index], dtype=float)
    frame = _WORKER["frames"][point_index]
    radius = float(_WORKER["radius"])
    width = float(_WORKER["width"])
    angles = _WORKER["angles"]
    config = _WORKER["collision_config"]
    candidates = _candidate_set(point, frame, radius, width, angles, point_index, 1.0)
    result = check_finite_cylinder_candidates_exact(
        candidates,
        _WORKER["workpiece"],
        width,
        inner_radius=0.0,
        clearance=float(config.get("clearance", 0.0)),
        allow_target_contact=bool(config.get("allow_target_contact", True)),
        target_contact_tolerance=float(config.get("target_contact_tolerance", 1e-4)),
        contact_patch_enabled=bool(config.get("contact_patch_enabled", True)),
        contact_patch_radius=float(_WORKER["contact_patch_radius"]),
        max_contact_penetration=float(_WORKER["max_contact_penetration"]),
        contact_normal_tolerance_degrees=float(config.get("contact_patch_normal_tolerance_degrees", 30.0)),
        narrow_phase=str(config.get("narrow_phase", "analytic_closest_point_flat")),
        cylinder_index=_WORKER["cylinder_index"],
    )
    return {
        "point_index": int(point_index),
        "contact_points": np.asarray(candidates.contact_points),
        "centers": np.asarray(candidates.centers),
        "radius_directions": np.asarray(candidates.radius_directions),
        "tangent_directions": np.asarray(candidates.tangent_directions),
        "axis_directions": np.asarray(candidates.axis_directions),
        "point_indices": np.asarray(candidates.point_indices),
        "radial_values": np.asarray(candidates.radial_values),
        "angles": np.asarray(candidates.angles),
        "radius": float(candidates.radius),
        "feasible": np.asarray(result.feasible, dtype=bool),
        "reason": np.asarray(result.reason, dtype=object),
        "contact_max_penetration": np.asarray(result.contact_max_penetration),
    }


# ----------------------------- arrow geometry -----------------------------
def _basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    reference = np.array([1.0, 0.0, 0.0]) if abs(float(direction[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(direction, reference)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.cross(direction, first)
    return first, second


def _arrow(handle, start: np.ndarray, direction: np.ndarray, length: float, radius: float, material: str, next_vertex: int, segments: int = 10) -> int:
    """写一个带 shaft 的圆锥箭头；所有坐标已经是 source_obj。"""
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    first, second = _basis(direction)
    shaft_end = start + direction * (length * 0.62)
    tip = start + direction * length
    shaft_radius = radius * 0.35
    handle.write(f"usemtl {material}\n")
    # shaft 圆环 + 两个端面
    shaft_ring_start = next_vertex
    for offset in (start, shaft_end):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            point = offset + shaft_radius * (math.cos(angle) * first + math.sin(angle) * second)
            handle.write(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
    for index in range(segments):
        nxt = (index + 1) % segments
        a = shaft_ring_start + index
        b = shaft_ring_start + nxt
        c = shaft_ring_start + segments + nxt
        d = shaft_ring_start + segments + index
        handle.write(f"f {a} {b} {c} {d}\n")
    next_vertex = shaft_ring_start + 2 * segments
    # 圆锥头
    base_ring_start = next_vertex
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        point = shaft_end + radius * (math.cos(angle) * first + math.sin(angle) * second)
        handle.write(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
    tip_index = next_vertex + segments
    handle.write(f"v {tip[0]:.9g} {tip[1]:.9g} {tip[2]:.9g}\n")
    for index in range(segments):
        nxt = (index + 1) % segments
        handle.write(f"f {base_ring_start + index} {base_ring_start + nxt} {tip_index}\n")
    return tip_index + 1


def _source(point: np.ndarray, scale: float, center: np.ndarray) -> np.ndarray:
    return np.asarray(point, dtype=float) * scale + center


def _write_point_patch_obj(
    path: Path,
    point_index: int,
    diagnostics: dict,
    mesh_data,
    sampled,
    normalization: dict[str, object],
) -> None:
    scale = float(normalization.get("scale", 1.0))
    center = np.asarray(normalization.get("center", [0.0, 0.0, 0.0]), dtype=float)
    point_source = _source(np.asarray(sampled.points[point_index], dtype=float), scale, center)
    patch_valid = bool(diagnostics.get("patch_valid", True))
    source_name = str(diagnostics.get("normal_source", ""))
    if patch_valid and source_name in {"two_side_bisector", "robust_two_side_bisector"}:
        # 真折边：按两个法向簇分别着色，不能把 secondary 从 patch 集合里排除。
        group_a = [int(index) for index in diagnostics.get("primary_face_indices", [])]
        group_b = [int(index) for index in diagnostics.get("secondary_face_indices", [])]
    else:
        group_a = [int(index) for index in diagnostics.get("patch_face_indices", [])]
        group_b = [
            int(index) for index in diagnostics.get("secondary_face_indices", [])
            if int(index) not in set(group_a)
        ] if patch_valid else []
    if not group_a:
        group_a = [int(index) for index in diagnostics.get("primary_face_indices", [])]
    path.write_text("", encoding="ascii")  # 创建/清空本次新文件（该路径本次刚创建）
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"# P1 point patch visualization point={point_index}\n")
        handle.write("# coordinate_frame: source_obj\n")
        handle.write(f"# normal_source: {diagnostics.get('normal_source', 'unknown')}\n")
        handle.write(f"# normal_confidence: {diagnostics.get('normal_confidence', 'unknown')}\n")
        handle.write(f"# angle_n1_n2_deg: {diagnostics.get('angle_n1_n2_deg')}\n")
        handle.write(f"# support_ratio: {diagnostics.get('support_ratio')}\n")
        handle.write(f"# patch_valid: {diagnostics.get('patch_valid', True)}\n")
        handle.write("mtllib point_patches.mtl\n")
        next_vertex = 1
        # 面片：两组不同颜色
        for material, group in (("side_a_mesh", group_a), ("side_b_mesh", group_b)):
            handle.write(f"usemtl {material}\n")
            for face_index in group:
                triangle = np.asarray(mesh_data.vertices[mesh_data.faces[face_index]], dtype=float)
                for vertex in triangle:
                    source = _source(vertex, scale, center)
                    handle.write(f"v {source[0]:.9g} {source[1]:.9g} {source[2]:.9g}\n")
                handle.write(f"f {next_vertex} {next_vertex + 1} {next_vertex + 2}\n")
                # 每个面一个小法向箭头
                centroid = triangle.mean(axis=0)
                face_normal = np.asarray(mesh_data.normals[face_index], dtype=float)
                next_vertex = _arrow(
                    handle,
                    _source(centroid, scale, center),
                    face_normal,
                    0.008 * scale,
                    0.0007 * scale,
                    f"face_normal_{material.split('_')[1]}",
                    next_vertex,
                    segments=8,
                )
        # 组法向（cluster mean）箭头
        for material, normal_key in (("cluster_normal_a", "primary_normal"), ("cluster_normal_b", "secondary_normal")):
            normal = diagnostics.get(normal_key)
            if normal is None:
                continue
            group = group_a if normal_key == "primary_normal" else group_b
            if not group:
                continue
            centroids = np.asarray([mesh_data.vertices[mesh_data.faces[index]].mean(axis=0) for index in group], dtype=float)
            start = _source(centroids.mean(axis=0), scale, center)
            next_vertex = _arrow(
                handle, start, np.asarray(normal, dtype=float),
                0.020 * scale, 0.0012 * scale, material, next_vertex, segments=12,
            )
        # 最终轴
        final_normal = np.asarray(diagnostics.get("normal"), dtype=float)
        next_vertex = _arrow(
            handle, point_source, final_normal,
            0.028 * scale, 0.0014 * scale, "final_normal", next_vertex, segments=12,
        )
        # 根点
        handle.write("usemtl root_point\n")
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            vertex = point_source + np.array([dx, dy, dz], dtype=float) * (0.0025 * scale)
            handle.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        handle.write("p " + " ".join(str(index) for index in range(next_vertex, next_vertex + 6)) + "\n")


def _write_point_mtl(path: Path) -> None:
    path.write_text(
        """# P1 point patch materials
newmtl side_a_mesh
Ka 0 0 0
Kd 0.10 0.45 0.90
illum 0
newmtl side_b_mesh
Ka 0 0 0
Kd 0.95 0.45 0.05
illum 0
newmtl face_normal_a
Ka 0 0 0
Kd 0.20 0.75 1.00
illum 0
newmtl face_normal_b
Ka 0 0 0
Kd 1.00 0.75 0.20
illum 0
newmtl cluster_normal_a
Ka 0 0 0
Kd 0.00 0.15 0.65
illum 0
newmtl cluster_normal_b
Ka 0 0 0
Kd 0.65 0.20 0.00
illum 0
newmtl final_normal
Ka 0 0 0
Kd 0.85 0.05 0.45
illum 0
newmtl root_point
Ka 0 0 0
Kd 0 0 0
illum 0
""",
        encoding="ascii",
    )


# ----------------------------- main -----------------------------
def _unique_axes(axes: list[np.ndarray]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for axis in axes:
        axis = np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            continue
        axis = axis / norm
        if all(float(np.dot(axis, other)) < math.cos(math.radians(1.0)) for other in result):
            result.append(axis)
    return result


def _fallback_axes(
    point: np.ndarray,
    diagnostic: dict,
    mesh_data,
    *,
    radius: float,
    min_faces: int,
) -> list[np.ndarray]:
    """为完全无解点收集局部可行轴候选。"""

    axes = [
        np.asarray(diagnostic.get("normal", [0.0, 0.0, 1.0]), dtype=float),
        np.asarray(diagnostic.get("primary_normal", [0.0, 0.0, 1.0]), dtype=float),
    ]
    secondary = diagnostic.get("secondary_normal")
    if secondary is not None:
        secondary = np.asarray(secondary, dtype=float)
        axes.append(secondary)
        axes.append(np.asarray(diagnostic["primary_normal"], dtype=float) + secondary)
    candidates, distances = query_candidates(
        point,
        mesh_data,
        k=96,
        radius=radius,
        min_faces=min_faces,
    )
    if len(candidates):
        clusters = cluster_normals(
            candidates,
            distances,
            mesh_data,
            angle_deg=15.0,
            min_support=2,
        )
        axes.extend(cluster.mean for cluster in clusters)
        face_normals = mesh_data.normals[candidates]
        bottom_like = face_normals[face_normals[:, 2] < -0.9]
        if len(bottom_like):
            bottom_mean = bottom_like.mean(axis=0)
            if float(np.linalg.norm(bottom_mean)) > 1e-12:
                axes.append(bottom_mean)
    return _unique_axes(axes)


def _evaluate_axis(
    point: np.ndarray,
    tangent: np.ndarray,
    axis: np.ndarray,
    point_index: int,
    *,
    radius: float,
    width: float,
    angles: np.ndarray,
    workpiece,
    cylinder_index,
    collision_config: dict,
    contact_patch: dict,
) -> dict[str, object]:
    frame = build_frames(
        np.asarray([point]),
        np.asarray([tangent]),
        np.asarray([axis]),
        normal_is_reliable=True,
    )[0]
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
    return {
        "point_index": int(point_index),
        "contact_points": np.asarray(candidates.contact_points),
        "centers": np.asarray(candidates.centers),
        "radius_directions": np.asarray(candidates.radius_directions),
        "tangent_directions": np.asarray(candidates.tangent_directions),
        "axis_directions": np.asarray(candidates.axis_directions),
        "point_indices": np.asarray(candidates.point_indices),
        "radial_values": np.asarray(candidates.radial_values),
        "angles": np.asarray(candidates.angles),
        "radius": float(candidates.radius),
        "feasible": np.asarray(result.feasible, dtype=bool),
        "reason": np.asarray(result.reason, dtype=object),
        "contact_max_penetration": np.asarray(result.contact_max_penetration),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 全量三 OBJ + 每点 patch/法向箭头")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--case", default="circle")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--points", default=None, help="逗号分隔的点编号，用于烟雾测试")
    parser.add_argument("--three-obj-only", action="store_true", help="只输出三个主 OBJ，不输出逐点 patch OBJ")
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args()

    config_path = (REPO / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    case_root = (config_path.parent / config["data_root"]).resolve() / arguments.case
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / "mesh_without_flash.obj")
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    normals, diagnostics = estimate_normals_experimental(
        sampled.points,
        workpiece,
        method="p1",
        config=dict(config.get("normal_estimation", {}).get("p1", {})),
        return_diagnostics=True,
    )
    radius = float(config["wheel"]["radius"])
    width = float(config["wheel"]["width"])
    angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)

    from multiprocessing import Pool

    point_indices = list(range(len(sampled.points)))
    if arguments.points:
        point_indices = [int(value) for value in arguments.points.split(",") if value.strip()]
    if arguments.max_points is not None:
        point_indices = point_indices[: int(arguments.max_points)]
    chunks = [point_indices[index:: arguments.workers] for index in range(arguments.workers)]
    results: dict[int, dict[str, object]] = {}
    with Pool(
        processes=max(1, int(arguments.workers)),
        initializer=_worker_init,
        initargs=(
            str(config_path),
            arguments.case,
            normalization["center"],
            float(normalization["scale"]),
            np.asarray(sampled.points, dtype=float),
            np.asarray(sampled.tangents, dtype=float),
            np.asarray(normals, dtype=float),
        ),
    ) as pool:
        for chunk_result in pool.imap_unordered(_worker_point, point_indices):
            results[int(chunk_result["point_index"])] = chunk_result
            if len(results) % 25 == 0 or len(results) == len(point_indices):
                print(f"collision_points={len(results)}/{len(point_indices)}", flush=True)

    mesh_data = prepare_mesh(workpiece)
    collision_config = dict(config.get("collision", {}))
    contact_patch = {
        "enabled": bool(collision_config.get("contact_patch_enabled", True)),
        "radius": float(collision_config.get("contact_patch_radius_ratio", 2.0)) * radius,
        "max_penetration": float(collision_config.get("contact_patch_max_penetration_ratio", 0.025)) * radius,
        "normal_tolerance_degrees": float(collision_config.get("contact_patch_normal_tolerance_degrees", 30.0)),
    }
    fallback_points: list[dict[str, object]] = []
    zero_indices = [
        point_index for point_index in point_indices
        if int(np.count_nonzero(np.asarray(results[point_index]["feasible"], dtype=bool))) == 0
    ]
    if zero_indices:
        fallback_index = build_cylinder_triangle_index(workpiece)
        for point_index in zero_indices:
            axes = _fallback_axes(
                np.asarray(sampled.points[point_index], dtype=float),
                diagnostics[point_index],
                mesh_data,
                radius=radius,
                min_faces=int(config.get("normal_estimation", {}).get("min_patch_faces", 10)),
            )
            best = None
            best_count = -1
            best_axis = None
            for axis in axes:
                evaluated = _evaluate_axis(
                    np.asarray(sampled.points[point_index], dtype=float),
                    np.asarray(sampled.tangents[point_index], dtype=float),
                    axis,
                    point_index,
                    radius=radius,
                    width=width,
                    angles=angles,
                    workpiece=workpiece,
                    cylinder_index=fallback_index,
                    collision_config=collision_config,
                    contact_patch=contact_patch,
                )
                count = int(np.count_nonzero(np.asarray(evaluated["feasible"], dtype=bool)))
                if count > best_count:
                    best_count = count
                    best = evaluated
                    best_axis = axis
            if best is not None and best_count > 0:
                results[point_index] = best
                diagnostics[point_index]["normal"] = np.asarray(best_axis, dtype=float).tolist()
                diagnostics[point_index]["normal_source"] = "fallback_axis"
                diagnostics[point_index]["normal_confidence"] = "low"
                reasons = list(diagnostics[point_index].get("low_confidence_reasons", []))
                if "zero_feasible_primary_axis" not in reasons:
                    reasons.append("zero_feasible_primary_axis")
                diagnostics[point_index]["low_confidence_reasons"] = reasons
                fallback_points.append(
                    {
                        "point_index": point_index,
                        "axis": np.asarray(best_axis, dtype=float).tolist(),
                        "feasible_angle_count": best_count,
                    }
                )

    branches = _branch_runs(np.asarray(sampled.branch_ids, dtype=np.int64))
    audits: list[PointAudit | None] = [None] * len(sampled.points)
    point_records = []
    for point_index in point_indices:
        raw = results[point_index]
        candidates = CandidateSet(
            contact_points=np.asarray(raw["contact_points"]),
            centers=np.asarray(raw["centers"]),
            radius_directions=np.asarray(raw["radius_directions"]),
            tangent_directions=np.asarray(raw["tangent_directions"]),
            axis_directions=np.asarray(raw["axis_directions"]),
            point_indices=np.asarray(raw["point_indices"]),
            radial_values=np.asarray(raw["radial_values"]),
            angles=np.asarray(raw["angles"]),
            radius=float(raw["radius"]),
        )
        result = SimpleNamespace(
            feasible=np.asarray(raw["feasible"], dtype=bool),
            reason=np.asarray(raw["reason"], dtype=object),
            contact_max_penetration=np.asarray(raw["contact_max_penetration"]),
        )
        audits[point_index] = PointAudit(point_index, candidates, result)
        infeasible_count = int(np.count_nonzero(~result.feasible))
        point_records.append(
            {
                "point_index": point_index,
                "branch_id": int(sampled.branch_ids[point_index]),
                "feasible_angle_count": int(np.count_nonzero(result.feasible)),
                "infeasible_angle_count": infeasible_count,
                "normal_confidence": diagnostics[point_index].get("normal_confidence"),
                "normal_source": diagnostics[point_index].get("normal_source"),
                "patch_valid": diagnostics[point_index].get("patch_valid"),
                "reason_counts": dict(Counter(str(reason) for reason in result.reason)),
            }
        )

    output_dir = (
        Path(arguments.output_dir).resolve()
        if arguments.output_dir is not None
        else REPO / "results" / "cases" / arguments.case / datetime.now().strftime("p1_full_delivery_%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    centers_obj = output_dir / f"{arguments.case}_rootline_circle_centers.obj"
    domains_obj = output_dir / f"{arguments.case}_feasible_center_domains.obj"
    wheels_obj = output_dir / f"{arguments.case}_inspection_wheels.obj"
    valid_audits = [audit for audit in audits if audit is not None]
    _export_circle_centers(centers_obj, sampled, valid_audits, branches, normalization, radius)
    _export_center_domains(domains_obj, sampled, valid_audits, branches, normalization, radius)
    infeasible_points = [row["point_index"] for row in point_records if row["infeasible_angle_count"] > 0]
    wheel_records = _export_inspection_wheels(
        wheels_obj,
        sampled,
        audits,
        branches,
        normalization,
        radius,
        width,
        int(config.get("visualization", {}).get("cylinder_segments", 64)),
        infeasible_points,
        per_point=2,
        include_contact_points=False,
    )

    if not arguments.three_obj_only:
        _write_point_mtl(output_dir / "point_patches.mtl")
        for point_index in infeasible_points:
            _write_point_patch_obj(
                output_dir / f"{point_index}.obj",
                point_index,
                diagnostics[point_index],
                mesh_data,
                sampled,
                normalization,
            )

    summary = {
        "case": arguments.case,
        "normal_method": "p1",
        "scope": "all 415 root-line points",
        "coordinate_frames": {"computation": "normalized", "obj": "source_obj"},
        "parameters": {
            "radius": radius,
            "width": width,
            "angle_count": 72,
            "delta": float(config["wheel"].get("delta", 0.0)),
            "p1": dict(config.get("normal_estimation", {}).get("p1", {})),
        },
        "point_count": len(point_indices),
        "infeasible_point_count": len(infeasible_points),
        "infeasible_points": infeasible_points,
        "fallback_axis_count": len(fallback_points),
        "fallback_axis_points": fallback_points,
        "normal_confidence_counts": dict(Counter(str(row["normal_confidence"]) for row in point_records)),
        "point_records": point_records,
        "wheel_records": wheel_records,
        "outputs": {
            "centers_obj": str(centers_obj),
            "domains_obj": str(domains_obj),
            "wheels_obj": str(wheels_obj),
            "point_patch_objs": (
                [] if arguments.three_obj_only
                else [str(output_dir / f"{index}.obj") for index in infeasible_points]
            ),
            "point_patches_mtl": (
                None if arguments.three_obj_only
                else str(output_dir / "point_patches.mtl")
            ),
        },
        "note": "Full P1 geometry visualization; not a process acceptance. Low-confidence/fallback normals are flagged in JSON/OBJ headers.",
    }
    (output_dir / "p1_full_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    point_section = "" if arguments.three_obj_only else """
- 逐点文件：
  - `<point_index>.obj`：选中的 patch 面（两组颜色，若存在两侧）、每面法向箭头、组法向箭头、最终法向箭头、根点；
  - `point_patches.mtl`：共享材质。
"""
    (output_dir / "README.md").write_text(
        f"""# P1 三个 OBJ 交付

- 方法：`p1`
- 工件：`circle/mesh_without_flash.obj`（原始 full mesh）
- 点：{len(point_indices)} 个 root-line 采样点；角度：72
- 不可行点：{len(infeasible_points)} 个；其中 {len(fallback_points)} 个点使用了 fallback axis
- 主文件：
  - `{centers_obj.name}`：根线 + 圆心；
  - `{domains_obj.name}`：可行/不可行中心域；
  - `{wheels_obj.name}`：不可行点的红色/绿色砂轮姿态；
{point_section}- 坐标：OBJ 为 `source_obj`。
""",
        encoding="utf-8",
    )
    print(f"output_dir={output_dir}")
    print(f"infeasible_points={len(infeasible_points)}")
    print(f"centers={centers_obj}\ndomains={domains_obj}\nwheels={wheels_obj}")
    return 0


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    raise SystemExit(main())
