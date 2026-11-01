"""路径可视化 OBJ/MTL 导出。

输出内容：
- root_line：灰色折线；
- 每个 branch 的 ``+q`` 路径：蓝色折线 + 蓝色小八面体节点；
- 每个 branch 的 ``-q`` 路径：橙色折线 + 橙色小八面体节点；
- ``no_path`` 的 branch/侧：只写注释，不画伪造路径。

所有坐标写为 ``source_obj``，可直接与原始工件 OBJ 在 MeshLab 中叠加。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _to_source(point: np.ndarray, normalization: dict[str, Any]) -> np.ndarray:
    scale = float(normalization.get("scale", 1.0))
    center = np.asarray(normalization.get("center", [0.0, 0.0, 0.0]), dtype=float)
    return np.asarray(point, dtype=float) * scale + center


def _write_mtl(path: Path) -> None:
    path.write_text(
        "# path planning visualization materials\n"
        "newmtl root_line_gray\n"
        "Ka 0.45 0.45 0.45\nKd 0.45 0.45 0.45\nillum 0\n"
        "newmtl path_positive_blue\n"
        "Ka 0.02 0.10 0.30\nKd 0.10 0.45 1.00\nKe 0.10 0.45 1.00\nillum 0\n"
        "newmtl path_negative_orange\n"
        "Ka 0.30 0.12 0.02\nKd 1.00 0.55 0.10\nKe 1.00 0.55 0.10\nillum 0\n"
        "newmtl path_node_positive\n"
        "Ka 0.02 0.10 0.30\nKd 0.20 0.60 1.00\nKe 0.20 0.60 1.00\nillum 0\n"
        "newmtl path_node_negative\n"
        "Ka 0.30 0.12 0.02\nKd 1.00 0.65 0.20\nKe 1.00 0.65 0.20\nillum 0\n",
        encoding="ascii",
    )


def _write_vertex(handle: Any, point: np.ndarray) -> None:
    handle.write(f"v {point[0]:.12g} {point[1]:.12g} {point[2]:.12g}\n")


def _write_root_line(handle: Any, sampled: Any, branches: list[tuple[int, int, int]], normalization: dict[str, Any]) -> int:
    """写 root_line 顶点与 branch 内折线，返回下一个可用顶点编号。"""

    next_vertex = 1
    root_ids: list[int] = []
    for point in np.asarray(sampled.points, dtype=float):
        root_ids.append(next_vertex)
        _write_vertex(handle, _to_source(point, normalization))
        next_vertex += 1
    for _branch_id, start, end in branches:
        for index in range(start, end - 1):
            handle.write(f"l {root_ids[index]} {root_ids[index + 1]}\n")
    return next_vertex


def _write_center_marker(handle: Any, center: np.ndarray, size: float, first: int) -> int:
    """写一个小八面体标记，返回下一个可用顶点编号。"""

    vertices = np.array(
        [
            center + [size, 0.0, 0.0],
            center + [-size, 0.0, 0.0],
            center + [0.0, size, 0.0],
            center + [0.0, -size, 0.0],
            center + [0.0, 0.0, size],
            center + [0.0, 0.0, -size],
        ]
    )
    for vertex in vertices:
        _write_vertex(handle, vertex)
    faces = (
        (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
        (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
    )
    for face in faces:
        handle.write("f " + " ".join(str(first + index) for index in face) + "\n")
    return first + 6


def _write_tube(
    handle: Any,
    start: np.ndarray,
    end: np.ndarray,
    tube_radius: float,
    first: int,
    *,
    segments: int = 8,
) -> int:
    """写一段细圆柱，把相邻路径中心连成 MeshLab 可见的实心线。

    OBJ 的 ``l`` 元素在 MeshLab 中默认线宽很细、且不一定按材质显示；
    为了让 ``·-·-·`` 在普通视图里可见，这里额外写一个多边形圆柱。
    圆柱只用于显示，不参与碰撞或路径计算。
    """

    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return first
    axis = direction / length
    reference = np.array([1.0, 0.0, 0.0]) if abs(float(axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    first_basis = np.cross(axis, reference)
    first_basis /= max(float(np.linalg.norm(first_basis)), 1e-12)
    second_basis = np.cross(axis, first_basis)
    second_basis /= max(float(np.linalg.norm(second_basis)), 1e-12)

    ring_start = first
    ring_end = first + segments
    for base in (np.asarray(start, dtype=float), np.asarray(end, dtype=float)):
        for index in range(segments):
            angle = 2.0 * np.pi * index / segments
            point = base + float(tube_radius) * (
                np.cos(angle) * first_basis + np.sin(angle) * second_basis
            )
            _write_vertex(handle, point)
    # 侧面四边形
    for index in range(segments):
        nxt = (index + 1) % segments
        handle.write(
            f"f {ring_start + index} {ring_start + nxt} {ring_end + nxt} {ring_end + index}\n"
        )
    # 两端封口，避免在 MeshLab 中看起来像空心管
    cap_start = ring_end + segments
    cap_end = cap_start + 1
    _write_vertex(handle, np.asarray(start, dtype=float))
    _write_vertex(handle, np.asarray(end, dtype=float))
    for index in range(segments):
        nxt = (index + 1) % segments
        handle.write(f"f {cap_start} {ring_start + index} {ring_start + nxt}\n")
        handle.write(f"f {cap_end} {ring_end + nxt} {ring_end + index}\n")
    return cap_end + 1


def _material_for_side(lateral_side: int) -> tuple[str, str]:
    if lateral_side > 0:
        return "path_positive_blue", "path_node_positive"
    return "path_negative_orange", "path_node_negative"


def write_path_obj(
    path: Path,
    sampled: Any,
    branches: list[tuple[int, int, int]],
    normalization: dict[str, Any],
    side_paths: list[Any],
    *,
    radius: float,
) -> None:
    """写出 root_line + 两侧路径的 MeshLab OBJ。

    ``side_paths`` 是 ``feilunqiege.planning.path_planning.SidePath`` 列表。
    只导出 ``status="ok"`` 的路径；``no_path`` 在文件头写注释。
    """

    material_path = path.with_suffix(".mtl")
    _write_mtl(material_path)
    scale = float(normalization.get("scale", 1.0))
    # 点标记和线段都适当加粗，便于在 MeshLab 的普通视图里看到 ·-·-·。
    marker_size = max(float(radius) * scale * 0.030, 1e-8)
    tube_radius = max(float(radius) * scale * 0.020, 1e-9)

    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# path planning visualization\n")
        handle.write("# coordinate_frame: source_obj\n")
        handle.write("# rendering: adjacent selected centers are connected by l elements and thin tube faces\n")
        handle.write("# rendering: marker_size and tube_radius are enlarged for visibility; not collision geometry\n")
        handle.write("# path_side_definition: lateral_side = sign(radius_direction . q), q = normalize(axis x tangent)\n")
        handle.write("# midline_candidates: theta=0/180 excluded from both lateral sides\n")
        handle.write("# cost: sum of Euclidean distances between consecutive selected wheel centers\n")
        handle.write("# coverage: each selected point covers [s-r, s+r] along branch arc length; points may be skipped\n")
        handle.write("# coverage_overlap: consecutive selected points satisfy s_next - s_prev <= 2r\n")
        handle.write("# coverage_restriction: coverage cannot cross a sample point without a feasible candidate on the same side\n")
        handle.write("# side_consistency_rule: only wheel center must be on the chosen side; wheel body may cross\n")
        handle.write("# edge_collision_audit: not performed in this run (node-level feasibility only)\n")
        handle.write("# method: coverage-constrained layered dynamic programming on 72-angle candidates\n")
        handle.write("# references: Zhang 2026 directed pose graph; Chen 2015 1-D orientation family; ")
        handle.write("Wu 2025 disconnected feasible domains; Schneyer 2023 coverage planning; Lu 2021 path-length objective\n")
        for item in side_paths:
            if item.status != "ok":
                handle.write(
                    f"# {item.branch_label} lateral_side={item.lateral_side:+d}: NO PATH "
                    f"(missing points {list(item.missing_point_indices)})\n"
                )
            else:
                handle.write(
                    f"# {item.branch_label} lateral_side={item.lateral_side:+d}: OK "
                    f"selected={len(item.nodes)} skipped={len(item.skipped_point_indices)} "
                    f"length_norm={item.length_norm}\n"
                )
        handle.write(f"mtllib {material_path.name}\n")
        handle.write("o root_line\nusemtl root_line_gray\n")
        next_vertex = _write_root_line(handle, sampled, branches, normalization)

        for item in side_paths:
            if item.status != "ok" or not item.nodes:
                continue
            line_material, node_material = _material_for_side(item.lateral_side)
            side_name = "pos" if item.lateral_side > 0 else "neg"
            points = np.asarray([node.center_norm for node in item.nodes], dtype=float)
            points_source = np.asarray([_to_source(point, normalization) for point in points], dtype=float)

            # 折线：每个选中中心写一个顶点，然后按顺序连 l。
            handle.write(
                f"o path_{item.branch_label}_{side_name}\nusemtl {line_material}\n"
            )
            vertex_ids: list[int] = []
            for point in points_source:
                vertex_ids.append(next_vertex)
                _write_vertex(handle, point)
                next_vertex += 1
            for first, second in zip(vertex_ids[:-1], vertex_ids[1:], strict=True):
                handle.write(f"l {first} {second}\n")

            # 节点标记：便于在 MeshLab 中看清每个采样点的选中中心。
            handle.write(
                f"o path_nodes_{item.branch_label}_{side_name}\nusemtl {node_material}\n"
            )
            for point in points_source:
                next_vertex = _write_center_marker(handle, point, marker_size, next_vertex)

            # 实心细管：MeshLab 对 OBJ l 元素的线宽支持有限，额外写入
            # 圆柱面，使相邻中心在普通视图中也显示为连续线段。
            if len(points_source) > 1:
                handle.write(
                    f"o path_tubes_{item.branch_label}_{side_name}\nusemtl {line_material}\n"
                )
                for first, second in zip(points_source[:-1], points_source[1:], strict=True):
                    next_vertex = _write_tube(
                        handle,
                        first,
                        second,
                        tube_radius,
                        next_vertex,
                    )
