"""将几何演示结果导出为 MeshLab 可直接打开的 OBJ 点/线文件。

OBJ 不只可以保存三角面，也可以用 ``p`` 保存点、用 ``l`` 保存线段。
本模块利用这两个原语，把可行砂轮中心和根部线写入同一个文件，便于
在 MeshLab 中直接检查空间位置关系。这里导出的不是砂轮实体网格。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def export_feasible_centers_obj(
    sampled_curve: Any,
    decisions: Any,
    output_path: str | Path,
    *,
    radius: float,
    delta: float,
    units: str = "normalized",
    coordinate_transform: dict[str, Any] | None = None,
) -> Path:
    """导出根部线和可行砂轮中心点。

    文件中先写采样后的根部线：每个采样点对应一个 ``v``，同一分支的
    相邻点用 ``l`` 连接。之后只写碰撞筛选通过的候选中心 ``v``，并按
    目标采样点分组，用 ``p`` 指向这些顶点。这样 MeshLab 无需依赖额外
    的材质文件也能显示结果；顶点颜色不写入，避免依赖非标准 OBJ 扩展。
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 将输入数组统一为只读式的数值视图，避免导出过程中意外修改算法结果。
    curve_points = np.asarray(sampled_curve.points, dtype=float)
    branch_ids = np.asarray(sampled_curve.branch_ids, dtype=np.int64)
    feasible = np.asarray(decisions.feasible, dtype=bool)
    candidates = decisions.candidates
    if candidates is None:
        raise ValueError("CollisionResult 缺少 candidates，无法导出可行中心")
    centers = np.asarray(candidates.centers, dtype=float)
    point_indices = np.asarray(candidates.point_indices, dtype=np.int64)
    angles = np.asarray(candidates.angles, dtype=float)
    radial_values = np.asarray(candidates.radial_values, dtype=float)
    if len(centers) != len(feasible) or len(centers) != len(point_indices):
        raise ValueError("候选中心、可行性标记和目标点索引长度不一致")
    if len(centers) != len(angles) or len(centers) != len(radial_values):
        raise ValueError("候选中心、角度和径向距离长度不一致")
    if len(branch_ids) != len(curve_points):
        raise ValueError("根部线点和分支编号长度不一致")

    # 算法在 normalized 坐标中计算；导出时可逆变换回原始 OBJ 坐标，
    # 从而让本文件能与 mesh_with_flash.obj 在 MeshLab 中直接叠加。
    coordinate_scale, coordinate_center = _parse_coordinate_transform(coordinate_transform)

    # 先按目标点收集可行中心的 OBJ 全局编号，随后再写 p 原语。
    feasible_indices = np.flatnonzero(feasible)
    point_to_obj_indices: dict[int, list[int]] = {}
    candidate_to_obj_index = np.full(len(centers), -1, dtype=np.int64)
    first_center_index = len(curve_points) + 1
    for local_index, candidate_index in enumerate(feasible_indices):
        point_index = int(point_indices[candidate_index])
        obj_index = first_center_index + local_index
        point_to_obj_indices.setdefault(point_index, []).append(obj_index)
        candidate_to_obj_index[candidate_index] = obj_index

    # 使用流式文本写入，避免额外复制几十万级候选数组。
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# FeiLunQieGe feasible-center export\n")
        handle.write(f"# units: {units}\n")
        handle.write("# wheel_model: finite_cylinder_collision_result_center_export\n")
        handle.write(f"# radius: {radius * coordinate_scale:.12g}\n")
        handle.write(f"# delta: {delta * coordinate_scale:.12g}\n")
        handle.write(f"# root_line_vertex_count: {len(curve_points)}\n")
        handle.write(f"# feasible_center_count: {len(feasible_indices)}\n")
        handle.write("# p primitives are feasible wheel-center points; l primitives are sampled root-line segments.\n")
        _write_coordinate_metadata(handle, coordinate_transform)

        # 根部线使用 l 原语保留分支的连续关系，分支之间不会被错误连接。
        handle.write("o root_line\n")
        handle.write("g root_line\n")
        for point in curve_points:
            handle.write(_vertex_line(_to_export_coordinates(point, coordinate_scale, coordinate_center)))
        for start, end in _branch_runs(branch_ids):
            for index in range(start, end - 1):
                handle.write(f"l {index + 1} {index + 2}\n")

        # 可行中心只写 p 原语，索引仍沿用 OBJ 文件中的全局顶点编号。
        handle.write("o feasible_centers\n")
        handle.write("g feasible_centers\n")
        for candidate_index in feasible_indices:
            handle.write(
                _vertex_line(_to_export_coordinates(centers[candidate_index], coordinate_scale, coordinate_center))
            )
        for point_index in sorted(point_to_obj_indices):
            handle.write(f"g feasible_centers_point_{point_index:06d}\n")
            indices = point_to_obj_indices[point_index]
            handle.write("p " + " ".join(str(index) for index in indices) + "\n")

        # 额外写出 rho=r 外圈的相邻可行中心连线，避免 MeshLab 默认不显示 p 点时
        # 只能看见根部线；这些 l 线段是“可行圆弧”的离散化，并非砂轮实体边界。
        handle.write("o feasible_center_arcs\n")
        handle.write("g feasible_center_arcs\n")
        for point_index in range(len(curve_points)):
            local = np.flatnonzero((point_indices == point_index) & np.isclose(radial_values, radius))
            if len(local) < 2:
                continue
            local = local[np.argsort(angles[local])]
            angle_step = _angle_step(angles[local])
            for local_index, current in enumerate(local):
                following = local[(local_index + 1) % len(local)]
                angle_gap = (angles[following] - angles[current]) % (2.0 * np.pi)
                if angle_gap > angle_step * 1.1:
                    continue
                first = int(candidate_to_obj_index[current])
                second = int(candidate_to_obj_index[following])
                if first > 0 and second > 0:
                    handle.write(f"l {first} {second}\n")

    return output


def export_pose_examples_obj(
    decisions: Any,
    output_path: str | Path,
    *,
    candidate_indices: dict[str, int],
    width: float,
    coordinate_transform: dict[str, Any] | None = None,
) -> Path:
    """导出可行/碰撞代表姿态的有限圆柱代理，供 MeshLab 叠加工件检查。"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = decisions.candidates
    if candidates is None:
        raise ValueError("CollisionResult 缺少 candidates，无法导出姿态")
    scale, center = _parse_coordinate_transform(coordinate_transform)

    def export_point(point: np.ndarray) -> np.ndarray:
        return np.asarray(point, dtype=float) * scale + center

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# FeiLunQieGe finite wheel pose examples\n")
        handle.write("# coordinate_frame: source_obj\n# units: source_obj coordinates\n")
        handle.write(f"# width: {width * scale:.12g}\n")
        handle.write(f"mtllib {output.with_suffix('.mtl').name}\n")
        next_vertex = 1
        for label, candidate_index in candidate_indices.items():
            if not 0 <= int(candidate_index) < len(candidates.centers):
                continue
            center_point = np.asarray(candidates.centers[int(candidate_index)], dtype=float)
            radial = np.asarray(candidates.radius_directions[int(candidate_index)], dtype=float)
            tangent = np.asarray(candidates.tangent_directions[int(candidate_index)], dtype=float)
            axis = np.asarray(candidates.axis_directions[int(candidate_index)], dtype=float)
            radius = float(candidates.radial_values[int(candidate_index)])
            theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
            vertices = []
            for axial in (-width / 2.0, width / 2.0):
                for angle in theta:
                    point = center_point + radius * (np.cos(angle) * radial + np.sin(angle) * tangent) + axial * axis
                    vertices.append(export_point(point))
            material = {"feasible": "feasible_pose", "collision": "collision_pose"}.get(label, "other_pose")
            handle.write(f"o pose_{label}\nusemtl {material}\n")
            for vertex in vertices:
                handle.write(_vertex_line(vertex))
            count = len(theta)
            for ring in range(count):
                following = (ring + 1) % count
                a, b, c, d = next_vertex + ring, next_vertex + following, next_vertex + count + following, next_vertex + count + ring
                handle.write(f"f {a} {b} {c} {d}\n")
            next_vertex += 2 * count
        _write_coordinate_metadata(handle, coordinate_transform)
    _write_pose_examples_mtl(output.with_suffix(".mtl"))
    return output


def _branch_runs(branch_ids: np.ndarray) -> list[tuple[int, int]]:
    """把连续的 branch_id 转成半开区间，供 ``l`` 原语逐段连接。"""

    if len(branch_ids) == 0:
        return []
    starts = [0]
    starts.extend((np.flatnonzero(branch_ids[1:] != branch_ids[:-1]) + 1).tolist())
    ends = starts[1:] + [len(branch_ids)]
    return list(zip(starts, ends, strict=True))


def _vertex_line(point: Any) -> str:
    """将三维坐标写为稳定、紧凑且保留足够精度的 OBJ ``v`` 行。"""

    x, y, z = (float(value) for value in np.asarray(point, dtype=float)[:3])
    return f"v {x:.12g} {y:.12g} {z:.12g}\n"


def _parse_coordinate_transform(transform: dict[str, Any] | None) -> tuple[float, np.ndarray]:
    """校验计算坐标到导出坐标的均匀缩放/平移变换。"""

    if transform is None:
        return 1.0, np.zeros(3, dtype=float)
    scale = float(transform["scale"])
    center = np.asarray(transform["center"], dtype=float)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("coordinate_transform.scale 必须是正的有限数")
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("coordinate_transform.center 必须是三个有限坐标")
    return scale, center


def _to_export_coordinates(point: Any, scale: float, center: np.ndarray) -> np.ndarray:
    """执行 source = normalized * scale + center，不修改算法内存中的数组。"""

    return np.asarray(point, dtype=float) * scale + center


def _write_coordinate_metadata(handle: Any, transform: dict[str, Any] | None) -> None:
    """将 OBJ 的实际坐标系写入文件头，防止查看时再次混用坐标框架。"""

    if transform is None:
        handle.write("# coordinate_frame: computation_normalized\n")
        return
    scale, center = _parse_coordinate_transform(transform)
    handle.write("# coordinate_frame: source_obj\n")
    handle.write("# computation_frame: normalized_bbox_longest_side\n")
    handle.write(
        "# source_transform: source = normalized * "
        f"{scale:.12g} + [{center[0]:.12g}, {center[1]:.12g}, {center[2]:.12g}]\n"
    )


def export_feasible_centers_display_obj(
    sampled_curve: Any,
    decisions: Any,
    output_path: str | Path,
    *,
    radius: float,
    delta: float,
    units: str = "normalized",
    coordinate_transform: dict[str, Any] | None = None,
    max_markers: int = 5000,
    marker_radius_factor: float = 0.08,
) -> Path:
    """导出适合 MeshLab 查看器的可行中心显示代理。

    原始 ``feasible_centers.obj`` 使用 OBJ 的 ``p`` 点原语保存全部算法点，
    但点的屏幕大小属于查看器设置，无法写进标准 OBJ。这里额外写入少量
    有三角面的八面体标记，因此红色中心不会因为点尺寸太小而难以辨认。
    八面体只是可视化代理，不是砂轮实体，也不参与碰撞计算。
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = _export_arrays(sampled_curve, decisions)
    curve_points, branch_ids, feasible, candidates = arrays
    centers = np.asarray(candidates.centers, dtype=float)
    point_indices = np.asarray(candidates.point_indices, dtype=np.int64)
    radial_values = np.asarray(candidates.radial_values, dtype=float)
    angles = np.asarray(candidates.angles, dtype=float)
    if max_markers < 1:
        raise ValueError("max_markers 必须至少为 1")
    if marker_radius_factor <= 0:
        raise ValueError("marker_radius_factor 必须为正数")

    # 所有位置和显示代理尺寸使用同一均匀逆变换，避免红色八面体与原始模型错位。
    coordinate_scale, coordinate_center = _parse_coordinate_transform(coordinate_transform)

    feasible_indices = np.flatnonzero(feasible)
    marker_indices = _evenly_spaced_indices(feasible_indices, max_markers)
    outer_indices = np.flatnonzero(feasible & np.isclose(radial_values, radius))
    outer_obj_index: dict[int, int] = {}
    marker_radius = float(radius) * float(marker_radius_factor) * coordinate_scale
    mtl_path = output.with_suffix(".mtl")
    _write_visualization_mtl(mtl_path)

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        _write_obj_metadata(
            handle,
            mtl_path.name,
            units=units,
            radius=radius * coordinate_scale,
            delta=delta * coordinate_scale,
            description="display proxy: sampled feasible centers as octahedron faces",
        )
        _write_coordinate_metadata(handle, coordinate_transform)
        # 根部线保留为 l 原语，便于把红色中心与真实目标位置对应起来。
        handle.write("o root_line\nusemtl root_line\ng root_line\n")
        for point in curve_points:
            handle.write(_vertex_line(_to_export_coordinates(point, coordinate_scale, coordinate_center)))
        for start, end in _branch_runs(branch_ids):
            for index in range(start, end - 1):
                handle.write(f"l {index + 1} {index + 2}\n")

        # 外半径上的可行点仍写成线段，作为可行圆弧的全局轮廓。
        handle.write("o feasible_center_arcs\nusemtl feasible_arc\ng feasible_center_arcs\n")
        next_vertex = len(curve_points) + 1
        for candidate_index in outer_indices:
            outer_obj_index[int(candidate_index)] = next_vertex
            handle.write(
                _vertex_line(_to_export_coordinates(centers[candidate_index], coordinate_scale, coordinate_center))
            )
            next_vertex += 1
        for point_index in range(len(curve_points)):
            local = outer_indices[point_indices[outer_indices] == point_index]
            if len(local) < 2:
                continue
            local = local[np.argsort(angles[local])]
            angle_step = _angle_step(angles[local])
            for local_index, current in enumerate(local):
                following = local[(local_index + 1) % len(local)]
                angle_gap = (angles[following] - angles[current]) % (2.0 * np.pi)
                if angle_gap <= angle_step * 1.1:
                    handle.write(f"l {outer_obj_index[int(current)]} {outer_obj_index[int(following)]}\n")

        # 只对均匀抽样后的可行中心写面片，控制文件大小并保持全局图可交互。
        handle.write("o feasible_center_markers\nusemtl feasible_marker\ng feasible_center_markers\n")
        for candidate_index in marker_indices:
            next_vertex = _write_octahedron(
                handle,
                _to_export_coordinates(centers[candidate_index], coordinate_scale, coordinate_center),
                candidates.tangent_directions[candidate_index],
                candidates.radius_directions[candidate_index],
                candidates.axis_directions[candidate_index],
                marker_radius,
                next_vertex,
            )

    return output


def export_local_feasible_centers_obj(
    sampled_curve: Any,
    decisions: Any,
    output_path: str | Path,
    *,
    radius: float,
    delta: float,
    units: str = "normalized",
    coordinate_transform: dict[str, Any] | None = None,
    point_index: int | None = None,
    marker_radius_factor: float = 0.15,
) -> Path:
    """导出一个目标点的局部放大 OBJ，便于 MeshLab 直接 Fit View。

    文件包含该目标点、该点的全部候选中心以及可行/不可行分类。可行中心
    使用红色八面体，不可行中心使用灰色八面体；颜色来自同目录 MTL 文件。
    该文件用于解释和人工验收，不替代包含全部点的精确 OBJ。
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    curve_points, _branch_ids, feasible, candidates = _export_arrays(sampled_curve, decisions)
    centers = np.asarray(candidates.centers, dtype=float)
    candidate_points = np.asarray(candidates.point_indices, dtype=np.int64)
    radial_values = np.asarray(candidates.radial_values, dtype=float)
    angles = np.asarray(candidates.angles, dtype=float)
    if point_index is None:
        point_index = _choose_explanation_point(candidate_points, feasible)
    if point_index < 0 or point_index >= len(curve_points):
        raise ValueError("point_index 超出采样点范围")
    local_indices = np.flatnonzero(candidate_points == int(point_index))
    if len(local_indices) == 0:
        raise ValueError("选定采样点没有候选中心")

    # 局部 OBJ 与全局 OBJ 使用相同导出坐标系，方便与原始工件直接叠加。
    coordinate_scale, coordinate_center = _parse_coordinate_transform(coordinate_transform)

    mtl_path = output.with_suffix(".mtl")
    _write_visualization_mtl(mtl_path)
    marker_radius = float(radius) * float(marker_radius_factor) * coordinate_scale
    target = curve_points[int(point_index)]
    next_vertex = 1
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        _write_obj_metadata(
            handle,
            mtl_path.name,
            units=units,
            radius=radius * coordinate_scale,
            delta=delta * coordinate_scale,
            description=f"local display proxy for sampled target point {point_index}",
        )
        _write_coordinate_metadata(handle, coordinate_transform)
        # 目标接触点使用黑色八面体，避免在大量候选中心中找不到原点。
        handle.write(f"o target_point_{point_index:06d}\nusemtl target_point\ng target_point\n")
        next_vertex = _write_octahedron(
            handle,
            _to_export_coordinates(target, coordinate_scale, coordinate_center),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            marker_radius * 1.35,
            next_vertex,
        )

        # 分离可行和不可行组，便于在 MeshLab 图层/材质中快速识别。
        for label, material, mask in (
            ("feasible_candidates", "feasible_marker", feasible),
            ("rejected_candidates", "rejected_marker", ~feasible),
        ):
            handle.write(f"o {label}\nusemtl {material}\ng {label}\n")
            for candidate_index in local_indices[mask[local_indices]]:
                next_vertex = _write_octahedron(
                    handle,
                    _to_export_coordinates(centers[candidate_index], coordinate_scale, coordinate_center),
                    candidates.tangent_directions[candidate_index],
                    candidates.radius_directions[candidate_index],
                    candidates.axis_directions[candidate_index],
                    marker_radius,
                    next_vertex,
                )

        # 只连接外半径上的可行中心，得到该点的离散可行圆弧。
        handle.write("o local_feasible_arc\nusemtl feasible_arc\ng local_feasible_arc\n")
        outer = local_indices[feasible[local_indices] & np.isclose(radial_values[local_indices], radius)]
        outer = outer[np.argsort(angles[outer])]
        outer_vertex_ids: dict[int, int] = {}
        for candidate_index in outer:
            outer_vertex_ids[int(candidate_index)] = next_vertex
            handle.write(
                _vertex_line(_to_export_coordinates(centers[candidate_index], coordinate_scale, coordinate_center))
            )
            next_vertex += 1
        if len(outer) > 1:
            angle_step = _angle_step(angles[local_indices])
            for local_index, current in enumerate(outer):
                following = outer[(local_index + 1) % len(outer)]
                angle_gap = (angles[following] - angles[current]) % (2.0 * np.pi)
                if angle_gap <= angle_step * 1.1:
                    handle.write(f"l {outer_vertex_ids[int(current)]} {outer_vertex_ids[int(following)]}\n")

    return output


def _export_arrays(sampled_curve: Any, decisions: Any) -> tuple[Any, Any, Any, Any]:
    """集中校验 OBJ 导出需要的数组，避免显示文件静默错位。"""

    curve_points = np.asarray(sampled_curve.points, dtype=float)
    branch_ids = np.asarray(sampled_curve.branch_ids, dtype=np.int64)
    feasible = np.asarray(decisions.feasible, dtype=bool)
    candidates = decisions.candidates
    if candidates is None:
        raise ValueError("CollisionResult 缺少 candidates，无法导出可行中心")
    centers = np.asarray(candidates.centers, dtype=float)
    point_indices = np.asarray(candidates.point_indices, dtype=np.int64)
    if len(centers) != len(feasible) or len(centers) != len(point_indices):
        raise ValueError("候选中心、可行性标签和目标点索引长度不一致")
    if len(branch_ids) != len(curve_points):
        raise ValueError("根部线点和分支编号长度不一致")
    return curve_points, branch_ids, feasible, candidates


def _angle_step(angles: np.ndarray) -> float:
    """根据原始角度网格估计步长，而不是按剩余可行点数量重算。"""

    unique = np.unique(np.asarray(angles, dtype=float))
    if len(unique) < 2:
        return 2.0 * np.pi
    gaps = np.diff(np.r_[unique, unique[0] + 2.0 * np.pi])
    return float(np.min(gaps))


def _evenly_spaced_indices(indices: np.ndarray, limit: int) -> np.ndarray:
    """从有序索引中均匀抽样，保证显示代理具有确定性。"""

    if len(indices) <= limit:
        return indices
    positions = np.linspace(0, len(indices) - 1, limit, dtype=np.int64)
    return indices[positions]


def _choose_explanation_point(point_indices: np.ndarray, feasible: np.ndarray) -> int:
    """选择可行率接近中间值的目标点，作为局部解释默认对象。"""

    point_count = int(point_indices.max(initial=-1)) + 1
    ratios = []
    for index in range(point_count):
        mask = point_indices == index
        ratios.append(float(feasible[mask].mean()) if mask.any() else 0.0)
    if not ratios:
        raise ValueError("没有可用的目标采样点")
    return int(np.argmin(np.abs(np.asarray(ratios) - 0.5)))


def _write_octahedron(
    handle: Any,
    center: Any,
    tangent: Any,
    radial: Any,
    axis: Any,
    marker_radius: float,
    first_vertex: int,
) -> int:
    """写入一个局部正交八面体，并返回下一个可用的 OBJ 顶点编号。"""

    basis = []
    for vector in (tangent, radial, axis):
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            raise ValueError("显示标记的局部方向不能为零向量")
        basis.append(vector / norm)
    center = np.asarray(center, dtype=float)
    vertices = [
        center + marker_radius * basis[0],
        center - marker_radius * basis[0],
        center + marker_radius * basis[1],
        center - marker_radius * basis[1],
        center + marker_radius * basis[2],
        center - marker_radius * basis[2],
    ]
    for vertex in vertices:
        handle.write(_vertex_line(vertex))
    a, b, c, d, e, f = range(first_vertex, first_vertex + 6)
    for triangle in ((a, c, e), (a, e, d), (a, d, f), (a, f, c), (b, e, c), (b, d, e), (b, f, d), (b, c, f)):
        handle.write(f"f {triangle[0]} {triangle[1]} {triangle[2]}\n")
    return first_vertex + 6


def _write_obj_metadata(handle: Any, mtl_name: str, *, units: str, radius: float, delta: float, description: str) -> None:
    """写入显示文件的来源、单位和参数，防止显示代理被误当成实体模型。"""

    handle.write("# FeiLunQieGe visualization export\n")
    handle.write(f"# units: {units}\n# radius: {radius:.12g}\n# delta: {delta:.12g}\n")
    handle.write(f"# {description}\nmtllib {mtl_name}\n")


def _write_visualization_mtl(path: Path) -> None:
    """写入常见 OBJ/MTL 查看器都能识别的基础颜色。"""

    path.write_text(
        """# FeiLunQieGe display materials
newmtl root_line
Kd 0.05 0.40 0.90
newmtl feasible_arc
Kd 1.00 0.55 0.00
newmtl feasible_marker
Kd 0.90 0.05 0.05
newmtl rejected_marker
Kd 0.55 0.55 0.55
newmtl target_point
Kd 0.05 0.05 0.05
""",
        encoding="utf-8",
    )


def _write_pose_examples_mtl(path: Path) -> None:
    """为姿态示例提供语义颜色：绿色可行，红橙色碰撞。"""

    path.write_text(
        """# FeiLunQieGe finite wheel pose semantics
# feasible_pose: collision check passed
newmtl feasible_pose
Kd 0.00 0.62 0.45
# collision_pose: collision or insufficient clearance detected
newmtl collision_pose
Kd 0.84 0.20 0.10
# fallback for future labels
newmtl other_pose
Kd 0.55 0.55 0.55
""",
        encoding="utf-8",
    )
