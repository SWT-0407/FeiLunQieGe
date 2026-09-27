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
    if len(centers) != len(feasible) or len(centers) != len(point_indices):
        raise ValueError("候选中心、可行性标记和目标点索引长度不一致")
    if len(branch_ids) != len(curve_points):
        raise ValueError("根部线点和分支编号长度不一致")

    # 先按目标点收集可行中心的 OBJ 全局编号，随后再写 p 原语。
    feasible_indices = np.flatnonzero(feasible)
    point_to_obj_indices: dict[int, list[int]] = {}
    first_center_index = len(curve_points) + 1
    for local_index, candidate_index in enumerate(feasible_indices):
        point_index = int(point_indices[candidate_index])
        obj_index = first_center_index + local_index
        point_to_obj_indices.setdefault(point_index, []).append(obj_index)

    # 使用流式文本写入，避免额外复制几十万级候选数组。
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# FeiLunQieGe feasible-center export\n")
        handle.write(f"# units: {units}\n")
        handle.write("# wheel_model: zero_thickness_disk\n")
        handle.write(f"# radius: {radius:.12g}\n")
        handle.write(f"# delta: {delta:.12g}\n")
        handle.write(f"# root_line_vertex_count: {len(curve_points)}\n")
        handle.write(f"# feasible_center_count: {len(feasible_indices)}\n")
        handle.write("# p primitives are feasible wheel-center points; l primitives are sampled root-line segments.\n")

        # 根部线使用 l 原语保留分支的连续关系，分支之间不会被错误连接。
        handle.write("o root_line\n")
        handle.write("g root_line\n")
        for point in curve_points:
            handle.write(_vertex_line(point))
        for start, end in _branch_runs(branch_ids):
            for index in range(start, end - 1):
                handle.write(f"l {index + 1} {index + 2}\n")

        # 可行中心只写 p 原语，索引仍沿用 OBJ 文件中的全局顶点编号。
        handle.write("o feasible_centers\n")
        handle.write("g feasible_centers\n")
        for candidate_index in feasible_indices:
            handle.write(_vertex_line(centers[candidate_index]))
        for point_index in sorted(point_to_obj_indices):
            handle.write(f"g feasible_centers_point_{point_index:06d}\n")
            indices = point_to_obj_indices[point_index]
            handle.write("p " + " ".join(str(index) for index in indices) + "\n")

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
