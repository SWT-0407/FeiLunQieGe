"""在已有离散候选中心上做 root_line 两侧的分层最短路径。

问题定义
--------
对每个 root_line branch，把 branch 内第 i 个采样点看成一层；该层每个
可行候选中心是节点。只允许相邻层节点连边，边权为两个砂轮中心的三维
欧氏距离。分别在 ``lateral_side = +1`` 和 ``-1`` 上求最短路径。

侧向定义
--------
当前候选中心写成

    e(theta) = cos(theta) * t + sin(theta) * q
    q = normalize(a x t)

其中 ``t`` 是投影到切平面内的曲线切线，``a`` 是砂轮轴向（工件表面法向），
``q`` 是切平面内垂直于 ``t`` 的横向基。路径左右侧用

    lateral_side = sign(e . q)

区分；``theta = 0/180`` 时 ``e . q = 0``，中心落在 root_line 的切线方向，
不属于任何一侧，因此在两侧路径中排除。注意代码中的 ``side=+1`` 是砂轮
轴向侧（有限圆柱中心沿 ``a`` 偏移 ``w/2``），不是这里的路径左右侧。

算法
----
分层图是 DAG，直接用动态规划求全局最优：

    D[i+1][k] = min_j ( D[i][j] + || c[i][j] - c[i+1][k] || )

回溯得到每层选中的候选索引。该实现还计算最优路径条数（上限 2），用于
说明是否存在多组同长解；若存在，只导出确定性的第一条（最低候选索引）。

边界
----
- 只做节点级可行性；本轮不做边级碰撞审计。
- 侧向一致性只要求“砂轮中心”在指定一侧；砂轮其他部分可以越过该侧，
  这是用户 2026-10-31 明确确认的规则。
- 某层没有该侧可行节点时，整侧标记为 no_path，不跳过、不伪造。

方法借鉴
--------
- Zhang et al. 2026 (Machines)：把路径点上的候选姿态组织成有向图；
  本实现采用同样的分层图结构，但用 DP 而不是 A*。
- Chen et al. 2015 (JCDE)：把刀轴限制在一维参考平面族内；本实现用
  ``lateral_side`` 把每侧限制为半圆角度族。
- Wu et al. 2025 (TMech)：可行姿态域可能断开；本实现不假设可行角连续。
- Lu et al. 2021 (JMSE)：路径长度目标；本实现最小化中心折线总长。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

# 单位向量点积接近 0 时视为中线；theta=0/180 的数值误差约 1e-16。
LATERAL_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PathNode:
    """路径上的一个选中中心。"""

    point_index: int
    candidate_index: int
    angle_deg: float
    lateral_projection: float
    center_norm: np.ndarray
    reason: str
    axis_norm: np.ndarray | None = None
    coverage_start_norm: float | None = None
    coverage_end_norm: float | None = None

    def to_record(self, normalization: dict[str, Any]) -> dict[str, Any]:
        """转换为 JSON 记录；同时给出 normalized 与 source_obj 坐标。"""

        scale = float(normalization.get("scale", 1.0))
        center = np.asarray(normalization.get("center", [0.0, 0.0, 0.0]), dtype=float)
        center_source = self.center_norm * scale + center
        return {
            "point_index": int(self.point_index),
            "candidate_index": int(self.candidate_index),
            "angle_deg": float(self.angle_deg),
            "lateral_projection": float(self.lateral_projection),
            "center_norm": [float(value) for value in self.center_norm],
            "center_source": [float(value) for value in center_source],
            "axis_norm": (
                None if self.axis_norm is None else [float(value) for value in self.axis_norm]
            ),
            "reason": str(self.reason),
            "coverage_start_norm": self.coverage_start_norm,
            "coverage_end_norm": self.coverage_end_norm,
        }


@dataclass(frozen=True)
class SidePath:
    """一个 branch、一个侧向的最短路径结果。"""

    branch_id: int
    branch_label: str
    lateral_side: int
    status: str
    nodes: tuple[PathNode, ...]
    length_norm: float | None
    max_segment_length_norm: float | None
    max_angle_step_deg: float | None
    mean_angle_step_deg: float | None
    max_axis_step_deg: float | None = None
    mean_axis_step_deg: float | None = None
    missing_point_indices: tuple[int, ...] = ()
    candidate_counts: tuple[int, ...] = ()
    optimal_path_count_capped: int = 0
    tie_break: str = "lowest_candidate_index"
    message: str = ""
    coverage_half_length: float | None = None
    coverage_gaps_norm: tuple[float, ...] = ()
    coverage_union_ok: bool | None = None
    skipped_point_indices: tuple[int, ...] = ()
    missing_feasible_point_indices: tuple[int, ...] = ()

    def to_record(self, normalization: dict[str, Any], *, include_nodes: bool = True) -> dict[str, Any]:
        record: dict[str, Any] = {
            "branch_id": int(self.branch_id),
            "branch_label": self.branch_label,
            "lateral_side": int(self.lateral_side),
            "lateral_side_name": "+q" if self.lateral_side > 0 else "-q",
            "status": self.status,
            "point_count": len(self.nodes) if self.status == "ok" else 0,
            "selected_point_count": len(self.nodes) if self.status == "ok" else 0,
            "skipped_point_count": len(self.skipped_point_indices),
            "skipped_point_indices": [int(value) for value in self.skipped_point_indices],
            "missing_feasible_point_indices": [int(value) for value in self.missing_feasible_point_indices],
            "length_norm": self.length_norm,
            "length_source": (
                None if self.length_norm is None else float(self.length_norm) * float(normalization.get("scale", 1.0))
            ),
            "max_segment_length_norm": self.max_segment_length_norm,
            "max_angle_step_deg": self.max_angle_step_deg,
            "mean_angle_step_deg": self.mean_angle_step_deg,
            "max_axis_step_deg": self.max_axis_step_deg,
            "mean_axis_step_deg": self.mean_axis_step_deg,
            "missing_point_indices": [int(value) for value in self.missing_point_indices],
            "candidate_counts": [int(value) for value in self.candidate_counts],
            "optimal_path_count_capped": int(self.optimal_path_count_capped),
            "tie_break": self.tie_break,
            "message": self.message,
            "coverage_half_length": self.coverage_half_length,
            "coverage_gaps_norm": [float(value) for value in self.coverage_gaps_norm],
            "coverage_union_ok": self.coverage_union_ok,
            "side_consistency_rule": "wheel_center_only",
        }
        if include_nodes and self.status == "ok":
            record["nodes"] = [node.to_record(normalization) for node in self.nodes]
        return record


def branch_runs(branch_ids: Any) -> list[tuple[int, int, int]]:
    """把连续 branch 编号切成 ``(branch_id, start, end)``；end 为开区间。"""

    values = np.asarray(branch_ids, dtype=np.int64)
    if len(values) == 0:
        return []
    starts = [0] + (np.flatnonzero(values[1:] != values[:-1]) + 1).tolist()
    ends = starts[1:] + [len(values)]
    return [(int(values[start]), int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _candidate_frame_q(candidates: Any) -> np.ndarray:
    """取一个点所有候选共用的 ``q = normalize(a x t)``。

    候选集由同一局部标架生成，``axis_directions`` 与 ``tangent_directions``
    在每个角度上重复。这里只取第一个候选，并检查其余候选的轴向/切向是否
    与它一致；不一致说明候选集混用了多个标架，不能静默做侧向分类。
    """

    axis = np.asarray(candidates.axis_directions, dtype=float)
    tangent = np.asarray(candidates.tangent_directions, dtype=float)
    if len(axis) == 0 or len(tangent) == 0:
        raise ValueError("候选集为空，无法定义侧向")
    axis_norm = np.linalg.norm(axis, axis=1)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    if np.any(axis_norm <= 1e-12) or np.any(tangent_norm <= 1e-12):
        raise ValueError("候选姿态含零向量，无法定义侧向")
    axis_unit = axis / axis_norm[:, None]
    tangent_unit = tangent / tangent_norm[:, None]
    if len(axis_unit) > 1:
        if np.min(np.einsum("ij,j->i", axis_unit, axis_unit[0])) < 1.0 - 1e-10:
            raise ValueError("同一采样点的候选砂轮轴向不一致")
        if np.min(np.einsum("ij,j->i", tangent_unit, tangent_unit[0])) < 1.0 - 1e-10:
            raise ValueError("同一采样点的候选曲线切线不一致")
    q = np.cross(axis_unit[0], tangent_unit[0])
    q_norm = float(np.linalg.norm(q))
    if q_norm <= 1e-12:
        raise ValueError("q = a x t 退化，无法定义侧向")
    return q / q_norm


def _classify_with_q(candidates: Any, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用指定的横向基 ``q`` 计算 ``e . q`` 和角度。"""

    tangent = np.asarray(candidates.tangent_directions, dtype=float)
    radial = np.asarray(candidates.radius_directions, dtype=float)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    radial_norm = np.linalg.norm(radial, axis=1)
    if np.any(tangent_norm <= 1e-12) or np.any(radial_norm <= 1e-12):
        raise ValueError("候选姿态含零向量，无法定义侧向")
    tangent = tangent / tangent_norm[:, None]
    radial = radial / radial_norm[:, None]
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    lateral = np.einsum("ij,j->i", radial, q)
    cosine = np.einsum("ij,ij->i", radial, tangent)
    angles = np.degrees(np.mod(np.arctan2(lateral, cosine), 2.0 * np.pi))
    angles = np.where(np.isclose(angles, 360.0, atol=1e-9), 0.0, angles)
    return lateral, angles


def classify_branch_lateral(
    point_indices: Iterable[int],
    audits: list[Any],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """按 branch 顺序计算侧向投影，并强制 ``q`` 符号连续。

    这是 2026-10-31 针对 branch 7 退化“往返”分支的修复。root_line 在
    某些 branch 上可能沿同一条空间线往返，导致相邻点的切线翻转 180°；
    若逐点独立取 ``q = a x t``，同一条物理侧向会被分成 +q/-q 两段，
    使两条路径都无解。这里在相邻点之间检查 ``q_i . q_{i-1}``，若为负则
    翻转 ``q_i``，保证同一 branch 的侧向语义一致。

    返回 ``(lateral_list, angle_list)``，顺序与 ``point_indices`` 一致。
    """

    indices = np.asarray(list(point_indices), dtype=np.int64)
    lateral_list: list[np.ndarray] = []
    angle_list: list[np.ndarray] = []
    previous_q: np.ndarray | None = None
    for point_index in indices:
        audit = audits[int(point_index)]
        if audit is None:
            raise ValueError(f"点 {int(point_index)} 缺少碰撞审计，无法分类侧向")
        candidates = audit.candidates
        q = _candidate_frame_q(candidates)
        if previous_q is not None and float(np.dot(q, previous_q)) < 0.0:
            q = -q
        previous_q = q
        lateral, angles = _classify_with_q(candidates, q)
        lateral_list.append(lateral)
        angle_list.append(angles)
    return lateral_list, angle_list


def classify_lateral(candidates: Any, tolerance: float = LATERAL_TOLERANCE) -> tuple[np.ndarray, np.ndarray]:
    """返回每个候选的 ``e . q`` 和角度（度）。

    ``e`` 是候选的 radial 方向，``q = normalize(a x t)``。返回值第一项
    的符号就是路径左右侧；第二项用于诊断相邻角度跳变。
    """

    if tolerance < 0.0:
        raise ValueError("lateral tolerance 不能为负")
    axis = np.asarray(candidates.axis_directions, dtype=float)
    tangent = np.asarray(candidates.tangent_directions, dtype=float)
    radial = np.asarray(candidates.radius_directions, dtype=float)
    if not (len(axis) == len(tangent) == len(radial)):
        raise ValueError("候选的 axis/tangent/radial 数量不一致")

    axis_norm = np.linalg.norm(axis, axis=1)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    radial_norm = np.linalg.norm(radial, axis=1)
    if np.any(axis_norm <= 1e-12) or np.any(tangent_norm <= 1e-12) or np.any(radial_norm <= 1e-12):
        raise ValueError("候选姿态含零向量，无法定义侧向")

    axis = axis / axis_norm[:, None]
    tangent = tangent / tangent_norm[:, None]
    radial = radial / radial_norm[:, None]
    q = np.cross(axis, tangent)
    q_norm = np.linalg.norm(q, axis=1)
    if np.any(q_norm <= 1e-12):
        raise ValueError("q = a x t 退化，无法定义侧向")
    q = q / q_norm[:, None]

    # e . q = sin(theta)，e . t = cos(theta)。
    lateral = np.einsum("ij,ij->i", radial, q)
    cosine = np.einsum("ij,ij->i", radial, tangent)
    angles = np.degrees(np.mod(np.arctan2(lateral, cosine), 2.0 * np.pi))
    # 排除浮点误差造成的 360.0，统一映射回 [0, 360)。
    angles = np.where(np.isclose(angles, 360.0, atol=1e-9), 0.0, angles)
    return lateral, angles


def _side_mask(lateral: np.ndarray, lateral_side: int, tolerance: float) -> np.ndarray:
    if lateral_side not in (-1, 1):
        raise ValueError("lateral_side 必须是 +1 或 -1")
    if lateral_side > 0:
        return lateral > tolerance
    return lateral < -tolerance


def plan_side_path(
    point_indices: Iterable[int],
    audits: list[Any],
    lateral_side: int,
    *,
    branch_id: int = -1,
    tolerance: float = LATERAL_TOLERANCE,
) -> SidePath:
    """求一个 branch 在一个侧向上的分层最短路径。

    ``audits[i]`` 需要提供 ``candidates``（CandidateSet）和 ``result``
    （至少含 ``feasible`` 与 ``reason``）。若某层没有该侧可行节点，返回
    ``status="no_path"`` 并记录所有缺失点。
    """

    indices = np.asarray(list(point_indices), dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("point_indices 不能为空")
    branch_label = f"branch_{int(branch_id):02d}" if branch_id >= 0 else "branch_unknown"

    layer_candidate_indices: list[np.ndarray] = []
    layer_centers: list[np.ndarray] = []
    layer_angles: list[np.ndarray] = []
    layer_lateral: list[np.ndarray] = []
    layer_reasons: list[np.ndarray] = []
    candidate_counts: list[int] = []
    missing: list[int] = []
    previous_q: np.ndarray | None = None

    for point_index in indices:
        audit = audits[int(point_index)]
        if audit is None:
            missing.append(int(point_index))
            candidate_counts.append(0)
            layer_candidate_indices.append(np.empty(0, dtype=np.int64))
            layer_centers.append(np.empty((0, 3), dtype=float))
            layer_angles.append(np.empty(0, dtype=float))
            layer_lateral.append(np.empty(0, dtype=float))
            layer_reasons.append(np.empty(0, dtype=object))
            continue
        candidates = audit.candidates
        feasible = np.asarray(audit.result.feasible, dtype=bool)
        # branch 内强制 q 符号连续：退化“往返”分支的切线会翻转 180°，
        # 若逐点独立取 q，会把同一物理侧向拆成 +q/-q 两段。
        q = _candidate_frame_q(candidates)
        if previous_q is not None and float(np.dot(q, previous_q)) < 0.0:
            q = -q
        previous_q = q
        lateral, angles = _classify_with_q(candidates, q)
        mask = feasible & _side_mask(lateral, lateral_side, tolerance)
        local_indices = np.flatnonzero(mask)
        centers = np.asarray(candidates.centers, dtype=float)[local_indices]
        reasons = np.asarray(audit.result.reason, dtype=object)[local_indices]
        candidate_counts.append(int(len(local_indices)))
        if len(local_indices) == 0:
            missing.append(int(point_index))
        layer_candidate_indices.append(local_indices.astype(np.int64))
        layer_centers.append(centers)
        layer_angles.append(angles[local_indices])
        layer_lateral.append(lateral[local_indices])
        layer_reasons.append(reasons)

    if missing:
        message = (
            f"no feasible candidate on lateral_side={lateral_side:+d} "
            f"for {len(missing)} point(s): {missing}"
        )
        return SidePath(
            branch_id=branch_id,
            branch_label=branch_label,
            lateral_side=int(lateral_side),
            status="no_path",
            nodes=(),
            length_norm=None,
            max_segment_length_norm=None,
            max_angle_step_deg=None,
            mean_angle_step_deg=None,
            missing_point_indices=tuple(missing),
            candidate_counts=tuple(candidate_counts),
            optimal_path_count_capped=0,
            tie_break="lowest_candidate_index",
            message=message,
        )

    # 分层动态规划。每层节点数最多 72，二维距离矩阵很小。
    layer_count = len(layer_centers)
    forward: list[np.ndarray] = [np.zeros(len(layer_centers[0]), dtype=float)]
    choices: list[np.ndarray] = []
    for position in range(1, layer_count):
        previous = layer_centers[position - 1]
        current = layer_centers[position]
        distances = np.linalg.norm(current[:, None, :] - previous[None, :, :], axis=2)
        costs = distances + forward[-1][None, :]
        choice = np.argmin(costs, axis=1)
        forward.append(costs[np.arange(len(current)), choice])
        choices.append(choice)

    best_last = int(np.argmin(forward[-1]))
    best_cost = float(forward[-1][best_last])
    selected_local: list[int] = [best_last]
    for choice in reversed(choices):
        selected_local.append(int(choice[selected_local[-1]]))
    selected_local.reverse()

    # 反向代价，用于统计同长最优路径条数（上限 2）。
    backward: list[np.ndarray] = [np.empty(0, dtype=float)] * layer_count
    backward[-1] = np.zeros(len(layer_centers[-1]), dtype=float)
    for position in range(layer_count - 2, -1, -1):
        current = layer_centers[position]
        following = layer_centers[position + 1]
        distances = np.linalg.norm(current[:, None, :] - following[None, :, :], axis=2)
        backward[position] = np.min(distances + backward[position + 1][None, :], axis=1)

    cost_tolerance = 1e-9 * max(1.0, abs(best_cost))
    counts: list[np.ndarray] = [np.ones(len(layer_centers[0]), dtype=np.int64)]
    for position in range(1, layer_count):
        previous = layer_centers[position - 1]
        current = layer_centers[position]
        distances = np.linalg.norm(current[:, None, :] - previous[None, :, :], axis=2)
        new_counts = np.zeros(len(current), dtype=np.int64)
        for node in range(len(current)):
            valid = (
                forward[position - 1]
                + distances[node, :]
                + backward[position][node]
                <= best_cost + cost_tolerance
            )
            new_counts[node] = min(2, int(counts[position - 1][valid].sum()))
        counts.append(new_counts)
    optimal_path_count = min(2, int(counts[-1].sum()))

    nodes: list[PathNode] = []
    for position, local_index in enumerate(selected_local):
        candidate_index = int(layer_candidate_indices[position][local_index])
        axis = np.asarray(
            audits[int(indices[position])].candidates.axis_directions[candidate_index],
            dtype=float,
        )
        nodes.append(
            PathNode(
                point_index=int(indices[position]),
                candidate_index=candidate_index,
                angle_deg=float(layer_angles[position][local_index]),
                lateral_projection=float(layer_lateral[position][local_index]),
                center_norm=np.asarray(layer_centers[position][local_index], dtype=float).copy(),
                reason=str(layer_reasons[position][local_index]),
                axis_norm=axis.copy(),
            )
        )

    selected_centers = np.asarray([node.center_norm for node in nodes], dtype=float)
    selected_angles = np.asarray([node.angle_deg for node in nodes], dtype=float)
    selected_axes = np.asarray([node.axis_norm for node in nodes], dtype=float)
    segments = np.linalg.norm(np.diff(selected_centers, axis=0), axis=1) if len(nodes) > 1 else np.empty(0)
    angle_delta = np.abs(np.diff(selected_angles)) if len(nodes) > 1 else np.empty(0)
    if len(angle_delta):
        angle_delta = np.minimum(angle_delta, 360.0 - angle_delta)
    axis_delta = (
        np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", selected_axes[:-1], selected_axes[1:]), -1.0, 1.0)))
        if len(nodes) > 1
        else np.empty(0)
    )

    tie_message = (
        f"ok; {optimal_path_count} optimal path(s) up to cap 2; exported lowest-candidate-index path"
        if optimal_path_count > 1
        else "ok"
    )
    return SidePath(
        branch_id=branch_id,
        branch_label=branch_label,
        lateral_side=int(lateral_side),
        status="ok",
        nodes=tuple(nodes),
        length_norm=best_cost,
        max_segment_length_norm=float(segments.max()) if len(segments) else 0.0,
        max_angle_step_deg=float(angle_delta.max()) if len(angle_delta) else 0.0,
        mean_angle_step_deg=float(angle_delta.mean()) if len(angle_delta) else 0.0,
        max_axis_step_deg=float(axis_delta.max()) if len(axis_delta) else 0.0,
        mean_axis_step_deg=float(axis_delta.mean()) if len(axis_delta) else 0.0,
        missing_point_indices=(),
        candidate_counts=tuple(candidate_counts),
        optimal_path_count_capped=optimal_path_count,
        tie_break="lowest_candidate_index",
        message=tie_message,
    )


def _build_side_layers(
    indices: np.ndarray,
    audits: list[Any],
    lateral_side: int,
    tolerance: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int], list[int]]:
    """构建一个 branch 某侧的可行候选层，并保持 branch 内 q 符号连续。

    返回 ``(candidate_indices, centers, angles, lateral, reasons,
    candidate_counts, missing_feasible_points)``。
    """

    layer_candidate_indices: list[np.ndarray] = []
    layer_centers: list[np.ndarray] = []
    layer_angles: list[np.ndarray] = []
    layer_lateral: list[np.ndarray] = []
    layer_reasons: list[np.ndarray] = []
    candidate_counts: list[int] = []
    missing_feasible: list[int] = []
    previous_q: np.ndarray | None = None
    for point_index in indices:
        audit = audits[int(point_index)]
        if audit is None:
            candidate_counts.append(0)
            missing_feasible.append(int(point_index))
            layer_candidate_indices.append(np.empty(0, dtype=np.int64))
            layer_centers.append(np.empty((0, 3), dtype=float))
            layer_angles.append(np.empty(0, dtype=float))
            layer_lateral.append(np.empty(0, dtype=float))
            layer_reasons.append(np.empty(0, dtype=object))
            continue
        candidates = audit.candidates
        feasible = np.asarray(audit.result.feasible, dtype=bool)
        q = _candidate_frame_q(candidates)
        if previous_q is not None and float(np.dot(q, previous_q)) < 0.0:
            q = -q
        previous_q = q
        lateral, angles = _classify_with_q(candidates, q)
        mask = feasible & _side_mask(lateral, lateral_side, tolerance)
        local_indices = np.flatnonzero(mask)
        centers = np.asarray(candidates.centers, dtype=float)[local_indices]
        reasons = np.asarray(audit.result.reason, dtype=object)[local_indices]
        candidate_counts.append(int(len(local_indices)))
        if len(local_indices) == 0:
            missing_feasible.append(int(point_index))
        layer_candidate_indices.append(local_indices.astype(np.int64))
        layer_centers.append(centers)
        layer_angles.append(angles[local_indices])
        layer_lateral.append(lateral[local_indices])
        layer_reasons.append(reasons)
    return (
        layer_candidate_indices,
        layer_centers,
        layer_angles,
        layer_lateral,
        layer_reasons,
        candidate_counts,
        missing_feasible,
    )


def _pairwise_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """返回 ``first`` 每点到 ``second`` 每点的欧氏距离矩阵。"""

    return np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)


def _coverage_bounds(
    arc: np.ndarray,
    missing_mask: np.ndarray,
    half: float,
) -> tuple[np.ndarray, np.ndarray]:
    """计算每个点的“保守覆盖区间”。

    几何覆盖本来是对称的 ``[s_i-h, s_i+h]``；但如果某个采样点在该侧
    完全没有可行候选（现有全域碰撞检测的结果），就不能声称它被覆盖。
    这里在最近的不可覆盖点与相邻可覆盖点之间取中点作为边界，阻止覆盖
    区间跨过该点。这样“被跳过的点也必须是该侧可碰撞可行的点”，避免
    用几何半径跨过整段碰撞点。
    """

    count = len(arc)
    left = np.zeros(count, dtype=float)
    right = np.full(count, float(arc[-1]), dtype=float)
    last_bad = -1
    for index in range(count):
        if bool(missing_mask[index]):
            last_bad = index
        if last_bad < 0:
            left[index] = 0.0
        elif last_bad + 1 < count:
            left[index] = 0.5 * (float(arc[last_bad]) + float(arc[last_bad + 1]))
        else:
            left[index] = float(arc[-1])
    next_bad = count
    for index in range(count - 1, -1, -1):
        if bool(missing_mask[index]):
            next_bad = index
        if next_bad == count:
            right[index] = float(arc[-1])
        elif next_bad > 0:
            right[index] = 0.5 * (float(arc[next_bad - 1]) + float(arc[next_bad]))
        else:
            right[index] = 0.0
    # 再与几何半径区间取交。
    coverage_left = np.maximum(arc - half, left)
    coverage_right = np.minimum(arc + half, right)
    return coverage_left, coverage_right


def _count_optimal_coverage_paths(
    layer_centers: list[np.ndarray],
    arc: np.ndarray,
    coverage_left: np.ndarray,
    coverage_right: np.ndarray,
    forward: list[np.ndarray],
    best_cost: float,
    tolerance: float,
) -> int:
    """统计覆盖约束下同长最优路径条数（上限 2）。

    反向代价和正向计数都只在“覆盖区间重叠且向前推进”的边上传播；
    计数使用 ``forward + edge + backward == best`` 的最优子图。
    """

    layer_count = len(layer_centers)
    sizes = [len(centers) for centers in layer_centers]
    total_length = float(arc[-1])
    goal = coverage_right >= total_length - tolerance
    backward: list[np.ndarray] = [np.full(size, np.inf) for size in sizes]
    for index in range(layer_count):
        if goal[index] and sizes[index]:
            backward[index][:] = 0.0
    for index in range(layer_count - 2, -1, -1):
        for following in range(index + 1, layer_count):
            if coverage_left[following] > coverage_right[index] + tolerance:
                break
            if sizes[following] == 0 or not np.isfinite(backward[following]).any():
                continue
            if coverage_right[following] <= coverage_right[index] + tolerance:
                continue
            distances = _pairwise_distances(layer_centers[index], layer_centers[following])
            candidate = distances + backward[following][None, :]
            best = np.min(candidate, axis=1)
            backward[index] = np.minimum(backward[index], best)

    counts: list[np.ndarray] = [np.zeros(size, dtype=np.int64) for size in sizes]
    for index in range(layer_count):
        if sizes[index] and coverage_left[index] <= tolerance:
            counts[index][:] = 1
    cost_tolerance = 1e-9 * max(1.0, abs(best_cost))
    for index in range(layer_count):
        if sizes[index] == 0 or not np.any(counts[index]):
            continue
        for following in range(index + 1, layer_count):
            if coverage_left[following] > coverage_right[index] + tolerance:
                break
            if sizes[following] == 0 or not np.isfinite(backward[following]).any():
                continue
            if coverage_right[following] <= coverage_right[index] + tolerance:
                continue
            distances = _pairwise_distances(layer_centers[index], layer_centers[following])
            for node in range(sizes[following]):
                valid = (
                    forward[index]
                    + distances[:, node]
                    + backward[following][node]
                    <= best_cost + cost_tolerance
                )
                if np.any(valid):
                    counts[following][node] = min(
                        2,
                        int(counts[following][node]) + int(counts[index][valid].sum()),
                    )
    total = 0
    for index in range(layer_count):
        if not goal[index] or sizes[index] == 0:
            continue
        finite = np.isfinite(forward[index]) & (forward[index] <= best_cost + cost_tolerance)
        total += int(counts[index][finite].sum())
    return min(2, total)


def plan_side_path_coverage(
    point_indices: Iterable[int],
    audits: list[Any],
    lateral_side: int,
    *,
    branch_id: int = -1,
    tolerance: float = LATERAL_TOLERANCE,
    arc_lengths: Any,
    coverage_half_length: float,
) -> SidePath:
    """在“允许跳过采样点”的覆盖约束下求一侧最短路径。

    覆盖模型（2026-10-31 用户确认的规划抽象）：选中第 i 个采样点时，
    其覆盖区间为沿 branch 弧长的 ``[s_i - h, s_i + h]``，其中
    ``h = coverage_half_length``。只要相邻选中点的覆盖区间重叠，中间
    采样点就可以跳过。起点/终点也只需被某个选中点的覆盖区间覆盖，
    不强制选首末点。

    保守限制：如果某个采样点在该侧完全没有可行候选（现有全域碰撞检测
    的结果），则不允许覆盖区间跨过它；覆盖边界取在该点与相邻可覆盖点的
    中点。这样“被跳过的点”至少要在现有碰撞结果中属于该侧可行点，
    避免用几何半径跨过整段碰撞点。

    与 ``plan_side_path`` 的区别：后者要求每一层都被选中（全采样点
    baseline）；本函数允许跳过被覆盖的层，因此图不再是严格的相邻层链，
    而是“覆盖区间重叠且向前推进”的 DAG。
    """

    indices = np.asarray(list(point_indices), dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("point_indices 不能为空")
    arc = np.asarray(arc_lengths, dtype=float)
    if len(arc) != len(indices):
        raise ValueError("arc_lengths 与 point_indices 长度不一致")
    if len(arc) > 1 and np.any(np.diff(arc) < -1e-12):
        raise ValueError("arc_lengths 必须非递减")
    arc = arc - arc[0]
    half = float(coverage_half_length)
    if half <= 0.0:
        raise ValueError("coverage_half_length 必须为正")
    branch_label = f"branch_{int(branch_id):02d}" if branch_id >= 0 else "branch_unknown"
    total_length = float(arc[-1])
    arc_tolerance = 1e-9 * max(1.0, total_length)
    if total_length <= 0.0:
        raise ValueError("branch 弧长必须为正")

    (
        layer_candidate_indices,
        layer_centers,
        layer_angles,
        layer_lateral,
        layer_reasons,
        candidate_counts,
        missing_feasible,
    ) = _build_side_layers(indices, audits, lateral_side, tolerance)
    sizes = [len(centers) for centers in layer_centers]
    layer_count = len(layer_centers)
    missing_mask = np.zeros(layer_count, dtype=bool)
    if missing_feasible:
        missing_global = set(int(value) for value in missing_feasible)
        missing_mask = np.asarray(
            [int(point_index) in missing_global for point_index in indices],
            dtype=bool,
        )
    # 保守覆盖：不能跨过该侧完全无可行候选的采样点；见 _coverage_bounds。
    coverage_left, coverage_right = _coverage_bounds(arc, missing_mask, half)
    active = coverage_left <= coverage_right + arc_tolerance

    # dp[i][j]：以第 i 点第 j 个候选结束，且从起点到该点的覆盖连续时，
    # 已选中心折线的最小长度。起点层的成本为 0（路径从第一个选中点开始）。
    forward: list[np.ndarray] = [np.full(size, np.inf) for size in sizes]
    predecessor: list[list[tuple[int, int] | None]] = [
        [None] * size for size in sizes
    ]
    for index in range(layer_count):
        if sizes[index] == 0 or not active[index]:
            continue
        if coverage_left[index] <= arc_tolerance:
            forward[index][:] = 0.0
        for previous in range(index - 1, -1, -1):
            if arc[index] - arc[previous] > 2.0 * half + arc_tolerance:
                break
            if sizes[previous] == 0 or not active[previous] or not np.isfinite(forward[previous]).any():
                continue
            if coverage_left[index] > coverage_right[previous] + arc_tolerance:
                continue
            if coverage_right[index] <= coverage_right[previous] + arc_tolerance:
                continue
            distances = _pairwise_distances(layer_centers[index], layer_centers[previous])
            costs = distances + forward[previous][None, :]
            best_previous = np.argmin(costs, axis=1)
            best_cost = costs[np.arange(sizes[index]), best_previous]
            better = best_cost < forward[index] - 1e-15
            if np.any(better):
                forward[index][better] = best_cost[better]
                for node in np.flatnonzero(better):
                    predecessor[index][int(node)] = (int(previous), int(best_previous[int(node)]))

    goal_indices = [
        index for index in range(layer_count)
        if sizes[index] and active[index]
        and coverage_right[index] >= total_length - arc_tolerance
        and np.isfinite(forward[index]).any()
    ]
    if not goal_indices:
        message = (
            f"coverage constraint cannot be satisfied on lateral_side={lateral_side:+d}: "
            f"no path reaches the last coverage bound of branch length {total_length:.6g}"
        )
        return SidePath(
            branch_id=branch_id,
            branch_label=branch_label,
            lateral_side=int(lateral_side),
            status="no_path",
            nodes=(),
            length_norm=None,
            max_segment_length_norm=None,
            max_angle_step_deg=None,
            mean_angle_step_deg=None,
            missing_point_indices=tuple(missing_feasible),
            candidate_counts=tuple(candidate_counts),
            optimal_path_count_capped=0,
            tie_break="lowest_candidate_index",
            message=message,
            coverage_half_length=half,
            coverage_union_ok=False,
            missing_feasible_point_indices=tuple(missing_feasible),
        )

    best_index = min(
        goal_indices,
        key=lambda index: float(np.min(forward[index])),
    )
    best_local = int(np.argmin(forward[best_index]))
    best_cost = float(forward[best_index][best_local])
    selected_layers: list[tuple[int, int]] = [(best_index, best_local)]
    current = (best_index, best_local)
    while predecessor[current[0]][current[1]] is not None:
        current = predecessor[current[0]][current[1]]  # type: ignore[assignment]
        selected_layers.append(current)
    selected_layers.reverse()

    nodes: list[PathNode] = []
    for layer, local in selected_layers:
        start = float(coverage_left[layer])
        end = float(coverage_right[layer])
        candidate_index = int(layer_candidate_indices[layer][local])
        axis = np.asarray(
            audits[int(indices[layer])].candidates.axis_directions[candidate_index],
            dtype=float,
        )
        nodes.append(
            PathNode(
                point_index=int(indices[layer]),
                candidate_index=candidate_index,
                angle_deg=float(layer_angles[layer][local]),
                lateral_projection=float(layer_lateral[layer][local]),
                center_norm=np.asarray(layer_centers[layer][local], dtype=float).copy(),
                reason=str(layer_reasons[layer][local]),
                axis_norm=axis.copy(),
                coverage_start_norm=start,
                coverage_end_norm=end,
            )
        )

    selected_points = {node.point_index for node in nodes}
    skipped = tuple(int(value) for value in indices if int(value) not in selected_points)
    intervals = [(node.coverage_start_norm, node.coverage_end_norm) for node in nodes]
    gaps: list[float] = []
    for previous, following in zip(intervals[:-1], intervals[1:], strict=True):
        gaps.append(max(0.0, float(following[0]) - float(previous[1])))
    union_ok = (
        intervals[0][0] <= arc_tolerance
        and intervals[-1][1] >= total_length - arc_tolerance
        and (max(gaps) if gaps else 0.0) <= arc_tolerance
    )
    if not union_ok:
        message = (
            f"internal coverage check failed on lateral_side={lateral_side:+d}: "
            f"start={intervals[0][0]:.6g}, end={intervals[-1][1]:.6g}, "
            f"max_gap={max(gaps) if gaps else 0.0:.6g}"
        )
        return SidePath(
            branch_id=branch_id,
            branch_label=branch_label,
            lateral_side=int(lateral_side),
            status="no_path",
            nodes=(),
            length_norm=None,
            max_segment_length_norm=None,
            max_angle_step_deg=None,
            mean_angle_step_deg=None,
            missing_point_indices=tuple(missing_feasible),
            candidate_counts=tuple(candidate_counts),
            optimal_path_count_capped=0,
            tie_break="lowest_candidate_index",
            message=message,
            coverage_half_length=half,
            coverage_gaps_norm=tuple(gaps),
            coverage_union_ok=False,
            skipped_point_indices=skipped,
            missing_feasible_point_indices=tuple(missing_feasible),
        )

    selected_centers = np.asarray([node.center_norm for node in nodes], dtype=float)
    selected_angles = np.asarray([node.angle_deg for node in nodes], dtype=float)
    selected_axes = np.asarray([node.axis_norm for node in nodes], dtype=float)
    segments = np.linalg.norm(np.diff(selected_centers, axis=0), axis=1) if len(nodes) > 1 else np.empty(0)
    angle_delta = np.abs(np.diff(selected_angles)) if len(nodes) > 1 else np.empty(0)
    if len(angle_delta):
        angle_delta = np.minimum(angle_delta, 360.0 - angle_delta)
    axis_delta = (
        np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", selected_axes[:-1], selected_axes[1:]), -1.0, 1.0)))
        if len(nodes) > 1
        else np.empty(0)
    )

    optimal_count = _count_optimal_coverage_paths(
        layer_centers,
        arc,
        coverage_left,
        coverage_right,
        forward,
        best_cost,
        arc_tolerance,
    )
    message = (
        f"ok; coverage radius {half:.6g}; selected {len(nodes)}/{len(indices)} points; "
        f"{optimal_count} optimal path(s) up to cap 2"
    )
    return SidePath(
        branch_id=branch_id,
        branch_label=branch_label,
        lateral_side=int(lateral_side),
        status="ok",
        nodes=tuple(nodes),
        length_norm=best_cost,
        max_segment_length_norm=float(segments.max()) if len(segments) else 0.0,
        max_angle_step_deg=float(angle_delta.max()) if len(angle_delta) else 0.0,
        mean_angle_step_deg=float(angle_delta.mean()) if len(angle_delta) else 0.0,
        max_axis_step_deg=float(axis_delta.max()) if len(axis_delta) else 0.0,
        mean_axis_step_deg=float(axis_delta.mean()) if len(axis_delta) else 0.0,
        missing_point_indices=(),
        candidate_counts=tuple(candidate_counts),
        optimal_path_count_capped=optimal_count,
        tie_break="lowest_candidate_index",
        message=message,
        coverage_half_length=half,
        coverage_gaps_norm=tuple(gaps),
        coverage_union_ok=True,
        skipped_point_indices=skipped,
        missing_feasible_point_indices=tuple(missing_feasible),
    )


def plan_branch_paths(
    sampled: Any,
    audits: list[Any],
    *,
    tolerance: float = LATERAL_TOLERANCE,
    allow_skips: bool = False,
    coverage_half_length: float | None = None,
) -> list[SidePath]:
    """对每个 branch 的 +q / -q 两侧分别求最短路径。

    ``sampled`` 需要提供 ``branch_ids``；``audits`` 按全局点编号索引。
    ``allow_skips=False`` 时保持旧的“每层必选”baseline；``True`` 时使用
    覆盖约束，允许跳过被相邻选中砂轮覆盖的采样点。覆盖半长默认取
    ``coverage_half_length``（本轮为砂轮半径 r）。
    """

    branch_ids = np.asarray(sampled.branch_ids, dtype=np.int64)
    if len(branch_ids) != len(audits):
        raise ValueError("sampled.branch_ids 与 audits 长度不一致")
    if allow_skips:
        if coverage_half_length is None or float(coverage_half_length) <= 0.0:
            raise ValueError("allow_skips=True 时必须提供正的 coverage_half_length")
        if not hasattr(sampled, "arc_lengths"):
            raise ValueError("allow_skips=True 时 sampled 必须提供 arc_lengths")
        arc_lengths = np.asarray(sampled.arc_lengths, dtype=float)
        if len(arc_lengths) != len(branch_ids):
            raise ValueError("sampled.arc_lengths 与 branch_ids 长度不一致")
    else:
        arc_lengths = None
    paths: list[SidePath] = []
    for branch_id, start, end in branch_runs(branch_ids):
        indices = np.arange(start, end, dtype=np.int64)
        for lateral_side in (1, -1):
            if allow_skips:
                paths.append(
                    plan_side_path_coverage(
                        indices,
                        audits,
                        lateral_side,
                        branch_id=int(branch_id),
                        tolerance=tolerance,
                        arc_lengths=arc_lengths[start:end],
                        coverage_half_length=float(coverage_half_length),
                    )
                )
            else:
                paths.append(
                    plan_side_path(
                        indices,
                        audits,
                        lateral_side,
                        branch_id=int(branch_id),
                        tolerance=tolerance,
                    )
                )
    return paths
