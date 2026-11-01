"""路径规划纯函数测试：侧向分类、分层 DP、无解与同长解处理。

这些测试不加载真实网格，也不调用碰撞后端；只用合成候选中心验证
``src/feilunqiege/planning/path_planning.py`` 的几何/图算法。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from feilunqiege.planning.path_planning import (
    LATERAL_TOLERANCE,
    branch_runs,
    classify_lateral,
    plan_branch_paths,
    plan_side_path,
    plan_side_path_coverage,
)
from feilunqiege.tooling.grinding_wheel import CandidateSet


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def make_audit(
    point: np.ndarray,
    angles_deg: np.ndarray,
    feasible_mask: np.ndarray | None = None,
    *,
    radius: float = 0.02,
    tangent: np.ndarray | None = None,
    axis: np.ndarray | None = None,
) -> SimpleNamespace:
    """构造一个合法的候选集与碰撞结果。"""

    tangent = _unit(np.array([1.0, 0.0, 0.0]) if tangent is None else tangent)
    axis = _unit(np.array([0.0, 0.0, 1.0]) if axis is None else axis)
    q = _unit(np.cross(axis, tangent))
    angles = np.asarray(angles_deg, dtype=float)
    theta = np.radians(angles)
    radial = np.cos(theta)[:, None] * tangent[None, :] + np.sin(theta)[:, None] * q[None, :]
    centers = np.asarray(point, dtype=float)[None, :] + float(radius) * radial
    count = len(angles)
    candidates = CandidateSet(
        contact_points=np.repeat(np.asarray(point, dtype=float)[None, :], count, axis=0),
        centers=centers,
        radius_directions=radial,
        tangent_directions=np.repeat(tangent[None, :], count, axis=0),
        axis_directions=np.repeat(axis[None, :], count, axis=0),
        point_indices=np.zeros(count, dtype=np.int64),
        radial_values=np.full(count, float(radius)),
        angles=theta,
        radius=float(radius),
    )
    feasible = np.ones(count, dtype=bool) if feasible_mask is None else np.asarray(feasible_mask, dtype=bool)
    reason = np.asarray(["contact_patch_only" if flag else "triangle_interior_collision" for flag in feasible], dtype=object)
    return SimpleNamespace(candidates=candidates, result=SimpleNamespace(feasible=feasible, reason=reason))


def make_custom_audit(centers: np.ndarray, *, radial: np.ndarray | None = None) -> SimpleNamespace:
    """构造任意中心坐标的候选集，用于验证 DP 的图性质。"""

    centers = np.asarray(centers, dtype=float)
    count = len(centers)
    tangent = np.tile(np.array([1.0, 0.0, 0.0]), (count, 1))
    axis = np.tile(np.array([0.0, 0.0, 1.0]), (count, 1))
    radial = np.tile(np.array([0.0, 1.0, 0.0]), (count, 1)) if radial is None else np.asarray(radial, dtype=float)
    candidates = CandidateSet(
        contact_points=centers.copy(),
        centers=centers,
        radius_directions=radial,
        tangent_directions=tangent,
        axis_directions=axis,
        point_indices=np.zeros(count, dtype=np.int64),
        radial_values=np.linalg.norm(centers, axis=1),
        angles=np.zeros(count, dtype=float),
        radius=0.02,
    )
    return SimpleNamespace(
        candidates=candidates,
        result=SimpleNamespace(
            feasible=np.ones(count, dtype=bool),
            reason=np.asarray(["contact_patch_only"] * count, dtype=object),
        ),
    )


def test_branch_runs_returns_contiguous_blocks() -> None:
    assert branch_runs(np.array([0, 0, 1, 1, 1, 2])) == [(0, 0, 2), (1, 2, 5), (2, 5, 6)]
    assert branch_runs(np.array([], dtype=int)) == []


def test_classify_lateral_and_midline_exclusion() -> None:
    audit = make_audit(np.zeros(3), np.array([0.0, 90.0, 180.0, 270.0]))
    lateral, angles = classify_lateral(audit.candidates)
    assert lateral[0] == pytest.approx(0.0, abs=1e-12)
    assert lateral[1] == pytest.approx(1.0, abs=1e-12)
    assert lateral[2] == pytest.approx(0.0, abs=1e-12)
    assert lateral[3] == pytest.approx(-1.0, abs=1e-12)
    assert angles.tolist() == pytest.approx([0.0, 90.0, 180.0, 270.0])

    positive = plan_side_path([0], [audit], +1)
    negative = plan_side_path([0], [audit], -1)
    assert positive.candidate_counts == (1,)
    assert negative.candidate_counts == (1,)
    assert positive.nodes[0].angle_deg == pytest.approx(90.0)
    assert negative.nodes[0].angle_deg == pytest.approx(270.0)


def test_straight_line_path_length_matches_root_spacing() -> None:
    spacing = 0.01
    points = np.array([[index * spacing, 0.0, 0.0] for index in range(5)])
    angles = np.arange(0.0, 360.0, 5.0)
    # 只保留两侧的正交方向，模拟“可行角很窄”的真实情况；此时最短中心路径
    # 就是与 root_line 平行、偏移 r 的直线。
    feasible = np.isclose(angles, 90.0) | np.isclose(angles, 270.0)
    audits = [make_audit(point, angles, feasible_mask=feasible) for point in points]
    sampled = SimpleNamespace(points=points, branch_ids=np.zeros(len(points), dtype=int))

    paths = plan_branch_paths(sampled, audits)
    positive = next(item for item in paths if item.lateral_side > 0)
    negative = next(item for item in paths if item.lateral_side < 0)

    assert positive.status == "ok"
    assert negative.status == "ok"
    assert positive.length_norm == pytest.approx(4 * spacing, abs=1e-9)
    assert negative.length_norm == pytest.approx(4 * spacing, abs=1e-9)
    assert positive.max_angle_step_deg == pytest.approx(0.0)
    assert len(positive.nodes) == len(points)


def test_pure_length_can_use_angle_variation_and_is_reported() -> None:
    """说明纯长度目标的一个真实性质：完整半圆可行时，中心折线可能短于 root 长度。

    这不是 bug，而是“只最小化中心距离”的数学结果；因此交付必须同时报告
    角度跳变，使用者才能判断路径是否适合加工。
    """

    spacing = 0.01
    points = np.array([[index * spacing, 0.0, 0.0] for index in range(5)])
    angles = np.arange(0.0, 360.0, 5.0)
    audits = [make_audit(point, angles) for point in points]
    path = plan_side_path([0, 1, 2, 3, 4], audits, +1)

    assert path.status == "ok"
    assert path.length_norm is not None
    assert path.length_norm < 4 * spacing
    # 相邻角度会发生变化，必须在诊断字段中可见。
    assert path.max_angle_step_deg > 0.0


def test_no_path_reports_missing_points_without_fabricating() -> None:
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    angles = np.arange(0.0, 360.0, 5.0)
    middle = make_audit(
        points[1],
        angles,
        feasible_mask=np.isclose(angles, 270.0),
    )
    audits = [make_audit(points[0], angles), middle, make_audit(points[2], angles)]
    sampled = SimpleNamespace(points=points, branch_ids=np.zeros(len(points), dtype=int))

    paths = plan_branch_paths(sampled, audits)
    positive = next(item for item in paths if item.lateral_side > 0)
    negative = next(item for item in paths if item.lateral_side < 0)

    assert positive.status == "no_path"
    assert positive.nodes == ()
    assert positive.missing_point_indices == (1,)
    assert "no feasible candidate" in positive.message
    assert negative.status == "ok"
    assert negative.length_norm is not None


def test_dp_matches_brute_force_for_small_layers() -> None:
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    angles = np.array([30.0, 90.0, 150.0])
    audits = [make_audit(point, angles) for point in points]
    path = plan_side_path([0, 1, 2], audits, +1)

    centers = [np.asarray(audit.candidates.centers) for audit in audits]
    brute_force = []
    for first in range(3):
        for second in range(3):
            for third in range(3):
                length = (
                    np.linalg.norm(centers[1][second] - centers[0][first])
                    + np.linalg.norm(centers[2][third] - centers[1][second])
                )
                brute_force.append(length)
    assert path.length_norm == pytest.approx(min(brute_force), abs=1e-12)


def test_multiple_optimal_paths_are_detected_and_one_is_exported() -> None:
    # 两层正方形：A-C 与 B-D 等长，均为一侧可行；DP 只导出一条。
    layer_zero = np.array([[0.0, 0.01, 0.0], [0.01, 0.01, 0.0]])
    layer_one = np.array([[0.0, 0.02, 0.0], [0.01, 0.02, 0.0]])
    audits = [make_custom_audit(layer_zero), make_custom_audit(layer_one)]
    path = plan_side_path([0, 1], audits, +1)
    assert path.status == "ok"
    assert path.optimal_path_count_capped == 2
    assert "lowest-candidate-index" in path.message
    assert len(path.nodes) == 2


def test_branch_labels_and_two_branches() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0], [0.01, 0.0, 0.0],
            [0.0, 0.02, 0.0], [0.01, 0.02, 0.0],
        ]
    )
    angles = np.arange(0.0, 360.0, 5.0)
    audits = [make_audit(point, angles) for point in points]
    sampled = SimpleNamespace(points=points, branch_ids=np.array([0, 0, 1, 1], dtype=int))

    paths = plan_branch_paths(sampled, audits)
    labels = {(item.branch_id, item.lateral_side, item.branch_label) for item in paths}
    assert labels == {
        (0, 1, "branch_00"), (0, -1, "branch_00"),
        (1, 1, "branch_01"), (1, -1, "branch_01"),
    }


def test_out_and_back_branch_keeps_consistent_lateral_side() -> None:
    """branch 7 式退化往返分支：切线翻转 180° 不能把同一物理侧向拆成两段。"""

    points = np.array(
        [
            [0.00, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.00, 0.0, 0.0],
        ]
    )
    angles = np.arange(0.0, 360.0, 5.0)
    forward = np.tile(np.array([[1.0, 0.0, 0.0]]), (3, 1))
    backward = np.tile(np.array([[-1.0, 0.0, 0.0]]), (3, 1))
    points_tangent = np.vstack([forward, backward])
    # 去程只有 -q（270°）；回程在自身局部标架里是 90°，但 q 符号翻转后
    # 仍属于同一物理侧 -q。
    feasible_masks = [np.isclose(angles, 270.0)] * 3 + [np.isclose(angles, 90.0)] * 3
    audits = [
        make_audit(points[index], angles, feasible_mask=feasible_masks[index], tangent=points_tangent[index])
        for index in range(len(points))
    ]
    sampled = SimpleNamespace(points=points, branch_ids=np.zeros(len(points), dtype=int))

    paths = plan_branch_paths(sampled, audits)
    positive = next(item for item in paths if item.lateral_side > 0)
    negative = next(item for item in paths if item.lateral_side < 0)

    assert negative.status == "ok"
    assert negative.missing_point_indices == ()
    assert negative.length_norm is not None
    assert positive.status == "no_path"
    assert positive.missing_point_indices == tuple(range(len(points)))


def test_coverage_can_skip_all_but_one_point_when_branch_is_short() -> None:
    spacing = 0.01
    points = np.array([[index * spacing, 0.0, 0.0] for index in range(5)])
    angles = np.arange(0.0, 360.0, 5.0)
    audits = [make_audit(point, angles) for point in points]
    arc = np.arange(len(points), dtype=float) * spacing

    path = plan_side_path_coverage(
        list(range(len(points))),
        audits,
        +1,
        branch_id=0,
        arc_lengths=arc,
        coverage_half_length=0.02,
    )

    assert path.status == "ok"
    assert path.coverage_union_ok is True
    assert path.length_norm == pytest.approx(0.0, abs=1e-12)
    assert len(path.nodes) == 1
    assert len(path.skipped_point_indices) == 4


def test_coverage_selects_sparse_points_on_long_branch() -> None:
    spacing = 0.01
    points = np.array([[index * spacing, 0.0, 0.0] for index in range(11)])
    angles = np.arange(0.0, 360.0, 5.0)
    audits = [make_audit(point, angles) for point in points]
    samples = SimpleNamespace(
        points=points,
        branch_ids=np.zeros(len(points), dtype=int),
        arc_lengths=np.arange(len(points), dtype=float) * spacing,
    )

    paths = plan_branch_paths(
        samples,
        audits,
        allow_skips=True,
        coverage_half_length=0.02,
    )
    positive = next(item for item in paths if item.lateral_side > 0)

    assert positive.status == "ok"
    assert positive.coverage_union_ok is True
    assert 2 <= len(positive.nodes) <= 4
    assert len(positive.skipped_point_indices) == len(points) - len(positive.nodes)
    assert positive.length_norm is not None
    assert positive.length_norm > 0.0


def test_coverage_does_not_cross_point_without_feasible_candidate() -> None:
    spacing = 0.01
    points = np.array([[index * spacing, 0.0, 0.0] for index in range(9)])
    angles = np.arange(0.0, 360.0, 5.0)
    audits = [make_audit(point, angles) for point in points]
    # 第 4 点在该侧没有可行候选。保守覆盖模型不允许跨过它；
    # 否则会声称一个在当前碰撞结果中不可行的点被覆盖。
    audits[4] = make_audit(points[4], angles, feasible_mask=np.zeros(len(angles), dtype=bool))
    arc = np.arange(len(points), dtype=float) * spacing

    path = plan_side_path_coverage(
        list(range(len(points))),
        audits,
        +1,
        branch_id=0,
        arc_lengths=arc,
        coverage_half_length=0.02,
    )

    assert path.status == "no_path"
    assert path.coverage_union_ok is False
    assert 4 in path.missing_feasible_point_indices
    assert "coverage constraint" in path.message


def test_coverage_reports_no_path_when_gap_is_too_large() -> None:
    spacing = 0.01
    points = np.array([[index * spacing, 0.0, 0.0] for index in range(11)])
    angles = np.arange(0.0, 360.0, 5.0)
    audits = [make_audit(point, angles) for point in points]
    # 5–8 全无可行候选；4 到 9 的间隔 0.05 > 2r=0.04，覆盖无法连续。
    for index in range(5, 9):
        audits[index] = make_audit(points[index], angles, feasible_mask=np.zeros(len(angles), dtype=bool))
    arc = np.arange(len(points), dtype=float) * spacing

    path = plan_side_path_coverage(
        list(range(len(points))),
        audits,
        +1,
        branch_id=0,
        arc_lengths=arc,
        coverage_half_length=0.02,
    )

    assert path.status == "no_path"
    assert path.coverage_union_ok is False
    assert "coverage constraint" in path.message


def test_coverage_does_not_require_first_or_last_sample() -> None:
    spacing = 0.01
    points = np.array([[index * spacing, 0.0, 0.0] for index in range(11)])
    angles = np.arange(0.0, 360.0, 5.0)
    audits = [make_audit(point, angles) for point in points]
    # 第 0 点仍可行，但把它的候选中心放得很远；DP 可以从前 h 内的
    # 后续点起步，不必选首点。末点同理可能不被选中。
    audits[0] = make_custom_audit(np.array([[100.0, 1.0, 0.0]]))
    arc = np.arange(len(points), dtype=float) * spacing

    path = plan_side_path_coverage(
        list(range(len(points))),
        audits,
        +1,
        branch_id=0,
        arc_lengths=arc,
        coverage_half_length=0.02,
    )

    assert path.status == "ok"
    assert path.coverage_union_ok is True
    assert path.nodes[0].point_index > 0
    assert path.nodes[-1].point_index < len(points) - 1
