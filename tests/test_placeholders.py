"""最小闭环的几何单元测试。"""

import numpy as np

from feilunqiege.collision.checker import CollisionResult
from feilunqiege.geometry.curve import split_and_resample
from feilunqiege.geometry.frame import build_frames
from feilunqiege.io.root_line import RootLineData
from feilunqiege.tooling.grinding_wheel import CandidateSet, generate_disk_candidates
from feilunqiege.visualization.export_obj import (
    export_feasible_centers_display_obj,
    export_feasible_centers_obj,
    export_local_feasible_centers_obj,
)


def test_candidate_radius_direction_is_orthogonal_to_tangent() -> None:
    """候选中心生成必须满足砂轮半径方向与曲线切线正交。"""

    root_line = RootLineData(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.5, 0.0]]),
        edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
    )
    curve = split_and_resample(root_line, point_count=0)
    frames = build_frames(curve.points, curve.tangents)
    candidates = generate_disk_candidates(curve.points, frames, 0.2, 0.01, 12, 2)
    dot_products = np.einsum("ij,ij->i", candidates.radius_directions, candidates.tangent_directions)
    assert np.max(np.abs(dot_products)) < 1e-10


def test_curve_loader_keeps_all_vertices_in_all_points_mode() -> None:
    """point_count=0 时不丢弃输入顶点。"""

    root_line = RootLineData(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
    )
    curve = split_and_resample(root_line, point_count=0)
    assert len(curve.points) == len(root_line.vertices)
    assert np.allclose(curve.points[[0, -1]], root_line.vertices[[0, -1]])


def test_obj_export_contains_root_line_and_feasible_points() -> None:
    """OBJ 导出应同时包含根部线的 l 原语和可行中心的 p 原语。"""

    # 用一个两点根部线和两个候选中心构造最小可验证样例。
    sampled = split_and_resample(
        RootLineData(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            edges=np.array([[0, 1]], dtype=np.int64),
        ),
        point_count=0,
    )
    candidates = CandidateSet(
        contact_points=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        centers=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
        radius_directions=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        tangent_directions=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        axis_directions=np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        point_indices=np.array([0, 0, 1], dtype=np.int64),
        radial_values=np.array([1.0, 1.0, 1.0]),
        angles=np.array([0.0, np.pi / 2.0, 0.0]),
        radius=1.0,
    )
    decisions = CollisionResult(
        feasible=np.array([True, True, False]),
        nearest_distance=np.array([2.0, 2.0, 0.5]),
        clearance=np.array([1.0, 1.0, -0.5]),
        method="test",
        candidates=candidates,
    )

    # 测试产物放在缓存目录，避免把测试中间文件混入正式结果目录；不执行删除。
    output = export_feasible_centers_obj(
        sampled,
        decisions,
        "__cache__/test_feasible_centers.obj",
        radius=1.0,
        delta=0.1,
    )
    text = output.read_text(encoding="utf-8")
    assert "g root_line" in text
    assert "l 1 2" in text
    assert "p 3 4" in text
    assert "g feasible_center_arcs" in text
    assert "l 3 4" in text


def test_display_and_local_obj_exports_contain_visible_faces() -> None:
    """显示 OBJ 应包含面片和材质，局部 OBJ 应同时包含可行与不可行分组。"""

    sampled = split_and_resample(
        RootLineData(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            edges=np.array([[0, 1]], dtype=np.int64),
        ),
        point_count=0,
    )
    candidates = CandidateSet(
        contact_points=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        centers=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
        radius_directions=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        tangent_directions=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        axis_directions=np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        point_indices=np.array([0, 0, 1], dtype=np.int64),
        radial_values=np.array([1.0, 1.0, 1.0]),
        angles=np.array([0.0, np.pi / 2.0, 0.0]),
        radius=1.0,
    )
    decisions = CollisionResult(
        feasible=np.array([True, True, False]),
        nearest_distance=np.array([2.0, 2.0, 0.5]),
        clearance=np.array([1.0, 1.0, -0.5]),
        method="test",
        candidates=candidates,
    )
    display = export_feasible_centers_display_obj(
        sampled, decisions, "__cache__/test_display.obj", radius=1.0, delta=0.1
    )
    local = export_local_feasible_centers_obj(
        sampled, decisions, "__cache__/test_local.obj", radius=1.0, delta=0.1, point_index=0
    )
    display_text = display.read_text(encoding="utf-8")
    local_text = local.read_text(encoding="utf-8")
    assert "mtllib test_display.mtl" in display_text
    assert "f " in display_text
    assert "o feasible_candidates" in local_text
    assert "o rejected_candidates" in local_text
