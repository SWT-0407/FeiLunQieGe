"""阶段二：无厚度圆盘连续碰撞判定的合成几何测试。"""

from types import SimpleNamespace

import numpy as np

from feilunqiege.collision.disk_exact import (
    build_disk_triangle_index,
    check_disk_candidates_exact,
    triangle_intersects_disk,
)
from feilunqiege.tooling.grinding_wheel import CandidateSet


DISK_CENTER = np.array([0.0, 0.0, 0.0])
DISK_NORMAL = np.array([0.0, 0.0, 1.0])


def test_triangle_interior_crossing_is_detected_when_all_vertices_are_outside_disk() -> None:
    """三角形边界包住圆心时，即使顶点均在圆外，也必须检测到连续相交。"""

    triangle = np.array([[-2.0, -1.0, 0.0], [2.0, -1.0, 0.0], [0.0, 2.0, 0.0]])
    intersects, witness, kind = triangle_intersects_disk(triangle, DISK_CENTER, DISK_NORMAL, 1.0)
    assert np.all(np.linalg.norm(triangle - DISK_CENTER, axis=1) > 1.0)
    assert intersects
    assert kind == "coplanar_triangle"
    assert np.allclose(witness, DISK_CENTER)


def test_triangle_edge_tangent_to_disk_is_collision() -> None:
    """与圆周相切属于接触，默认规则下必须判为碰撞。"""

    triangle = np.array([[-2.0, 1.0, 0.0], [2.0, 1.0, 0.0], [0.0, 3.0, 0.0]])
    intersects, witness, kind = triangle_intersects_disk(triangle, DISK_CENTER, DISK_NORMAL, 1.0)
    assert intersects
    assert kind == "coplanar_triangle"
    assert np.allclose(witness, [0.0, 1.0, 0.0])


def test_triangle_vertex_inside_disk_is_collision() -> None:
    """顶点位于圆盘内部是最直接的连续相交情形。"""

    triangle = np.array([[0.25, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    intersects, witness, kind = triangle_intersects_disk(triangle, DISK_CENTER, DISK_NORMAL, 1.0)
    assert intersects
    assert kind == "coplanar_triangle"
    assert np.allclose(witness, triangle[0])


def test_separated_triangle_is_not_collision() -> None:
    """与圆盘平面分离的三角形必须被连续判定为安全。"""

    triangle = np.array([[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0]])
    intersects, witness, kind = triangle_intersects_disk(triangle, DISK_CENTER, DISK_NORMAL, 1.0)
    assert not intersects
    assert witness is None
    assert kind == "separated_by_disk_plane"


def test_non_coplanar_triangle_crossing_disk_plane_is_detected() -> None:
    """非共面三角形与圆盘平面的交线段穿过圆心时必须检测到。"""

    triangle = np.array([[-2.0, 0.0, -1.0], [2.0, 0.0, 1.0], [0.0, 2.0, 1.0]])
    intersects, witness, kind = triangle_intersects_disk(triangle, DISK_CENTER, DISK_NORMAL, 1.0)
    assert intersects
    assert kind == "segment_plane_intersection"
    assert np.linalg.norm(witness - DISK_CENTER) <= 1.0


def test_disk_center_inside_closed_workpiece_is_collision_without_surface_contact() -> None:
    """圆盘整体位于闭合盒体内部时，必须靠实体包含测试拒绝该候选。"""

    vertices = np.array(
        [
            [-2.0, -2.0, -2.0], [2.0, -2.0, -2.0], [2.0, 2.0, -2.0], [-2.0, 2.0, -2.0],
            [-2.0, -2.0, 2.0], [2.0, -2.0, 2.0], [2.0, 2.0, 2.0], [-2.0, 2.0, 2.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    mesh = SimpleNamespace(vertices=vertices, faces=faces)
    candidates = _single_candidate(DISK_CENTER, DISK_NORMAL, radius=0.5)
    result = check_disk_candidates_exact(candidates, mesh, clearance=0.0)
    assert not bool(result.feasible[0])
    assert result.reason[0] == "disk_center_inside_closed_workpiece"


def test_non_closed_mesh_is_reported_as_surface_only() -> None:
    """非封闭网格不能伪装为完成实体包含检测。"""

    mesh = SimpleNamespace(
        vertices=np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    index = build_disk_triangle_index(mesh)
    assert not index.mesh_is_closed
    result = check_disk_candidates_exact(_single_candidate(DISK_CENTER, DISK_NORMAL, radius=0.5), mesh, clearance=0.0, disk_index=index)
    assert bool(result.feasible[0])
    assert result.reason[0] == "safe_surface_only_non_closed_mesh"


def test_target_contact_exemption_does_not_hide_another_triangle_crossing() -> None:
    """允许根部点接触时，后续三角形的真实穿入仍必须被拒绝。"""

    mesh = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0], [0.3, 0.0, 1.0], [-0.3, 0.0, 1.0],
                [0.0, 1.0, -1.0], [0.3, 1.0, 1.0], [-0.3, 1.0, 1.0],
            ]
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
    )
    candidates = _single_candidate(np.array([0.0, 1.0, 0.0]), DISK_NORMAL, radius=1.0)
    candidates = CandidateSet(
        contact_points=np.array([[0.0, 0.0, 0.0]]),
        centers=candidates.centers,
        radius_directions=candidates.radius_directions,
        tangent_directions=candidates.tangent_directions,
        axis_directions=candidates.axis_directions,
        point_indices=candidates.point_indices,
        radial_values=candidates.radial_values,
        angles=candidates.angles,
        radius=candidates.radius,
    )
    result = check_disk_candidates_exact(candidates, mesh, clearance=0.0, allow_target_contact=True)
    assert not bool(result.feasible[0])
    assert result.reason[0] != "target_contact_only_surface_only_non_closed_mesh"


def _single_candidate(center: np.ndarray, axis: np.ndarray, *, radius: float) -> CandidateSet:
    """构造一个 x-y 圆盘平面的最小候选集。"""

    return CandidateSet(
        contact_points=np.asarray([center], dtype=float),
        centers=np.asarray([center], dtype=float),
        radius_directions=np.array([[1.0, 0.0, 0.0]]),
        tangent_directions=np.array([[0.0, 1.0, 0.0]]),
        axis_directions=np.asarray([axis], dtype=float),
        point_indices=np.array([0], dtype=np.int64),
        radial_values=np.array([radius]),
        angles=np.array([0.0]),
        radius=radius,
    )
