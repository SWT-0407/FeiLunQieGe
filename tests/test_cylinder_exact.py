"""阶段三有限厚度圆柱连续碰撞的合成几何测试。"""

from types import SimpleNamespace

import numpy as np

from feilunqiege.collision.cylinder_exact import (
    build_cylinder_triangle_index,
    check_finite_cylinder_candidates_exact,
    move_candidates_to_lower_face,
    triangle_cylinder_contact_state,
    triangle_intersects_cylinder,
)
from feilunqiege.tooling.grinding_wheel import CandidateSet


def _candidate(center: np.ndarray, *, radius: float = 1.0, width: float = 0.4) -> CandidateSet:
    return CandidateSet(
        contact_points=np.array([[0.0, 0.0, 0.0]]),
        centers=np.asarray([center], dtype=float),
        radius_directions=np.array([[1.0, 0.0, 0.0]]),
        tangent_directions=np.array([[0.0, 1.0, 0.0]]),
        axis_directions=np.array([[0.0, 0.0, 1.0]]),
        point_indices=np.array([0], dtype=np.int64),
        radial_values=np.array([radius]),
        angles=np.array([0.0]),
        radius=radius,
    )


def _box_mesh() -> SimpleNamespace:
    vertices = np.array([
        [-2, -2, -2], [2, -2, -2], [2, 2, -2], [-2, 2, -2],
        [-2, -2, 2], [2, -2, 2], [2, 2, 2], [-2, 2, 2],
    ], dtype=float)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=np.int64)
    return SimpleNamespace(vertices=vertices, faces=faces)


def test_triangle_interior_collision_when_all_vertices_are_outside() -> None:
    triangle = np.array([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]])
    hit, witness, kind = triangle_intersects_cylinder(
        triangle, np.zeros(3), np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 1.0, 0.4,
    )
    assert hit
    assert witness is not None
    assert kind == "triangle_interior_collision"


def test_cylinder_side_contact_is_detected() -> None:
    triangle = np.array([[1.0, -2.0, -0.1], [1.0, 2.0, -0.1], [1.0, 0.0, 0.1]])
    hit, witness, kind = triangle_intersects_cylinder(
        triangle, np.zeros(3), np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 1.0, 0.4,
    )
    assert hit and witness is not None
    assert kind == "cylinder_side_collision"


def test_cylinder_cap_contact_is_detected() -> None:
    triangle = np.array([[-0.2, -0.2, 0.2], [0.2, -0.2, 0.2], [0.0, 0.2, 0.2]])
    hit, witness, kind = triangle_intersects_cylinder(
        triangle, np.zeros(3), np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 1.0, 0.4,
    )
    assert hit and witness is not None
    assert kind == "cylinder_cap_collision"


def test_cylinder_edge_contact_is_detected() -> None:
    triangle = np.array([[1.0, 0.0, 0.2], [1.2, 0.0, 0.2], [1.0, 0.2, 0.2]])
    hit, witness, kind = triangle_intersects_cylinder(
        triangle, np.zeros(3), np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 1.0, 0.4,
    )
    assert hit and witness is not None
    assert kind == "cylinder_edge_collision"


def test_closed_workpiece_containment_is_not_safe_without_surface_intersection() -> None:
    result = check_finite_cylinder_candidates_exact(_candidate(np.zeros(3)), _box_mesh(), width=0.4)
    assert not bool(result.feasible[0])
    assert result.reason[0] == "workpiece_contains_wheel"


def test_lower_face_offset_is_half_width_along_axis() -> None:
    original = _candidate(np.array([0.0, 0.0, 0.0]), width=0.4)
    moved = move_candidates_to_lower_face(original, width=0.4, side=1.0)
    assert np.allclose(moved.centers[0], [0.0, 0.0, 0.2])
    assert np.isclose(np.dot(original.contact_points[0] - moved.centers[0], moved.axis_directions[0]), -0.2)


def test_annular_cylinder_ignores_triangle_inside_inner_void() -> None:
    triangle = np.array([[0.1, 0.0, -0.1], [0.2, 0.0, 0.1], [0.0, 0.2, 0.0]])
    hit, witness, reason = triangle_intersects_cylinder(
        triangle, np.zeros(3), np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        1.0, 0.4, inner_radius=0.8,
    )
    assert not hit
    assert witness is None
    assert reason == "inside_inner_contact_void"


def test_annular_cylinder_detects_triangle_in_contact_band() -> None:
    triangle = np.array([[0.85, -0.05, -0.1], [0.95, 0.05, 0.1], [0.9, 0.0, 0.0]])
    hit, witness, _reason = triangle_intersects_cylinder(
        triangle, np.zeros(3), np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        1.0, 0.4, inner_radius=0.8,
    )
    assert hit
    assert witness is not None


def test_annular_collision_reason_uses_contact_band_witness_radius() -> None:
    triangle = np.array([[0.7, 0.0, 0.0], [1.0, 0.0, 0.0], [0.85, 0.05, 0.0]])
    hit, witness, reason = triangle_intersects_cylinder(
        triangle,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        1.0,
        0.4,
        inner_radius=0.8,
    )
    assert hit
    assert witness is not None
    assert reason == "cylinder_side_collision"


def test_nonorthogonal_candidate_directions_do_not_distort_radial_distance() -> None:
    """e/t 不正交时，横向距离必须由去轴向分量的长度给出。"""

    triangle = np.array([[1.0, -1.0, -0.1], [1.01, -1.0, -0.1], [1.0, -1.01, 0.1]])
    hit, witness, reason = triangle_intersects_cylinder(
        triangle,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),  # Deliberately non-orthogonal to basis_1.
        1.2,
        0.4,
    )
    assert not hit
    assert witness is None
    assert reason == "separated_by_cylinder_side"


def _plane_mesh() -> SimpleNamespace:
    """单三角形平面，法向 +z，覆盖 z=0 处半径约 1.5 的圆盘。"""
    return SimpleNamespace(
        vertices=np.array([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )


def _side_mesh() -> SimpleNamespace:
    """位于圆柱侧面 x=2 的竖直三角形，法向 +x。"""
    return SimpleNamespace(
        vertices=np.array([[2.0, -1.0, 0.5], [2.0, 1.0, 0.5], [2.0, 0.0, 1.5]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )


def test_solid_cylinder_counts_geometry_inside_old_inner_void() -> None:
    """实心圆柱必须把旧 annular 模型内孔中的三角形也算作相交。"""

    triangle = np.array([[0.1, -0.05, -0.1], [0.2, 0.05, 0.1], [0.0, 0.2, 0.0]])
    solid = triangle_cylinder_contact_state(
        triangle, np.zeros(3), np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        1.0, 0.4, inner_radius=0.0,
    )
    annular = triangle_cylinder_contact_state(
        triangle, np.zeros(3), np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        1.0, 0.4, inner_radius=0.8,
    )
    assert solid.hit
    assert not annular.hit
    assert annular.kind == "inside_inner_contact_void"


def test_solid_coplanar_cap_is_collision_without_contact_patch() -> None:
    """纯相交判定下，端面与平面共面仍应判为碰撞。"""

    result = check_finite_cylinder_candidates_exact(
        _candidate(np.array([1.0, 0.0, 1.0]), radius=1.0),
        _plane_mesh(),
        width=2.0,
        inner_radius=0.0,
        contact_patch_enabled=False,
    )
    assert not bool(result.feasible[0])
    assert str(result.reason[0]) in {"cylinder_cap_collision", "triangle_interior_collision"}


def test_contact_patch_allows_shallow_coplanar_contact() -> None:
    """浅层共面接触位于目标 patch 内、法向对齐时应允许。"""

    result = check_finite_cylinder_candidates_exact(
        _candidate(np.array([1.0, 0.0, 1.0]), radius=1.0),
        _plane_mesh(),
        width=2.0,
        inner_radius=0.0,
        contact_patch_enabled=True,
        contact_patch_radius=2.0,
        max_contact_penetration=0.01,
        contact_normal_tolerance_degrees=30.0,
    )
    assert bool(result.feasible[0])
    assert str(result.reason[0]).startswith("contact_patch_only")
    assert float(result.contact_max_penetration[0]) <= 0.01


def test_contact_patch_rejects_penetration_beyond_threshold() -> None:
    """端面嵌入超过 max_contact_penetration 时仍应拒绝。"""

    result = check_finite_cylinder_candidates_exact(
        _candidate(np.array([1.0, 0.0, 0.98]), radius=1.0),
        _plane_mesh(),
        width=2.0,
        inner_radius=0.0,
        contact_patch_enabled=True,
        contact_patch_radius=2.5,
        max_contact_penetration=0.01,
        contact_normal_tolerance_degrees=30.0,
    )
    assert not bool(result.feasible[0])


def test_contact_patch_rejects_side_contact() -> None:
    """侧面接触的三角形法向与砂轮轴垂直，不能按端面接触斑豁免。"""

    result = check_finite_cylinder_candidates_exact(
        _candidate(np.array([1.0, 0.0, 1.0]), radius=1.0),
        _side_mesh(),
        width=2.0,
        inner_radius=0.0,
        contact_patch_enabled=True,
        contact_patch_radius=2.5,
        max_contact_penetration=0.01,
        contact_normal_tolerance_degrees=30.0,
    )
    assert not bool(result.feasible[0])


def test_contact_patch_rejects_contact_far_from_target() -> None:
    """相交区域超出 contact_patch_radius 时不能豁免。"""

    result = check_finite_cylinder_candidates_exact(
        _candidate(np.array([1.0, 0.0, 1.0]), radius=1.0),
        _plane_mesh(),
        width=2.0,
        inner_radius=0.0,
        contact_patch_enabled=True,
        contact_patch_radius=0.5,
        max_contact_penetration=0.01,
        contact_normal_tolerance_degrees=30.0,
    )
    assert not bool(result.feasible[0])
