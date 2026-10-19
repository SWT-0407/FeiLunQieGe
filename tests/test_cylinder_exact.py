"""阶段三有限厚度圆柱连续碰撞的合成几何测试。"""

from types import SimpleNamespace

import numpy as np

from feilunqiege.collision.cylinder_exact import (
    build_cylinder_triangle_index,
    check_finite_cylinder_candidates_exact,
    move_candidates_to_lower_face,
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
