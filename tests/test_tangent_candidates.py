from types import SimpleNamespace

import numpy as np

from feilunqiege.collision.cylinder_exact import (
    build_cylinder_triangle_index,
    check_finite_cylinder_candidates_exact,
)
from feilunqiege.tooling.grinding_wheel import generate_tangent_cylinder_candidates


def _frame() -> SimpleNamespace:
    return SimpleNamespace(
        tangent=np.array([1.0, 0.0, 0.0]),
        axis=np.array([0.0, 0.0, 1.0]),
        surface_normal=np.array([0.0, 0.0, 1.0]),
    )


def test_tangent_candidate_geometry_and_lower_face_offset() -> None:
    candidates = generate_tangent_cylinder_candidates(
        np.zeros(3), _frame(), radius=1.0, width=0.4,
        angles=np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False), point_index=0,
    )
    radial = candidates.radius_directions
    tangent = candidates.tangent_directions
    axis = candidates.axis_directions
    assert np.max(np.abs(np.einsum("ij,ij->i", radial, tangent))) < 1e-12
    assert np.max(np.abs(np.einsum("ij,ij->i", axis, tangent))) < 1e-12
    assert np.max(np.abs(np.einsum("ij,ij->i", axis, radial))) < 1e-12
    assert np.max(np.abs(np.einsum("ij,ij->i", candidates.contact_points - candidates.centers, axis) + 0.2)) < 1e-12


def test_target_contact_band_exempts_only_contact_adjacent_triangle() -> None:
    candidates = generate_tangent_cylinder_candidates(
        np.zeros(3), _frame(), radius=1.0, width=0.4,
        angles=np.array([0.0]), point_index=0,
    )
    mesh = SimpleNamespace(
        vertices=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, -0.01, 0.0]]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    result = check_finite_cylinder_candidates_exact(
        candidates,
        mesh,
        width=0.4,
        inner_radius=0.0,
        target_contact_inner_radius=0.9,
        allow_target_contact=True,
        target_contact_tolerance=0.02,
        cylinder_index=build_cylinder_triangle_index(mesh),
    )
    assert bool(result.feasible[0])
    assert "target_contact_only" in str(result.reason[0])
