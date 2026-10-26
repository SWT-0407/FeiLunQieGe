from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from feilunqiege.geometry.frame import estimate_surface_normals
from feilunqiege.geometry.normal_p1 import estimate_normals_experimental


def _plane_mesh(size: float = 1.0, segments: int = 8) -> SimpleNamespace:
    xs = np.linspace(-size / 2, size / 2, segments + 1)
    vertices = np.asarray([[x, y, 0.0] for x in xs for y in xs], dtype=float)
    faces = []
    for ix in range(segments):
        for iy in range(segments):
            a = ix * (segments + 1) + iy
            b = a + 1
            c = a + segments + 1
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return SimpleNamespace(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))


def _edge_mesh(size: float = 1.0, segments: int = 8) -> SimpleNamespace:
    # z=0 底面 + y=0 侧壁，共享 y=0,z=0 的边；两个面片都是四边形网格。
    vertices = []
    faces = []
    xs = np.linspace(-size / 2, size / 2, segments + 1)
    ys = np.linspace(0.0, size / 2, segments + 1)
    # bottom: z=0, y in [0, size/2]
    bottom_offset = len(vertices)
    for x in xs:
        for y in ys:
            vertices.append([x, y, 0.0])
    for ix in range(segments):
        for iy in range(segments):
            a = bottom_offset + ix * (segments + 1) + iy
            b = a + 1
            c = a + segments + 1
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    # side: y=0, z in [0, size/2]
    zs = np.linspace(0.0, size / 2, segments + 1)
    side_offset = len(vertices)
    for x in xs:
        for z in zs:
            vertices.append([x, 0.0, z])
    for ix in range(segments):
        for iz in range(segments):
            a = side_offset + ix * (segments + 1) + iz
            b = a + 1
            c = a + segments + 1
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return SimpleNamespace(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64))


def test_p1_plane_normals_are_unit_and_axis_aligned() -> None:
    mesh = _plane_mesh()
    points = np.asarray([[-0.2, -0.2, 0.0], [0.0, 0.0, 0.0], [0.2, 0.2, 0.0]], dtype=float)
    normals, diagnostics = estimate_normals_experimental(points, mesh, method="p1", config={})
    assert normals.shape == points.shape
    assert np.all(np.isfinite(normals))
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)
    assert np.max(np.abs(normals[:, 2])) > 0.99
    assert len(diagnostics) == len(points)


def test_p1_edge_mesh_returns_finite_unit_normals() -> None:
    mesh = _edge_mesh()
    points = np.asarray([[-0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=float)
    for method in ("p1", "zhong2021", "zhang2015", "cazals2005", "li2010", "all"):
        normals = estimate_normals_experimental(points, mesh, method=method, config={}, return_diagnostics=False)
        assert normals.shape == points.shape
        assert np.all(np.isfinite(normals))
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)


def test_frame_dispatch_to_p1_and_baseline_raw_fit_diagnostic() -> None:
    mesh = _plane_mesh()
    points = np.asarray([[0.0, 0.0, 0.0], [0.2, -0.1, 0.0]], dtype=float)
    normals, diagnostics = estimate_surface_normals(
        points, mesh, normal_method="p1", p1_config={"candidate_faces": 32}, return_diagnostics=True
    )
    assert normals.shape == points.shape
    assert len(diagnostics) == len(points)
    assert "fit" in diagnostics[0]

    baseline, baseline_diagnostics = estimate_surface_normals(
        points, mesh, normal_method="surface_patch_continuity", return_diagnostics=True
    )
    assert baseline.shape == points.shape
    assert np.allclose(np.linalg.norm(baseline, axis=1), 1.0, atol=1e-9)
    assert "surface_patch_fit_normal" in baseline_diagnostics[0]
    assert "surface_patch_chosen_normal" in baseline_diagnostics[0]
