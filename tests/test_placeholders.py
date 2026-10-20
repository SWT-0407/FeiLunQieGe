"""最小闭环的几何单元测试。"""

import numpy as np

from feilunqiege.collision.checker import CollisionResult
from feilunqiege.geometry.curve import split_and_resample
from feilunqiege.geometry.frame import build_frames, estimate_surface_normals
from feilunqiege.io.mesh import MeshData
from feilunqiege.io.root_line import RootLineData
from feilunqiege.tooling.grinding_wheel import CandidateSet, generate_disk_candidates
from feilunqiege.visualization.export_obj import (
    export_feasible_centers_display_obj,
    export_feasible_centers_obj,
    export_local_feasible_centers_obj,
    export_pose_examples_obj,
)
from feilunqiege.visualization.interactive import export_interactive_demo_html


def test_candidate_disk_is_parallel_to_tangent_plane() -> None:
    """候选中心偏移在切平面内，砂轮轴向垂直于该切平面。"""

    root_line = RootLineData(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.5, 0.0]]),
        edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
    )
    curve = split_and_resample(root_line, point_count=0)
    frames = build_frames(curve.points, curve.tangents)
    candidates = generate_disk_candidates(curve.points, frames, 0.2, 0.01, 12, 2)
    axis_tangent = np.einsum("ij,ij->i", candidates.axis_directions, candidates.tangent_directions)
    axis_radius = np.einsum("ij,ij->i", candidates.axis_directions, candidates.radius_directions)
    assert np.max(np.abs(axis_tangent)) < 1e-10
    assert np.max(np.abs(axis_radius)) < 1e-10
    assert np.allclose(np.linalg.norm(candidates.radius_directions, axis=1), 1.0)


def test_explicit_surface_normal_controls_wheel_axis() -> None:
    """显式工件法向应成为砂轮轴向，候选圆轨迹应留在切平面。"""

    points = np.array([[0.0, 0.0, 0.0]])
    tangents = np.array([[1.0, 0.0, 0.0]])
    normals = np.array([[0.0, 0.0, 1.0]])
    frames = build_frames(points, tangents, normals)
    candidates = generate_disk_candidates(points, frames, 0.2, 0.01, 16, 1)
    assert np.allclose(np.abs(candidates.axis_directions[:, 2]), 1.0)
    assert np.max(np.abs(candidates.radius_directions[:, 2])) < 1e-10


def test_angle_bisector_surface_normal_uses_two_face_directions() -> None:
    """平面退化为同一侧时，角平分估计应恢复该平面的法向。"""

    vertices = np.array(
        [
            [-1.0, -1.0, 0.0], [0.0, -1.0, 0.0], [1.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]])
    mesh = MeshData(vertices=vertices, faces=faces)
    normals = estimate_surface_normals(np.array([[0.0, 0.0, 0.0]]), mesh, neighbor_count=9)
    assert np.allclose(np.abs(normals[0]), [0.0, 0.0, 1.0], atol=1e-10)


def test_labelled_flash_point_sets_normal_sign_without_branch_dynamic_programming() -> None:
    """飞边标签应按 O->F 对每个局部面法向定正负。"""

    vertices = np.array(
        [
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],
            [2.0, -1.0, 0.0], [3.0, -1.0, 0.0], [2.5, 1.0, 0.0],
        ]
    )
    # 两个三角形都刻意使用朝 -z 的绕序，验证代码会用 F-O 翻转它们。
    mesh = MeshData(vertices=vertices, faces=np.array([[0, 2, 1], [3, 5, 4]], dtype=np.int64))
    normals, diagnostics = estimate_surface_normals(
        np.array([[0.0, 0.0, 0.0]]),
        mesh,
        neighbor_count=2,
        candidate_count=2,
        flash_points=np.array([[0.0, 0.0, 1.0]]),
        return_diagnostics=True,
    )
    assert float(np.dot(normals[0], [0.0, 0.0, 1.0])) > 0.999
    assert diagnostics[0]["orientation_rule"] == "nearest_flash_OF_smaller_angle"
    assert not diagnostics[0]["continuity_state"]["applied"]


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
        coordinate_transform={"scale": 10.0, "center": [10.0, 20.0, 30.0]},
    )
    text = output.read_text(encoding="utf-8")
    assert "g root_line" in text
    assert "l 1 2" in text
    assert "p 3 4" in text
    assert "g feasible_center_arcs" in text
    assert "l 3 4" in text
    assert "# coordinate_frame: source_obj" in text
    assert "# radius: 10" in text
    assert "v 10 20 30" in text
    assert "v 10 30 30" in text


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
    transform = {"scale": 2.0, "center": [3.0, 4.0, 5.0]}
    display = export_feasible_centers_display_obj(
        sampled,
        decisions,
        "__cache__/test_display.obj",
        radius=1.0,
        delta=0.1,
        coordinate_transform=transform,
    )
    local = export_local_feasible_centers_obj(
        sampled,
        decisions,
        "__cache__/test_local.obj",
        radius=1.0,
        delta=0.1,
        coordinate_transform=transform,
        point_index=0,
    )
    display_text = display.read_text(encoding="utf-8")
    local_text = local.read_text(encoding="utf-8")
    assert "mtllib test_display.mtl" in display_text
    assert "f " in display_text
    assert "# coordinate_frame: source_obj" in display_text
    assert "v 3 4 5" in display_text
    assert "o feasible_candidates" in local_text
    assert "o rejected_candidates" in local_text
    assert "# coordinate_frame: source_obj" in local_text
    assert "v 3 4 5" in local_text

    pose = export_pose_examples_obj(
        decisions,
        "__cache__/test_pose_examples.obj",
        candidate_indices={"feasible": 0, "collision": 2},
        width=0.1,
        coordinate_transform=transform,
    )
    pose_text = pose.read_text(encoding="utf-8")
    pose_mtl = pose.with_suffix(".mtl").read_text(encoding="utf-8")
    assert "mtllib test_pose_examples.mtl" in pose_text
    assert "usemtl feasible_pose" in pose_text
    assert "usemtl collision_pose" in pose_text
    assert "v 3 8 4.9 0.6 1 0.8" in pose_text
    assert "v 5 8 4.9 1 0.78 0.7" in pose_text
    assert "newmtl feasible_pose" in pose_mtl
    assert "newmtl collision_pose" in pose_mtl
    assert "Ke 0.60 1.00 0.80" in pose_mtl
    assert "Ke 1.00 0.78 0.70" in pose_mtl
    assert pose_mtl.count("illum 0") == 2


def test_interactive_demo_contains_selected_points_and_local_script() -> None:
    """交互演示应包含少量点数据，并保留点击后重绘局部图的脚本。"""

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
    output = export_interactive_demo_html(
        sampled, decisions, "__cache__/test_interactive_demo.html", radius=1.0, delta=0.1, max_demo_points=2
    )
    html = output.read_text(encoding="utf-8")
    assert "砂轮可行中心交互演示" in html
    assert "selectPoint" in html
    assert "selected_points" in html
    assert "${point.index}" in html
