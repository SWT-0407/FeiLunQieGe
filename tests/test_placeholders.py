"""最小闭环的几何单元测试。"""

import numpy as np

from feilunqiege.geometry.curve import split_and_resample
from feilunqiege.geometry.frame import build_frames
from feilunqiege.io.root_line import RootLineData
from feilunqiege.tooling.grinding_wheel import generate_disk_candidates


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
