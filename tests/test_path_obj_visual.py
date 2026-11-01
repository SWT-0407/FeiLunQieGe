"""路径 OBJ 可视化测试：相邻点必须有线和实心管，no_path 只写注释。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from feilunqiege.visualization.path_obj import write_path_obj


def _node(point_index: int, center: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        point_index=point_index,
        candidate_index=0,
        angle_deg=90.0,
        center_norm=np.asarray(center, dtype=float),
    )


def test_path_obj_contains_lines_and_tubes(tmp_path: Path) -> None:
    sampled = SimpleNamespace(
        points=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=float),
        branch_ids=np.array([0, 0], dtype=int),
    )
    path = SimpleNamespace(
        branch_id=0,
        branch_label="branch_00",
        lateral_side=1,
        status="ok",
        nodes=[_node(0, [0.0, 0.02, 0.0]), _node(1, [0.01, 0.02, 0.0])],
        length_norm=0.01,
        skipped_point_indices=(),
        missing_point_indices=(),
    )
    output = tmp_path / "paths.obj"
    write_path_obj(
        output,
        sampled,
        [(0, 0, 2)],
        {"scale": 1.0, "center": [0.0, 0.0, 0.0]},
        [path],
        radius=0.02,
    )
    text = output.read_text(encoding="ascii")
    assert "o path_branch_00_pos" in text
    assert "l 3 4" in text
    assert "o path_tubes_branch_00_pos" in text
    assert "f " in text
    assert "path_positive_blue" in (tmp_path / "paths.mtl").read_text(encoding="ascii")


def test_path_obj_no_path_writes_comment_only(tmp_path: Path) -> None:
    sampled = SimpleNamespace(
        points=np.array([[0.0, 0.0, 0.0]], dtype=float),
        branch_ids=np.array([7], dtype=int),
    )
    path = SimpleNamespace(
        branch_id=7,
        branch_label="branch_07",
        lateral_side=1,
        status="no_path",
        nodes=[],
        length_norm=None,
        skipped_point_indices=(),
        missing_point_indices=(7,),
    )
    output = tmp_path / "paths.obj"
    write_path_obj(
        output,
        sampled,
        [(7, 0, 1)],
        {"scale": 1.0, "center": [0.0, 0.0, 0.0]},
        [path],
        radius=0.02,
    )
    text = output.read_text(encoding="ascii")
    assert "branch_07 lateral_side=+1: NO PATH" in text
    assert "o path_branch_07_pos" not in text
    assert "o path_nodes_branch_07_pos" not in text
