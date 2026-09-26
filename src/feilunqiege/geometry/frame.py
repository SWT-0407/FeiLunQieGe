"""曲线切线和砂轮局部标架接口占位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LocalFrame:
    """采样点处的局部单位标架。"""

    tangent: Any
    radial_basis_1: Any
    radial_basis_2: Any
    axis: Any | None = None


def build_frames(points: Any, tangents: Any, surface_normals: Any | None = None) -> list[LocalFrame]:
    """建立满足砂轮半径方向垂直于曲线切线的局部标架。

    TODO: 后续可用表面法向替换全局参考向量，以减少复杂曲面上的标架翻转。
    """

    _ = points
    _ = surface_normals
    frames: list[LocalFrame] = []
    for tangent in np.asarray(tangents, dtype=float):
        norm = np.linalg.norm(tangent)
        if norm <= 1e-12:
            raise ValueError("存在无法归一化的曲线切线")
        t = tangent / norm
        references = np.eye(3)
        reference = references[int(np.argmin(np.abs(references @ t)))]
        basis_1 = np.cross(t, reference)
        basis_1 /= np.linalg.norm(basis_1)
        basis_2 = np.cross(t, basis_1)
        basis_2 /= np.linalg.norm(basis_2)
        frames.append(LocalFrame(tangent=t, radial_basis_1=basis_1, radial_basis_2=basis_2))
    return frames
