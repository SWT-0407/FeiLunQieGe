"""曲线切线和砂轮局部标架接口占位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LocalFrame:
    """采样点处的局部单位标架。"""

    tangent: Any
    radial_basis_1: Any
    radial_basis_2: Any
    axis: Any | None = None


def build_frames(points: Any, tangents: Any, surface_normals: Any | None = None) -> list[LocalFrame]:
    """建立满足砂轮半径方向垂直于曲线切线的局部标架。

    TODO: 优先使用工件表面法向构造稳定标架；法向缺失时再定义明确的退化策略。
    TODO: 逐点检查单位长度、正交性和相邻标架翻转，不能静默接受 NaN。
    TODO: 明确自由轴向与固定轴向两种模式的输入输出差异。
    """

    raise NotImplementedError("TODO: 尚未实现局部标架构造")
