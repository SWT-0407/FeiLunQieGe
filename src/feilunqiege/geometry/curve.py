"""根部交线排序、分支处理和弧长采样接口占位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SampledCurve:
    """离散曲线采样结果。"""

    points: Any
    tangents: Any
    branch_ids: Any
    arc_lengths: Any


def split_and_resample(root_line: Any, point_count: int) -> SampledCurve:
    """按 edge 拆分分支，并按弧长重采样曲线。

    TODO: 找出端点、闭环和分叉点，避免对分叉曲线使用错误的单一路径顺序。
    TODO: 明确 point_count 是每个分支的数量还是全部分支共享的预算。
    TODO: 对等距采样和端点切线使用可测试的确定性规则。
    """

    raise NotImplementedError("TODO: 尚未实现曲线拆分和重采样")
