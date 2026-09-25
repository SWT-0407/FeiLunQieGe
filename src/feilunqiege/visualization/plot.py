"""可行砂轮中心总览图接口占位。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_feasible_centers(
    sampled_curve: Any,
    decisions: Any,
    output_path: str | Path,
    *,
    radius: float,
    delta: float,
) -> Path:
    """绘制全部目标采样点对应的可行中心总览图。

    TODO: 默认一张总览图覆盖全部点，必要时再增加局部放大图。
    TODO: 图中注明 normalized units、r、delta、采样数、无解点和碰撞容差。
    TODO: 明确二维圆弧图是无厚度圆盘中心近似，不冒充三维实体加工结果。
    """

    raise NotImplementedError("TODO: 尚未实现可行中心可视化")
