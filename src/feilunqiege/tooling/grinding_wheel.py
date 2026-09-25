"""无厚度砂轮圆盘的候选中心接口占位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiskCandidate:
    """单个候选圆盘的几何描述。"""

    contact_point: Any
    center: Any
    radius_direction: Any
    tangent_direction: Any
    axis_direction: Any
    radius: float


def generate_disk_candidates(
    points: Any,
    frames: Any,
    radius: float,
    delta: float,
    angle_count: int,
    radial_samples: int,
) -> Any:
    """生成候选无厚度圆盘中心和姿态。

    TODO: 只生成 normalized units 下 `rho in [r-delta, r]` 的候选。
    TODO: 强制检查 `radius_direction dot tangent == 0`。
    TODO: 明确自由轴向模式下的圆周切线、圆盘轴向和符号约定。
    TODO: 保留未来有限厚度圆盘/圆柱模型的接口，不在初版实现实体扫掠。
    """

    raise NotImplementedError("TODO: 尚未实现无厚度砂轮候选生成")
