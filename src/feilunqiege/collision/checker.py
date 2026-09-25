"""砂轮候选与工件本体的碰撞检测接口占位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CollisionDecision:
    """单个候选的可行性及审计信息。"""

    feasible: bool
    minimum_clearance: float | None
    reason: str


def check_candidates(candidates: Any, workpiece_mesh: Any, clearance: float) -> list[CollisionDecision]:
    """筛选不与工件本体碰撞的候选。

    TODO: 明确无厚度圆盘的碰撞包络和目标接触豁免规则。
    TODO: 记录相交、距离不足、合法接触和输入异常等原因，不能只返回布尔值。
    TODO: 首版可用可解释的保守判定，后续再评估 trimesh/open3d 的精确接口。
    """

    raise NotImplementedError("TODO: 尚未实现碰撞检测")
