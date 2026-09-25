"""端到端流程编排接口占位。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_from_config(config_path: str | Path) -> Any:
    """按配置执行输入、采样、候选生成、碰撞筛选和可视化流程。

    TODO: 这里只负责编排，不在流程函数中隐藏参数、单位转换或异常过滤。
    TODO: 每个阶段保存可审计的输入摘要和参数快照。
    TODO: 在算法实现完成前保持占位状态，避免误生成结果。
    """

    raise NotImplementedError("TODO: 尚未实现端到端流程")
