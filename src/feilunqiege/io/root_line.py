"""`root_line.ply` 输入契约和读取接口。

该文件通常包含 ASCII PLY 顶点和 edge。实现时必须保留原始顶点索引，
再根据 edge 连通关系拆分多个分支，不能仅按文件中的点顺序猜测曲线顺序。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RootLineData:
    """根部交线的最小数据契约，坐标单位固定为 normalized units。"""

    vertices: Any
    edges: Any
    colors: Any | None = None
    source_path: Path | None = None


def load_root_line(path: str | Path) -> RootLineData:
    """读取根部线 PLY。

    TODO: 实现 ASCII PLY header 解析、顶点/edge 读取和格式校验。
    TODO: 拒绝缺少 vertex 或 edge 的输入，或将缺失 edge 明确标为待排序点集。
    TODO: 保留原始索引，禁止在读取阶段静默删除重复点或异常点。
    """

    raise NotImplementedError("TODO: 尚未实现 root_line.ply 读取")
