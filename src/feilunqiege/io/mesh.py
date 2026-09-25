"""OBJ/PLY 网格和点云输入接口占位。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MeshData:
    """工件网格数据契约；坐标不换算，保留 normalized units。"""

    vertices: Any
    faces: Any
    normals: Any | None = None
    colors: Any | None = None
    source_path: Path | None = None


def load_mesh(path: str | Path) -> MeshData:
    """读取 OBJ 三角网格的接口。

    TODO: 支持 OBJ 顶点、法向和面索引，处理 OBJ 的 1-based 索引。
    TODO: 检查空面、退化三角形和异常坐标；不要在这里改变坐标尺度。
    TODO: 后续可由 trimesh/open3d 实现，但依赖版本必须记录。
    """

    raise NotImplementedError("TODO: 尚未实现网格读取")
