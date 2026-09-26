"""OBJ/PLY 网格和点云输入接口占位。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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

    当前最小实现读取 OBJ 中的 `v` 和 `f`。多边形面会用扇形三角化，
    顶点坐标不做缩放，保留 normalized units。
    """

    source = Path(path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    normals: list[list[float]] = []
    with source.open("r", encoding="ascii", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "vn" and len(fields) >= 4:
                normals.append([float(value) for value in fields[1:4]])
            elif fields[0] == "f" and len(fields) >= 4:
                indices = [_obj_vertex_index(value, len(vertices)) for value in fields[1:]]
                for index in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[index], indices[index + 1]])

    if not vertices:
        raise ValueError(f"OBJ 没有顶点：{source}")
    if not faces:
        raise ValueError(f"OBJ 没有可用面：{source}")
    return MeshData(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        normals=np.asarray(normals, dtype=float) if normals else None,
        source_path=source,
    )


def _obj_vertex_index(token: str, vertex_count: int) -> int:
    """解析 OBJ 面元素中的顶点索引。"""

    raw_index = int(token.split("/", 1)[0])
    index = raw_index - 1 if raw_index > 0 else vertex_count + raw_index
    if not 0 <= index < vertex_count:
        raise ValueError(f"OBJ 顶点索引越界：{raw_index}")
    return index
