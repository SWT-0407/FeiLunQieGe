"""`root_line.ply` 输入契约和读取接口。

该文件通常包含 ASCII PLY 顶点和 edge。实现时必须保留原始顶点索引，
再根据 edge 连通关系拆分多个分支，不能仅按文件中的点顺序猜测曲线顺序。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RootLineData:
    """根部交线的最小数据契约，坐标单位固定为 normalized units。"""

    vertices: Any
    edges: Any
    colors: Any | None = None
    source_path: Path | None = None


def load_root_line(path: str | Path) -> RootLineData:
    """读取根部线 PLY。

    当前最小实现只接受 ASCII PLY，因为项目中的 `root_line.ply` 正是这种格式。
    坐标按输入原值读取，单位保持为 normalized units。
    """

    # 读取 ASCII 文本头，暂不支持二进制 PLY，避免把格式误判成可读数据。
    source = Path(path)
    lines = source.read_text(encoding="ascii").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"不是 PLY 文件：{source}")
    if "format ascii 1.0" not in lines[:12]:
        raise ValueError("最小实现只支持 ASCII PLY root_line 文件")

    # 头部给出顶点和 edge 数量；后续数据行按这两个数量切分。
    vertex_count = _element_count(lines, "vertex")
    edge_count = _element_count(lines, "edge")
    end_header = _header_end(lines)
    data_lines = lines[end_header + 1 :]
    if len(data_lines) < vertex_count + edge_count:
        raise ValueError("PLY 数据行数少于 header 声明的数量")

    # 顶点前三列始终作为坐标；若有 RGB 列则保留，但不参与几何判定。
    vertices = []
    colors = []
    for line in data_lines[:vertex_count]:
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f"顶点行字段不足：{line!r}")
        vertices.append([float(value) for value in fields[:3]])
        if len(fields) >= 6:
            colors.append([int(value) for value in fields[3:6]])

    # edge 索引使用 PLY 的零基编号，读取时检查范围以防损坏拓扑。
    edges = []
    for line in data_lines[vertex_count : vertex_count + edge_count]:
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"边行字段不足：{line!r}")
        a, b = int(fields[0]), int(fields[1])
        if not (0 <= a < vertex_count and 0 <= b < vertex_count):
            raise ValueError(f"edge 索引越界：{a}, {b}")
        edges.append([a, b])

    return RootLineData(
        vertices=np.asarray(vertices, dtype=float),
        edges=np.asarray(edges, dtype=np.int64).reshape((-1, 2)),
        colors=np.asarray(colors, dtype=np.uint8) if len(colors) == vertex_count else None,
        source_path=source,
    )


def _element_count(lines: list[str], element_name: str) -> int:
    """读取 PLY header 中某个 element 的数量。"""

    prefix = f"element {element_name} "
    for line in lines:
        if line.startswith(prefix):
            return int(line[len(prefix) :])
    return 0


def _header_end(lines: list[str]) -> int:
    """返回 `end_header` 所在行号。"""

    try:
        return lines.index("end_header")
    except ValueError as exc:
        raise ValueError("PLY 缺少 end_header") from exc
