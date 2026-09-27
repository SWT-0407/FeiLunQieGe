"""根部交线排序、分支处理和弧长采样接口占位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SampledCurve:
    """离散曲线采样结果。"""

    points: Any
    tangents: Any
    branch_ids: Any
    arc_lengths: Any


def split_and_resample(root_line: Any, point_count: int) -> SampledCurve:
    """按 edge 拆分分支，并按弧长重采样曲线。

    `point_count <= 0` 表示保留每个输入顶点；这是当前配置的默认值。
    正数表示在所有分支之间按长度分配的总采样点数。
    """

    # PLY 的原始顶点索引保留到这里，edge 只用于恢复分支拓扑。
    vertices = np.asarray(root_line.vertices, dtype=float)
    edges = np.asarray(root_line.edges, dtype=np.int64).reshape((-1, 2))

    # 先建立无向邻接表，再通过连通分量和端点顺序恢复每条曲线分支。
    adjacency = [[] for _ in range(len(vertices))]
    for a, b in edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))

    components = _connected_components(adjacency)
    ordered_components = [_order_component(component, adjacency) for component in components]

    # point_count=0 是当前默认模式：不丢弃原始点；正数才触发按弧长重采样。
    if point_count > 0:
        ordered_components = _resample_components(vertices, ordered_components, point_count)

    points_out: list[np.ndarray] = []
    tangents_out: list[np.ndarray] = []
    branch_ids: list[int] = []
    arc_lengths: list[float] = []
    for branch_id, order in enumerate(ordered_components):
        # 每条分支独立估计切线，避免把分支末端错误连接成一条曲线。
        points = vertices[order] if point_count <= 0 else np.asarray(order, dtype=float)
        tangents = _estimate_tangents(points)
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.empty(0)
        cumulative = np.concatenate(([0.0], np.cumsum(distances)))
        points_out.extend(points)
        tangents_out.extend(tangents)
        branch_ids.extend([branch_id] * len(points))
        arc_lengths.extend(cumulative.tolist())

    return SampledCurve(
        points=np.asarray(points_out, dtype=float),
        tangents=np.asarray(tangents_out, dtype=float),
        branch_ids=np.asarray(branch_ids, dtype=np.int64),
        arc_lengths=np.asarray(arc_lengths, dtype=float),
    )


def _connected_components(adjacency: list[list[int]]) -> list[list[int]]:
    """按 edge 邻接关系拆分连通分支。"""

    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(len(adjacency)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _order_component(component: list[int], adjacency: list[list[int]]) -> list[int]:
    """将无分叉路径按邻接关系排序；遇到分叉显式报错。"""

    component_set = set(component)
    if any(len([n for n in adjacency[node] if n in component_set]) > 2 for node in component):
        raise ValueError("root_line 含分叉点；当前最小实现不能将分叉硬接成单一路径")
    endpoints = [node for node in component if len([n for n in adjacency[node] if n in component_set]) == 1]
    start = min(endpoints) if endpoints else min(component)
    ordered = [start]
    visited = {start}
    previous = None
    while True:
        candidates = sorted(
            neighbor
            for neighbor in adjacency[ordered[-1]]
            if neighbor in component_set and neighbor != previous and neighbor not in visited
        )
        if not candidates:
            break
        next_node = candidates[0]
        previous = ordered[-1]
        ordered.append(next_node)
        visited.add(next_node)
    if len(visited) != len(component_set):
        raise ValueError("root_line 连通分支未完整排序")
    return ordered


def _resample_components(vertices: np.ndarray, components: list[list[int]], point_count: int) -> list[np.ndarray]:
    """按分支折线长度分配采样数，并返回采样坐标。"""

    if point_count < 2 * len(components):
        raise ValueError("采样点数不足，每个连通分支至少需要两个点")
    lengths = []
    for order in components:
        points = vertices[order]
        lengths.append(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0)
    total = sum(lengths)
    result: list[np.ndarray] = []
    remaining = point_count
    for index, order in enumerate(components):
        if index == len(components) - 1:
            count = max(2, remaining)
        else:
            fraction = lengths[index] / total if total > 0 else 1 / len(components)
            count = max(2, round(point_count * fraction))
            remaining -= count
        points = vertices[order]
        if len(points) <= 1:
            result.append(np.repeat(points, count, axis=0))
            continue
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        target = np.linspace(0.0, cumulative[-1], count)
        sampled = np.column_stack([np.interp(target, cumulative, points[:, axis]) for axis in range(3)])
        result.append(sampled)
    return result


def _estimate_tangents(points: np.ndarray) -> np.ndarray:
    """用相邻点差分估计单位切线。"""

    if len(points) == 1:
        return np.array([[1.0, 0.0, 0.0]])
    raw = np.empty_like(points)
    raw[0] = points[1] - points[0]
    raw[-1] = points[-1] - points[-2]
    if len(points) > 2:
        raw[1:-1] = points[2:] - points[:-2]
    norms = np.linalg.norm(raw, axis=1)
    valid = norms > 1e-12
    result = np.zeros_like(raw)
    result[valid] = raw[valid] / norms[valid, None]
    result[~valid] = np.array([1.0, 0.0, 0.0])
    return result
