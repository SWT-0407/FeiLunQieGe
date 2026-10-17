"""曲线切线和砂轮局部标架接口占位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LocalFrame:
    """采样点处的局部单位标架。"""

    tangent: Any
    radial_basis_1: Any
    radial_basis_2: Any
    axis: Any | None = None
    surface_normal: Any | None = None


def build_frames(points: Any, tangents: Any, surface_normals: Any | None = None) -> list[LocalFrame]:
    """建立切平面标架。

    ``radial_basis_1`` 和 ``radial_basis_2`` 这两个历史字段名继续保留，
    但它们现在表示切平面内的两个基向量，而不是法平面基向量。候选中心
    因而在切平面内形成圆轨迹，砂轮轴向取工件表面法向，砂轮圆盘平面与
    切平面平行、与法平面垂直。

    未传入表面法向时使用一个稳定的合成法向，仅用于旧单元测试；正式流程
    必须通过 ``estimate_surface_normals`` 从工件三角网格提供法向。
    """

    point_array = np.asarray(points, dtype=float)
    tangent_array = np.asarray(tangents, dtype=float)
    if len(point_array) != len(tangent_array):
        raise ValueError("points 与 tangents 的数量必须一致")
    if surface_normals is None:
        normal_array = None
    else:
        normal_array = np.asarray(surface_normals, dtype=float)
        if normal_array.shape != tangent_array.shape:
            raise ValueError("surface_normals 必须与 tangents 具有相同形状")
    frames: list[LocalFrame] = []
    # 每个采样点分别建立局部切平面，避免曲线弯折时使用全局固定平面。
    for index, tangent in enumerate(tangent_array):
        norm = np.linalg.norm(tangent)
        if norm <= 1e-12:
            raise ValueError("存在无法归一化的曲线切线")
        t = tangent / norm
        if normal_array is None:
            # 兼容只提供曲线的旧测试：构造一个不平行于 t 的合成表面法向。
            references = np.eye(3)
            normal = references[int(np.argmin(np.abs(references @ t)))]
        else:
            normal = normal_array[index]
        normal = normal - np.dot(normal, t) * t
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= 1e-12:
            raise ValueError(f"第 {index} 个表面法向与曲线切线退化或平行")
        normal = normal / normal_norm
        # q 是切平面中垂直于 root_line 切线的方向；(t, q) 张成切平面。
        q = np.cross(normal, t)
        q /= np.linalg.norm(q)
        frames.append(LocalFrame(tangent=t, radial_basis_1=t, radial_basis_2=q, axis=normal, surface_normal=normal))
    return frames


# Estimate each normal from a local vertex neighborhood with PCA. KD Tree is
# only the neighborhood index; the nearest face is used solely for sign fixing.
def estimate_surface_normals(points: Any, mesh: Any, neighbor_count: int = 24) -> np.ndarray:
    """用工件网格中最近三角形的面法向估计根部采样点法向。

    这是几何姿态的输入估计，不是法向重建或曲面拟合。使用三角形质心
    KDTree 只加速最近面查询；面法向本身由原始三角形叉乘直接计算。
    """

    points = np.asarray(points, dtype=float)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    if int(neighbor_count) < 3:
        raise ValueError("PCA neighborhood requires at least 3 vertices")
    if len(triangles) == 0:
        raise ValueError("工件网格没有三角形，无法估计表面法向")
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > 1e-12
    if not np.any(valid):
        raise ValueError("工件网格所有三角形均退化，无法估计表面法向")
    face_normals[valid] /= lengths[valid, None]
    centroids = triangles.mean(axis=1)
    # The nearest face is used only to orient PCA normals consistently.
    orientation_tree = cKDTree(centroids[valid])
    vertex_tree = cKDTree(vertices)
    query_count = min(int(neighbor_count), len(vertices))
    if query_count < 3:
        raise ValueError("Mesh has too few vertices for local PCA")

    normals = np.empty((len(points), 3), dtype=float)
    for index, point in enumerate(points):
        _, vertex_indices = vertex_tree.query(point, k=query_count)
        neighborhood = vertices[np.asarray(vertex_indices, dtype=np.int64)]
        centered = neighborhood - neighborhood.mean(axis=0, keepdims=True)
        covariance = (centered.T @ centered) / float(query_count)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, int(np.argmin(eigenvalues))]
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= 1e-12:
            raise ValueError(f"Local PCA is degenerate at point {index}")
        normal = normal / normal_norm

        # PCA has an arbitrary sign; align it with the nearest face normal so
        # neighboring samples do not randomly flip by 180 degrees.
        _, nearest_face = orientation_tree.query(point)
        reference = face_normals[valid][int(nearest_face)]
        if float(np.dot(normal, reference)) < 0.0:
            normal = -normal
        normals[index] = normal
    return normals
