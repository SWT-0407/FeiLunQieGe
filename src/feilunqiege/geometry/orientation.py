"""利用带标签飞边点云确定法向绝对符号。"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def estimate_flash_body_directions(
    points: Any,
    tangents: Any,
    flash_points: Any,
    body_points: Any,
    *,
    neighbor_count: int = 32,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Estimate a pointwise flash-to-body side direction.

    The direction is computed before normal selection, so it can distinguish
    ``n1+n2`` from ``n1-n2`` instead of only flipping the final result.
    Median-neighborhood vectors are used to reduce the influence of isolated
    labels near branch endpoints.
    """
    points = np.asarray(points, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    flash = np.asarray(flash_points, dtype=float)
    body = np.asarray(body_points, dtype=float)
    if points.shape != tangents.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and tangents must have shape (n, 3)")
    if len(flash) == 0 or len(body) == 0 or neighbor_count < 2:
        raise ValueError("flash/body point clouds must be non-empty and neighbor_count >= 2")
    flash_tree = cKDTree(flash)
    body_tree = cKDTree(body)
    k_flash = min(int(neighbor_count), len(flash))
    k_body = min(int(neighbor_count), len(body))
    directions = np.zeros_like(points)
    diagnostics: list[dict[str, Any]] = []
    for index, (point, tangent) in enumerate(zip(points, tangents, strict=True)):
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-12)
        flash_distance, flash_index = flash_tree.query(point, k=k_flash)
        body_distance, body_index = body_tree.query(point, k=k_body)
        flash_local = flash[np.atleast_1d(flash_index)]
        body_local = body[np.atleast_1d(body_index)]
        # The median is less sensitive than an inverse-distance mean when a
        # root point lies close to mixed body/flash samples.
        flash_center = np.median(flash_local, axis=0)
        body_center = np.median(body_local, axis=0)
        raw = flash_center - body_center
        raw -= float(np.dot(raw, tangent)) * tangent
        norm = float(np.linalg.norm(raw))
        direction = raw / max(norm, 1e-12)
        directions[index] = direction
        diagnostics.append({
            "flash_neighbor_distance_median": float(np.median(np.atleast_1d(flash_distance))),
            "body_neighbor_distance_median": float(np.median(np.atleast_1d(body_distance))),
            "direction": direction.tolist(),
            "confidence": norm,
        })
    return directions, diagnostics


def orient_axes_to_flash_side(
    points: Any,
    tangents: Any,
    axes: Any,
    flash_points: Any,
    body_points: Any,
    branch_ids: Any,
    *,
    neighbor_count: int = 32,
    confidence_threshold: float = 0.08,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """把法向轴向统一到飞边所在的外侧，并返回逐点审计信息。

    局部飞边-本体质心连线提供绝对方向证据；每条分支再用多数证据决定一个
    全局符号，避免稀疏点云导致单点翻转。低置信度点沿用分支符号，但在审计
    中明确标记，便于人工检查。
    """

    points = np.asarray(points, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    axes = np.asarray(axes, dtype=float)
    branch_ids = np.asarray(branch_ids, dtype=np.int64).reshape(-1)
    flash = np.asarray(flash_points, dtype=float)
    body = np.asarray(body_points, dtype=float)
    if points.shape != tangents.shape or points.shape != axes.shape or len(branch_ids) != len(points):
        raise ValueError("points, tangents, axes and branch_ids shapes must match")
    if neighbor_count < 2 or confidence_threshold < 0.0:
        raise ValueError("neighbor_count must be >= 2 and confidence_threshold must be non-negative")

    point_directions, point_direction_diagnostics = estimate_flash_body_directions(
        points, tangents, flash, body, neighbor_count=neighbor_count,
    )
    evidence = np.zeros(len(points), dtype=float)
    audits: list[dict[str, Any]] = []
    for index, (point, tangent, axis) in enumerate(zip(points, tangents, axes, strict=True)):
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-12)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        direction = point_directions[index]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-12:
            score = 0.0
        else:
            score = float(np.dot(axis, direction / direction_norm))
        evidence[len(audits)] = score
        audits.append({
            "flash_neighbor_distance": point_direction_diagnostics[index]["flash_neighbor_distance_median"],
            "body_neighbor_distance": point_direction_diagnostics[index]["body_neighbor_distance_median"],
            "flash_body_direction": direction.tolist(),
            "alignment_score": score,
        })

    oriented = axes.copy()
    for branch_id in np.unique(branch_ids):
        indices = np.flatnonzero(branch_ids == branch_id)
        reliable = evidence[indices][np.abs(evidence[indices]) >= confidence_threshold]
        branch_score = float(np.median(reliable)) if len(reliable) else float(np.median(evidence[indices]))
        sign = 1.0 if branch_score >= 0.0 else -1.0
        oriented[indices] *= sign
        for index in indices:
            audits[int(index)]["branch_id"] = int(branch_id)
            audits[int(index)]["branch_score"] = branch_score
            audits[int(index)]["branch_sign"] = int(sign)
            audits[int(index)]["low_confidence"] = bool(abs(evidence[index]) < confidence_threshold)
    return oriented, audits
