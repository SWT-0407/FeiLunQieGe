"""端到端流程编排接口占位。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
from datetime import datetime
import numpy as np

from feilunqiege.collision.checker import (
    build_collision_index,
    check_candidates,
    check_finite_cylinder_candidates,
)
from feilunqiege.geometry.curve import split_and_resample
from feilunqiege.geometry.frame import build_frames
from feilunqiege.io.mesh import load_mesh
from feilunqiege.io.root_line import load_root_line
from feilunqiege.tooling.grinding_wheel import generate_disk_candidates
from feilunqiege.visualization.export_obj import (
    export_feasible_centers_display_obj,
    export_feasible_centers_obj,
    export_local_feasible_centers_obj,
    export_pose_examples_obj,
)
from feilunqiege.visualization.plot import (
    plot_feasible_centers,
    plot_local_feasibility_explanation,
    plot_workpiece_orientation_guide,
    plot_finite_wheel_pose_overview,
    plot_pose_example,
)


def run_from_config(config_path: str | Path, *, case_name: str | None = None) -> Any:
    """按配置执行输入、采样、候选生成、碰撞筛选和可视化流程。

    这里保留输入原坐标值，不进行毫米换算。``case_name`` 用于批处理时
    在不生成临时配置文件的情况下切换外部数据目录中的工件案例。
    """

    # 配置文件位于仓库的 configs/ 目录；所有外部原始数据都通过相对路径引用。
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # 批处理只覆盖内存中的案例名，不修改磁盘上的默认配置文件。
    if case_name is not None:
        config["case_name"] = str(case_name)
    repo_root = config_path.parent.parent
    data_root = (config_path.parent / config["data_root"]).resolve()
    case_root = data_root / config["case_name"]

    # 读取根部线和工件本体，保留输入坐标，不在这里做毫米换算或缩放。
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / config["inputs"]["workpiece_mesh"])
    # 统一把每个案例变换到以工件包围盒中心为原点、最长边为 1 的 normalized units。
    # 变换只发生在内存中，外部原始 OBJ/PLY 保持不变；摘要会记录中心和缩放因子。
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)

    # 先按 PLY edge 拆分分支，再按弧长规则获得目标采样点和切线。
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    frames = build_frames(sampled.points, sampled.tangents)

    # 完整工件网格只建立一次碰撞索引，半径扫描和最终计算复用同一个 KDTree。
    collision_index = build_collision_index(workpiece)

    # 根据案例几何扫描候选半径；也允许配置按案例直接覆盖，便于人工复现实验。
    wheel = config["wheel"]
    radius_selection = _select_case_radius(
        config,
        sampled,
        frames,
        workpiece,
        collision_index,
    )
    wheel["radius"] = radius_selection["selected_radius"]
    wheel["delta"] = radius_selection["selected_delta"]

    # 在每个目标点的切线法平面内生成候选中心，半径方向天然满足 e·t=0。
    candidates = generate_disk_candidates(
        sampled.points,
        frames,
        float(wheel["radius"]),
        float(wheel["delta"]),
        int(config["sampling"]["candidate_angles"]),
        int(config["sampling"]["radial_samples"]),
    )

    # 当前使用表面采样点的保守球形包络做快速碰撞筛选，结果必须标注为近似。
    baseline_collisions = check_candidates(
        candidates,
        workpiece,
        float(config["collision"]["clearance"]),
        collision_index=collision_index,
    )
    width = float(wheel.get("width") or 0.0)
    if width <= 0:
        raise ValueError("第二阶段要求 wheel.width 为正数")
    collisions = check_finite_cylinder_candidates(
        candidates,
        workpiece,
        float(config["collision"]["clearance"]),
        width,
        collision_index=collision_index,
        nearest_samples=int(config["collision"].get("finite_nearest_samples", 12)),
        chunk_size=int(config["collision"].get("finite_chunk_size", 20000)),
    )
    # 每次运行建立独立目录，避免用户正在 MeshLab/图片查看器中打开旧文件时
    # Windows 锁住目标文件，也避免新的实验覆盖旧结果；旧目录绝不删除。
    # 结果先按工件案例分目录，再按时间戳建立独立运行目录，便于后续批量处理 12 个案例。
    output_root = _create_run_output_root(repo_root / config["outputs"]["directory"] / config["case_name"])

    # 输出一张用于快速核验的三维总览图。
    image_path = output_root / config["outputs"]["overview_image"]
    plot_feasible_centers(
        sampled,
        collisions,
        image_path,
        radius=candidates.radius,
        delta=float(wheel["delta"]),
        workpiece_mesh=workpiece,
        max_surface_faces=int(config.get("visualization", {}).get("max_surface_faces", 20000)),
    )

    # 局部解释图放大一个代表性目标点，专门展示 r、delta 和可行圆弧。
    local_explanation_path = output_root / config["outputs"]["local_explanation_image"]
    plot_local_feasibility_explanation(
        sampled,
        collisions,
        local_explanation_path,
        radius=candidates.radius,
        delta=float(wheel["delta"]),
    )

    # 输出同一工件在透视视角和正视投影下的对照图，帮助理解 T 形外观差异。
    orientation_path = output_root / config["outputs"]["orientation_guide_image"]
    plot_workpiece_orientation_guide(
        sampled,
        workpiece,
        orientation_path,
        max_surface_faces=int(config.get("visualization", {}).get("max_surface_faces", 20000)),
    )

    pose_overview_path = output_root / config["outputs"]["pose_overview_image"]
    plot_finite_wheel_pose_overview(
        sampled,
        collisions,
        pose_overview_path,
        width=width,
        workpiece_mesh=workpiece,
        max_surface_faces=int(config.get("visualization", {}).get("max_surface_faces", 20000)),
    )
    feasible_indices = np.flatnonzero(np.asarray(collisions.feasible, dtype=bool))
    collision_indices = np.flatnonzero(~np.asarray(collisions.feasible, dtype=bool))
    pose_feasible_path = output_root / config["outputs"]["pose_feasible_image"]
    pose_collision_path = output_root / config["outputs"]["pose_collision_image"]
    pose_examples = {}
    if len(feasible_indices):
        pose_examples["feasible"] = int(feasible_indices[0])
        plot_pose_example(sampled, collisions, pose_feasible_path, candidate_index=int(feasible_indices[0]), width=width, workpiece_mesh=workpiece, title="Representative feasible finite wheel pose")
    if len(collision_indices):
        pose_examples["collision"] = int(collision_indices[0])
        plot_pose_example(sampled, collisions, pose_collision_path, candidate_index=int(collision_indices[0]), width=width, workpiece_mesh=workpiece, title="Representative collision finite wheel pose")
    pose_obj_path = output_root / config["outputs"]["pose_examples_obj"]
    export_pose_examples_obj(collisions, pose_obj_path, candidate_indices=pose_examples, width=width, coordinate_transform=normalization)

    # 输出 MeshLab 可直接打开的 OBJ：l 是根部线，p 是可行砂轮中心点。
    obj_path = output_root / config["outputs"]["feasible_centers_obj"]
    export_feasible_centers_obj(
        sampled,
        collisions,
        obj_path,
        radius=candidates.radius,
        delta=float(wheel["delta"]),
        units="source_obj coordinates (derived from normalized computation)",
        coordinate_transform=normalization,
    )

    # 额外导出带三角标记的全局显示代理，避免 MeshLab 中 p 点过小而难以辨认。
    display_obj_path = output_root / config["outputs"]["feasible_centers_display_obj"]
    export_feasible_centers_display_obj(
        sampled,
        collisions,
        display_obj_path,
        radius=candidates.radius,
        delta=float(wheel["delta"]),
        units="source_obj coordinates (derived from normalized computation)",
        coordinate_transform=normalization,
    )

    # 额外导出单个目标点的局部 OBJ；单独打开后 Fit View 即可查看放大结果。
    local_obj_path = output_root / config["outputs"]["local_feasible_centers_obj"]
    export_local_feasible_centers_obj(
        sampled,
        collisions,
        local_obj_path,
        radius=candidates.radius,
        delta=float(wheel["delta"]),
        units="source_obj coordinates (derived from normalized computation)",
        coordinate_transform=normalization,
    )

    # 生成少量采样点可点击的离线 HTML，先用于人工验收交互方式。
    # 最后写入 JSON 审计摘要，记录参数、计数、输出文件和当前方法边界。
    summary = _build_summary(
        config,
        root_line,
        sampled,
        candidates,
        collisions,
        image_path,
        local_explanation_path,
        orientation_path,
        obj_path,
        display_obj_path,
        local_obj_path,
        output_root,
        repo_root,
        normalization,
        radius_selection,
        baseline_collisions,
        pose_overview_path,
        pose_feasible_path,
        pose_collision_path,
        pose_examples,
        width,
        pose_obj_path,
    )
    summary_path = output_root / config["outputs"]["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _build_summary(
    config: dict[str, Any],
    root_line: Any,
    sampled: Any,
    candidates: Any,
    collisions: Any,
    image_path: Path,
    local_explanation_path: Path,
    orientation_path: Path,
    obj_path: Path,
    display_obj_path: Path,
    local_obj_path: Path,
    output_root: Path,
    repo_root: Path,
    normalization: dict[str, Any],
    radius_selection: dict[str, Any],
    baseline_collisions: Any,
    pose_overview_path: Path,
    pose_feasible_path: Path,
    pose_collision_path: Path,
    pose_examples: dict[str, int],
    width: float,
    pose_obj_path: Path,
) -> dict[str, Any]:
    """生成可审计的 JSON 摘要。"""

    import numpy as np

    # 每个目标点单独统计可行候选数量和离散角度区间，便于检查局部无解点。
    per_point = []
    angle_count = int(config["sampling"]["candidate_angles"])
    for point_index in range(len(sampled.points)):
        mask = candidates.point_indices == point_index
        feasible = np.asarray(collisions.feasible)[mask]
        distances = np.asarray(collisions.clearance)[mask]
        angle_mask = np.zeros(angle_count, dtype=bool)
        local_angles = np.asarray(candidates.angles)[mask]
        angle_indices = np.mod(np.rint(local_angles / (2 * np.pi) * angle_count).astype(int), angle_count)
        angle_mask[angle_indices[feasible]] = True
        per_point.append(
            {
                "point_index": point_index,
                "branch_id": int(sampled.branch_ids[point_index]),
                "candidate_count": int(mask.sum()),
                "feasible_candidate_count": int(feasible.sum()),
                "feasible_angle_count": int(angle_mask.sum()),
                "minimum_candidate_clearance": float(distances.min()) if len(distances) else None,
                "feasible_angle_intervals_degree": _angle_intervals(angle_mask),
            }
        )
    # 汇总全局计数，并保留 OBJ 顶点语义，避免把点云误解为砂轮实体。
    feasible_count = int(np.asarray(collisions.feasible).sum())
    # 统一把输出路径写成相对仓库根目录的路径，便于复制摘要或跨机器查看。
    relative_output_root = output_root.relative_to(repo_root)
    reason_values = np.asarray(collisions.reason, dtype=object) if collisions.reason is not None else np.full(len(collisions.feasible), "unknown", dtype=object)
    reason_counts = {str(key): int(value) for key, value in zip(*np.unique(reason_values, return_counts=True), strict=True)}
    return {
        "status": "preliminary_geometry_run",
        "units": config["units"],
        "tool_model": config["tool_model"],
        "collision_method": collisions.method,
        "baseline_collision_method": baseline_collisions.method,
        "case_name": config["case_name"],
        "inputs": {
            "root_line": str(Path(config["data_root"]) / config["case_name"] / config["inputs"]["root_line"]),
            "root_line_vertex_count": int(len(root_line.vertices)),
        },
        "coordinate_frames": {
            "computation": "normalized_bbox_longest_side",
            "meshlab_obj_export": "source_obj; directly overlay with the original case OBJ files",
            "png_and_json": "normalized computation coordinates",
            "source_transform": "source = normalized * scale + center",
        },
        "parameters": {
            "radius": candidates.radius,
            "delta": float(config["wheel"]["delta"]),
            "width": width,
            "curve_points": int(len(sampled.points)),
            "candidate_angles": int(config["sampling"]["candidate_angles"]),
            "radial_samples": int(config["sampling"]["radial_samples"]),
            "collision_clearance": float(config["collision"]["clearance"]),
            "normalization": normalization,
            "radius_selection": radius_selection,
            "source_coordinate_equivalent_radius": candidates.radius * float(normalization["scale"]),
            "source_coordinate_equivalent_delta": float(config["wheel"]["delta"]) * float(normalization["scale"]),
            "source_coordinate_equivalent_width": width * float(normalization["scale"]),
        },
        "candidate_count": int(len(candidates.centers)),
        "feasible_candidate_count": feasible_count,
        "feasible_ratio": feasible_count / len(candidates.centers) if len(candidates.centers) else 0.0,
        "baseline_feasible_ratio": float(np.asarray(baseline_collisions.feasible, dtype=bool).mean()) if len(baseline_collisions.feasible) else 0.0,
        "collision_reason_counts": reason_counts,
        "pose_examples": pose_examples,
        "output_directory": str(relative_output_root),
        "overview_image": str(relative_output_root / image_path.name),
        "local_explanation_image": str(relative_output_root / local_explanation_path.name),
        "orientation_guide_image": str(relative_output_root / orientation_path.name),
        "pose_overview_image": str(relative_output_root / pose_overview_path.name),
        "pose_feasible_image": str(relative_output_root / pose_feasible_path.name),
        "pose_collision_image": str(relative_output_root / pose_collision_path.name),
        "pose_examples_obj": str(relative_output_root / pose_obj_path.name),
        "feasible_centers_obj": str(relative_output_root / obj_path.name),
        "feasible_centers_display_obj": str(relative_output_root / display_obj_path.name),
        "local_feasible_centers_obj": str(relative_output_root / local_obj_path.name),
        "obj_contents": [
            "all OBJ vertex coordinates are exported back to source_obj coordinates for MeshLab overlay",
            "sampled root-line vertices and l line primitives",
            "feasible wheel-center vertices and p point primitives grouped by target point",
            "outer-radius feasible-center arcs as l line primitives",
            "display OBJ: sampled feasible centers as red octahedron face proxies",
            "local OBJ: one target point with feasible red and rejected gray face proxies",
            "pose_examples.obj: feasible_pose is green and collision_pose is red-orange; see adjacent pose_examples.mtl",
        ],
        "per_point": per_point,
        "limitations": [
            "第一阶段 spherical-envelope 结果作为 baseline_collision_method 保留；主结果使用有限圆柱表面采样，仍可能漏检三角面内部。",
            "有限圆柱碰撞原因和见证点来自 sampled_surface_nearest，不是精确实体布尔碰撞。",
            "算法计算使用 normalized units，不能直接解释为毫米加工公差；MeshLab OBJ 仅为便于叠加而逆变换到 source_obj 坐标。",
        ],
    }


def _select_case_radius(
    config: dict[str, Any],
    sampled: Any,
    frames: Any,
    workpiece: Any,
    collision_index: Any,
) -> dict[str, Any]:
    """扫描相对工件尺度的半径，并按明确覆盖规则选择本案例主半径。"""

    import numpy as np

    settings = config.get("radius_selection", {})
    mode = str(settings.get("mode", "fixed"))
    case_name = str(config["case_name"])
    delta_ratio = float(settings.get("delta_ratio", 0.05))
    if not 0.0 <= delta_ratio < 1.0:
        raise ValueError("radius_selection.delta_ratio 必须位于 [0, 1)")

    # 人工覆盖优先级最高；这使特定砂轮规格可以直接映射到单个案例。
    overrides = settings.get("case_overrides", {})
    if case_name in overrides:
        selected = float(overrides[case_name])
        return {
            "mode": "case_override",
            "selected_radius": selected,
            "selected_delta": selected * delta_ratio,
            "selection_reason": f"case_overrides.{case_name}",
            "scan_results": [],
        }

    # fixed 模式保持单案例旧行为，但仍把选择来源写入摘要。
    if mode == "fixed":
        selected = float(config["wheel"]["radius"])
        return {
            "mode": "fixed",
            "selected_radius": selected,
            "selected_delta": float(config["wheel"]["delta"]),
            "selection_reason": "wheel.radius",
            "scan_results": [],
        }
    if mode != "scan_largest_usable":
        raise ValueError(f"不支持的 radius_selection.mode: {mode}")

    radii = sorted({float(value) for value in settings["candidate_radii"]})
    if not radii or radii[0] <= 0:
        raise ValueError("candidate_radii 必须包含正数")
    scan_angles = int(settings.get("scan_angles", 72))
    minimum_point_coverage = float(settings.get("minimum_point_coverage", 0.95))
    minimum_candidate_ratio = float(settings.get("minimum_candidate_ratio", 0.20))
    scan_results: list[dict[str, Any]] = []

    # 扫描只使用外半径的一层候选；主结果仍按正式角度数和 3 个径向层重新计算。
    for radius in radii:
        scan_candidates = generate_disk_candidates(
            sampled.points,
            frames,
            radius,
            0.0,
            scan_angles,
            1,
        )
        scan_collisions = check_candidates(
            scan_candidates,
            workpiece,
            float(config["collision"]["clearance"]),
            collision_index=collision_index,
        )
        feasible = np.asarray(scan_collisions.feasible, dtype=bool)
        feasible_by_point = np.bincount(
            np.asarray(scan_candidates.point_indices, dtype=np.int64)[feasible],
            minlength=len(sampled.points),
        )
        point_coverage = float(np.count_nonzero(feasible_by_point) / len(sampled.points))
        candidate_ratio = float(feasible.mean()) if len(feasible) else 0.0
        qualifies = point_coverage >= minimum_point_coverage and candidate_ratio >= minimum_candidate_ratio
        scan_results.append(
            {
                "radius": radius,
                "point_coverage": point_coverage,
                "feasible_candidate_ratio": candidate_ratio,
                "qualifies": bool(qualifies),
            }
        )

    # 优先选满足规则的最大半径；若全部失败，则选择覆盖率/候选率最佳且更小的半径。
    qualifying = [item for item in scan_results if item["qualifies"]]
    if qualifying:
        selected_item = max(qualifying, key=lambda item: item["radius"])
        reason = "largest radius meeting point coverage and feasible-candidate thresholds"
    else:
        selected_item = max(
            scan_results,
            key=lambda item: (item["point_coverage"], item["feasible_candidate_ratio"], -item["radius"]),
        )
        reason = "no radius met thresholds; selected best coverage with smaller-radius tie break"
    selected = float(selected_item["radius"])
    return {
        "mode": mode,
        "selected_radius": selected,
        "selected_delta": selected * delta_ratio,
        "selection_reason": reason,
        "minimum_point_coverage": minimum_point_coverage,
        "minimum_candidate_ratio": minimum_candidate_ratio,
        "scan_angles": scan_angles,
        "scan_results": scan_results,
    }


def _normalize_case_geometry(root_line: Any, workpiece: Any, config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """按配置对根部线和工件应用同一个包围盒归一化变换。"""

    import numpy as np
    from feilunqiege.io.mesh import MeshData
    from feilunqiege.io.root_line import RootLineData

    normalization = config.get("normalization", {})
    mode = str(normalization.get("mode", "bbox_longest_side_centered"))
    if mode == "none":
        return root_line, workpiece, {"mode": "none", "scale": 1.0, "center": [0.0, 0.0, 0.0]}
    if mode != "bbox_longest_side_centered":
        raise ValueError(f"不支持的 normalization.mode: {mode}")

    vertices = np.asarray(workpiece.vertices, dtype=float)
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    center = (low + high) / 2.0
    scale = float(np.max(high - low))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("工件包围盒最长边必须为正数")

    # 根部线和工件必须共用同一中心/尺度，否则候选中心会与碰撞网格错位。
    normalized_root = (np.asarray(root_line.vertices, dtype=float) - center) / scale
    normalized_vertices = (vertices - center) / scale
    normalized_mesh = MeshData(
        vertices=normalized_vertices,
        faces=workpiece.faces,
        normals=workpiece.normals,
        colors=workpiece.colors,
        source_path=workpiece.source_path,
    )
    normalized_root_line = RootLineData(
        vertices=normalized_root,
        edges=root_line.edges,
        colors=root_line.colors,
        source_path=root_line.source_path,
    )
    return normalized_root_line, normalized_mesh, {
        "mode": mode,
        "scale": scale,
        "center": [float(value) for value in center],
        "source_bbox_min": [float(value) for value in low],
        "source_bbox_max": [float(value) for value in high],
    }


def _create_run_output_root(base_root: Path) -> Path:
    """创建不覆盖旧结果的时间戳目录；同秒重复运行时追加序号。"""

    base_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = base_root / stamp
    suffix = 1
    while candidate.exists():
        candidate = base_root / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _angle_intervals(mask: Any) -> list[list[float]]:
    """把角度布尔数组合并成可读的角区间。"""

    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    if mask.all():
        return [[0.0, 360.0]]
    linear_runs = []
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        start = index
        while index < len(mask) and mask[index]:
            index += 1
        linear_runs.append((start, index))
    step = 360.0 / len(mask)
    intervals = [[round(start * step, 3), round(end * step, 3)] for start, end in linear_runs]
    if mask[0] and mask[-1] and len(intervals) >= 2:
        first = intervals.pop(0)
        last = intervals.pop(-1)
        intervals = [[last[0], 360.0], [0.0, first[1]]] + intervals
    return intervals
