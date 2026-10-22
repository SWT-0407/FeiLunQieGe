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
)
from feilunqiege.collision.cylinder_exact import (
    build_cylinder_triangle_index,
    check_finite_cylinder_candidates_exact,
    move_candidates_to_lower_face,
)
from feilunqiege.geometry.curve import split_and_resample
from feilunqiege.geometry.frame import build_frames, estimate_surface_normals
from feilunqiege.io.mesh import load_mesh
from feilunqiege.io.flash_cloud import load_labeled_flash_cloud
from feilunqiege.io.root_line import load_root_line
from feilunqiege.tooling.grinding_wheel import generate_disk_candidates
from feilunqiege.visualization.export_obj import (
    export_feasible_centers_display_obj,
    export_local_feasible_centers_obj,
    export_pose_examples_obj,
)
from feilunqiege.visualization.plot import (
    plot_feasible_centers,
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
    flash_cloud = load_labeled_flash_cloud(case_root / config["inputs"].get("flash_cloud", "original_flash_cloud.obj"))
    # 统一把每个案例变换到以工件包围盒中心为原点、最长边为 1 的 normalized units。
    # 变换只发生在内存中，外部原始 OBJ/PLY 保持不变；摘要会记录中心和缩放因子。
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    source_center = np.asarray(normalization["center"], dtype=float)
    source_scale = float(normalization["scale"])
    flash_points = (np.asarray(flash_cloud.flash_points, dtype=float) - source_center) / source_scale

    # 先按 PLY edge 拆分分支，再按弧长规则获得目标采样点和切线。
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    normal_config = config.get("normal_estimation", {})
    normal_method = str(normal_config.get("method", "two_face_angle_bisector"))
    surface_normals = estimate_surface_normals(
        sampled.points,
        workpiece,
        neighbor_count=int(normal_config.get("neighbor_count", 24)),
        candidate_count=int(normal_config.get("candidate_count", 96)),
        side_count=int(normal_config.get("side_count", 8)),
        coherence_angle_degrees=float(normal_config.get("coherence_angle_degrees", 25.0)),
        offset_epsilon=float(normal_config.get("offset_epsilon", 1e-5)),
        weight_distance_floor=float(normal_config.get("weight_distance_floor", 1e-5)),
        continuity_jump_threshold_degrees=float(normal_config.get("continuity_jump_threshold_degrees", 20.0)),
        continuity_normal_weight=float(normal_config.get("continuity_normal_weight", 1.0)),
        continuity_patch_weight=float(normal_config.get("continuity_patch_weight", 0.15)),
        continuity_distance_weight=float(normal_config.get("continuity_distance_weight", 0.1)),
        branch_ids=sampled.branch_ids,
        flash_points=flash_points,
        normal_method=normal_method,
        surface_patch_neighbors=int(normal_config.get("surface_patch_neighbors", 32)),
        surface_patch_distance_factor=float(normal_config.get("surface_patch_distance_factor", 3.0)),
        surface_patch_coherence_degrees=float(normal_config.get("surface_patch_coherence_degrees", 25.0)),
        max_bisector_deviation_degrees=float(normal_config.get("max_bisector_deviation_degrees", 10.0)),
        continuity_repair_degrees=float(normal_config.get("continuity_repair_degrees", 30.0)),
        continuity_repair_iterations=int(normal_config.get("continuity_repair_iterations", 3)),
    )
    frames = build_frames(
        sampled.points,
        sampled.tangents,
        surface_normals,
        normal_is_reliable=(normal_method == "surface_patch_continuity"),
        tangent_normal_threshold=float(normal_config.get("tangent_normal_threshold", 0.2)),
    )

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

    # 在每个目标点的局部切平面内生成候选中心；砂轮轴向由带标签飞边点云定向。
    candidates = generate_disk_candidates(
        sampled.points,
        frames,
        float(wheel["radius"]),
        float(wheel["delta"]),
        int(config["sampling"]["candidate_angles"]),
        int(config["sampling"]["radial_samples"]),
    )

    # 旧的 KDTree 球形包络仅保留作半径扫描/对照，不再决定最终可行域。
    baseline_collisions = check_candidates(
        candidates,
        workpiece,
        float(config["collision"]["clearance"]),
        collision_index=collision_index,
    )
    width = float(wheel.get("width") or 0.0)
    if width <= 0:
        raise ValueError("第二阶段要求 wheel.width 为正数")
    # 将切平面中的候选圆心移动到有限圆柱几何中心；指定端面仍与接触切平面共面。
    candidates = move_candidates_to_lower_face(candidates, width, side=1.0)
    cylinder_index = build_cylinder_triangle_index(workpiece)
    # 砂轮是实心圆柱：delta 不再参与碰撞体几何，只保留在候选中心半径带中。
    wheel_radius = float(wheel["radius"])
    collision_config = config["collision"]
    contact_patch_enabled = bool(collision_config.get("contact_patch_enabled", True))
    contact_patch_radius = float(collision_config.get("contact_patch_radius_ratio", 2.0)) * wheel_radius
    max_contact_penetration = float(collision_config.get("contact_patch_max_penetration_ratio", 0.025)) * wheel_radius
    contact_normal_tolerance_degrees = float(
        collision_config.get("contact_patch_normal_tolerance_degrees", 30.0)
    )
    narrow_phase = str(collision_config.get("narrow_phase", "analytic_closest_point_flat"))
    # AABB 树只剔除不相交三角面，连续窄相再裁剪三角形并计算到圆柱轴的真实横向距离。
    collisions = check_finite_cylinder_candidates_exact(
        candidates,
        workpiece,
        width,
        inner_radius=0.0,
        clearance=float(collision_config["clearance"]),
        allow_target_contact=bool(collision_config.get("allow_target_contact", False)),
        contact_patch_enabled=contact_patch_enabled,
        contact_patch_radius=contact_patch_radius,
        max_contact_penetration=max_contact_penetration,
        contact_normal_tolerance_degrees=contact_normal_tolerance_degrees,
        narrow_phase=narrow_phase,
        cylinder_index=cylinder_index,
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

    feasible_indices = np.flatnonzero(np.asarray(collisions.feasible, dtype=bool))
    collision_indices = np.flatnonzero(~np.asarray(collisions.feasible, dtype=bool))
    pose_examples = {}
    if len(feasible_indices):
        # 绿色展示姿态优先取已判定可行集合中余量最大的候选，而不是数组中的第一个。
        # 这避免把仅擦过离散判定边界的姿态作为人工核验的代表案例。
        feasible_clearance = np.asarray(collisions.clearance, dtype=float)[feasible_indices]
        pose_examples["feasible"] = int(feasible_indices[int(np.argmax(feasible_clearance))])
    if len(collision_indices):
        pose_examples["collision"] = int(collision_indices[0])
    # 个别案例可指定经过更高密度独立复核的展示姿态；这只影响示例环，不改变候选筛选结果。
    pose_overrides = config.get("visualization", {}).get("pose_examples", {}).get("case_overrides", {})
    case_override = pose_overrides.get(config["case_name"], {})
    override_feasible = case_override.get("feasible")
    if override_feasible is not None:
        override_feasible = int(override_feasible)
        if 0 <= override_feasible < len(collisions.feasible) and bool(collisions.feasible[override_feasible]):
            pose_examples["feasible"] = override_feasible
    pose_obj_path = output_root / config["outputs"]["pose_examples_obj"]
    export_pose_examples_obj(collisions, pose_obj_path, candidate_indices=pose_examples, width=width, coordinate_transform=normalization)

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
        display_obj_path,
        local_obj_path,
        output_root,
        repo_root,
        normalization,
        radius_selection,
        baseline_collisions,
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
    display_obj_path: Path,
    local_obj_path: Path,
    output_root: Path,
    repo_root: Path,
    normalization: dict[str, Any],
    radius_selection: dict[str, Any],
    baseline_collisions: Any,
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
            "flash_cloud": str(Path(config["data_root"]) / config["case_name"] / config["inputs"].get("flash_cloud", "original_flash_cloud.obj")),
        },
        "coordinate_frames": {
            "computation": "normalized_bbox_longest_side",
            "meshlab_obj_export": "source_obj; directly overlay with the original case OBJ files",
            "png_and_json": "normalized computation coordinates",
            "source_transform": "source = normalized * scale + center",
        },
        "normal_orientation": {
            "method": "nearest_labelled_flash_point_per_face",
            "formula": "choose sign(n) so dot(n, F-O) >= 0",
            "dynamic_programming": "not applied when labelled flash points are available",
        },
        "parameters": {
            "radius": candidates.radius,
            "delta": float(config["wheel"]["delta"]),
            "width": width,
            "curve_points": int(len(sampled.points)),
            "candidate_angles": int(config["sampling"]["candidate_angles"]),
            "radial_samples": int(config["sampling"]["radial_samples"]),
            "collision_clearance": float(config["collision"]["clearance"]),
            "collision_body": "solid_cylinder",
            "contact_patch_enabled": contact_patch_enabled,
            "contact_patch_radius": contact_patch_radius,
            "contact_patch_max_penetration": max_contact_penetration,
            "contact_patch_normal_tolerance_degrees": contact_normal_tolerance_degrees,
            "narrow_phase": narrow_phase,
            "broad_phase": "aabb_bvh",
            "finite_nearest_samples": int(config["collision"].get("finite_nearest_samples", 2048)),
            "finite_chunk_size": int(config["collision"].get("finite_chunk_size", 20000)),
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
        "contact_patch_candidate_count": int(
            np.sum([str(value).startswith("contact_patch_only") for value in reason_values])
        ),
        "pose_examples": pose_examples,
        "output_directory": str(relative_output_root),
        "overview_image": str(relative_output_root / image_path.name),
        "pose_examples_obj": str(relative_output_root / pose_obj_path.name),
        "feasible_centers_display_obj": str(relative_output_root / display_obj_path.name),
        "local_feasible_centers_obj": str(relative_output_root / local_obj_path.name),
        "obj_contents": [
            "all OBJ vertex coordinates are exported back to source_obj coordinates for MeshLab overlay",
            "display OBJ: sampled feasible centers as red octahedron face proxies",
            "local OBJ: one target point with feasible red and rejected gray face proxies",
            "pose_examples.obj: feasible_pose is mint green and collision_pose is coral red; OBJ vertex RGB and adjacent MTL agree",
        ],
        "per_point": per_point,
        "limitations": [
            "spherical-envelope 结果仅作为半径扫描/历史 baseline 保留；主结果使用三角形 AABB 宽相和固定姿态下的连续有限圆柱窄相。",
            "候选角度与径向位置仍为离散采样；固定姿态连续相交不等于所有连续角度均已证明安全。",
            "非封闭网格只能给出三角面相交结论，不能完成砂轮整体位于工件内部的实体包含证明。",
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
