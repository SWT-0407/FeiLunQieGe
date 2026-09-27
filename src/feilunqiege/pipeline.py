"""端到端流程编排接口占位。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

from feilunqiege.collision.checker import check_candidates
from feilunqiege.geometry.curve import split_and_resample
from feilunqiege.geometry.frame import build_frames
from feilunqiege.io.mesh import load_mesh
from feilunqiege.io.root_line import load_root_line
from feilunqiege.tooling.grinding_wheel import generate_disk_candidates
from feilunqiege.visualization.export_obj import export_feasible_centers_obj
from feilunqiege.visualization.plot import plot_feasible_centers


def run_from_config(config_path: str | Path) -> Any:
    """按配置执行输入、采样、候选生成、碰撞筛选和可视化流程。

    这里保留输入原坐标值，不进行毫米换算。
    """

    # 配置文件位于仓库的 configs/ 目录；所有外部原始数据都通过相对路径引用。
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.parent.parent
    data_root = (config_path.parent / config["data_root"]).resolve()
    case_root = data_root / config["case_name"]

    # 读取根部线和工件本体，保留输入坐标，不在这里做毫米换算或缩放。
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / config["inputs"]["workpiece_mesh"])

    # 先按 PLY edge 拆分分支，再按弧长规则获得目标采样点和切线。
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    frames = build_frames(sampled.points, sampled.tangents)

    # 在每个目标点的切线法平面内生成候选中心，半径方向天然满足 e·t=0。
    wheel = config["wheel"]
    candidates = generate_disk_candidates(
        sampled.points,
        frames,
        float(wheel["radius"]),
        float(wheel["delta"]),
        int(config["sampling"]["candidate_angles"]),
        int(config["sampling"]["radial_samples"]),
    )

    # 当前使用表面采样点的保守球形包络做快速碰撞筛选，结果必须标注为近似。
    collisions = check_candidates(candidates, workpiece, float(config["collision"]["clearance"]))
    output_root = repo_root / config["outputs"]["directory"]

    # 输出一张用于快速核验的三维总览图。
    image_path = output_root / config["outputs"]["overview_image"]
    plot_feasible_centers(
        sampled,
        collisions,
        image_path,
        radius=candidates.radius,
        delta=float(wheel["delta"]),
        workpiece_mesh=workpiece,
    )

    # 输出 MeshLab 可直接打开的 OBJ：l 是根部线，p 是可行砂轮中心点。
    obj_path = output_root / config["outputs"]["feasible_centers_obj"]
    export_feasible_centers_obj(
        sampled,
        collisions,
        obj_path,
        radius=candidates.radius,
        delta=float(wheel["delta"]),
        units=str(config["units"]),
    )

    # 最后写入 JSON 审计摘要，记录参数、计数、输出文件和当前方法边界。
    summary = _build_summary(config, root_line, sampled, candidates, collisions, image_path, obj_path)
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
    obj_path: Path,
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
    return {
        "status": "preliminary_geometry_run",
        "units": config["units"],
        "tool_model": config["tool_model"],
        "collision_method": collisions.method,
        "case_name": config["case_name"],
        "inputs": {
            "root_line": str(Path(config["data_root"]) / config["case_name"] / config["inputs"]["root_line"]),
            "root_line_vertex_count": int(len(root_line.vertices)),
        },
        "parameters": {
            "radius": candidates.radius,
            "delta": float(config["wheel"]["delta"]),
            "curve_points": int(len(sampled.points)),
            "candidate_angles": int(config["sampling"]["candidate_angles"]),
            "radial_samples": int(config["sampling"]["radial_samples"]),
            "collision_clearance": float(config["collision"]["clearance"]),
        },
        "candidate_count": int(len(candidates.centers)),
        "feasible_candidate_count": feasible_count,
        "feasible_ratio": feasible_count / len(candidates.centers) if len(candidates.centers) else 0.0,
        "overview_image": str(Path(config["outputs"]["directory"]) / image_path.name),
        "feasible_centers_obj": str(Path(config["outputs"]["directory"]) / obj_path.name),
        "obj_contents": [
            "sampled root-line vertices and l line primitives",
            "feasible wheel-center vertices and p point primitives grouped by target point",
        ],
        "per_point": per_point,
        "limitations": [
            "碰撞筛选使用工件表面采样点的近似球形包络，可能误判可行；不是精确的圆盘-三角面相交计算。",
            "砂轮没有轴向厚度，结果不能代表有限宽度砂轮的侧面干涉结论。",
            "结果使用 normalized units，不能直接解释为毫米加工公差。",
        ],
    }


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
