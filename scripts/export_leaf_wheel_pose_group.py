"""为 leaf 的多个采样点导出一组 MeshLab 砂轮姿态 OBJ。

脚本只执行法向估计、局部标架构建和无厚度圆盘姿态生成，不执行碰撞筛选。
输出默认只有 OBJ、MTL 和 JSON，便于与 ``mesh_without_flash.obj`` 叠加验收。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from feilunqiege.geometry.curve import split_and_resample
from feilunqiege.geometry.frame import build_frames, estimate_surface_normals
from feilunqiege.io.mesh import load_mesh
from feilunqiege.io.root_line import load_root_line
from feilunqiege.pipeline import _normalize_case_geometry
from feilunqiege.tooling.grinding_wheel import generate_disk_candidates
from feilunqiege.visualization.export_obj import export_wheel_pose_group_obj


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出 leaf 多采样点砂轮姿态 OBJ")
    parser.add_argument("--config", default="configs/default.json", help="配置文件路径")
    parser.add_argument(
        "--point-indices",
        default="241,780,809,831,905",
        help="逗号分隔的 root_line 重采样点索引",
    )
    parser.add_argument("--segments", type=int, default=48, help="每个砂轮圆周的离散段数")
    return parser


def _parse_indices(raw: str, count: int) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or len(set(values)) != len(values):
        raise ValueError("采样点索引不能为空且不能重复")
    if any(value < 0 or value >= count for value in values):
        raise ValueError(f"采样点索引必须位于 [0, {count - 1}]")
    return values


def main() -> int:
    arguments = build_parser().parse_args()
    config_path = (REPO_ROOT / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("case_name") != "leaf":
        raise ValueError("该脚本固定使用 leaf 工件")

    data_root = (config_path.parent / config["data_root"]).resolve()
    case_root = data_root / "leaf"
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / config["inputs"]["workpiece_mesh"])
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    point_indices = _parse_indices(arguments.point_indices, len(sampled.points))

    normal_config = config.get("normal_estimation", {})
    surface_normals = estimate_surface_normals(
        sampled.points,
        workpiece,
        neighbor_count=int(normal_config.get("neighbor_count", 24)),
        candidate_count=int(normal_config.get("candidate_count", 96)),
        side_count=int(normal_config.get("side_count", 8)),
        coherence_angle_degrees=float(normal_config.get("coherence_angle_degrees", 25.0)),
        offset_epsilon=float(normal_config.get("offset_epsilon", 1e-5)),
        weight_distance_floor=float(normal_config.get("weight_distance_floor", 1e-5)),
    )
    frames = build_frames(sampled.points, sampled.tangents, surface_normals)
    wheel = config["wheel"]
    radius = float(wheel["radius"])
    delta = float(wheel["delta"])
    width = float(wheel.get("width", 0.0))
    if width <= 0.0:
        raise ValueError("wheel.width 必须为正数，才能导出有限厚度砂轮")

    poses: list[dict[str, object]] = []
    for point_index in point_indices:
        candidates = generate_disk_candidates(
            sampled.points[point_index : point_index + 1],
            [frames[point_index]],
            radius,
            delta,
            4,
            2,
        )
        selected_index = int(np.argmax(np.asarray(candidates.radial_values, dtype=float)))
        poses.append(
            {
                "point_index": point_index,
                "contact": np.asarray(candidates.contact_points[selected_index], dtype=float).tolist(),
                "center": np.asarray(candidates.centers[selected_index], dtype=float).tolist(),
                "radial": np.asarray(candidates.radius_directions[selected_index], dtype=float).tolist(),
                "axis": np.asarray(candidates.axis_directions[selected_index], dtype=float).tolist(),
                "radius": float(candidates.radial_values[selected_index]),
                "candidate_index": selected_index,
            }
        )

    output_root = REPO_ROOT / "results" / "cases" / "leaf" / datetime.now().strftime("wheel_pose_obj_%Y%m%d_%H%M%S_%f")
    output_root.mkdir(parents=True, exist_ok=False)
    obj_path = output_root / "leaf_wheel_pose_group.obj"
    export_wheel_pose_group_obj(
        sampled,
        poses,
        obj_path,
        width=width,
        coordinate_transform=normalization,
        segments=int(arguments.segments),
    )
    summary = {
        "status": "preliminary_wheel_pose_visualization",
        "case_name": "leaf",
        "point_indices": point_indices,
        "units": "normalized for computation; source_obj for OBJ",
        "wheel": {"radius": radius, "delta": delta, "width": width},
        "normal_estimation": normal_config,
        "normalization": normalization,
        "poses": poses,
        "outputs": {
            "obj": str(obj_path.relative_to(REPO_ROOT)),
            "mtl": str(obj_path.with_suffix(".mtl").relative_to(REPO_ROOT)),
        },
        "limitations": [
            "该 OBJ 仅用于姿态人工验收，不包含碰撞检测结论。",
            "砂轮使用有限厚度圆柱代理，姿态中心取 rho=r 的代表候选。",
        ],
    }
    summary_path = output_root / "leaf_wheel_pose_group.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status=completed case=leaf points={','.join(str(index) for index in point_indices)}")
    print(f"obj={obj_path}")
    print(f"mtl={obj_path.with_suffix('.mtl')}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
