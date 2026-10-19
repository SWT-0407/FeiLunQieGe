"""导出 ``leaf`` 问题采样点的法向诊断结果。

本脚本只读取当前配置和原始输入，不执行碰撞检测。
它把两个侧面法向及其内角平分法向写入 JSON、MeshLab 可打开的 OBJ，
以及便于人工查看的 PNG。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from feilunqiege.geometry.curve import split_and_resample
from feilunqiege.geometry.frame import estimate_surface_normals
from feilunqiege.geometry.orientation import estimate_flash_body_directions
from feilunqiege.io.flash_cloud import load_labeled_flash_cloud
from feilunqiege.io.mesh import load_mesh
from feilunqiege.io.root_line import load_root_line
from feilunqiege.pipeline import _normalize_case_geometry
from feilunqiege.visualization.export_obj import export_normal_diagnostic_obj


COLORS = {
    "n1": "#E41A1C",
    "n2": "#377EB8",
    "normal": "#009E73",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出 leaf 问题点法向量的 OBJ、PNG 和 JSON")
    parser.add_argument("--config", default="configs/default.json", help="项目配置文件")
    parser.add_argument(
        "--point-indices",
        default="241,780,809,831,905",
        help="逗号分隔的 root_line 重采样点索引",
    )
    parser.add_argument("--arrow-length", type=float, default=0.08, help="归一化坐标中的箭头长度")
    return parser


def _parse_indices(raw: str, count: int) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("至少需要一个问题点索引")
    if len(set(values)) != len(values):
        raise ValueError("问题点索引不能重复")
    if any(value < 0 or value >= count for value in values):
        raise ValueError(f"问题点索引必须位于 [0, {count - 1}]")
    return values


def _set_equal_axes(axis, values: np.ndarray) -> None:
    low = values.min(axis=0)
    high = values.max(axis=0)
    center = (low + high) / 2.0
    half = max(float((high - low).max()) / 2.0, 1e-6) * 1.15
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)


def _nearby_vertices(mesh, point: np.ndarray, radius: float, maximum: int = 5000) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    mask = np.linalg.norm(vertices - point[None, :], axis=1) <= radius
    selected = vertices[mask]
    if len(selected) <= maximum:
        return selected
    indices = np.linspace(0, len(selected) - 1, maximum, dtype=int)
    return selected[indices]


def _draw_point(axis, sampled, diagnostics, point_index: int, mesh, arrow_length: float) -> None:
    point = np.asarray(sampled.points[point_index], dtype=float)
    local_vertices = _nearby_vertices(mesh, point, max(0.12, arrow_length * 1.8))
    if len(local_vertices):
        axis.scatter(
            local_vertices[:, 0],
            local_vertices[:, 1],
            local_vertices[:, 2],
            s=3,
            c="#B8BDC5",
            alpha=0.28,
            label="nearby workpiece vertices",
        )

    start = max(0, point_index - 20)
    end = min(len(sampled.points), point_index + 21)
    root_segment = np.asarray(sampled.points[start:end], dtype=float)
    if len(root_segment) >= 2:
        axis.plot(
            root_segment[:, 0],
            root_segment[:, 1],
            root_segment[:, 2],
            color="#666666",
            linewidth=1.2,
            label="root_line nearby",
        )
    axis.scatter(*point, s=45, c="black", label=f"problem point {point_index}")

    item = diagnostics[point_index]
    directions: list[tuple[str, np.ndarray, float]] = []
    sides = item.get("side_details", [])
    for label, side_number in (("n1", 0), ("n2", 1)):
        selected_key = "selected_n1" if side_number == 0 else "selected_n2"
        if item.get(selected_key) is not None:
            directions.append((label, np.asarray(item[selected_key], dtype=float), arrow_length))
        elif side_number < len(sides):
            directions.append((label, np.asarray(sides[side_number]["normal"], dtype=float), arrow_length))
    selected = np.asarray(item.get("selected_normal", [0.0, 0.0, 0.0]), dtype=float)
    if np.linalg.norm(selected) > 1e-12:
        directions.append(("normal", selected, arrow_length * 1.12))

    for label, direction, length in directions:
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        axis.quiver(
            *point,
            *direction,
            length=length,
            color=COLORS[label],
            linewidth=2.4,
            arrow_length_ratio=0.16,
            label=label,
        )
    axis.set_title(f"leaf normal diagnostic, point {point_index}\nn1 + n2 angle bisector; normalized units")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    axis.legend(unique.values(), unique.keys(), loc="upper left", fontsize=7)
    values = np.vstack([local_vertices, root_segment, point[None, :]]) if len(local_vertices) else np.vstack([root_segment, point])
    for _, direction, length in directions:
        values = np.vstack((values, point + length * direction))
    _set_equal_axes(axis, values)
    axis.view_init(elev=23, azim=-58)


def _save_figures(output_root: Path, sampled, diagnostics, point_indices, mesh, arrow_length: float) -> list[Path]:
    paths: list[Path] = []
    figure = plt.figure(figsize=(16, 9), dpi=160, layout="constrained")
    for subplot_index, point_index in enumerate(point_indices, start=1):
        axis = figure.add_subplot(2, 3, subplot_index, projection="3d")
        _draw_point(axis, sampled, diagnostics, point_index, mesh, arrow_length)
    overview = output_root / "leaf_problem_normals_overview.png"
    figure.savefig(overview, facecolor="white")
    plt.close(figure)
    paths.append(overview)

    for point_index in point_indices:
        figure = plt.figure(figsize=(9, 8), dpi=170, layout="constrained")
        axis = figure.add_subplot(111, projection="3d")
        _draw_point(axis, sampled, diagnostics, point_index, mesh, arrow_length)
        path = output_root / f"leaf_problem_normal_{point_index:04d}.png"
        figure.savefig(path, facecolor="white")
        plt.close(figure)
        paths.append(path)
    return paths


def main() -> int:
    arguments = build_parser().parse_args()
    config_path = (REPO_ROOT / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("case_name") != "leaf":
        raise ValueError("本诊断脚本固定使用 leaf 工件")

    data_root = (config_path.parent / config["data_root"]).resolve()
    case_root = data_root / "leaf"
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / config["inputs"]["workpiece_mesh"])
    flash_cloud = load_labeled_flash_cloud(case_root / "original_flash_cloud.obj")
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    point_indices = _parse_indices(arguments.point_indices, len(sampled.points))
    normal_config = config.get("normal_estimation", {})
    center = np.asarray(normalization.get("center", [0.0, 0.0, 0.0]), dtype=float)
    scale = float(normalization.get("scale", 1.0))
    side_hints, side_hint_diagnostics = estimate_flash_body_directions(
        sampled.points, sampled.tangents,
        (flash_cloud.flash_points - center) / scale,
        (flash_cloud.body_points - center) / scale,
        neighbor_count=int(normal_config.get("flash_neighbor_count", 32)),
    )
    surface_normals, diagnostics = estimate_surface_normals(
        sampled.points,
        workpiece,
        neighbor_count=int(normal_config.get("neighbor_count", 24)),
        candidate_count=int(normal_config.get("candidate_count", 96)),
        side_count=int(normal_config.get("side_count", 8)),
        coherence_angle_degrees=float(normal_config.get("coherence_angle_degrees", 25.0)),
        offset_epsilon=float(normal_config.get("offset_epsilon", 1e-5)),
        weight_distance_floor=float(normal_config.get("weight_distance_floor", 1e-5)),
        branch_ids=sampled.branch_ids,
        side_direction_hints=side_hints,
        side_direction_weight=float(normal_config.get("side_direction_weight", 2.0)),
        return_diagnostics=True,
    )

    output_root = REPO_ROOT / "results" / "cases" / "leaf" / datetime.now().strftime("normal_diagnostic_%Y%m%d_%H%M%S_%f")
    output_root.mkdir(parents=True, exist_ok=False)
    obj_path = output_root / "leaf_problem_normals.obj"
    export_normal_diagnostic_obj(
        sampled,
        diagnostics,
        point_indices,
        obj_path,
        coordinate_transform=normalization,
        arrow_length=float(arguments.arrow_length),
    )
    image_paths = _save_figures(output_root, sampled, diagnostics, point_indices, workpiece, float(arguments.arrow_length))

    records = []
    for point_index in point_indices:
        item = diagnostics[point_index]
        records.append(
            {
                "point_index": point_index,
                "point_normalized": np.asarray(sampled.points[point_index], dtype=float).tolist(),
                "tangent": item.get("tangent"),
                "side_axis": item.get("side_axis"),
                "group_sizes": item.get("group_sizes"),
                "grouping_method": item.get("grouping_method"),
                "side_details": item.get("side_details"),
                "n1": item.get("side_details", [{}])[0].get("normal") if item.get("side_details") else None,
                "n2": item.get("side_details", [{}, {}])[1].get("normal") if len(item.get("side_details", [])) > 1 else None,
                "selected_n1": item.get("selected_n1"),
                "selected_n2": item.get("selected_n2"),
                "selected_normal": item.get("selected_normal"),
            }
        )
    summary = {
        "status": "preliminary_normal_diagnostic",
        "case_name": "leaf",
        "units": "normalized for computation; source_obj for OBJ",
        "point_indices": point_indices,
        "normal_estimation": normal_config,
        "normalization": normalization,
        "outputs": {
            "obj": str(obj_path.relative_to(REPO_ROOT)),
            "mtl": str(obj_path.with_suffix(".mtl").relative_to(REPO_ROOT)),
            "png": [str(path.relative_to(REPO_ROOT)) for path in image_paths],
        },
        "points": records,
        "flash_side_hint_diagnostics": side_hint_diagnostics,
        "limitations": [
            "本输出只记录当前法向估计过程，不执行碰撞检测。",
            "OBJ 中的箭头是诊断几何，不是砂轮实体。",
            "人工验收应在 MeshLab 中叠加 mesh_without_flash.obj，并检查两侧法向和最终法向的局部方向。",
        ],
    }
    summary_path = output_root / "leaf_problem_normals.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status=completed case=leaf points={','.join(str(index) for index in point_indices)}")
    print(f"obj={obj_path}")
    print(f"mtl={obj_path.with_suffix('.mtl')}")
    print(f"png_overview={image_paths[0]}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
