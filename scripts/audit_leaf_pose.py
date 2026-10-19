"""生成 ``leaf`` 案例的砂轮姿态审计图。

本脚本只验证阶段一的几何姿态，不执行碰撞筛选，也不覆盖历史运行结果。
图中同时画出：

* 根部曲线切线 ``t``；
* 以 ``t`` 为法向的法平面；
* 位于切平面内的半径方向 ``e`` 和砂轮中心轨迹示意；
* 与切平面平行的圆盘平面及工件表面法向轴向。
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
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from feilunqiege.geometry.curve import split_and_resample
from feilunqiege.geometry.frame import build_frames, estimate_surface_normals
from feilunqiege.io.mesh import load_mesh
from feilunqiege.io.root_line import load_root_line
from feilunqiege.pipeline import _normalize_case_geometry
from feilunqiege.tooling.grinding_wheel import generate_disk_candidates


def build_parser() -> argparse.ArgumentParser:
    """构造阶段一审计脚本参数。"""

    parser = argparse.ArgumentParser(description="生成 leaf 砂轮姿态审计图")
    parser.add_argument("--config", default="configs/default.json", help="配置文件路径")
    parser.add_argument("--point-index", type=int, default=None, help="代表性根部采样点；默认取中间点")
    return parser


def _normalise(vector: np.ndarray) -> np.ndarray:
    """归一化向量并在退化时给出明确错误。"""

    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("姿态审计遇到零长度向量")
    return vector / length


def _create_output_root(repo_root: Path) -> Path:
    """创建不可覆盖的时间戳目录；旧审计结果始终保留。"""

    base = repo_root / "results" / "cases" / "leaf"
    stamp = datetime.now().strftime("pose_audit_%Y%m%d_%H%M%S_%f")
    output = base / stamp
    output.mkdir(parents=True, exist_ok=False)
    return output


def _draw_disk_plane(axis, center: np.ndarray, basis_1: np.ndarray, basis_2: np.ndarray, radius: float) -> None:
    """绘制与切平面平行的砂轮圆盘平面。"""

    grid = np.linspace(-radius, radius, 2)
    vertices = []
    for u in grid:
        for v in grid:
            vertices.append(center + u * basis_1 + v * basis_2)
    vertices = np.asarray(vertices).reshape(2, 2, 3)
    corners = [vertices[0, 0], vertices[1, 0], vertices[1, 1], vertices[0, 1]]
    axis.add_collection3d(
        Poly3DCollection([corners], facecolor="#56B4E9", edgecolor="#0072B2", alpha=0.25, linewidth=1.2)
    )


def _draw_normal_plane(axis, point: np.ndarray, basis_1: np.ndarray, basis_2: np.ndarray, extent: float) -> None:
    """绘制以 ``t`` 为法向的法平面，作为橙色参考平面。"""

    corners = [
        point - extent * basis_1 - extent * basis_2,
        point + extent * basis_1 - extent * basis_2,
        point + extent * basis_1 + extent * basis_2,
        point - extent * basis_1 + extent * basis_2,
    ]
    axis.add_collection3d(
        Poly3DCollection([corners], facecolor="#E69F00", edgecolor="#A66100", alpha=0.12, linewidth=1.0)
    )


def main() -> int:
    """读取 leaf、生成姿态审计图并写入 JSON 数值报告。"""

    arguments = build_parser().parse_args()
    config_path = (REPO_ROOT / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("case_name") != "leaf":
        raise ValueError("阶段一固定使用 leaf；请将 configs/default.json 的 case_name 保持为 leaf")

    # 只读取外部原始数据并在内存中做与主流程一致的归一化，不改写输入文件。
    data_root = (config_path.parent / config["data_root"]).resolve()
    case_root = data_root / "leaf"
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / config["inputs"]["workpiece_mesh"])
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
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

    point_index = arguments.point_index
    if point_index is None:
        point_index = len(sampled.points) // 2
    if not 0 <= point_index < len(sampled.points):
        raise ValueError(f"point-index 必须在 [0, {len(sampled.points) - 1}] 内")

    wheel = config["wheel"]
    candidates = generate_disk_candidates(
        sampled.points[point_index : point_index + 1],
        [frames[point_index]],
        float(wheel["radius"]),
        float(wheel["delta"]),
        4,
        2,
    )
    # 候选按径向层分组；选取最外层 rho=r，避免把代表图误读为内侧余量半径。
    selected_index = int(np.argmax(np.asarray(candidates.radial_values, dtype=float)))
    contact = np.asarray(candidates.contact_points[selected_index], dtype=float)
    center = np.asarray(candidates.centers[selected_index], dtype=float)
    radial = _normalise(np.asarray(candidates.radius_directions[selected_index], dtype=float))
    tangent = _normalise(np.asarray(candidates.tangent_directions[selected_index], dtype=float))
    axis_direction = _normalise(np.asarray(candidates.axis_directions[selected_index], dtype=float))
    disk_basis_2 = _normalise(np.cross(axis_direction, radial))
    radius = float(candidates.radial_values[selected_index])

    # 计算姿态正交性，数值结果会写入 JSON，供人工验收时核对。
    metrics = {
        "radius_dot_tangent": float(np.dot(radial, tangent)),
        "axis_dot_tangent": float(np.dot(axis_direction, tangent)),
        "axis_dot_radius": float(np.dot(axis_direction, radial)),
        "radius_norm": float(np.linalg.norm(radial)),
        "axis_norm": float(np.linalg.norm(axis_direction)),
        "contact_to_center_distance": float(np.linalg.norm(center - contact)),
    }
    tolerance = 1e-10
    passed = (
        abs(metrics["axis_dot_tangent"]) <= tolerance
        and abs(metrics["axis_dot_radius"]) <= tolerance
        and abs(metrics["radius_norm"] - 1.0) <= tolerance
        and abs(metrics["axis_norm"] - 1.0) <= tolerance
    )

    output_root = _create_output_root(REPO_ROOT)
    image_path = output_root / "leaf_pose_orientation_audit.png"
    summary_path = output_root / "leaf_pose_orientation_audit.json"

    # 绘制局部姿态：橙色是法平面，蓝色是与切平面平行的砂轮圆盘平面。
    figure = plt.figure(figsize=(12, 8), dpi=160, layout="constrained")
    axis = figure.add_subplot(111, projection="3d")
    mesh_vertices = np.asarray(workpiece.vertices, dtype=float)
    near_radius = max(6.0 * radius, 0.06)
    nearby = np.linalg.norm(mesh_vertices - contact[None, :], axis=1) <= near_radius
    if np.count_nonzero(nearby) >= 3:
        near_vertices = mesh_vertices[nearby]
        axis.scatter(near_vertices[:, 0], near_vertices[:, 1], near_vertices[:, 2], s=5, c="#B8BDC5", alpha=0.28, label="nearby workpiece vertices")

    normal_plane_basis_1 = _normalise(np.asarray(frames[point_index].radial_basis_2, dtype=float))
    _draw_normal_plane(axis, contact, normal_plane_basis_1, axis_direction, max(4.0 * radius, 0.04))
    _draw_disk_plane(axis, center, radial, disk_basis_2, radius)

    # 用绿色圆周直接强调最终砂轮方向；橙色圆仅作为法平面参考，不代表砂轮。
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    circle = center[None, :] + radius * (
        np.cos(theta)[:, None] * radial[None, :] + np.sin(theta)[:, None] * disk_basis_2[None, :]
    )
    axis.plot(circle[:, 0], circle[:, 1], circle[:, 2], color="#009E73", linewidth=3.0, label="correct wheel circle")
    axis.scatter(*contact, c="black", s=45, label="root-line contact point")
    axis.scatter(*center, c="#D55E00", s=45, label="wheel center")
    vector_length = max(5.0 * radius, 0.05)
    for vector, color, label in (
        (tangent, "#0072B2", "tangent t"),
        (radial, "#E69F00", "radius e"),
        (axis_direction, "#CC79A7", "wheel axis a"),
    ):
        origin = contact if label == "tangent t" else center
        axis.quiver(*origin, *vector, length=vector_length, color=color, linewidth=2.0, arrow_length_ratio=0.16, label=label)

    values = np.vstack([near_vertices if np.count_nonzero(nearby) >= 3 else np.empty((0, 3)), circle, contact, center])
    lower = values.min(axis=0)
    upper = values.max(axis=0)
    midpoint = (lower + upper) / 2.0
    half = max(float((upper - lower).max()) / 2.0, 1e-3) * 1.15
    axis.set_xlim(midpoint[0] - half, midpoint[0] + half)
    axis.set_ylim(midpoint[1] - half, midpoint[1] + half)
    axis.set_zlim(midpoint[2] - half, midpoint[2] + half)
    axis.set_xlabel("X (normalized units)")
    axis.set_ylabel("Y (normalized units)")
    axis.set_zlabel("Z (normalized units)")
    axis.set_title("leaf stage-1 pose audit: wheel plane parallel to tangent plane")
    axis.text2D(0.02, 0.02, "Orange plane: normal plane (reference)\nBlue/green circle: wheel plane, parallel to tangent plane", transform=axis.transAxes, fontsize=10)
    axis.legend(loc="upper left", fontsize=8)
    axis.view_init(elev=24, azim=-58)
    figure.savefig(image_path, dpi=160, facecolor="white")
    plt.close(figure)

    summary = {
        "case_name": "leaf",
        "stage": "stage_1_pose_and_input_contract",
        "point_index": int(point_index),
        "candidate_index": selected_index,
        "units": "normalized",
        "tool_model": "zero_thickness_disk_orientation_audit",
        "pose_definition": {
            "tangent": "t",
            "radius_direction": "e",
            "wheel_plane": "local tangent plane, perpendicular to normal plane",
            "axis": "surface normal n",
            "center": "contact + rho * e, e in local tangent plane",
        },
        "wheel": {"radius": radius, "delta": float(wheel["delta"]), "width_reference": float(wheel.get("width", 0.0))},
        "metrics": metrics,
        "orthogonality_tolerance": tolerance,
        "pose_checks_passed": passed,
        "normalization": normalization,
        "outputs": {"image": str(image_path.relative_to(REPO_ROOT)), "summary": str(summary_path.relative_to(REPO_ROOT))},
        "limitations": [
            "本审计只确认姿态与坐标契约，不确认碰撞安全性。",
            "工件显示为局部顶点参考，不代表三角面碰撞判定。",
            "绿色圆表示砂轮圆盘方向；橙色平面只是法平面参考。",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status={'passed' if passed else 'failed'} case=leaf point_index={point_index}")
    print(f"image={image_path}")
    print(f"summary={summary_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
