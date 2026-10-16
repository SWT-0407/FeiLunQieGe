"""绘制阶段一的轴对齐砂轮姿态验收示意图。

这张图故意不显示真实工件点云，只表达姿态逻辑：

* 灰色平面：工件平面 ``z=-0.6``；
* 红色平面：飞边平面 ``y=0``；
* 绿色窄面：飞边根部面，位于飞边平面上；
* 黑色直线：``root_line``，沿 ``x`` 轴；
* 蓝色圆盘：砂轮圆盘，圆盘平面由 ``x`` 和 ``y`` 轴张成；
* 紫色箭头：砂轮轴向 ``z``。

示意中砂轮圆盘平面与切平面平行、与法平面垂直；该图只用于姿态验收，
不用于碰撞结论。
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """构造示意图命令行参数。"""

    parser = argparse.ArgumentParser(description="绘制 leaf 砂轮姿态轴对齐验收示意图")
    parser.add_argument("--case-name", default="leaf", help="固定工件名称；阶段一默认 leaf")
    return parser


def _plane(ax, corners, *, face, edge, alpha, label):
    """绘制带标签语义的透明平面。"""

    ax.add_collection3d(
        Poly3DCollection([corners], facecolor=face, edgecolor=edge, alpha=alpha, linewidth=1.4, label=label)
    )


def _arrow(ax, origin, direction, *, color, label, length=0.85):
    """绘制一个坐标或姿态方向箭头。"""

    ax.quiver(
        *origin,
        *direction,
        length=length,
        color=color,
        linewidth=2.2,
        arrow_length_ratio=0.14,
        label=label,
    )


def main() -> int:
    """生成不依赖真实网格的阶段一验收图和 JSON。"""

    arguments = build_parser().parse_args()
    if arguments.case_name != "leaf":
        raise ValueError("阶段一后续固定使用 leaf")

    output_base = REPO_ROOT / "results" / "cases" / "leaf"
    stamp = datetime.now().strftime("pose_schematic_%Y%m%d_%H%M%S_%f")
    output_root = output_base / stamp
    output_root.mkdir(parents=True, exist_ok=False)
    image_path = output_root / "leaf_pose_schematic.png"
    summary_path = output_root / "leaf_pose_schematic.json"

    # 标准轴对齐示意几何。所有平面边界均平行于 x/y/z 轴，避免真实模型透视造成歧义。
    x = np.linspace(-1.15, 1.15, 2)
    y = np.linspace(-0.75, 1.15, 2)
    z = np.linspace(-0.6, 0.85, 2)
    workpiece_plane = [(-1.15, -0.75, -0.6), (1.15, -0.75, -0.6), (1.15, 1.15, -0.6), (-1.15, 1.15, -0.6)]
    flash_plane = [(-1.15, 0.0, -0.55), (1.15, 0.0, -0.55), (1.15, 0.0, 0.75), (-1.15, 0.0, 0.75)]
    root_face = [(-0.95, -0.012, -0.18), (0.95, -0.012, -0.18), (0.95, -0.012, 0.18), (-0.95, -0.012, 0.18)]

    # 根部线沿 x 轴；接触点为原点。正确姿态取 t=x、e=y、a=z。
    contact = np.array([0.0, 0.0, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])
    radius_direction = np.array([0.0, 1.0, 0.0])
    axis_direction = np.array([0.0, 0.0, 1.0])
    radius = 0.45
    center = contact + radius * radius_direction

    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    circle = center[None, :] + radius * (
        np.cos(theta)[:, None] * tangent[None, :] + np.sin(theta)[:, None] * radius_direction[None, :]
    )

    figure = plt.figure(figsize=(12, 8), dpi=170, layout="constrained")
    ax = figure.add_subplot(111, projection="3d")
    _plane(ax, workpiece_plane, face="#B8BDC5", edge="#4D5966", alpha=0.25, label="workpiece plane")
    _plane(ax, flash_plane, face="#E69F00", edge="#A66100", alpha=0.18, label="flash plane")
    _plane(ax, root_face, face="#009E73", edge="#006D52", alpha=0.55, label="flash-root face")

    # 根部线是飞边平面与工件平面附近的目标交线示意，明确画在 x 轴方向。
    ax.plot([-1.0, 1.0], [0.0, 0.0], [0.0, 0.0], color="#006D52", linewidth=4.0, label="root_line")
    ax.scatter(*contact, color="black", s=48, label="contact point")

    # 砂轮圆盘平面为 x-y 平面；圆周在接触点处与 root_line 的切线平行。
    ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color="#0072B2", linewidth=3.6, label="wheel circle")
    wheel_corners = [
        center - radius * tangent - radius * radius_direction,
        center + radius * tangent - radius * radius_direction,
        center + radius * tangent + radius * radius_direction,
        center - radius * tangent + radius * radius_direction,
    ]
    _plane(ax, wheel_corners, face="#56B4E9", edge="#0072B2", alpha=0.18, label="wheel disk plane")
    ax.scatter(*center, color="#0072B2", s=48, label="wheel center")

    # 只保留必要的三个姿态/坐标方向箭头。
    _arrow(ax, contact, tangent, color="#005AB5", label="tangent t = +x")
    _arrow(ax, center, radius_direction, color="#E69F00", label="radius e = +y")
    _arrow(ax, center, axis_direction, color="#CC79A7", label="wheel axis a = +z")

    # 标准 x-y-z 坐标系放在左前下角，避免和砂轮姿态箭头混淆。
    origin = np.array([-1.0, -0.65, -0.52])
    _arrow(ax, origin, np.array([1.0, 0.0, 0.0]), color="#222222", label="x axis", length=0.35)
    _arrow(ax, origin, np.array([0.0, 1.0, 0.0]), color="#222222", label="y axis", length=0.35)
    _arrow(ax, origin, np.array([0.0, 0.0, 1.0]), color="#222222", label="z axis", length=0.35)
    ax.text(*(origin + np.array([0.38, 0.0, 0.0])), "x", color="#222222")
    ax.text(*(origin + np.array([0.0, 0.38, 0.0])), "y", color="#222222")
    ax.text(*(origin + np.array([0.0, 0.0, 0.38])), "z", color="#222222")

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.85, 1.25)
    ax.set_zlim(-0.75, 1.0)
    ax.set_box_aspect((2.5, 2.1, 1.8))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("leaf stage-1 schematic: corrected wheel orientation")
    ax.text2D(
        0.02,
        0.02,
        "root_line // x\nwheel disk plane = span(x, y)\nwheel axis = z\nworkpiece/flash planes are schematic only",
        transform=ax.transAxes,
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.view_init(elev=25, azim=-58)
    figure.savefig(image_path, dpi=170, facecolor="white")
    plt.close(figure)

    summary = {
        "case_name": "leaf",
        "stage": "stage_1_pose_and_input_contract",
        "figure_type": "axis_aligned_schematic",
        "units": "schematic coordinates",
        "geometry": {
            "workpiece_plane": "z=-0.6 (xy plane)",
            "flash_plane": "y=0 (xz plane)",
            "flash_root_face": "narrow highlighted region on y=0",
            "root_line": "x axis at y=0,z=0",
            "wheel_disk_plane": "z=0 (xy plane)",
            "wheel_center": [0.0, 0.45, 0.0],
            "wheel_radius": radius,
            "tangent": [1.0, 0.0, 0.0],
            "radius_direction": [0.0, 1.0, 0.0],
            "axis": [0.0, 0.0, 1.0],
        },
        "checks": {
            "e_dot_t": float(np.dot(radius_direction, tangent)),
            "a_dot_t": float(np.dot(axis_direction, tangent)),
            "a_dot_e": float(np.dot(axis_direction, radius_direction)),
            "wheel_circle_tangent_parallel_to_root_line": True,
        },
        "outputs": {
            "image": str(image_path.relative_to(REPO_ROOT)),
            "summary": str(summary_path.relative_to(REPO_ROOT)),
        },
        "limitations": [
            "这是轴对齐逻辑示意图，不是 leaf 原始网格的测量图。",
            "平面位置和比例用于清楚表达姿态，不用于碰撞或加工尺寸结论。",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status=passed case=leaf")
    print(f"image={image_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
