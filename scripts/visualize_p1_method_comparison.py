"""把 P1 方法对比结果做成 PNG + OBJ/MTL 可视化。

输入：`p1_method_comparison_*/p1_method_comparison.json`
输出：
- `p1_summary_dashboard.png`
- `p1_<method>_branch7.png`
- `p1_<method>_normals.obj/.mtl`（source_obj 坐标，可叠加原始工件）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from feilunqiege.geometry.curve import split_and_resample  # noqa: E402
from feilunqiege.io.mesh import load_mesh  # noqa: E402
from feilunqiege.io.root_line import load_root_line  # noqa: E402
from feilunqiege.pipeline import _normalize_case_geometry  # noqa: E402


def safe_name(method: str) -> str:
    return method.replace("+", "_plus_").replace("/", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 方法对比可视化")
    parser.add_argument("--comparison-dir", default=None)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--case", default="circle")
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args()

    results_root = REPO / "results" / "cases" / arguments.case
    if arguments.comparison_dir is not None:
        comparison_dir = Path(arguments.comparison_dir).resolve()
    else:
        candidates = sorted(
            results_root.glob("p1_method_comparison_*"),
            key=lambda path: path.name,
            reverse=True,
        )
        comparison_dir = candidates[0]
    data = json.loads((comparison_dir / "p1_method_comparison.json").read_text(encoding="utf-8"))
    records = data["records"]
    methods = list(dict.fromkeys(record["method"] for record in records))
    point_indices = data["metadata"]["points"]
    point_set = set(point_indices)

    config_path = (REPO / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    case_root = (config_path.parent / config["data_root"]).resolve() / arguments.case
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / "mesh_without_flash.obj")
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    center = np.asarray(normalization["center"], dtype=float)
    scale = float(normalization["scale"])
    source_points = sampled.points * scale + center
    normalized_points = np.asarray(sampled.points)

    output_dir = (
        Path(arguments.output_dir).resolve()
        if arguments.output_dir is not None
        else comparison_dir / "visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 汇总 dashboard
    figure, axes = plt.subplots(2, 2, figsize=(17, 12), dpi=170)
    ax = axes[0, 0]
    x = np.arange(len(methods))
    branch7_means = []
    branch4_means = []
    for method in methods:
        rows = [record for record in records if record["method"] == method]
        branch7 = [record["feasible_count"] or 0 for record in rows if record["branch_id"] == 7]
        branch4 = [record["feasible_count"] or 0 for record in rows if record["branch_id"] == 4]
        branch7_means.append(float(np.mean(branch7)) if branch7 else 0.0)
        branch4_means.append(float(np.mean(branch4)) if branch4 else 0.0)
    ax.bar(x - 0.2, branch7_means, width=0.4, color="#D62728", label="branch 7 mean feasible")
    ax.bar(x + 0.2, branch4_means, width=0.4, color="#1F77B4", label="branch 4 mean feasible")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("feasible angles / 72")
    ax.set_title("Branch-wise mean feasible angles")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    branch7_points = [index for index in point_indices if int(sampled.branch_ids[index]) == 7]
    matrix = np.zeros((len(methods), len(branch7_points)), dtype=float)
    for row_index, method in enumerate(methods):
        for column_index, point_index in enumerate(branch7_points):
            record = next(
                record for record in records
                if record["method"] == method and record["point_index"] == point_index
            )
            matrix[row_index, column_index] = record["feasible_count"] or 0
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=72)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_xticks(range(len(branch7_points)))
    ax.set_xticklabels([str(index) for index in branch7_points], fontsize=8)
    ax.set_title("Branch 7: feasible angles per point")
    figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    low_counts = []
    fit_counts = []
    for method in methods:
        rows = [record for record in records if record["method"] == method]
        low_counts.append(sum(1 for record in rows if record["normal_confidence"] == "low"))
        fit_counts.append(sum(1 for record in rows if record["normal_source"] == "jet_fit"))
    ax.bar(x - 0.2, low_counts, width=0.4, color="#FF7F0E", label="low confidence points")
    ax.bar(x + 0.2, fit_counts, width=0.4, color="#2CA02C", label="jet fit points")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_title("Confidence / source composition")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    baseline = {
        record["point_index"]: np.asarray(record["normal"], dtype=float)
        for record in records if record["method"] == "baseline"
    }
    for method in methods:
        angles = []
        for record in records:
            if record["method"] != method or record["point_index"] not in baseline:
                continue
            normal = np.asarray(record["normal"], dtype=float)
            reference = baseline[record["point_index"]]
            cosine = float(np.clip(np.dot(normal, reference), -1.0, 1.0))
            angles.append(float(np.degrees(np.arccos(cosine))))
        ax.plot(range(len(angles)), angles, marker="o", linewidth=1.2, label=method)
    ax.set_xlabel("selected point order")
    ax.set_ylabel("angle vs baseline normal (deg)")
    ax.set_title("Normal deviation from baseline")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    figure.suptitle("P1 literature-inspired normal methods on circle", fontsize=14)
    figure.tight_layout()
    dashboard_path = output_dir / "p1_summary_dashboard.png"
    figure.savefig(dashboard_path, facecolor="white")
    plt.close(figure)

    # 每个方法：branch7 局部图 + OBJ/MTL
    for method in methods:
        method_rows = [record for record in records if record["method"] == method]
        branch7_rows = [record for record in method_rows if record["branch_id"] == 7]
        branch4_rows = [record for record in method_rows if record["branch_id"] == 4]
        if not branch7_rows:
            continue
        figure = plt.figure(figsize=(13, 9), dpi=170)
        ax_normals = figure.add_subplot(1, 2, 1)
        # branch 7 的点几乎在一条线上，3D 视角容易边看边重叠；这里直接画
        # 法向的 (y,z) 分量，能清楚看到不同方法的支撑方向差异。
        ax_normals.add_patch(plt.Circle((0.0, 0.0), 1.0, fill=False, color="#999999", linewidth=0.8))
        ax_normals.axhline(0.0, color="#CCCCCC", linewidth=0.7)
        ax_normals.axvline(0.0, color="#CCCCCC", linewidth=0.7)
        for record in branch7_rows:
            index = record["point_index"]
            normal = np.asarray(record["normal"], dtype=float)
            feasible = record["feasible_count"] or 0
            color = "#2CA02C" if feasible >= 60 else "#FF7F0E" if feasible > 0 else "#D62728"
            ax_normals.quiver(
                0.0, 0.0, normal[1], normal[2],
                angles="xy", scale_units="xy", scale=1.0,
                color=color, width=0.006, headwidth=4, headlength=5,
            )
        ax_normals.set_xlim(-1.1, 1.1)
        ax_normals.set_ylim(-1.1, 1.1)
        ax_normals.set_aspect("equal")
        ax_normals.set_xlabel("normal y component")
        ax_normals.set_ylabel("normal z component")
        ax_normals.set_title(f"{method}: branch 7 normal directions (y-z)")
        ax_normals.grid(True, alpha=0.25, linewidth=0.4)
        ax_normals.legend(
            handles=[
                Line2D([0], [0], color="#2CA02C", lw=3, label=">=60 feasible"),
                Line2D([0], [0], color="#FF7F0E", lw=3, label="1–59 feasible"),
                Line2D([0], [0], color="#D62728", lw=3, label="0 feasible"),
            ],
            fontsize=8,
            loc="upper left",
        )
        ax2d = figure.add_subplot(1, 2, 2)
        labels = [str(record["point_index"]) for record in branch7_rows]
        values = [record["feasible_count"] or 0 for record in branch7_rows]
        colors = ["#2CA02C" if value >= 60 else "#FF7F0E" if value > 0 else "#D62728" for value in values]
        ax2d.bar(labels, values, color=colors)
        ax2d.set_ylim(0, 72)
        ax2d.set_ylabel("feasible angles / 72")
        ax2d.set_title(f"{method}: branch 7 feasible counts")
        ax2d.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        png_path = output_dir / f"p1_{safe_name(method)}_branch7.png"
        figure.savefig(png_path, facecolor="white")
        plt.close(figure)

        # OBJ：source_obj，点 + 法向线 + branch 线，便于 MeshLab 叠加原工件。
        obj_lines = [
            f"# P1 method visualization: {method}",
            "# coordinate_frame: source_obj",
            f"# normalization_scale: {scale:.12g}",
            f"# normalization_center: {center[0]:.12g} {center[1]:.12g} {center[2]:.12g}",
            "# Lines show sampled normals; point labels are in the accompanying JSON.",
            f"mtllib p1_{safe_name(method)}_normals.mtl",
        ]
        vertices: list[str] = []
        lines: list[str] = []
        next_index = 1
        for record in branch7_rows + branch4_rows:
            index = record["point_index"]
            point = source_points[index]
            normal = np.asarray(record["normal"], dtype=float)
            endpoint = point + normal * (0.018 * scale)
            vertices.append(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}")
            vertices.append(f"v {endpoint[0]:.9g} {endpoint[1]:.9g} {endpoint[2]:.9g}")
            lines.append(f"usemtl normal_line")
            lines.append(f"l {next_index} {next_index + 1}")
            next_index += 2
        # branch line
        branch_indices = [index for index in point_indices if int(sampled.branch_ids[index]) == 7]
        start = next_index
        for index in branch_indices:
            point = source_points[index]
            vertices.append(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}")
            next_index += 1
        lines.append("usemtl branch_line")
        lines.append("l " + " ".join(str(index) for index in range(start, next_index)))
        obj_path = output_dir / f"p1_{safe_name(method)}_normals.obj"
        obj_path.write_text("\n".join(obj_lines + vertices + lines) + "\n", encoding="utf-8")
        (output_dir / f"p1_{safe_name(method)}_normals.mtl").write_text(
            "\n".join(
                [
                    "newmtl normal_line",
                    "Ka 0 0 0",
                    "Kd 0.85 0.10 0.10",
                    "illum 0",
                    "newmtl branch_line",
                    "Ka 0 0 0",
                    "Kd 0.10 0.10 0.10",
                    "illum 0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    summary = {
        "comparison_dir": str(comparison_dir),
        "output_dir": str(output_dir),
        "methods": methods,
        "point_indices": point_indices,
        "dashboard": str(dashboard_path),
        "note": "PNG uses normalized coordinates; OBJ/MTL use source_obj coordinates.",
    }
    (output_dir / "visualization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output_dir={output_dir}")
    print(f"dashboard={dashboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
