"""核验 MeshLab 导出 OBJ 是否能与原始工件坐标直接叠加。

算法为便于比较不同工件，会先在内存中把数据变为 normalized 坐标；而
MeshLab 通常打开的是外部目录中的原始 OBJ。本脚本检查每个结果 OBJ 开头写入
的 root_line 顶点，是否已经被逆变换回原始 root_line.ply 的 source_obj 坐标。
它验证的是“坐标框架一致性”，不把通过误认为碰撞判定或工艺可行性已被验证。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

# 允许直接以 ``python scripts/audit_meshlab_alignment.py`` 运行，不要求先安装包。
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from feilunqiege.io.root_line import load_root_line


def _parse_arguments() -> argparse.Namespace:
    """读取批量审计需要的配置、案例范围和可选正式报告位置。"""

    parser = argparse.ArgumentParser(description="审计 OBJ 与原始 root_line 的坐标一致性")
    parser.add_argument("--config", default="configs/default.json", help="仓库内配置 JSON 路径")
    parser.add_argument("--case", action="append", dest="cases", help="只审计指定案例，可重复传入")
    parser.add_argument("--output", help="把 JSON 审计报告写入一个尚不存在的正式文件")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="逐坐标最大绝对误差上限")
    return parser.parse_args()


def _latest_run(case_directory: Path) -> Path:
    """选择一个案例最新的时间戳运行目录，不修改或清理任何历史结果。"""

    runs = sorted((path for path in case_directory.glob("run_*") if path.is_dir()), reverse=True)
    if not runs:
        raise FileNotFoundError(f"未找到运行目录: {case_directory}")
    return runs[0]


def _read_obj_header_and_root_vertices(path: Path, root_count: int) -> tuple[str | None, np.ndarray]:
    """仅读取 OBJ 头部和首段 root_line 顶点，避免加载数量很大的可行中心点。"""

    coordinate_frame: str | None = None
    vertices: list[list[float]] = []
    in_root_line = False
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("# coordinate_frame:"):
                coordinate_frame = line.split(":", maxsplit=1)[1].strip()
            if line == "o root_line":
                in_root_line = True
                continue
            if in_root_line and line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
                if len(vertices) == root_count:
                    break
    if len(vertices) != root_count:
        raise ValueError(f"{path} 的 root_line 顶点数为 {len(vertices)}，预期为 {root_count}")
    return coordinate_frame, np.asarray(vertices, dtype=float)


def _symmetric_nearest_error(source_points: np.ndarray, exported_points: np.ndarray) -> float:
    """比较两个无序点集的双向最近邻误差，避免分支重排后的错误逐项配对。"""

    source_tree = cKDTree(source_points)
    exported_tree = cKDTree(exported_points)
    # 两个方向都检查，既能发现导出点偏离原始输入，也能发现原始点被遗漏。
    exported_to_source = float(np.max(source_tree.query(exported_points)[0]))
    source_to_exported = float(np.max(exported_tree.query(source_points)[0]))
    return max(exported_to_source, source_to_exported)


def audit_case(case_name: str, *, data_root: Path, results_root: Path, tolerance: float) -> dict[str, Any]:
    """审计一个案例最新运行目录的显示 OBJ 与原始根部交线。"""

    run_directory = _latest_run(results_root / case_name)
    summary_path = run_directory / "feasible_centers_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    root_path = data_root / case_name / "root_line.ply"
    source_root = np.asarray(load_root_line(root_path).vertices, dtype=float)
    display_path = run_directory / "feasible_centers_display.obj"
    frame, exported_root = _read_obj_header_and_root_vertices(display_path, len(source_root))

    # 算法会按 edge 重排根部线；双向最近邻检查顶点集合而不假设两者顶点顺序相同。
    coordinate_error = _symmetric_nearest_error(source_root, exported_root)
    frame_ok = frame == "source_obj"
    coordinates_ok = coordinate_error <= tolerance
    return {
        "case_name": case_name,
        "run_directory": str(run_directory),
        "display_obj": str(display_path),
        "coordinate_frame": frame,
        "root_vertex_count": int(len(source_root)),
        "max_root_coordinate_error": coordinate_error,
        "tolerance": tolerance,
        "status": "pass" if frame_ok and coordinates_ok else "fail",
        "reason": (
            "显示 OBJ 的根部线已在 source_obj 坐标中，与原始 root_line.ply 一致。"
            if frame_ok and coordinates_ok
            else "OBJ 未标为 source_obj 坐标，或其根部线与原始 root_line.ply 不一致。"
        ),
    }


def main() -> int:
    """执行全部目标案例的审计，并在发生坐标错误时返回非零退出码。"""

    arguments = _parse_arguments()
    config_path = Path(arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.parent.parent
    data_root = (config_path.parent / config["data_root"]).resolve()
    # 配置中的 directory 已经是 results/cases；不再额外拼接一次 cases。
    results_root = repo_root / config["outputs"]["directory"]
    cases = arguments.cases or sorted(path.name for path in results_root.iterdir() if path.is_dir())

    records = [
        audit_case(case_name, data_root=data_root, results_root=results_root, tolerance=arguments.tolerance)
        for case_name in cases
    ]
    report = {
        "purpose": "Verify source-coordinate alignment for MeshLab OBJ overlays; not a collision-validation certificate.",
        "case_count": len(records),
        "passed_case_count": sum(record["status"] == "pass" for record in records),
        "failed_case_count": sum(record["status"] != "pass" for record in records),
        "cases": records,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)

    if arguments.output:
        output_path = Path(arguments.output).resolve()
        if output_path.exists():
            raise FileExistsError(f"为避免覆盖已有审计记录，输出文件必须不存在: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["failed_case_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
