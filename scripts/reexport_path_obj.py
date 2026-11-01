"""从已有路径规划结果重新导出 `circle_paths.obj`（仅可视化）。

用途：当只修改 `path_obj.py` 的显示方式（例如把 `l` 线换成加粗的
圆柱管、放大节点标记）时，不必重新运行 415 点碰撞筛选。脚本读取同一
运行目录的 `path_planning_summary.json` 和 `path_candidates.npz`，
重建 `write_path_obj` 所需的输入，并写出新的 OBJ/MTL。

注意：
- 不改变路径节点、角度、覆盖区间或 JSON；
- 默认拒绝覆盖已有 OBJ；确认是可视化重导出时显式传 `--overwrite`；
- 输出仍是 `source_obj` 坐标。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from feilunqiege.planning.path_planning import branch_runs  # noqa: E402
from feilunqiege.visualization.path_obj import write_path_obj  # noqa: E402


def _build_side_paths(records: list[dict]) -> list[SimpleNamespace]:
    """把 JSON 记录转换为 `write_path_obj` 需要的简单对象。"""

    side_paths: list[SimpleNamespace] = []
    for record in records:
        nodes = [
            SimpleNamespace(
                point_index=int(node["point_index"]),
                candidate_index=int(node["candidate_index"]),
                angle_deg=float(node["angle_deg"]),
                center_norm=np.asarray(node["center_norm"], dtype=float),
            )
            for node in record.get("nodes", [])
        ]
        side_paths.append(
            SimpleNamespace(
                branch_id=int(record["branch_id"]),
                branch_label=str(record["branch_label"]),
                lateral_side=int(record["lateral_side"]),
                status=str(record["status"]),
                nodes=nodes,
                length_norm=record.get("length_norm"),
                skipped_point_indices=tuple(int(value) for value in record.get("skipped_point_indices", [])),
                missing_point_indices=tuple(int(value) for value in record.get("missing_point_indices", [])),
            )
        )
    return side_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="从已有 JSON/NPZ 重新导出路径 OBJ（仅可视化）")
    parser.add_argument("--run-dir", required=True, help="已有路径规划运行目录")
    parser.add_argument("--radius", type=float, default=0.02, help="用于显示标记/管径的砂轮半径（normalized）")
    parser.add_argument("--output", default=None, help="输出 OBJ 路径；默认 <run-dir>/circle_paths.obj")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有 OBJ")
    arguments = parser.parse_args()

    run_dir = Path(arguments.run_dir).resolve()
    summary = json.loads((run_dir / "path_planning_summary.json").read_text(encoding="utf-8"))
    data = np.load(run_dir / "path_candidates.npz")
    sampled = SimpleNamespace(
        points=np.asarray(data["point_norm"], dtype=float),
        branch_ids=np.asarray(data["branch_id"], dtype=np.int64),
    )
    branches = branch_runs(sampled.branch_ids)
    normalization = {
        "scale": float(np.asarray(data["normalization_scale"])),
        "center": np.asarray(data["normalization_center"], dtype=float).tolist(),
    }
    side_paths = _build_side_paths(summary["branch_paths"])

    output = Path(arguments.output).resolve() if arguments.output else run_dir / "circle_paths.obj"
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"输出已存在；如确认是可视化重导出，请传 --overwrite: {output}")
    write_path_obj(output, sampled, branches, normalization, side_paths, radius=float(arguments.radius))
    print(f"output={output}")
    print(f"paths={len(side_paths)} ok={sum(1 for item in side_paths if item.status == 'ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
