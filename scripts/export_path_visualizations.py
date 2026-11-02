"""从已有路径规划结果生成四个可视化 OBJ。

输出：
1. `circle_feasible_center_domains.obj`：从源运行目录原样复制，不重算；
2. `circle_paths_dots.obj`：只输出选中路径点，亮黄色标记；
3. `circle_paths_lines.obj`：root_line + 选中点连线（l + tube），保留蓝/橙分侧；
4. `circle_path_wheels.obj`：每个 branch 只取一侧（优先 +q，若 +q no_path 则用 -q），
   在每个选中点画一个有限圆柱砂轮。

本脚本只做显示，不修改路径节点、角度、覆盖区间或 JSON。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from feilunqiege.planning.path_planning import branch_runs  # noqa: E402
from feilunqiege.visualization.path_obj import (  # noqa: E402
    write_path_dots_obj,
    write_path_obj,
    write_path_wheels_obj,
)


def _build_side_paths(records: list[dict]) -> list[SimpleNamespace]:
    side_paths: list[SimpleNamespace] = []
    for record in records:
        nodes = [
            SimpleNamespace(
                point_index=int(node["point_index"]),
                candidate_index=int(node["candidate_index"]),
                angle_deg=float(node["angle_deg"]),
                center_norm=np.asarray(node["center_norm"], dtype=float),
                axis_norm=(
                    None
                    if node.get("axis_norm") is None
                    else np.asarray(node["axis_norm"], dtype=float)
                ),
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


def _select_one_side_per_branch(side_paths: list[SimpleNamespace]) -> tuple[list[SimpleNamespace], list[str]]:
    """每个 branch 优先 +q；+q no_path 时回退 -q。返回选中路径和说明行。"""

    by_branch: dict[int, list[SimpleNamespace]] = {}
    for item in side_paths:
        by_branch.setdefault(int(item.branch_id), []).append(item)
    selected: list[SimpleNamespace] = []
    notes: list[str] = []
    for branch_id in sorted(by_branch):
        items = by_branch[branch_id]
        positive = next((item for item in items if item.lateral_side > 0), None)
        negative = next((item for item in items if item.lateral_side < 0), None)
        chosen = positive if positive is not None and positive.status == "ok" else None
        source = "+q"
        if chosen is None and negative is not None and negative.status == "ok":
            chosen = negative
            source = "-q (fallback because +q has no path)"
        if chosen is None:
            notes.append(f"branch_{branch_id:02d}: no wheel output (both sides no_path)")
            continue
        chosen._wheel_side_source = source  # type: ignore[attr-defined]
        selected.append(chosen)
        notes.append(
            f"branch_{branch_id:02d}: {source}, {len(chosen.nodes)} wheel(s)"
        )
    return selected, notes


def _build_wheels(selected: list[SimpleNamespace], data: np.lib.npyio.NpzFile) -> list[SimpleNamespace]:
    wheels: list[SimpleNamespace] = []
    for item in selected:
        for node in item.nodes:
            point_index = int(node.point_index)
            candidate_index = int(node.candidate_index)
            axis = data["axis"][point_index, candidate_index]
            radial = data["radial"][point_index, candidate_index]
            wheels.append(
                SimpleNamespace(
                    branch_label=item.branch_label,
                    lateral_side=int(item.lateral_side),
                    point_index=point_index,
                    candidate_index=candidate_index,
                    center_norm=np.asarray(node.center_norm, dtype=float),
                    axis_norm=np.asarray(axis, dtype=float),
                    radial_norm=np.asarray(radial, dtype=float),
                )
            )
    return wheels


def main() -> int:
    parser = argparse.ArgumentParser(description="生成四个路径可视化 OBJ（仅显示）")
    parser.add_argument("--run-dir", required=True, help="已有路径规划运行目录")
    parser.add_argument("--output-dir", default=None, help="输出目录；默认在 run-dir 同级新建时间戳目录")
    parser.add_argument("--radius", type=float, default=None, help="砂轮半径；默认读 path_planning_summary.json 参数")
    parser.add_argument("--width", type=float, default=None, help="砂轮宽度；默认读 path_planning_summary.json 参数")
    parser.add_argument("--segments", type=int, default=48, help="显示圆柱的圆周分段数")
    arguments = parser.parse_args()

    run_dir = Path(arguments.run_dir).resolve()
    summary = json.loads((run_dir / "path_planning_summary.json").read_text(encoding="utf-8"))
    data = np.load(run_dir / "path_candidates.npz")
    parameters = summary.get("parameters", {})
    radius = float(arguments.radius if arguments.radius is not None else parameters.get("radius", 0.02))
    width = float(arguments.width if arguments.width is not None else parameters.get("width", 0.004))

    output_dir = (
        Path(arguments.output_dir).resolve()
        if arguments.output_dir is not None
        else run_dir.parent / datetime.now().strftime("path_viz_4obj_%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

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
    selected, wheel_notes = _select_one_side_per_branch(side_paths)
    wheels = _build_wheels(selected, data)

    # ① 可行中心域：原样复制，不重算、不改内容。
    for suffix in (".obj", ".mtl"):
        source = run_dir / f"circle_feasible_center_domains{suffix}"
        if not source.exists():
            raise FileNotFoundError(f"缺少可行中心域文件: {source}")
        shutil.copy2(source, output_dir / source.name)

    # ② 只输出选中点，亮黄色。
    dots_obj = output_dir / "circle_paths_dots.obj"
    write_path_dots_obj(dots_obj, side_paths, normalization, radius=radius)

    # ③ 只保留连线/tube 和 root_line，不重复输出点标记。
    lines_obj = output_dir / "circle_paths_lines.obj"
    write_path_obj(
        lines_obj,
        sampled,
        branches,
        normalization,
        side_paths,
        radius=radius,
        include_nodes=False,
    )

    # ④ 每个 branch 只输出一侧的砂轮圆柱。
    wheels_obj = output_dir / "circle_path_wheels.obj"
    write_path_wheels_obj(
        wheels_obj,
        sampled,
        branches,
        normalization,
        wheels,
        radius=radius,
        width=width,
        segments=int(arguments.segments),
    )

    readme = output_dir / "README.md"
    readme.write_text(
        f"""# 路径可视化四 OBJ（2026-11-02）

来源运行目录：

```text
{run_dir}
```

文件：

| 文件 | 内容 |
|---|---|
| `circle_feasible_center_domains.obj` | 从来源目录原样复制；绿色可行/红色碰撞中心域，未重算 |
| `circle_paths_dots.obj` | 只输出选中路径点；亮黄色 `path_dot_yellow` |
| `circle_paths_lines.obj` | root_line + 选中点连线（`l` + tube）；`+q` 蓝、`-q` 橙；不重复点标记 |
| `circle_path_wheels.obj` | 每个 branch 只取一侧的选中点，画有限圆柱砂轮；`wheel_teal` |

参数：

- 坐标：`source_obj`；
- 砂轮半径：`{radius}` normalized（source 等效 `{radius * float(normalization['scale']):.9g}`）；
- 砂轮宽度：`{width}` normalized（source 等效 `{width * float(normalization['scale']):.9g}`）；
- 圆柱分段：`{arguments.segments}`；
- 路径数据未改；点/线/砂轮都是显示几何。

砂轮侧向选择（每个 branch 优先 `+q`，若 `+q` 无解则用 `-q`）：

{chr(10).join('- ' + note for note in wheel_notes)}

限制：

- 未做边级扫掠碰撞审计；
- 砂轮可视化只表示选中中心处的有限圆柱姿态，不是加工仿真；
- 同一 branch 只输出一侧砂轮，另一侧不输出（按要求）。
""",
        encoding="utf-8",
    )

    print(f"output_dir={output_dir}")
    print(f"dots={dots_obj}")
    print(f"lines={lines_obj}")
    print(f"wheels={wheels_obj} wheel_count={len(wheels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
