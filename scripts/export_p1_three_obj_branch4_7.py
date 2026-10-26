"""导出 P1 方法的三个 OBJ（branch 4/7 代表性点版本）。

三个文件：
1. circle_rootline_circle_centers.obj：根线 + 圆心；
2. circle_feasible_center_domains.obj：可行/不可行候选中心域；
3. circle_inspection_wheels.obj：代表性点的砂轮实体可视化。

全量 415 点 p1 碰撞很慢；本脚本先覆盖 branch 4/7 的 20 个关键点。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from feilunqiege.collision.cylinder_exact import (  # noqa: E402
    build_cylinder_triangle_index,
    check_finite_cylinder_candidates_exact,
)
from feilunqiege.geometry.curve import split_and_resample  # noqa: E402
from feilunqiege.geometry.frame import build_frames  # noqa: E402
from feilunqiege.geometry.normal_p1 import estimate_normals_experimental  # noqa: E402
from feilunqiege.io.mesh import load_mesh  # noqa: E402
from feilunqiege.io.root_line import load_root_line  # noqa: E402
from feilunqiege.pipeline import _normalize_case_geometry  # noqa: E402
from run_leaf_stage3_visualization import (  # noqa: E402
    PointAudit,
    _branch_runs,
    _candidate_set,
    _export_center_domains,
    _export_circle_centers,
    _export_inspection_wheels,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 branch4/7 三 OBJ 导出")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--case", default="circle")
    parser.add_argument(
        "--points",
        default=",".join(str(index) for index in list(range(365, 375)) + list(range(395, 405))),
    )
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args()

    config_path = (REPO / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    case_root = (config_path.parent / config["data_root"]).resolve() / arguments.case
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / "mesh_without_flash.obj")
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    points = [int(value) for value in arguments.points.split(",") if value.strip()]
    if any(index < 0 or index >= len(sampled.points) for index in points):
        raise ValueError("points 越界")

    normal_config = dict(config.get("normal_estimation", {}))
    p1_config = dict(normal_config.get("p1", {}))
    normals, _ = estimate_normals_experimental(
        sampled.points,
        workpiece,
        method="p1",
        config={**p1_config, "orientation_repair": True},
        return_diagnostics=True,
    )
    frames = build_frames(
        sampled.points,
        sampled.tangents,
        normals,
        normal_is_reliable=True,
    )
    radius = float(config["wheel"]["radius"])
    width = float(config["wheel"]["width"])
    angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    collision_config = dict(config.get("collision", {}))
    patch = dict(collision_config)
    contact_patch_radius = float(collision_config.get("contact_patch_radius_ratio", 2.0)) * radius
    max_contact_penetration = float(collision_config.get("contact_patch_max_penetration_ratio", 0.025)) * radius
    cylinder_index = build_cylinder_triangle_index(workpiece)
    audits: list[PointAudit | None] = [None] * len(sampled.points)
    records = []
    for point_index in points:
        candidates = _candidate_set(
            np.asarray(sampled.points[point_index]),
            frames[point_index],
            radius,
            width,
            angles,
            point_index,
            1.0,
        )
        result = check_finite_cylinder_candidates_exact(
            candidates,
            workpiece,
            width,
            inner_radius=0.0,
            clearance=float(collision_config.get("clearance", 0.0)),
            allow_target_contact=bool(collision_config.get("allow_target_contact", True)),
            target_contact_tolerance=float(collision_config.get("target_contact_tolerance", 1e-4)),
            contact_patch_enabled=bool(collision_config.get("contact_patch_enabled", True)),
            contact_patch_radius=contact_patch_radius,
            max_contact_penetration=max_contact_penetration,
            contact_normal_tolerance_degrees=float(collision_config.get("contact_patch_normal_tolerance_degrees", 30.0)),
            narrow_phase=str(collision_config.get("narrow_phase", "analytic_closest_point_flat")),
            cylinder_index=cylinder_index,
        )
        audits[point_index] = PointAudit(point_index, candidates, result)
        reasons = Counter(str(reason) for reason in np.asarray(result.reason, dtype=object))
        records.append(
            {
                "point_index": point_index,
                "branch_id": int(sampled.branch_ids[point_index]),
                "feasible_angle_count": int(np.count_nonzero(result.feasible)),
                "reason_counts": dict(sorted(reasons.items())),
            }
        )
    branches = _branch_runs(np.asarray(sampled.branch_ids, dtype=np.int64))
    output_dir = (
        Path(arguments.output_dir).resolve()
        if arguments.output_dir is not None
        else REPO / "results" / "cases" / arguments.case / datetime.now().strftime("p1_circle_three_obj_branch4_7_%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    centers_obj = output_dir / f"{arguments.case}_rootline_circle_centers.obj"
    domains_obj = output_dir / f"{arguments.case}_feasible_center_domains.obj"
    wheels_obj = output_dir / f"{arguments.case}_inspection_wheels.obj"
    _export_circle_centers(centers_obj, sampled, [audit for audit in audits if audit is not None], branches, normalization, radius)
    _export_center_domains(domains_obj, sampled, [audit for audit in audits if audit is not None], branches, normalization, radius)
    wheel_records = _export_inspection_wheels(
        wheels_obj,
        sampled,
        audits,
        branches,
        normalization,
        radius,
        width,
        int(config.get("visualization", {}).get("cylinder_segments", 64)),
        points,
        per_point=2,
    )
    metadata = {
        "case": arguments.case,
        "normal_method": "p1",
        "scope": "branch 4 (365–374) + branch 7 (395–404) representative points",
        "coordinate_frames": {
            "computation": "normalized",
            "obj": "source_obj",
        },
        "parameters": {
            "radius": radius,
            "width": width,
            "angle_count": 72,
            "delta": float(config["wheel"].get("delta", 0.0)),
            "contact_patch_radius": contact_patch_radius,
            "max_contact_penetration": max_contact_penetration,
            "p1": p1_config,
        },
        "points": records,
        "wheel_records": wheel_records,
        "outputs": {
            "centers_obj": str(centers_obj),
            "domains_obj": str(domains_obj),
            "wheels_obj": str(wheels_obj),
        },
        "note": "P1 three-OBJ representative visualization; not a full 415-point batch and not a process acceptance.",
    }
    (output_dir / "p1_three_obj_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        f"""# P1 三 OBJ 可视化（branch 4/7 代表性点）

- 方法：`p1`（与 `zhong+cazals` 当前同路径）
- 工件：`circle/mesh_without_flash.obj`（原始 full mesh）
- 范围：branch 4 的 365–374 + branch 7 的 395–404，共 {len(points)} 点
- 角度：72；半径：{radius}；宽度：{width}；单位：normalized（OBJ 为 source_obj）
- 文件：
  - `{centers_obj.name}`：根线 + 圆心；
  - `{domains_obj.name}`：可行/不可行候选中心域；
  - `{wheels_obj.name}`：每个检查点最多一个红色不可行、一个绿色可行砂轮实体；
  - `p1_three_obj_summary.json`：逐点可行角数和碰撞原因。

**说明**：全量 415 点 p1 碰撞成本高，本文件夹先给 branch 4/7 关键点的三 OBJ；不是全量批处理，也不是加工验收结论。
""",
        encoding="utf-8",
    )
    print(f"output_dir={output_dir}")
    print(f"centers={centers_obj}")
    print(f"domains={domains_obj}")
    print(f"wheels={wheels_obj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
