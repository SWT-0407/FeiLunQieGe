"""重写已有 P1 全量交付中的逐点 patch OBJ（不重跑碰撞）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from feilunqiege.geometry.curve import split_and_resample  # noqa: E402
from feilunqiege.geometry.normal_p1 import estimate_normals_experimental, prepare_mesh  # noqa: E402
from feilunqiege.io.mesh import load_mesh  # noqa: E402
from feilunqiege.io.root_line import load_root_line  # noqa: E402
from feilunqiege.pipeline import _normalize_case_geometry  # noqa: E402
from export_p1_full_delivery import _write_point_patch_obj  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="重写 P1 逐点 patch OBJ")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--case", default="circle")
    parser.add_argument("--delivery-dir", required=True)
    arguments = parser.parse_args()

    config_path = (REPO / arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    delivery = Path(arguments.delivery_dir).resolve()
    summary = json.loads((delivery / "p1_full_summary.json").read_text(encoding="utf-8"))
    case_root = (config_path.parent / config["data_root"]).resolve() / arguments.case
    root_line = load_root_line(case_root / config["inputs"]["root_line"])
    workpiece = load_mesh(case_root / "mesh_without_flash.obj")
    root_line, workpiece, normalization = _normalize_case_geometry(root_line, workpiece, config)
    sampled = split_and_resample(root_line, int(config["sampling"]["curve_points"]))
    normals, diagnostics = estimate_normals_experimental(
        sampled.points,
        workpiece,
        method="p1",
        config=dict(config.get("normal_estimation", {}).get("p1", {})),
        return_diagnostics=True,
    )
    mesh_data = prepare_mesh(workpiece)
    for point_index in summary["infeasible_points"]:
        _write_point_patch_obj(
            delivery / f"{point_index}.obj",
            int(point_index),
            diagnostics[int(point_index)],
            mesh_data,
            sampled,
            normalization,
        )
    print(f"rewrote {len(summary['infeasible_points'])} point OBJ files in {delivery}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
