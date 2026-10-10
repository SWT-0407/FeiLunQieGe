"""按统一参数顺序运行 12 个工件案例。

该脚本复用单案例 ``run_from_config``，每个案例单独创建时间戳结果目录。
脚本不复制原始 OBJ/PLY，也不会删除或覆盖任何已有运行结果。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path


# 将仓库内的 src 加入模块搜索路径，使脚本可以直接从仓库根目录运行。
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from feilunqiege.pipeline import run_from_config


# 12 个案例名称与外部 flash_pipeline_12_models_20260924 目录保持一致。
DEFAULT_CASES = (
    "1041432",
    "226633",
    "252632",
    "439142",
    "5head",
    "5pipe",
    "719790",
    "804299",
    "804301",
    "circle",
    "gear",
    "leaf",
)


def build_parser() -> argparse.ArgumentParser:
    """构造批处理命令行参数。"""

    parser = argparse.ArgumentParser(description="批量生成 12 个工件的砂轮可行中心结果")
    parser.add_argument(
        "--config",
        default="configs/default.json",
        help="基础配置文件路径；半径、delta、采样数等参数由此文件统一提供",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(DEFAULT_CASES),
        help="要运行的案例名；省略时运行全部 12 个案例",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某个案例失败后继续运行其余案例，并在最后返回失败状态",
    )
    return parser


def main() -> int:
    """按顺序运行案例并打印每个案例的输出目录和关键计数。"""

    arguments = build_parser().parse_args()
    config_path = (REPO_ROOT / arguments.config).resolve()
    completed: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []

    # 顺序执行可以让每个大网格在单次运行结束后释放，降低批量处理的内存峰值。
    for case_name in arguments.cases:
        print(f"starting case={case_name}", flush=True)
        try:
            summary = run_from_config(config_path, case_name=case_name)
        except Exception as exc:  # noqa: BLE001 - 批处理需记录单案例异常并决定是否继续。
            failed.append({"case_name": case_name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"failed case={case_name} error={type(exc).__name__}: {exc}", flush=True)
            if not arguments.continue_on_error:
                return 1
            continue

        # 只打印审计摘要中的关键字段；完整参数和每点统计已写入案例 JSON。
        per_point = summary["per_point"]
        result = {
            "case_name": summary["case_name"],
            "status": summary["status"],
            "root_line_point_count": summary["parameters"]["curve_points"],
            "selected_radius": summary["parameters"]["radius"],
            "selected_delta": summary["parameters"]["delta"],
            "source_coordinate_equivalent_radius": summary["parameters"]["source_coordinate_equivalent_radius"],
            "radius_selection_reason": summary["parameters"]["radius_selection"]["selection_reason"],
            "candidate_count": summary["candidate_count"],
            "feasible_candidate_count": summary["feasible_candidate_count"],
            "feasible_ratio": summary["feasible_ratio"],
            "points_with_feasible_candidate": sum(
                1 for item in per_point if item["feasible_candidate_count"] > 0
            ),
            "points_without_feasible_candidate": sum(
                1 for item in per_point if item["feasible_candidate_count"] == 0
            ),
            "output_directory": summary["output_directory"],
            "feasible_centers_obj": summary["feasible_centers_obj"],
        }
        completed.append(result)
        print(
            f"completed case={case_name} "
            f"candidates={result['candidate_count']} "
            f"feasible={result['feasible_candidate_count']} "
            f"output={result['output_directory']}",
            flush=True,
        )

    # 批次清单集中记录 12 个案例的参数和结果路径，便于一次性验收。
    batch_output = _write_batch_summary(config_path, completed, failed)
    print(f"batch_summary={batch_output}", flush=True)
    # 用机器可读的退出码区分“全部成功”和“部分失败”。
    print(f"batch_completed={len(completed)} batch_failed={len(failed)}", flush=True)
    return 1 if failed else 0


def _write_batch_summary(
    config_path: Path,
    completed: list[dict[str, object]],
    failed: list[dict[str, str]],
) -> Path:
    """写入批次级 JSON/CSV；每次建立新目录，不覆盖以往批次。"""

    batch_root = REPO_ROOT / "results" / "batches"
    batch_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    output = batch_root / stamp
    suffix = 1
    while output.exists():
        output = batch_root / f"{stamp}_{suffix:02d}"
        suffix += 1
    output.mkdir(parents=True)

    # JSON 保留完整字段和失败原因，适合后续程序化审计。
    payload = {
        "status": "completed" if not failed else "completed_with_failures",
        "config": str(config_path.relative_to(REPO_ROOT)),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "completed": completed,
        "failed": failed,
        "limitations": [
            "结果是无厚度砂轮和近似表面采样碰撞筛选的 preliminary geometry run。",
            "可行中心不等于真实设备运动学、有限厚度和加工工艺均可行。",
        ],
    }
    (output / "batch_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # CSV 只保留一行一个案例的扁平字段，便于 Excel 直接打开和横向比较。
    fieldnames = [
        "case_name",
        "status",
        "root_line_point_count",
        "selected_radius",
        "selected_delta",
        "source_coordinate_equivalent_radius",
        "candidate_count",
        "feasible_candidate_count",
        "feasible_ratio",
        "points_with_feasible_candidate",
        "points_without_feasible_candidate",
        "radius_selection_reason",
        "output_directory",
        "feasible_centers_obj",
    ]
    with (output / "batch_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(completed)
    return output.relative_to(REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
