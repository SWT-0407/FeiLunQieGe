"""最小几何闭环的命令行入口。

脚本只负责解析配置并调用 ``pipeline``；几何计算和文件输出均由源代码包
完成，便于后续测试和复用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from feilunqiege.pipeline import run_from_config


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="砂轮切除飞边可行区域实验入口")
    parser.add_argument("--config", default="configs/default.json", help="实验配置文件路径")
    return parser


def main() -> int:
    """执行最小几何闭环。"""

    # 解析用户指定的配置路径；默认配置使用 leaf 小案例和 normalized units。
    arguments = build_parser().parse_args()
    summary = run_from_config(REPO_ROOT / arguments.config)

    # 控制台只打印最重要的计数，完整参数和输出文件写在 JSON 摘要中。
    print(f"status=completed case={summary['case_name']}")
    print(f"candidate_count={summary['candidate_count']}")
    print(f"feasible_candidate_count={summary['feasible_candidate_count']}")
    print(f"output_directory={summary['output_directory']}")
    print(f"overview_image={summary['overview_image']}")
    print(f"local_explanation_image={summary['local_explanation_image']}")
    print(f"orientation_guide_image={summary['orientation_guide_image']}")
    print(f"feasible_centers_obj={summary['feasible_centers_obj']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
