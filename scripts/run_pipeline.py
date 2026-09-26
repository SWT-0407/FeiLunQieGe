"""后续实验命令行入口占位。

当前不执行任何算法。正式实现后应从命令行接收配置路径，并将参数快照
和输出位置写入正式结果目录，而不是写入缓存目录。
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

    arguments = build_parser().parse_args()
    summary = run_from_config(REPO_ROOT / arguments.config)
    print(f"status=completed case={summary['case_name']}")
    print(f"candidate_count={summary['candidate_count']}")
    print(f"feasible_candidate_count={summary['feasible_candidate_count']}")
    print(f"overview_image={summary['overview_image']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
