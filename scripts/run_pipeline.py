"""后续实验命令行入口占位。

当前不执行任何算法。正式实现后应从命令行接收配置路径，并将参数快照
和输出位置写入正式结果目录，而不是写入缓存目录。
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="砂轮切除飞边可行区域实验入口")
    parser.add_argument("--config", default="configs/default.json", help="实验配置文件路径")
    return parser


def main() -> int:
    """预留程序入口；算法尚未实现。"""

    _ = build_parser().parse_args()
    raise NotImplementedError("TODO: 算法实现完成后再启用实验入口")


if __name__ == "__main__":
    raise SystemExit(main())
