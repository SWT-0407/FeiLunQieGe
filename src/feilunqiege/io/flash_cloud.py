"""读取带 body/flash 标签的 OBJ 点云，用于确定根部线外侧方向。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LabeledFlashCloud:
    """保留 source_obj 坐标中的铸件本体点和飞边点。"""

    body_points: np.ndarray
    flash_points: np.ndarray
    source_path: Path


def load_labeled_flash_cloud(path: str | Path) -> LabeledFlashCloud:
    """按 OBJ 的 ``g casting_body_label_0_blue``/``g flash_label_1_red`` 分组读取点云。

    文件中的 ``p`` 原语不参与几何计算；顶点所属分组决定其标签。这样不会把
    点云误读为三角网格，也不会依赖 OBJ 顶点颜色扩展语法。
    """

    source = Path(path)
    body: list[list[float]] = []
    flash: list[list[float]] = []
    current_label: int | None = None
    with source.open("r", encoding="ascii", errors="replace") as handle:
        for raw_line in handle:
            fields = raw_line.strip().split()
            if not fields:
                continue
            if fields[0] == "g":
                group = " ".join(fields[1:]).lower()
                if "casting_body_label_0" in group:
                    current_label = 0
                elif "flash_label_1" in group:
                    current_label = 1
            elif fields[0] == "v" and len(fields) >= 4:
                point = [float(fields[1]), float(fields[2]), float(fields[3])]
                if current_label == 0:
                    body.append(point)
                elif current_label == 1:
                    flash.append(point)
    if not body or not flash:
        raise ValueError(f"labeled flash cloud must contain both body and flash points: {source}")
    return LabeledFlashCloud(
        body_points=np.asarray(body, dtype=float),
        flash_points=np.asarray(flash, dtype=float),
        source_path=source,
    )
