import sys, os
sys.stdout.reconfigure(encoding="utf-8")

content = """from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


GROUP_COLORS = [
    (255, 0, 0),    # red
    (0, 255, 0),    # green
    (0, 0, 255),    # blue
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 0, 0),    # dark red
    (0, 128, 0),    # dark green
    (0, 0, 128),    # dark blue
    (128, 128, 0),  # olive
]


def save_preview(image_bgr: np.ndarray, detect_result: dict[str, Any], output_path: Path, config: dict[str, Any]) -> Path:
    preview_config = config.get("preview", {})
    draw_box = bool(preview_config.get("draw_box", True))
    draw_index = bool(preview_config.get("draw_index", True))
    draw_groups = bool(preview_config.get("draw_groups", True))
    draw_rows = bool(preview_config.get("draw_rows", True))
    thickness = int(preview_config.get("box_thickness", 2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = image_bgr.copy()

    boxes = detect_result.get("boxes", [])
    groups = detect_result.get("groups", [])

    # Draw row group backgrounds
    if draw_rows and groups:
        overlay = canvas.copy()
        for gi, group in enumerate(groups):
            color = GROUP_COLORS[gi % len(GROUP_COLORS)]
            ry = group["row_y"]
            rh = group["row_h"]
            cv2.rectangle(overlay, (0, ry), (canvas.shape[1] - 1, ry + rh), color, -1)
        cv2.addWeighted(overlay, 0.08, canvas, 0.92, 0, canvas)

    # Draw boxes with group colors
    color_map: dict[int, tuple] = {}
    for gi, group in enumerate(groups):
        color_map[group["id"]] = GROUP_COLORS[gi % len(GROUP_COLORS)]

    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        gid = box.get("group_id", 0)
        color = color_map.get(gid, (0, 0, 255))

        if draw_box:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        if draw_index:
            label = str(box["id"])
            label_x = max(0, x1)
            label_y = max(20, y1 - 6)
            cv2.putText(canvas, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw row labels
    if draw_rows and groups:
        for gi, group in enumerate(groups):
            color = GROUP_COLORS[gi % len(GROUP_COLORS)]
            label = group["name"]
            ly = max(20, group["row_y"] + 4)
            cv2.putText(canvas, label, (8, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    ok = cv2.imwrite(str(output_path), canvas)
    if not ok:
        raise RuntimeError(f"预览图保存失败：{output_path}")
    return output_path
"""

dst = os.path.expandvars(r"%USERPROFILE%\Desktop\KoutuTools\auto_psd_cutout\src\preview.py")
with open(dst, "w", encoding="utf-8") as f:
    f.write(content)
print("preview.py written OK")

import ast
with open(dst, "r", encoding="utf-8") as f:
    ast.parse(f.read())
print("Syntax OK")
