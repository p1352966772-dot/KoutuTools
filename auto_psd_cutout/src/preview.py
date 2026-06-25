from __future__ import annotations

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
    """绘制检测结果预览图（含行列框线和列间隙标注）。"""
    preview_config = config.get("preview", {})
    draw_box = bool(preview_config.get("draw_box", True))
    draw_index = bool(preview_config.get("draw_index", True))
    draw_groups = bool(preview_config.get("draw_groups", True))
    draw_rows = bool(preview_config.get("draw_rows", True))
    thickness = int(preview_config.get("box_thickness", 2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = image_bgr.copy()
    h_img, w_img = canvas.shape[:2]

    boxes = detect_result.get("boxes", [])
    groups = detect_result.get("groups", [])

    # ── 1. 行背景色（半透明彩色条） ──
    if draw_rows and groups:
        overlay = canvas.copy()
        for gi, group in enumerate(groups):
            color = GROUP_COLORS[gi % len(GROUP_COLORS)]
            ry = group["row_y"]
            rh = group["row_h"]
            cv2.rectangle(overlay, (0, ry), (w_img - 1, ry + rh), color, -1)
        cv2.addWeighted(overlay, 0.08, canvas, 0.92, 0, canvas)

    # ── 2. 行内列间隙竖线 ──
    # 在每个行区域内，画出垂直的列分隔线（半透明）
    if draw_groups and groups:
        col_overlay = canvas.copy()
        for gi, group in enumerate(groups):
            gid = group["id"]
            row_boxes = sorted(
                [b for b in boxes if b.get("group_id") == gid],
                key=lambda b: b["x1"],
            )
            # 相邻 box 之间的间隙中点画竖线
            for i in range(len(row_boxes) - 1):
                gap = row_boxes[i + 1]["x1"] - row_boxes[i]["x2"]
                if gap > 3:  # 有明显间隙才画
                    cx = (row_boxes[i]["x2"] + row_boxes[i + 1]["x1"]) // 2
                    ry = group["row_y"]
                    rh = group["row_h"]
                    color = GROUP_COLORS[gi % len(GROUP_COLORS)]
                    cv2.line(col_overlay, (cx, ry), (cx, ry + rh), color, 1)
        cv2.addWeighted(col_overlay, 0.3, canvas, 0.7, 0, canvas)

    # ── 3. 元素框 + 编号 ──
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
            label = f"{box['id']}"
            # 如果存在行列号则显示 "row-col"
            col_num = box.get("col")
            if col_num:
                label = f"{box.get('group_id', '?')}-{col_num}"
            label_x = max(0, x1)
            label_y = max(20, y1 - 6)
            cv2.putText(canvas, label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ── 4. 行标签 ──
    if draw_rows and groups:
        for gi, group in enumerate(groups):
            color = GROUP_COLORS[gi % len(GROUP_COLORS)]
            label = f"{group['name']}  ({group['count']})"
            ly = max(20, group["row_y"] + 4)
            cv2.putText(canvas, label, (8, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 覆盖已有预览文件
    if output_path.exists():
        output_path.unlink()
    ok = cv2.imwrite(str(output_path), canvas)
    if not ok:
        raise RuntimeError(f"预览图保存失败：{output_path}")
    return output_path
