"""
grid_cutter.py — 行列分割算法
==============================

把一张包含多行多列小图的大图，按行列间隙切分成独立的小图。

核心思路（两遍扫描）：
  1. 行分割：从上到下扫描，找到连续多行为白色的间隙 → 切出行
  2. 列分割：在每个行区域内，从左到右扫描，找到连续多列为白色的间隙 → 切出小图

适用场景：
  - 密集排版图、产品排列图、表格截图、SKU 排列图
  - 白底或浅色背景，元素之间有清晰空白间隙

用法：
  from grid_cutter import grid_split, grid_split_and_save

  boxes = grid_split("input.jpg")
  # boxes = [(x, y, w, h), ...]

  grid_split_and_save("input.jpg", "output_dir")
  # 保存 crop_001.png, crop_002.png ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


# ============================================================
# 核心参数
# ============================================================

DEFAULT_CONFIG: dict[str, Any] = {
    # 白色判定 —— 灰度值 > white_threshold 即为"空白"
    "white_threshold": 240,
    # 行间隙：连续 min_gap_rows 行及以上为白色，才判定为行分割线
    "min_gap_rows": 3,
    # 列间隙：连续 min_gap_cols 列及以上为白色，才判定为列分割线
    "min_gap_cols": 3,
    # 最小行高：过滤掉低于此值的行区域（噪点/极细线）
    "min_row_height": 10,
    # 最小区域面积（像素）：过滤掉小于此面积的小图（防噪点）
    "min_crop_area": 200,
    # 边缘裁剪：去掉四周边沿的空白 (top, bottom, left, right)
    "crop_border": 0,
}


# ============================================================
# 核心函数
# ============================================================

def grid_split(
    image: str | Path | np.ndarray,
    config: dict[str, Any] | None = None,
) -> list[dict[str, int]]:
    """行列分割：返回每个小图的 bounding box。

    Parameters
    ----------
    image : str | Path | np.ndarray
        输入图片路径或 numpy 数组（BGR/HWC uint8）。
    config : dict | None
        配置参数，覆盖 DEFAULT_CONFIG 中的对应项。

    Returns
    -------
    list[dict[str, int]]
        每个元素：{"id": 序号, "row": 行号, "col": 列号,
                   "x": int, "y": int, "w": int, "h": int}
        按先行后列排序。
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    white_th = int(cfg["white_threshold"])
    min_gap_rows = int(cfg["min_gap_rows"])
    min_gap_cols = int(cfg["min_gap_cols"])
    min_row_h = int(cfg["min_row_height"])
    min_area = int(cfg["min_crop_area"])
    crop_border = int(cfg.get("crop_border", 0))

    # ── 读图 ──
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(f"无法读取图片: {image}")
    elif isinstance(image, np.ndarray):
        bgr = image
    else:
        raise TypeError("image 须为路径或 numpy 数组")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 边缘裁剪：去掉四周边沿空白
    if crop_border > 0:
        gray = gray[crop_border : h - crop_border, crop_border : w - crop_border]
        h, w = gray.shape
        top_offset = crop_border
        left_offset = crop_border
    else:
        top_offset = 0
        left_offset = 0

    # ════════════════════════════════════════════════════════
    # 第一步：行分割
    # ════════════════════════════════════════════════════════

    # 每行平均灰度值
    row_means = np.mean(gray, axis=1)
    row_is_white = row_means > white_th

    # 找"连续多行为白色"的段 → 行间隙
    row_gaps = _find_gap_ranges(row_is_white, min_gap_rows, h)

    # 用间隙切出行区域
    row_regions = _ranges_from_gaps(row_gaps, 0, h, min_row_h)

    if not row_regions:
        # 没找到行间隙 → 整张图就是一行
        row_regions = [(0, h)]

    # ════════════════════════════════════════════════════════
    # 第二步：在每个行区域内做列分割
    # ════════════════════════════════════════════════════════

    results: list[dict[str, int]] = []
    crop_id = 0

    for row_idx, (ry1, ry2) in enumerate(row_regions):
        # 取行区域的灰度切片
        row_slice = gray[ry1:ry2, :]

        # 每列平均灰度值
        col_means = np.mean(row_slice, axis=0)
        col_is_white = col_means > white_th

        # 找"连续多列为白色"的段 → 列间隙
        col_gaps = _find_gap_ranges(col_is_white, min_gap_cols, w)

        # 用间隙切出列区域
        col_regions = _ranges_from_gaps(col_gaps, 0, w, 1)

        if not col_regions:
            # 没找到列间隙 → 整行就是一个元素
            col_regions = [(0, w)]

        for col_idx, (cx1, cx2) in enumerate(col_regions):
            crop_w = cx2 - cx1
            crop_h = ry2 - ry1
            if crop_w * crop_h < min_area:
                continue

            crop_id += 1
            results.append({
                "id": crop_id,
                "row": row_idx + 1,
                "col": col_idx + 1,
                "x": cx1 + left_offset,
                "y": ry1 + top_offset,
                "w": crop_w,
                "h": crop_h,
            })

    return results


def grid_split_and_save(
    image: str | Path | np.ndarray,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    prefix: str = "crop",
    fmt: str = "png",
) -> list[dict[str, int]]:
    """行列分割 + 直接裁剪保存为独立图片。

    Parameters
    ----------
    image : str | Path | np.ndarray
        输入图片路径或 numpy 数组。
    output_dir : str | Path
        输出目录（不存在则自动创建）。
    config : dict | None
        配置参数。
    prefix : str
        输出文件前缀，默认 "crop"。
    fmt : str
        输出格式，默认 "png"。

    Returns
    -------
    list[dict[str, int]]
        每个元素含坐标和保存路径。
    """
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(f"无法读取图片: {image}")
    else:
        bgr = image

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boxes = grid_split(bgr, config)

    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        crop_bgr = bgr[y : y + h, x : x + w]
        filename = f"{prefix}_{box['id']:03d}.{fmt}"
        cv2.imwrite(str(out_dir / filename), crop_bgr)
        box["path"] = str(out_dir / filename)

    return boxes


def draw_grid(
    image: str | Path | np.ndarray,
    boxes: list[dict[str, int]],
    output_path: str | Path | None = None,
) -> np.ndarray:
    """在图上绘制行列分割框线，用于可视化验证。

    Parameters
    ----------
    image : str | Path | np.ndarray
        图片或路径。
    boxes : list[dict]
        grid_split 返回的结果。
    output_path : str | Path | None
        不为 None 则保存到文件。

    Returns
    -------
    np.ndarray
        带标注的 BGR 图。
    """
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
    else:
        bgr = image.copy()

    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        color = (0, 200, 0)  # BGR 绿
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
        label = f"{box['row']}-{box['col']}"
        cv2.putText(bgr, label, (x + 2, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if output_path:
        cv2.imwrite(str(output_path), bgr)

    return bgr


# ============================================================
# 内部工具函数
# ============================================================

def _find_gap_ranges(
    is_white: np.ndarray,
    min_gap: int,
    total_len: int,
) -> list[tuple[int, int]]:
    """找到连续 is_white==True 且长度 >= min_gap 的段。

    Returns
    -------
    list[(start, end)]
        gap 的起始/结束位置（左闭右开），与坐标轴方向一致。
        对行间隙：上下方向；对列间隙：左右方向。
    """
    if total_len == 0:
        return []

    gaps: list[tuple[int, int]] = []
    start: int | None = None

    for i in range(total_len):
        if is_white[i] and start is None:
            start = i
        elif not is_white[i] and start is not None:
            if i - start >= min_gap:
                gaps.append((start, i))
            start = None

    # 最后一段
    if start is not None and total_len - start >= min_gap:
        gaps.append((start, total_len))

    return gaps


def _ranges_from_gaps(
    gaps: list[tuple[int, int]],
    start: int,
    end: int,
    min_size: int,
) -> list[tuple[int, int]]:
    """用间隙切分区间，返回有效内容段。

    例如 full=[0..100), gaps=[(30,35), (60,65)]
    → content=[(0,30), (35,60), (65,100)]

    Parameters
    ----------
    gaps : list[(s, e)]
        间隙（空白）段。
    start, end : int
        总区间的起止。
    min_size : int
        内容段最短长度，小于此值丢弃。

    Returns
    -------
    list[(s, e)]
        内容段。
    """
    regions: list[tuple[int, int]] = []
    prev = start

    for gs, ge in gaps:
        if gs > prev:
            region_len = gs - prev
            if region_len >= min_size:
                regions.append((prev, gs))
        prev = ge

    if prev < end and end - prev >= min_size:
        regions.append((prev, end))

    return regions


# ============================================================
# CLI 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="行列分割：把多行多列小图的大图切分成独立图片"
    )
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("--output", "-o", default="crops", help="输出目录（默认 crops）")
    parser.add_argument("--white-threshold", type=int, default=240,
                        help="白色灰度阈值（0-255，越大越严格）")
    parser.add_argument("--min-gap-rows", type=int, default=3,
                        help="行间隙最少行数")
    parser.add_argument("--min-gap-cols", type=int, default=3,
                        help="列间隙最少列数")
    parser.add_argument("--min-row-height", type=int, default=10,
                        help="最小行高像素")
    parser.add_argument("--min-crop-area", type=int, default=200,
                        help="最小裁剪面积像素")
    parser.add_argument("--prefix", default="crop", help="输出文件前缀")
    parser.add_argument("--fmt", default="png", help="输出格式（png/jpg）")
    parser.add_argument("--draw", action="store_true",
                        help="生成带框标注的预览图")

    args = parser.parse_args()

    cfg = {
        "white_threshold": args.white_threshold,
        "min_gap_rows": args.min_gap_rows,
        "min_gap_cols": args.min_gap_cols,
        "min_row_height": args.min_row_height,
        "min_crop_area": args.min_crop_area,
    }

    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"参数: white_th={args.white_threshold}, "
          f"min_gap_rows={args.min_gap_rows}, "
          f"min_gap_cols={args.min_gap_cols}")

    boxes = grid_split_and_save(
        args.input, args.output, cfg,
        prefix=args.prefix, fmt=args.fmt,
    )

    print(f"分割完成: {len(boxes)} 个小图")
    for box in boxes:
        path = box.get("path", "")
        print(f"  [{box['id']:03d}] row={box['row']} col={box['col']}  "
              f"({box['w']}x{box['h']}) @ ({box['x']},{box['y']})  → {path}")

    if args.draw and boxes:
        draw_path = Path(args.output) / "_grid_preview.jpg"
        draw_grid(args.input, boxes, draw_path)
        print(f"预览图: {draw_path}")


if __name__ == "__main__":
    main()

# ============================================================
# Pipeline 适配器：与 detector.detect_ui_elements 接口兼容
# ============================================================

def detect_ui_elements_grid(
    image_bgr: np.ndarray,
    config: dict[str, Any],
    rmbg_prob_map: np.ndarray | None = None,
) -> dict[str, Any]:
    """行列投影分割，输出格式与 detector.detect_ui_elements 一致。

    配置参数（config["grid"]）：
      white_threshold   — 白色灰度阈值 (0-255)，默认 240
      min_gap_rows      — 行间空白最少行数，默认 3
      min_gap_cols      — 列间空白最少列数，默认 3
      min_row_height    — 最小行高，默认 10
      min_crop_area     — 最小裁剪面积，默认 200

    Returns
    -------
    dict {"boxes": [...], "groups": [...], "canvas_width": w, "canvas_height": h}
    """
    h, w = image_bgr.shape[:2]
    grid_cfg = config.get("grid", {})

    # 构建 grid_cutter 配置
    gc_config = {
        "white_threshold": int(grid_cfg.get("white_threshold", 240)),
        "min_gap_rows": int(grid_cfg.get("min_gap_rows", 3)),
        "min_gap_cols": int(grid_cfg.get("min_gap_cols", 3)),
        "min_row_height": int(grid_cfg.get("min_row_height", 10)),
        "min_crop_area": int(grid_cfg.get("min_crop_area", 200)),
        "crop_border": int(grid_cfg.get("crop_border", 0)),
    }

    # 调用行列分割
    raw_boxes = grid_split(image_bgr, gc_config)

    if not raw_boxes:
        return {"boxes": [], "groups": [], "canvas_width": w, "canvas_height": h}

    # 按 row 分组
    rows: dict[int, list[dict]] = {}
    for rb in raw_boxes:
        rows.setdefault(rb["row"], []).append(rb)

    # 生成 groups
    groups: list[dict[str, Any]] = []
    for row_idx in sorted(rows):
        row_boxes = rows[row_idx]
        ry1 = min(b["y"] for b in row_boxes)
        ry2 = max(b["y"] + b["h"] for b in row_boxes)
        groups.append({
            "id": row_idx,
            "name": f"Row_{row_idx}",
            "row_y": ry1,
            "row_h": ry2 - ry1,
            "count": len(row_boxes),
        })

    # 生成 boxes（标准格式）
    boxes: list[dict[str, Any]] = []
    for rb in raw_boxes:
        boxes.append({
            "id": rb["id"],
            "x1": rb["x"],
            "y1": rb["y"],
            "x2": rb["x"] + rb["w"],
            "y2": rb["y"] + rb["h"],
            "group_id": rb["row"],
            "name": f"Row_{rb['row']}_{rb['col']:02d}",
            "rgba_path": None,
            "score": 1.0,
            "area": rb["w"] * rb["h"],
        })

    return {
        "boxes": boxes,
        "groups": groups,
        "canvas_width": w,
        "canvas_height": h,
    }
