"""
grid_cutter.py — 推荐方案：OCR 去文字 + 非白占比行列检测
============================================================

算法流程：
  1. OCR 识别文字区域并涂白（防止文字干扰行列判断）
  2. 行检测：逐行计算「非白像素占比」，找到内容行区间
  3. 列检测：在每个行区域内按列计算「非白像素占比」，找到内容列区间
  4. 输出每个小图的 bounding box

核心改进：
  - 非白像素占比代替灰度均值（对白色/浅色背景更鲁棒）
  - OCR 去文字避免文字像素干扰间隙判断
  - 内容区间 + 间隙区间双扫描，抗噪能力强

依赖：
  pip install rapidocr_onnxruntime

用法：
  python src/grid_cutter.py input.jpg --output crops/ --draw
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR


# ============================================================
# 默认参数
# ============================================================

DEFAULT_CONFIG: dict[str, Any] = {
    # OCR 去文字
    "ocr_enabled": True,
    "ocr_pad": 3,
    # 行检测
    "white_threshold": 240,
    "content_threshold": 0,  # 保留，不再使用
    "min_gap_rows": 3,          # 连续 3 行全白 → 行间隙
    "min_row_height": 30,       # 过滤太矮的行（文字标注）
    # 列检测
    "min_gap_cols": 3,          # 连续 3 列全白 → 列间隙
    "min_col_width": 20,        # 过滤太窄的列
    "min_crop_area": 200,
}


# ============================================================
# 核心类
# ============================================================

class SmartGridSplitter:
    """OCR 去文字 + 非白占比行列检测。"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self._ocr: RapidOCR | None = None

    # ── OCR 去文字 ──────────────────────────────────────────

    def _lazy_ocr(self) -> RapidOCR:
        if self._ocr is None:
            self._ocr = RapidOCR()
        return self._ocr

    def remove_text(self, image: Image.Image) -> Image.Image:
        """OCR 检测文字区域并涂白（填充白色背景）。"""
        if not self.cfg.get("ocr_enabled", True):
            return image

        ocr = self._lazy_ocr()
        result, _ = ocr(np.array(image))

        if result is None:
            return image  # 没检测到文字

        img_array = np.array(image).copy()
        pad = int(self.cfg.get("ocr_pad", 3))

        for box, text, conf in result:
            # 只涂白中文文字，英文内容保留（排版图上英文通常是设计元素）
            has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
            if not has_chinese:
                continue
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            x1 = max(0, min(xs) - pad)
            y1 = max(0, min(ys) - pad)
            x2 = min(img_array.shape[1], max(xs) + pad)
            y2 = min(img_array.shape[0], max(ys) + pad)
            img_array[y1:y2, x1:x2] = [255, 255, 255]

        return Image.fromarray(img_array)

    # ── 行检测 ──────────────────────────────────────────────

    def find_rows(
        self,
        image: Image.Image,
    ) -> list[tuple[int, int]]:
        """行检测：全白行判定。

        原理：逐行扫描，有任一非白像素（gray < white_threshold）即为内容行。
        间隙必须连续 min_gap_rows 行全部为白色才视为行分割线。
        这确保细线（1px 宽）也不会被切掉。

        返回 [(y1, y2), ...] 每个内容行的上下边界（闭区间）。
        """
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        cfg = self.cfg

        white_th = int(cfg.get("white_threshold", 240))
        min_gap = int(cfg.get("min_gap_rows", 5))
        min_row_h = int(cfg.get("min_row_height", 30))

        # 逐行统计非白像素数
        # 一行有任意非白像素（> 0）即视为内容
        non_white = np.sum(gray < white_th, axis=1)
        is_content = non_white > 0

        # 找间隙（连续全白行）
        gaps = self._find_runs(is_content, False, min_gap)

        # 反推内容行
        rows = self._content_from_gaps(gaps, 0, h - 1)

        # 过滤太矮的行
        rows = [(s, e) for s, e in rows if e - s + 1 >= min_row_h]

        return rows

    # ── 列检测 ──────────────────────────────────────────────

    def find_cols_in_row(
        self,
        row_image: Image.Image,
    ) -> list[tuple[int, int]]:
        """行内列检测：全白列判定。

        原理：逐列扫描，有任一非白像素即视为内容列。
        间隙必须连续 min_gap_cols 列全部为白色才视为列分割线。
        这确保水平细线也不会被切掉。

        返回 [(x1, x2), ...] 每个内容列的左右边界（闭区间）。
        """
        gray = cv2.cvtColor(np.array(row_image), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        cfg = self.cfg

        white_th = int(cfg.get("white_threshold", 240))
        min_gap = int(cfg.get("min_gap_cols", 3))
        min_col_w = int(cfg.get("min_col_width", 20))

        # 逐列统计非白像素数
        # 一列有任意非白像素即视为内容
        non_white = np.sum(gray < white_th, axis=0)
        is_content = non_white > 0

        # 找间隙（连续全白列）
        gaps = self._find_runs(is_content, False, min_gap)

        # 反推内容列
        cols = self._content_from_gaps(gaps, 0, w - 1)

        # 过滤太窄的列
        cols = [(s, e) for s, e in cols if e - s + 1 >= min_col_w]

        return cols

    # ── 完整流程 ─────────────────────────────────────────────

    def split(
        self,
        image: str | Path | np.ndarray | Image.Image,
    ) -> tuple[list[dict[str, int]], Image.Image]:
        """完整流程：OCR 去文字 → 行检测 → 行内列检测。

        Returns
        -------
        boxes : list[dict]
            [{"row":1, "col":1, "x":int, "y":int, "w":int, "h":int}, ...]
        clean : PIL.Image
            去文字后的图片（用于可视化验证）。
        """
        # ── 读图 ──
        if isinstance(image, (str, Path)):
            img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError("不支持的图片类型")

        img_w, img_h = img.size

        # 1. OCR 去文字
        clean = self.remove_text(img)

        # 2. 找行
        rows = self.find_rows(clean)
        print(f"行检测: {len(rows)} 行")

        # 3. 每行找列
        boxes: list[dict[str, int]] = []
        for row_idx, (y1, y2) in enumerate(rows):
            row_img = clean.crop((0, y1, img_w, y2 + 1))
            cols = self.find_cols_in_row(row_img)
            print(f"  第{row_idx+1}行: {len(cols)} 个小图")

            for col_idx, (x1, x2) in enumerate(cols):
                boxes.append({
                    "row": row_idx + 1,
                    "col": col_idx + 1,
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1 + 1,
                    "h": y2 - y1 + 1,
                })

        return boxes, clean

    # ── 工具方法 ────────────────────────────────────────────

    @staticmethod
    def _find_runs(
        array: np.ndarray,
        target: bool,
        min_len: int,
    ) -> list[tuple[int, int]]:
        """找到连续 target 值且长度 >= min_len 的段。

        返回 [(start, end), ...] 闭区间。
        """
        runs: list[tuple[int, int]] = []
        start: int | None = None

        for i in range(len(array)):
            if array[i] == target and start is None:
                start = i
            elif array[i] != target and start is not None:
                if i - start >= min_len:
                    runs.append((start, i - 1))
                start = None

        if start is not None and len(array) - start >= min_len:
            runs.append((start, len(array) - 1))

        return runs

    @staticmethod
    def _content_from_gaps(
        gaps: list[tuple[int, int]],
        start: int,
        end: int,
    ) -> list[tuple[int, int]]:
        """从间隙反推内容段。

        Parameters
        ----------
        gaps : list[(s, e)]
            间隙段（闭区间）。
        start, end : int
            整段总范围（闭区间）。

        Returns
        -------
        list[(s, e)]
            内容段（闭区间）。
        """
        regions: list[tuple[int, int]] = []
        prev = start

        for gs, ge in gaps:
            if gs > prev:
                regions.append((prev, gs - 1))
            prev = ge + 1

        if prev <= end:
            regions.append((prev, end))

        return regions


# ============================================================
# 简便调用函数（兼容旧接口）
# ============================================================

def grid_split(
    image: str | Path | np.ndarray,
    config: dict[str, Any] | None = None,
) -> list[dict[str, int]]:
    """行列分割：返回每个小图的 bounding box。

    参数与 SmartGridSplitter.split() 相同。
    """
    splitter = SmartGridSplitter(config)
    boxes, _ = splitter.split(image)
    return boxes


def grid_split_and_save(
    image: str | Path | np.ndarray,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    prefix: str = "crop",
    fmt: str = "png",
    draw: bool = False,
) -> list[dict[str, int]]:
    """行列分割 + 裁剪保存 + 可选预览图。"""
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
    else:
        bgr = image

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splitter = SmartGridSplitter(config)
    boxes, clean_img = splitter.split(image)

    # 保存去文字后的图片
    clean_img.save(str(out_dir / "_ocr_cleaned.png"))

    # 裁剪保存
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        crop = bgr[y : y + h, x : x + w]
        filename = f"{prefix}_{box['row']:02d}_{box['col']:02d}.{fmt}"
        cv2.imwrite(str(out_dir / filename), crop)
        box["path"] = str(out_dir / filename)

    return boxes


def draw_grid(
    image: str | Path | np.ndarray,
    boxes: list[dict[str, int]],
    output_path: str | Path | None = None,
) -> np.ndarray:
    """绘制行/列分割框线。"""
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
    else:
        bgr = image.copy()

    # 收集行区间
    rows: dict[int, list[dict]] = {}
    for box in boxes:
        rows.setdefault(box["row"], []).append(box)

    colors = [
        (0, 200, 0), (0, 0, 200), (200, 0, 0),
        (0, 200, 200), (200, 0, 200), (200, 200, 0),
    ]

    for row_idx in sorted(rows):
        row_boxes = rows[row_idx]
        color = colors[(row_idx - 1) % len(colors)]

        for box in row_boxes:
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
            label = f"{box['row']}-{box['col']}"
            cv2.putText(bgr, label, (x + 2, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if output_path:
        cv2.imwrite(str(output_path), bgr)

    return bgr


# ============================================================
# Pipeline 适配器（与 detector.detect_ui_elements 接口兼容）
# ============================================================

def detect_ui_elements_grid(
    image_bgr: np.ndarray,
    config: dict[str, Any],
    rmbg_prob_map: np.ndarray | None = None,
) -> dict[str, Any]:
    """SmartGridSplitter 适配器，输出与 detector.detect_ui_elements 一致。"""
    h, w = image_bgr.shape[:2]
    grid_cfg = config.get("grid", {})

    splitter = SmartGridSplitter(grid_cfg)
    raw_boxes, _ = splitter.split(image_bgr)

    if not raw_boxes:
        return {"boxes": [], "groups": [], "canvas_width": w, "canvas_height": h}

    # 按 row 分组
    rows_map: dict[int, list[dict]] = {}
    for rb in raw_boxes:
        rows_map.setdefault(rb["row"], []).append(rb)

    # 生成 groups
    groups: list[dict[str, Any]] = []
    for row_idx in sorted(rows_map):
        row_boxes = rows_map[row_idx]
        ry1 = min(b["y"] for b in row_boxes)
        ry2 = max(b["y"] + b["h"] for b in row_boxes)
        groups.append({
            "id": row_idx,
            "name": f"Row_{row_idx}",
            "row_y": ry1,
            "row_h": ry2 - ry1,
            "count": len(row_boxes),
        })

    # 生成标准 boxes
    boxes: list[dict[str, Any]] = []
    for rb in raw_boxes:
        boxes.append({
            "id": rb["row"] * 100 + rb["col"],
            "x1": rb["x"],
            "y1": rb["y"],
            "x2": rb["x"] + rb["w"],
            "y2": rb["y"] + rb["h"],
            "group_id": rb["row"],
            "col": rb["col"],
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


# ============================================================
# CLI 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="行列分割：OCR 去文字 + 非白占比行列检测"
    )
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("--output", "-o", default="crops", help="输出目录")
    parser.add_argument("--no-ocr", action="store_true", help="跳过 OCR 去文字")
    parser.add_argument("--white-threshold", type=int, default=240)
    parser.add_argument("--content-threshold", type=float, default=0.05)
    parser.add_argument("--min-gap-rows", type=int, default=5)
    parser.add_argument("--min-gap-cols", type=int, default=3)
    parser.add_argument("--min-row-height", type=int, default=30)
    parser.add_argument("--min-col-width", type=int, default=20)
    parser.add_argument("--prefix", default="crop")
    parser.add_argument("--fmt", default="png")
    parser.add_argument("--draw", action="store_true", help="生成预览图")

    args = parser.parse_args()

    cfg = {
        "ocr_enabled": not args.no_ocr,
        "white_threshold": args.white_threshold,
        "content_threshold": args.content_threshold,
        "min_gap_rows": args.min_gap_rows,
        "min_gap_cols": args.min_gap_cols,
        "min_row_height": args.min_row_height,
        "min_col_width": args.min_col_width,
    }

    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"OCR: {'跳过' if args.no_ocr else '启用'}")
    print(f"参数: white_th={args.white_threshold} "
          f"content_th={args.content_threshold} "
          f"min_gap_rows={args.min_gap_rows} "
          f"min_gap_cols={args.min_gap_cols}")

    boxes = grid_split_and_save(
        args.input, args.output, cfg,
        prefix=args.prefix, fmt=args.fmt,
    )
    total = len(boxes)
    rows = set(b["row"] for b in boxes)
    print(f"\n分割完成: {total} 个小图, {len(rows)} 行")

    for b in boxes[:20]:
        print(f"  [{b['row']}-{b['col']}] ({b['w']}x{b['h']}) @ ({b['x']},{b['y']})")

    if args.draw and boxes:
        draw_path = Path(args.output) / "_grid_preview.jpg"
        draw_grid(args.input, boxes, draw_path)
        print(f"预览图: {draw_path}")


if __name__ == "__main__":
    main()
