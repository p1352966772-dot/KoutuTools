import sys, os
sys.stdout.reconfigure(encoding="utf-8")

content = """from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def detect_ui_elements(image_bgr: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    \"\"\"Multi-scale UI element detection.
    
    Returns:
        boxes: list of dicts with x1,y1,x2,y2,w,h,id,group_id,name
        groups: list of dicts with group_id, name, row_y, boxes
    \"\"\"
    detect_config = config.get(\"detect\", {})
    white_threshold = int(detect_config.get(\"white_threshold\", 240))
    l1_row_kernel = int(detect_config.get(\"l1_row_kernel\", 35))
    l2_element_kernel = int(detect_config.get(\"l2_element_kernel\", 10))
    row_gap_min = int(detect_config.get(\"row_gap_min\", 15))
    col_gap_min = int(detect_config.get(\"col_gap_min\", 10))
    merge_h_gap = int(detect_config.get(\"merge_h_gap\", 30))
    merge_v_tol = int(detect_config.get(\"merge_v_tol\", 15))
    merge_iou = float(detect_config.get(\"merge_iou\", 0.25))
    merge_aspect_ratio_tol = float(detect_config.get(\"merge_aspect_ratio_tol\", 0.20))
    min_area_ratio = float(detect_config.get(\"min_area_ratio\", 0.0015))
    min_element_size = int(detect_config.get(\"min_element_size\", 20))
    max_aspect_ratio = float(detect_config.get(\"max_aspect_ratio\", 10.0))
    min_aspect_ratio = float(detect_config.get(\"min_aspect_ratio\", 0.1))

    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    total_px = width * height

    # Step 1: Foreground mask (non-white pixels)
    fg_mask = (gray < white_threshold).astype(np.uint8) * 255

    # Step 2: L1 - Row detection via horizontal projection
    # Use tall kernel to connect elements within same row vertically
    l1_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, l1_row_kernel))
    l1_merged = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, l1_kernel)

    h_proj = np.count_nonzero(l1_merged, axis=1)
    row_boundaries = _projection_split(h_proj, row_gap_min, 2, height)

    # Step 3: L2 - Element detection within each row
    l2_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (l2_element_kernel, 1))
    l2_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, l2_kernel)

    raw_boxes: list[dict[str, int]] = []
    for ry1, ry2 in row_boundaries:
        if ry2 - ry1 < min_element_size:
            continue
        row_slice = l2_mask[ry1:ry2, :]
        v_proj = np.count_nonzero(row_slice, axis=0)
        col_boundaries = _projection_split(v_proj, col_gap_min, 2, width)

        for cx1, cx2 in col_boundaries:
            if cx2 - cx1 < min_element_size:
                continue
            crop = fg_mask[ry1:ry2, cx1:cx2]
            pts = cv2.findNonZero(crop)
            if pts is None:
                continue
            bx, by, bw, bh = cv2.boundingRect(pts)
            x1 = cx1 + bx
            y1 = ry1 + by
            x2 = x1 + bw
            y2 = y1 + bh
            pw = x2 - x1
            ph = y2 - y1
            if pw < min_element_size or ph < min_element_size:
                continue
            raw_boxes.append({
                \"id\": 0, \"group_id\": 0,
                \"x\": x1, \"y\": y1, \"w\": pw, \"h\": ph,
                \"x1\": x1, \"y1\": y1, \"x2\": x2, \"y2\": y2,
                \"area\": pw * ph,
                \"name\": \"\",
            })

    if not raw_boxes:
        return {\"boxes\": [], \"groups\": []}

    # Step 4: Connected component refinement
    # For each raw box, find exact connected components inside
    refined_boxes = _refine_with_cc(raw_boxes, fg_mask, min_element_size)

    # Step 5: Merge boxes
    merged_boxes = _merge_boxes(refined_boxes, merge_h_gap, merge_v_tol, merge_iou, merge_aspect_ratio_tol)

    # Step 6: Noise filtering
    filtered = _filter_noise(merged_boxes, total_px, min_area_ratio, min_element_size, max_aspect_ratio, min_aspect_ratio)

    if not filtered:
        return {\"boxes\": [], \"groups\": []}

    # Step 7: Row clustering
    grouped_boxes, groups = _cluster_rows(filtered, row_gap_min * 2)

    # Step 8: Semantic naming
    for group in groups:
        gid = group[\"id\"]
        row_boxes = [b for b in grouped_boxes if b[\"group_id\"] == gid]
        row_boxes.sort(key=lambda b: b[\"x1\"])

        for idx, box in enumerate(row_boxes):
            pos_h = _h_position(box, width)
            pos_v = _v_position(box, height)
            size_label = _size_label(box)
            box[\"name\"] = f\"row_{gid}_{pos_h}_{size_label}_{idx + 1}\"

    # Sort final list
    grouped_boxes.sort(key=lambda b: (b[\"y1\"], b[\"x1\"]))
    for i, box in enumerate(grouped_boxes, 1):
        box[\"id\"] = i

    return {\"boxes\": grouped_boxes, \"groups\": groups}


def _projection_split(proj: np.ndarray, min_gap: int, noise: int, total: int) -> list[tuple[int, int]]:
    \"\"\"Find segments in a 1D projection separated by gaps >= min_gap.\"\"\"
    low = (proj <= noise).astype(np.uint8)
    runs = _true_runs(low)
    gaps = [(s, e) for s, e in runs if e - s >= min_gap]

    if not gaps:
        return [(0, total)]

    segs: list[tuple[int, int]] = []
    prev = 0
    for s, e in gaps:
        if s > prev:
            segs.append((prev, s))
        prev = e
    if prev < total:
        segs.append((prev, total))
    return segs


def _true_runs(arr: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(arr):
        if bool(v) and start is None:
            start = i
        elif not bool(v) and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(arr)))
    return runs


def _refine_with_cc(boxes: list[dict], mask: np.ndarray, min_size: int) -> list[dict]:
    \"\"\"Split boxes into exact connected components for finer granularity.\"\"\"
    result: list[dict] = []
    for box in boxes:
        x1, y1, x2, y2 = box[\"x1\"], box[\"y1\"], box[\"x2\"], box[\"y2\"]
        crop = mask[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        total, labels, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
        if total <= 1:
            result.append(box)
            continue
        components = []
        for label in range(1, total):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_size:
                continue
            cx = int(stats[label, cv2.CC_STAT_LEFT]) + x1
            cy = int(stats[label, cv2.CC_STAT_TOP]) + y1
            cw = int(stats[label, cv2.CC_STAT_WIDTH])
            ch = int(stats[label, cv2.CC_STAT_HEIGHT])
            w2 = cx + cw
            h2 = cy + ch
            if cw < min_size or ch < min_size:
                continue
            components.append({
                \"id\": 0, \"group_id\": 0,
                \"x\": cx, \"y\": cy, \"w\": cw, \"h\": ch,
                \"x1\": cx, \"y1\": cy, \"x2\": w2, \"y2\": h2,
                \"area\": cw * ch,
                \"name\": \"\",
            })
        if len(components) >= 2:
            result.extend(components)
        else:
            result.append(box)
    return result


def _merge_boxes(boxes: list[dict], h_gap: int, v_tol: int, iou_th: float, ar_tol: float) -> list[dict]:
    \"\"\"Merge boxes that should logically be one UI element.\"\"\"
    if len(boxes) < 2:
        return boxes

    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        new_list: list[dict] = []
        used = [False] * len(merged)
        for i, a in enumerate(merged):
            if used[i]:
                continue
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                b = merged[j]
                if _should_merge(a, b, h_gap, v_tol, iou_th, ar_tol):
                    a = _union_box(a, b)
                    used[j] = True
                    changed = True
            new_list.append(a)
            used[i] = True
        merged = new_list
    return merged


def _should_merge(a: dict, b: dict, h_gap: int, v_tol: int, iou_th: float, ar_tol: float) -> bool:
    ax1, ay1, ax2, ay2 = a[\"x1\"], a[\"y1\"], a[\"x2\"], a[\"y2\"]
    bx1, by1, bx2, by2 = b[\"x1\"], b[\"y1\"], b[\"x2\"], b[\"y2\"]

    # Compute IoU
    ox1 = max(ax1, bx1); oy1 = max(ay1, by1)
    ox2 = min(ax2, bx2); oy2 = min(ay2, by2)
    ow = max(0, ox2 - ox1); oh = max(0, oy2 - oy1)
    oarea = ow * oh
    aarea = (ax2 - ax1) * (ay2 - ay1)
    barea = (bx2 - bx1) * (by2 - by1)
    iou = oarea / max(1, aarea + barea - oarea)
    if iou > iou_th:
        return True

    # Horizontal gap check
    if ax1 > bx1:
        a, b = b, a
        ax1, ay1, ax2, ay2 = a[\"x1\"], a[\"y1\"], a[\"x2\"], a[\"y2\"]
        bx1, by1, bx2, by2 = b[\"x1\"], b[\"y1\"], b[\"x2\"], b[\"y2\"]
    hgap = bx1 - ax2
    if hgap < 0 or hgap > h_gap:
        return False

    # Vertical alignment
    a_center_y = (ay1 + ay2) / 2
    b_center_y = (by1 + by2) / 2
    if abs(a_center_y - b_center_y) > v_tol:
        return False

    # Aspect ratio similarity
    a_ar = (ax2 - ax1) / max(1, ay2 - ay1)
    b_ar = (bx2 - bx1) / max(1, by2 - by1)
    if a_ar > 0 and b_ar > 0:
        ratio = max(a_ar, b_ar) / max(0.001, min(a_ar, b_ar))
        if ratio > 1.0 + ar_tol:
            return False

    return True


def _union_box(a: dict, b: dict) -> dict:
    x1 = min(a[\"x1\"], b[\"x1\"])
    y1 = min(a[\"y1\"], b[\"y1\"])
    x2 = max(a[\"x2\"], b[\"x2\"])
    y2 = max(a[\"y2\"], b[\"y2\"])
    return {
        \"id\": 0, \"group_id\": 0,
        \"x\": x1, \"y\": y1, \"w\": x2 - x1, \"h\": y2 - y1,
        \"x1\": x1, \"y1\": y1, \"x2\": x2, \"y2\": y2,
        \"area\": (x2 - x1) * (y2 - y1),
        \"name\": \"\",
    }


def _filter_noise(boxes: list[dict], total_px: int, min_area_ratio: float, min_size: int, max_ar: float, min_ar: float) -> list[dict]:
    \"\"\"Remove noise boxes that don't meet quality criteria.\"\"\"
    result: list[dict] = []
    for box in boxes:
        w = box[\"x2\"] - box[\"x1\"]
        h = box[\"y2\"] - box[\"y1\"]
        area = w * h
        if area < total_px * min_area_ratio:
            continue
        if w < min_size and h < min_size:
            continue
        ar = max(w, h) / max(1, min(w, h))
        if ar > max_ar or ar < min_ar:
            continue
        result.append(box)
    return result


def _cluster_rows(boxes: list[dict], row_tol: int) -> tuple[list[dict], list[dict]]:
    \"\"\"Group boxes into visual rows by y-coordinate proximity.\"\"\"
    if not boxes:
        return [], []

    sorted_boxes = sorted(boxes, key=lambda b: (b[\"y1\"] + b[\"y2\"]) // 2)

    groups: list[list[dict]] = []
    group_centers: list[int] = []

    for box in sorted_boxes:
        cy = (box[\"y1\"] + box[\"y2\"]) // 2
        placed = False
        for gi, gc in enumerate(group_centers):
            if abs(cy - gc) <= row_tol:
                groups[gi].append(box)
                new_cy = sum((b[\"y1\"] + b[\"y2\"]) // 2 for b in groups[gi]) // len(groups[gi])
                group_centers[gi] = new_cy
                placed = True
                break
        if not placed:
            groups.append([box])
            group_centers.append(cy)

    # Sort groups by y
    group_data = sorted(enumerate(groups), key=lambda x: group_centers[x[0]])
    group_list: list[dict] = []
    box_list: list[dict] = []

    for gi, (orig_idx, gboxes) in enumerate(group_data):
        gid = gi + 1
        ry = min(b[\"y1\"] for b in gboxes)
        rh = max(b[\"y2\"] for b in gboxes) - ry
        group_list.append({
            \"id\": gid,
            \"name\": f\"Row_{gid}\",
            \"row_y\": ry,
            \"row_h\": rh,
            \"count\": len(gboxes),
        })
        for box in gboxes:
            box[\"group_id\"] = gid
            box_list.append(box)

    return box_list, group_list


def _h_position(box: dict, width: int) -> str:
    cx = (box[\"x1\"] + box[\"x2\"]) / 2
    if cx < width * 0.3:
        return \"left\"
    elif cx < width * 0.7:
        return \"center\"
    return \"right\"


def _v_position(box: dict, height: int) -> str:
    cy = (box[\"y1\"] + box[\"y2\"]) / 2
    if cy < height * 0.3:
        return \"top\"
    elif cy < height * 0.7:
        return \"mid\"
    return \"bot\"


def _size_label(box: dict) -> str:
    area = (box[\"x2\"] - box[\"x1\"]) * (box[\"y2\"] - box[\"y1\"])
    w = box[\"x2\"] - box[\"x1\"]
    h = box[\"y2\"] - box[\"y1\"]
    if area < 5000:
        return \"sml\"
    elif area < 30000:
        return \"med\"
    return \"lrg\"
"""

dst = os.path.expandvars(r"%USERPROFILE%\Desktop\KoutuTools\auto_psd_cutout\src\detector.py")
with open(dst, "w", encoding="utf-8") as f:
    f.write(content)
print("detector.py written OK")

import ast
with open(dst, "r", encoding="utf-8") as f:
    ast.parse(f.read())
print("Syntax OK")
