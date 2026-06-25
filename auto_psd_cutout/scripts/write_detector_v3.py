import sys, os
sys.stdout.reconfigure(encoding="utf-8")

content = """from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def detect_ui_elements(image_bgr: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    detect_config = config.get("detect", {})
    white_threshold = int(detect_config.get("white_threshold", 240))
    l1_row_kernel = int(detect_config.get("l1_row_kernel", 35))
    row_gap_min = int(detect_config.get("row_gap_min", 15))
    min_element_size = int(detect_config.get("min_element_size", 20))
    min_area_ratio = float(detect_config.get("min_area_ratio", 0.0015))
    max_aspect_ratio = float(detect_config.get("max_aspect_ratio", 8.0))
    min_aspect_ratio = float(detect_config.get("min_aspect_ratio", 0.1))
    text_max_height_pct = float(detect_config.get("text_max_height_pct", 0.08))
    text_max_area_pct = float(detect_config.get("text_max_area_pct", 0.005))
    text_y_alignment_tol = int(detect_config.get("text_y_alignment_tol", 10))
    text_merge_gap = int(detect_config.get("text_merge_gap", 30))
    merge_h_gap = int(detect_config.get("merge_h_gap", 25))
    merge_y_overlap = float(detect_config.get("merge_y_overlap_threshold", 0.70))
    merge_iou = float(detect_config.get("merge_iou", 0.20))
    min_char_size = int(detect_config.get("min_char_size", 18))

    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    total_px = width * height

    # Step 1: Foreground mask
    fg_mask = (gray < white_threshold).astype(np.uint8) * 255

    # Step 2: Row detection (L1)
    l1_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, l1_row_kernel))
    l1_merged = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, l1_kernel)
    h_proj = np.count_nonzero(l1_merged, axis=1)
    row_boundaries = _projection_split(h_proj, row_gap_min, 2, height)

    # Step 3: Within each row, find components and merge text
    all_boxes: list[dict] = []
    for ry1, ry2 in row_boundaries:
        if ry2 - ry1 < min_element_size:
            continue
        row_boxes = _process_row(fg_mask, ry1, ry2, width, height, total_px, detect_config)
        all_boxes.extend(row_boxes)

    if not all_boxes:
        return {"boxes": [], "groups": []}

    # Step 4: General merge pass (catch remaining merges across rows)
    all_boxes = _merge_general(all_boxes, merge_h_gap, merge_y_overlap, merge_iou)

    # Step 5: Character suppression — remove sub-18px isolated boxes
    all_boxes = _suppress_characters(all_boxes, min_char_size, max_aspect_ratio)

    # Step 6: Noise filtering
    all_boxes = _filter_noise(all_boxes, total_px, min_area_ratio, min_element_size, max_aspect_ratio, min_aspect_ratio)

    if not all_boxes:
        return {"boxes": [], "groups": []}

    # Step 7: Row clustering (re-cluster after merges)
    grouped_boxes, groups = _cluster_rows(all_boxes, row_gap_min * 2)

    # Step 8: Semantic naming
    for group in groups:
        gid = group["id"]
        gboxes = [b for b in grouped_boxes if b["group_id"] == gid]
        gboxes.sort(key=lambda b: b["x1"])
        for idx, box in enumerate(gboxes):
            ph = _h_position(box, width)
            pv = _v_position(box, height)
            sz = _size_label(box)
            box["name"] = f"row_{gid}_{ph}_{sz}_{idx + 1}"

    grouped_boxes.sort(key=lambda b: (b["y1"], b["x1"]))
    for i, box in enumerate(grouped_boxes, 1):
        box["id"] = i

    return {"boxes": grouped_boxes, "groups": groups}


def _process_row(
    fg_mask: np.ndarray, ry1: int, ry2: int, img_w: int, img_h: int,
    total_px: int, cfg: dict,
) -> list[dict]:
    min_comp = int(cfg.get("min_element_size", 20))
    row_slice = fg_mask[ry1:ry2, :].copy()
    h = ry2 - ry1
    if row_slice.size == 0:
        return []

    total, labels, stats, _ = cv2.connectedComponentsWithStats(row_slice, connectivity=8)

    comps: list[dict] = []
    for label in range(1, total):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 10:
            continue
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = ry1 + int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if cw < 3 or ch < 3:
            continue
        comps.append({
            "id": 0, "group_id": 0,
            "x": cx, "y": cy, "w": cw, "h": ch,
            "x1": cx, "y1": cy, "x2": cx + cw, "y2": cy + ch,
            "area": cw * ch,
            "name": "",
        })

    if not comps:
        return []

    comps.sort(key=lambda c: c["x1"])

    # Classify text vs icon
    is_text_arr = [_is_likely_text(c, img_h, total_px, cfg) for c in comps]
    text_comps = [c for i, c in enumerate(comps) if is_text_arr[i]]
    icon_comps = [c for i, c in enumerate(comps) if not is_text_arr[i]]

    # Merge text components into blocks
    text_blocks = _merge_text_components(text_comps, cfg) if text_comps else []

    # Merge icon components that are very close
    icon_blocks = _merge_icon_components(icon_comps, cfg) if icon_comps else []

    row_boxes = text_blocks + icon_blocks
    row_boxes.sort(key=lambda b: b["x1"])
    return row_boxes


def _is_likely_text(comp: dict, img_h: int, total_px: int, cfg: dict) -> bool:
    h = comp["y2"] - comp["y1"]
    w = comp["x2"] - comp["x1"]
    area = comp["area"]
    max_text_h = max(10, int(img_h * float(cfg.get("text_max_height_pct", 0.08))))
    max_text_area = max(50, int(total_px * float(cfg.get("text_max_area_pct", 0.005))))
    if h > max_text_h:
        return False
    if area > max_text_area:
        return False
    if w < 3 or h < 3:
        return False
    ar = max(w, h) / max(1, min(w, h))
    if ar > 6.0:
        return False
    return True


def _merge_text_components(comps: list[dict], cfg: dict) -> list[dict]:
    if not comps:
        return []
    tol = int(cfg.get("text_y_alignment_tol", 10))
    gap = int(cfg.get("text_merge_gap", 30))
    comps.sort(key=lambda c: c["x1"])

    groups: list[list[dict]] = []
    for c in comps:
        placed = False
        for g in groups:
            last = g[-1]
            if _text_can_merge(last, c, tol, gap):
                g.append(c)
                placed = True
                break
        if not placed:
            groups.append([c])

    result: list[dict] = []
    for g in groups:
        x1 = min(c["x1"] for c in g)
        y1 = min(c["y1"] for c in g)
        x2 = max(c["x2"] for c in g)
        y2 = max(c["y2"] for c in g)
        w = x2 - x1
        h = y2 - y1
        result.append({
            "id": 0, "group_id": 0,
            "x": x1, "y": y1, "w": w, "h": h,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "area": w * h, "name": "",
        })
    return result


def _text_can_merge(prev: dict, curr: dict, y_tol: int, max_gap: int) -> bool:
    prev_cy = (prev["y1"] + prev["y2"]) / 2
    curr_cy = (curr["y1"] + curr["y2"]) / 2
    if abs(prev_cy - curr_cy) > y_tol:
        return False
    prev_h = prev["y2"] - prev["y1"]
    curr_h = curr["y2"] - curr["y1"]
    if prev_h > 0 and curr_h > 0:
        hr = max(prev_h, curr_h) / max(1, min(prev_h, curr_h))
        if hr > 2.5:
            return False
    gap = curr["x1"] - prev["x2"]
    if gap > max_gap:
        return False
    return True


def _merge_icon_components(comps: list[dict], cfg: dict) -> list[dict]:
    if not comps:
        return []
    gap = int(cfg.get("merge_h_gap", 25))
    y_overlap = float(cfg.get("merge_y_overlap_threshold", 0.70))
    iou_th = float(cfg.get("merge_iou", 0.20))
    comps.sort(key=lambda c: c["x1"])

    groups: list[list[dict]] = []
    for c in comps:
        placed = False
        for g in groups:
            last = g[-1]
            if _icon_can_merge(last, c, gap, y_overlap, iou_th):
                g.append(c)
                placed = True
                break
        if not placed:
            groups.append([c])

    result: list[dict] = []
    for g in groups:
        x1 = min(c["x1"] for c in g)
        y1 = min(c["y1"] for c in g)
        x2 = max(c["x2"] for c in g)
        y2 = max(c["y2"] for c in g)
        w = x2 - x1
        h = y2 - y1
        result.append({
            "id": 0, "group_id": 0,
            "x": x1, "y": y1, "w": w, "h": h,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "area": w * h, "name": "",
        })
    return result


def _icon_can_merge(prev: dict, curr: dict, h_gap: int, y_overlap: float, iou_th: float) -> bool:
    ax1, ay1, ax2, ay2 = prev["x1"], prev["y1"], prev["x2"], prev["y2"]
    bx1, by1, bx2, by2 = curr["x1"], curr["y1"], curr["x2"], curr["y2"]
    ox1 = max(ax1, bx1); oy1 = max(ay1, by1)
    ox2 = min(ax2, bx2); oy2 = min(ay2, by2)
    ow = max(0, ox2 - ox1); oh = max(0, oy2 - oy1)
    oarea = ow * oh
    aarea = (ax2 - ax1) * (ay2 - ay1)
    barea = (bx2 - bx1) * (by2 - by1)
    iou = oarea / max(1, aarea + barea - oarea)
    if iou > iou_th:
        return True
    if ow > 0:
        min_h = min(ay2 - ay1, by2 - by1)
        if min_h > 0 and oh / min_h >= y_overlap:
            if ax1 > bx1:
                ax1, ax2 = bx1, bx2
            hgap = bx1 - ax2 if bx1 > ax2 else ax1 - (bx2 if bx2 < ax1 else bx1)
            if hgap < 0:
                hgap = 0
            if hgap <= h_gap:
                return True
    if ax1 > bx1:
        prev, curr = curr, prev
        ax1, ay1, ax2, ay2 = prev["x1"], prev["y1"], prev["x2"], prev["y2"]
        bx1, by1, bx2, by2 = curr["x1"], curr["y1"], curr["x2"], curr["y2"]
    hgap = bx1 - ax2
    if hgap <= h_gap:
        prev_cy = (ay1 + ay2) / 2
        curr_cy = (by1 + by2) / 2
        if abs(prev_cy - curr_cy) <= 15:
            return True
    return False


def _merge_general(boxes: list[dict], h_gap: int, y_overlap: float, iou_th: float) -> list[dict]:
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
                if _general_should_merge(a, b, h_gap, y_overlap, iou_th):
                    a = _union_box(a, b)
                    used[j] = True
                    changed = True
            new_list.append(a)
            used[i] = True
        merged = new_list
    return merged


def _general_should_merge(a: dict, b: dict, h_gap: int, y_overlap: float, iou_th: float) -> bool:
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ox1 = max(ax1, bx1); oy1 = max(ay1, by1)
    ox2 = min(ax2, bx2); oy2 = min(ay2, by2)
    ow = max(0, ox2 - ox1); oh = max(0, oy2 - oy1)
    oarea = ow * oh
    aarea = (ax2 - ax1) * (ay2 - ay1)
    barea = (bx2 - bx1) * (by2 - by1)
    iou = oarea / max(1, aarea + barea - oarea)
    if iou > iou_th:
        return True
    if ow > 0:
        min_h = min(ay2 - ay1, by2 - by1)
        if min_h > 0 and oh / min_h >= y_overlap:
            if bx1 > ax2:
                hg = bx1 - ax2
                if hg <= h_gap:
                    return True
            elif ax1 > bx2:
                hg = ax1 - bx2
                if hg <= h_gap:
                    return True
    if ax1 > bx1:
        a, b = b, a
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    hgap = bx1 - ax2
    if hgap <= h_gap:
        prev_cy = (ay1 + ay2) / 2
        curr_cy = (by1 + by2) / 2
        if abs(prev_cy - curr_cy) <= 15:
            return True
    return False


def _suppress_characters(boxes: list[dict], min_size: int, max_ar: float) -> list[dict]:
    keep: list[dict] = []
    discard: list[dict] = []
    for b in boxes:
        w = b["x2"] - b["x1"]
        h = b["y2"] - b["y1"]
        if w >= min_size and h >= min_size:
            keep.append(b)
        else:
            ar = max(w, h) / max(1, min(w, h))
            if ar > max_ar:
                discard.append(b)
            else:
                discard.append(b)
    if not discard or not keep:
        return keep if keep else boxes
    for db in discard:
        best_dist = 999999
        best = None
        for kb in keep:
            dx = max(kb["x1"] - db["x2"], db["x1"] - kb["x2"], 0)
            dy = max(kb["y1"] - db["y2"], db["y1"] - kb["y2"], 0)
            dist = dx + dy
            if dist < best_dist:
                best_dist = dist
                best = kb
        if best and best_dist < 100:
            best["x1"] = min(best["x1"], db["x1"])
            best["y1"] = min(best["y1"], db["y1"])
            best["x2"] = max(best["x2"], db["x2"])
            best["y2"] = max(best["y2"], db["y2"])
            best["w"] = best["x2"] - best["x1"]
            best["h"] = best["y2"] - best["y1"]
            best["area"] = best["w"] * best["h"]
    return keep


def _union_box(a: dict, b: dict) -> dict:
    x1 = min(a["x1"], b["x1"])
    y1 = min(a["y1"], b["y1"])
    x2 = max(a["x2"], b["x2"])
    y2 = max(a["y2"], b["y2"])
    return {
        "id": 0, "group_id": 0,
        "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "area": (x2 - x1) * (y2 - y1), "name": "",
    }


def _projection_split(proj: np.ndarray, min_gap: int, noise: int, total: int) -> list[tuple[int, int]]:
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


def _filter_noise(boxes: list[dict], total_px: int, min_area_ratio: float, min_size: int, max_ar: float, min_ar: float) -> list[dict]:
    result: list[dict] = []
    for box in boxes:
        w = box["x2"] - box["x1"]
        h = box["y2"] - box["y1"]
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
    if not boxes:
        return [], []
    sorted_boxes = sorted(boxes, key=lambda b: (b["y1"] + b["y2"]) // 2)
    groups: list[list[dict]] = []
    centers: list[int] = []
    for box in sorted_boxes:
        cy = (box["y1"] + box["y2"]) // 2
        placed = False
        for gi, gc in enumerate(centers):
            if abs(cy - gc) <= row_tol:
                groups[gi].append(box)
                centers[gi] = sum((b["y1"] + b["y2"]) // 2 for b in groups[gi]) // len(groups[gi])
                placed = True
                break
        if not placed:
            groups.append([box])
            centers.append(cy)
    gdata = sorted(enumerate(groups), key=lambda x: centers[x[0]])
    glist: list[dict] = []
    blist: list[dict] = []
    for gi, (_, gboxes) in enumerate(gdata):
        gid = gi + 1
        ry = min(b["y1"] for b in gboxes)
        rh = max(b["y2"] for b in gboxes) - ry
        glist.append({"id": gid, "name": f"Row_{gid}", "row_y": ry, "row_h": rh, "count": len(gboxes)})
        for box in gboxes:
            box["group_id"] = gid
            blist.append(box)
    return blist, glist


def _h_position(box: dict, width: int) -> str:
    cx = (box["x1"] + box["x2"]) / 2
    if cx < width * 0.3:
        return "left"
    elif cx < width * 0.7:
        return "center"
    return "right"


def _v_position(box: dict, height: int) -> str:
    cy = (box["y1"] + box["y2"]) / 2
    if cy < height * 0.3:
        return "top"
    elif cy < height * 0.7:
        return "mid"
    return "bot"


def _size_label(box: dict) -> str:
    area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
    if area < 5000:
        return "sml"
    elif area < 30000:
        return "med"
    return "lrg"
"""

dst = os.path.expandvars(r"%USERPROFILE%\Desktop\KoutuTools\auto_psd_cutout\src\detector.py")
with open(dst, "w", encoding="utf-8") as f:
    f.write(content)
print("detector.py written OK")
import ast
with open(dst, "r", encoding="utf-8") as f:
    ast.parse(f.read())
print("Syntax OK")
