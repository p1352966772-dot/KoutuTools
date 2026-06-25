from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def detect_ui_elements(image_bgr: np.ndarray, config: dict[str, Any], rmbg_prob_map: np.ndarray | None = None) -> dict[str, Any]:
    detect_config = config.get("detect", {})
    white_threshold = int(detect_config.get("white_threshold", 225))
    adaptive_block = int(detect_config.get("adaptive_block", 31))
    adaptive_c = int(detect_config.get("adaptive_c", 2))
    canny_low = int(detect_config.get("canny_low", 50))
    canny_high = int(detect_config.get("canny_high", 150))
    l1_row_kernel = int(detect_config.get("l1_row_kernel", 30))
    row_gap_min = int(detect_config.get("row_gap_min", 12))
    sm_kernel = int(detect_config.get("sm_kernel", 5))
    md_kernel = int(detect_config.get("md_kernel", 13))
    lg_kernel = int(detect_config.get("lg_kernel", 25))
    min_area_ratio = float(detect_config.get("min_area_ratio", 0.0005))
    min_element_size = int(detect_config.get("min_element_size", 10))
    max_aspect_ratio = float(detect_config.get("max_aspect_ratio", 12.0))
    min_aspect_ratio = float(detect_config.get("min_aspect_ratio", 0.05))
    text_max_height_pct = float(detect_config.get("text_max_height_pct", 0.08))
    text_max_area_pct = float(detect_config.get("text_max_area_pct", 0.005))
    text_y_alignment_tol = int(detect_config.get("text_y_alignment_tol", 10))
    text_merge_gap = int(detect_config.get("text_merge_gap", 30))
    merge_h_gap = int(detect_config.get("merge_h_gap", 20))
    merge_y_overlap = float(detect_config.get("merge_y_overlap_threshold", 0.60))
    merge_iou = float(detect_config.get("merge_iou", 0.30))
    min_char_size = int(detect_config.get("min_char_size", 14))
    two_stage_min_size = int(detect_config.get("two_stage_min_size", 150))
    scoring_config = config.get("rmbg_scoring", {})
    scoring_enabled = bool(scoring_config.get("enabled", False))
    edge_weight = float(scoring_config.get("edge_weight", 0.3))
    area_weight = float(scoring_config.get("area_weight", 0.15))
    shape_weight = float(scoring_config.get("shape_weight", 0.15))
    align_weight = float(scoring_config.get("align_weight", 0.15))
    rmbg_weight = float(scoring_config.get("rmbg_weight", 0.25))
    score_threshold = float(scoring_config.get("score_threshold", 0.35))

    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    total_px = width * height

    # ===== Step 1: Multi-mask fusion =====
    mask_fused = _generate_fused_mask(gray, white_threshold, adaptive_block, adaptive_c, canny_low, canny_high)

    # ===== Step 2: Multi-scale morphology =====
    mask_multi = _multi_scale_morphology(mask_fused, sm_kernel, md_kernel, lg_kernel)

    # ===== Step 3: Hole filling =====
    mask_filled = _fill_mask_holes(mask_multi)

    # ===== Step 4: Row detection on filled mask =====
    l1_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, l1_row_kernel))
    l1_merged = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, l1_kernel)
    h_proj = np.count_nonzero(l1_merged, axis=1)
    row_boundaries = _projection_split(h_proj, row_gap_min, 2, height)

    # ===== Step 5: Per-row detection (CC + text/icon merge) =====
    all_boxes: list[dict] = []
    for ry1, ry2 in row_boundaries:
        if ry2 - ry1 < min_element_size:
            continue
        row_boxes = _process_row(mask_filled, ry1, ry2, width, height, total_px, detect_config)
        if row_boxes:
            all_boxes.extend(row_boxes)

    if not all_boxes:
        return {"boxes": [], "groups": []}

    # ===== Step 6: Two-stage detection for large bbox refinement =====
    extra_boxes = _two_stage_detection(all_boxes, mask_filled, gray, width, height, detect_config)
    if extra_boxes:
        all_boxes.extend(extra_boxes)

    # ===== Step 7: Aggressive merge =====
    all_boxes = _merge_aggressive(all_boxes, merge_h_gap, merge_y_overlap, merge_iou)

    # ===== Step 8: Character suppression (merged text blocks) =====
    all_boxes = _suppress_characters(all_boxes, min_char_size, max_aspect_ratio)

    # ===== Step 9: Noise filtering (low bar—recall first) =====
    all_boxes = _filter_noise(all_boxes, total_px, min_area_ratio, min_element_size, max_aspect_ratio, min_aspect_ratio)

    if scoring_enabled and rmbg_prob_map is not None and all_boxes:
        all_boxes = _score_bboxes(
            all_boxes, gray, rmbg_prob_map, width, height,
            edge_weight, area_weight, shape_weight, align_weight, rmbg_weight, score_threshold,
        )

    if not all_boxes:
        return {"boxes": [], "groups": []}

    # ===== Step 10: Row clustering =====
    grouped_boxes, groups = _cluster_rows(all_boxes, row_gap_min * 2, mask_filled)

    # ===== Step 11: Semantic naming =====
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


# ============================================================
# Mask Generation
# ============================================================

def _generate_fused_mask(gray: np.ndarray, white_th: int, adapt_block: int, adapt_c: int, canny_low: int, canny_high: int) -> np.ndarray:
    h, w = gray.shape
    # Mask 1a: White threshold (foreground = non-white)
    m_white = (gray < white_th).astype(np.uint8) * 255

    # Mask 1b: Adaptive threshold
    block = adapt_block if adapt_block % 2 == 1 else adapt_block + 1
    block = max(3, min(block, min(h, w) - 2))
    m_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, adapt_c)

    # Mask 1c: Otsu
    _, m_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Mask 2: Canny edges
    m_canny = cv2.Canny(gray, canny_low, canny_high)
    ce = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m_canny = cv2.dilate(m_canny, ce, iterations=1)

    # Fuse: OR all masks
    fused = cv2.bitwise_or(m_white, m_adaptive)
    fused = cv2.bitwise_or(fused, m_otsu)
    fused = cv2.bitwise_or(fused, m_canny)

    return fused


def _multi_scale_morphology(mask: np.ndarray, sm: int, md: int, lg: int) -> np.ndarray:
    # Use small kernel only to preserve fine details
    # Medium/large kernels cause over-merging of nearby elements
    if sm >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sm, sm))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k)
        return cv2.bitwise_or(mask, opened)
    return mask


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    # Morphological close to fill small holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Flood-fill interior holes (connected to border remains 0)
    h, w = closed.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:h + 1, 1:w + 1] = closed
    fill_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(padded, fill_mask, (0, 0), 255)
    interior = padded[1:h + 1, 1:w + 1]
    holes = cv2.bitwise_not(interior) & closed
    result = cv2.bitwise_or(closed, holes)
    return result


# ============================================================
# Per-row processing (preserved from v3 with multi-mask)
# ============================================================

def _process_row(fg_mask: np.ndarray, ry1: int, ry2: int, img_w: int, img_h: int, total_px: int, cfg: dict) -> list[dict]:
    min_comp = int(cfg.get("min_element_size", 10))
    row_slice = fg_mask[ry1:ry2, :].copy()
    if row_slice.size == 0:
        return []

    total, labels, stats, _ = cv2.connectedComponentsWithStats(row_slice, connectivity=8)

    comps: list[dict] = []
    for label in range(1, total):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = ry1 + int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if cw < 2 or ch < 2:
            continue
        comps.append({
            "id": 0, "group_id": 0,
            "x": cx, "y": cy, "w": cw, "h": ch,
            "x1": cx, "y1": cy, "x2": cx + cw, "y2": cy + ch,
            "area": cw * ch, "name": "",
        })

    if not comps:
        return []
    comps.sort(key=lambda c: c["x1"])

    is_text_arr = [_is_likely_text(c, img_h, total_px, cfg) for c in comps]
    text_comps = [c for i, c in enumerate(comps) if is_text_arr[i]]
    icon_comps = [c for i, c in enumerate(comps) if not is_text_arr[i]]

    text_blocks = _merge_text_components(text_comps, cfg) if text_comps else []
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
    if w < 2 or h < 2:
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
    gap = int(cfg.get("merge_h_gap", 20))
    y_overlap = float(cfg.get("merge_y_overlap_threshold", 0.60))
    iou_th = float(cfg.get("merge_iou", 0.30))
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
            if bx1 > ax2:
                hg = bx1 - ax2
                if hg <= h_gap:
                    return True
            elif ax1 > bx2:
                hg = ax1 - bx2
                if hg <= h_gap:
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


# ============================================================
# Two-stage detection
# ============================================================

def _two_stage_detection(boxes: list[dict], mask: np.ndarray, gray: np.ndarray, img_w: int, img_h: int, cfg: dict) -> list[dict]:
    min_size = int(cfg.get("two_stage_min_size", 150))
    extra: list[dict] = []
    for box in boxes:
        bw = box["x2"] - box["x1"]
        bh = box["y2"] - box["y1"]
        if bw < min_size or bh < min_size:
            continue

        x1 = max(0, box["x1"] - 5)
        y1 = max(0, box["y1"] - 5)
        x2 = min(img_w, box["x2"] + 5)
        y2 = min(img_h, box["y2"] + 5)
        crop_mask = mask[y1:y2, x1:x2].copy()
        crop_gray = gray[y1:y2, x1:x2].copy()
        if crop_mask.size == 0:
            continue

        # Local adaptive threshold on the gray crop
        ch, cw = crop_gray.shape
        block = min(21, max(3, min(ch, cw) - 2))
        if block % 2 == 0:
            block += 1
        try:
            local_mask = cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 3)
        except Exception:
            local_mask = crop_mask

        # Combine with existing mask
        local_combined = cv2.bitwise_or(crop_mask, local_mask)

        # Close small gaps
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        local_combined = cv2.morphologyEx(local_combined, cv2.MORPH_CLOSE, k)

        total, labels, stats, _ = cv2.connectedComponentsWithStats(local_combined, connectivity=8)
        for label in range(1, total):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 30:
                continue
            cx = int(stats[label, cv2.CC_STAT_LEFT]) + x1
            cy = int(stats[label, cv2.CC_STAT_TOP]) + y1
            cw = int(stats[label, cv2.CC_STAT_WIDTH])
            ch = int(stats[label, cv2.CC_STAT_HEIGHT])
            if cw < 5 or ch < 5:
                continue

            # Only keep if significantly smaller than parent
            parent_area = bw * bh
            comp_area = cw * ch
            if comp_area > parent_area * 0.7:
                continue

            # Only keep if NOT already covered by an existing box
            already_covered = False
            for ob in boxes:
                ox1, oy1 = max(cx, ob["x1"]), max(cy, ob["y1"])
                ox2, oy2 = min(cx + cw, ob["x2"]), min(cy + ch, ob["y2"])
                ow = max(0, ox2 - ox1)
                oh = max(0, oy2 - oy1)
                if ow * oh > comp_area * 0.5:
                    already_covered = True
                    break

            if not already_covered:
                extra.append({
                    "id": 0, "group_id": 0,
                    "x": cx, "y": cy, "w": cw, "h": ch,
                    "x1": cx, "y1": cy, "x2": cx + cw, "y2": cy + ch,
                    "area": cw * ch, "name": "",
                })

    return extra


# ============================================================
# Aggressive merge
# ============================================================

def _merge_aggressive(boxes: list[dict], h_gap: int, y_overlap: float, iou_th: float) -> list[dict]:
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
                if _aggressive_should_merge(a, b, h_gap, y_overlap, iou_th):
                    a = _union_box(a, b)
                    used[j] = True
                    changed = True
            new_list.append(a)
            used[i] = True
        merged = new_list
    return merged


def _aggressive_should_merge(a: dict, b: dict, h_gap: int, y_overlap: float, iou_th: float) -> bool:
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ox1 = max(ax1, bx1); oy1 = max(ay1, by1)
    ox2 = min(ax2, bx2); oy2 = min(ay2, by2)
    ow = max(0, ox2 - ox1); oh = max(0, oy2 - oy1)
    oarea = ow * oh
    aarea = (ax2 - ax1) * (ay2 - ay1)
    barea = (bx2 - bx1) * (by2 - by1)
    iou = oarea / max(1, aarea + barea - oarea)

    # IoU > threshold
    if iou > iou_th:
        return True

    # Containment: one fully inside the other
    if ax1 <= bx1 and ay1 <= by1 and ax2 >= bx2 and ay2 >= by2:
        return True
    if bx1 <= ax1 and by1 <= ay1 and bx2 >= ax2 and by2 >= ay2:
        return True

    # Overlap > 60% of the smaller box
    if oarea > 0:
        min_area = min(aarea, barea)
        if min_area > 0 and oarea / min_area >= y_overlap:
            return True

    # Horizontal gap + same row
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


# ============================================================
# Character suppression
# ============================================================

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


# ============================================================
# Utilities
# ============================================================

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


def _cluster_rows(
    boxes: list[dict],
    row_tol: int,
    mask_filled: np.ndarray | None = None,
    col_gap_min: int = 5,
) -> tuple[list[dict], list[dict]]:
    """两阶段聚类：先分行（Y 范围交叠），再行内竖缝检测切分元素。

    分行策略（解决高低差问题）：
      同一行的元素可能高低不平（白底图上常见），用两个条件共同判断：
      1. 交叠条件：Y 范围交叠比例 >= 30%（你上我下也算同一行）
      2. 距离条件：Y 中心点距离 <= row_tol（挨得近也算同一行）

    行内竖缝切分：
      在每个行的 Y 区域内，计算 mask 的竖直投影（列方向），
      找到内容为零的间隙，用间隙位置为元素标注列编号。

    Parameters
    ----------
    row_tol : int
        行聚类容差（像素）。元素 Y 中心距离 <= row_tol 视为同一行。
    mask_filled : np.ndarray | None
        填充后的前景掩码。有值时才启用行内竖缝检测。
    col_gap_min : int
        列间隙最小像素数，小于此值的缝隙忽略（防误切）。
    """
    if not boxes:
        return [], []

    # ── Phase 1: Y-overlap row clustering ─────────────────────
    # 用 Y-overlap + 中心距离双重判断，解决高低差问题
    sorted_boxes = sorted(boxes, key=lambda b: (b["y1"] + b["y2"]) // 2)
    groups: list[list[dict]] = []
    group_bboxes: list[tuple[int, int]] = []  # (y1, y2) of each group

    for box in sorted_boxes:
        y1, y2 = box["y1"], box["y2"]
        bh = max(1, y2 - y1)
        cy = (y1 + y2) // 2
        placed = False

        for gi, (gy1, gy2) in enumerate(group_bboxes):
            gh = max(1, gy2 - gy1)
            # 条件 1: Y 范围交叠 >= 30%
            overlap = max(0, min(y2, gy2) - max(y1, gy1))
            overlap_ratio = overlap / max(1, min(bh, gh))

            # 条件 2: 中心距离 <= row_tol
            gcy = (gy1 + gy2) // 2
            center_dist = abs(cy - gcy)

            if overlap_ratio >= 0.3 or center_dist <= row_tol:
                groups[gi].append(box)
                # 更新该行的总范围（取并集，最高到最低）
                group_bboxes[gi] = (min(y1, gy1), max(y2, gy2))
                placed = True
                break

        if not placed:
            groups.append([box])
            group_bboxes.append((y1, y2))

    # ── Phase 2: Merge small fragmented groups ────────────────
    # 小群组（高度小、元素少）合并到相邻大行
    merged_groups: list[list[dict]] = []
    merged_bboxes: list[tuple[int, int]] = []

    for gboxes, (gy1, gy2) in zip(groups, group_bboxes):
        gh = gy2 - gy1
        is_small = gh < row_tol and len(gboxes) <= 2

        if is_small and merged_groups:
            prev_y1, prev_y2 = merged_bboxes[-1]
            gap = gy1 - prev_y2
            if 0 < gap < row_tol * 1.5:  # 离上一行很近
                merged_groups[-1].extend(gboxes)
                merged_bboxes[-1] = (min(gy1, prev_y1), max(gy2, prev_y2))
                continue

        merged_groups.append(gboxes)
        merged_bboxes.append((gy1, gy2))

    groups = merged_groups
    group_bboxes = merged_bboxes

    # ── Phase 3: Internal column detection within each row ────
    # 在行内检测竖缝间隙，为元素标注列编号
    if mask_filled is not None:
        _, img_w = mask_filled.shape[:2]
        for gi, gboxes in enumerate(groups):
            ry1, ry2 = group_bboxes[gi]
            row_slice = mask_filled[ry1:ry2, :] if ry2 > ry1 else None
            if row_slice is None:
                continue
            v_proj = np.count_nonzero(row_slice, axis=0)

            # 找列间隙（连续 col_gap_min 列及以上内容为零）
            is_empty = (v_proj == 0).astype(np.uint8)
            empty_start = None
            col_gaps: list[int] = []
            for cx in range(img_w):
                if is_empty[cx] and empty_start is None:
                    empty_start = cx
                elif not is_empty[cx] and empty_start is not None:
                    if cx - empty_start >= col_gap_min:
                        col_gaps.append((empty_start + cx) // 2)
                    empty_start = None
            if empty_start is not None and img_w - empty_start >= col_gap_min:
                col_gaps.append((empty_start + img_w) // 2)

            if col_gaps:
                # 给行内元素标注列区间编号
                for box in gboxes:
                    cx = (box["x1"] + box["x2"]) // 2
                    col_idx = sum(1 for gap in col_gaps if cx > gap) + 1
                    box["col"] = col_idx

    # ── Phase 4: Format output ────────────────────────────────
    gdata = sorted(enumerate(groups), key=lambda x: group_bboxes[x[0]][0])
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


# ============================================================
# Dual-channel bbox scoring (Path A + Path B fusion)
# ============================================================

def _score_bboxes(
    boxes: list[dict],
    gray: np.ndarray,
    rmbg_prob: np.ndarray,
    img_w: int,
    img_h: int,
    edge_w: float,
    area_w: float,
    shape_w: float,
    align_w: float,
    rmbg_w: float,
    thresh: float,
) -> list[dict]:
    """Score each bbox using multi-dimensional quality metrics.
    Combines edge strength, area validity, shape consistency,
    alignment score, and RMBG foreground overlap ratio.
    Removes boxes below threshold.
    """
    if not boxes:
        return boxes

    total_px = img_w * img_h
    scores: list[float] = []
    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        w = max(1, x2 - x1)
        h_val = max(1, y2 - y1)
        area = w * h_val

        # 1. Edge strength
        roi_gray = gray[y1:y2, x1:x2]
        if roi_gray.size > 0:
            gx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            edge_strength = float(np.mean(mag)) / 255.0
        else:
            edge_strength = 0.0
        edge_strength = min(1.0, edge_strength * 2.0)

        # 2. Area validity
        area_ratio = area / max(1, total_px)
        if area_ratio < 0.001:
            area_validity = max(0.0, area_ratio / 0.001)
        elif area_ratio > 0.5:
            area_validity = max(0.0, 1.0 - (area_ratio - 0.5) / 0.5)
        else:
            area_validity = 1.0

        # 3. Shape consistency
        ar = max(w, h_val) / max(1, min(w, h_val))
        if ar > 8.0:
            shape_consistency = max(0.0, 1.0 - (ar - 8.0) / 12.0)
        elif ar < 0.15:
            shape_consistency = max(0.0, ar / 0.15)
        else:
            shape_consistency = 1.0

        # 4. Alignment score
        cy = (y1 + y2) / 2.0
        count_in_band = 0
        for ob in boxes:
            ocy = (ob["y1"] + ob["y2"]) / 2.0
            if abs(ocy - cy) <= 20:
                count_in_band += 1
        alignment_score = min(1.0, count_in_band / max(1, len(boxes)) * 3.0)

        # 5. RMBG overlap ratio
        if rmbg_prob is not None and rmbg_prob.shape[:2] == gray.shape:
            roi_prob = rmbg_prob[y1:y2, x1:x2]
            if roi_prob.size > 0:
                rmbg_overlap = float(np.mean(roi_prob))
            else:
                rmbg_overlap = 0.5
        else:
            rmbg_overlap = 0.5

        score = (
            edge_w * edge_strength
            + area_w * area_validity
            + shape_w * shape_consistency
            + align_w * alignment_score
            + rmbg_w * rmbg_overlap
        )
        scores.append(score)

    result: list[dict] = []
    for i, box in enumerate(boxes):
        if scores[i] >= thresh:
            box["score"] = round(scores[i], 4)
            result.append(box)

    if not result:
        sorted_pairs = sorted(zip(boxes, scores), key=lambda x: -x[1])
        for box, sc in sorted_pairs[:3]:
            box["score"] = round(sc, 4)
            if box not in result:
                result.append(box)

    return result
