from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class CutoutRecord:
    id: int
    name: str
    path: Path
    box: dict[str, int]
    layer_x: int
    layer_y: int
    alpha_x: int
    alpha_y: int
    alpha_w: int
    alpha_h: int
    rgba: np.ndarray


def merge_cutouts_after_cutout(cutouts: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    merge_config = config.get("post_cutout_merge", {})
    if not bool(merge_config.get("enabled", True)):
        return cutouts

    records: list[CutoutRecord] = []
    for cutout in cutouts:
        rgba = np.array(Image.open(cutout["path"]).convert("RGBA"))
        alpha_bounds = _alpha_bounds(rgba)
        if alpha_bounds is None:
            continue
        alpha_x, alpha_y, alpha_w, alpha_h = alpha_bounds
        records.append(
            CutoutRecord(
                id=int(cutout["id"]),
                name=str(cutout["name"]),
                path=Path(cutout["path"]),
                box=dict(cutout["box"]),
                layer_x=int(cutout["layer_x"]),
                layer_y=int(cutout["layer_y"]),
                alpha_x=int(alpha_x),
                alpha_y=int(alpha_y),
                alpha_w=int(alpha_w),
                alpha_h=int(alpha_h),
                rgba=rgba,
            )
        )

    if len(records) < 2:
        return cutouts

    max_gap = int(merge_config.get("max_gap", 65))
    min_vertical_overlap = float(merge_config.get("min_vertical_overlap", 0.45))
    max_height_ratio = float(merge_config.get("max_height_ratio", 1.9))
    max_merged_width_ratio = float(merge_config.get("max_merged_width_ratio", 3.2))

    groups = _group_records(records, max_gap, min_vertical_overlap, max_height_ratio, max_merged_width_ratio)
    if len(groups) == len(records):
        return cutouts

    merged_cutouts: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups, start=1):
        if len(group) == 1:
            record = group[0]
            merged_cutouts.append(
                {
                    "id": record.id,
                    "name": record.name,
                    "path": record.path.resolve(),
                    "box": record.box,
                    "layer_x": record.layer_x,
                    "layer_y": record.layer_y,
                }
            )
            continue

        merged_record = _compose_group(group, group_index, records[0].path.parent)
        merged_cutouts.append(merged_record)

    merged_cutouts = sorted(merged_cutouts, key=lambda item: (item["box"]["y"], item["box"]["x"]))
    return merged_cutouts


def _group_records(
    records: list[CutoutRecord],
    max_gap: int,
    min_vertical_overlap: float,
    max_height_ratio: float,
    max_merged_width_ratio: float,
) -> list[list[CutoutRecord]]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            if _should_merge(records[left_index], records[right_index], max_gap, min_vertical_overlap, max_height_ratio, max_merged_width_ratio):
                union(left_index, right_index)

    groups: dict[int, list[CutoutRecord]] = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)
    return list(groups.values())


def _should_merge(
    left: CutoutRecord,
    right: CutoutRecord,
    max_gap: int,
    min_vertical_overlap: float,
    max_height_ratio: float,
    max_merged_width_ratio: float,
) -> bool:
    left_visible = _visible_box(left)
    right_visible = _visible_box(right)
    if left_visible[0] > right_visible[0]:
        left, right = right, left
        left_visible, right_visible = right_visible, left_visible

    horizontal_gap = right_visible[0] - left_visible[2]
    if horizontal_gap < 0 or horizontal_gap > max_gap:
        return False

    overlap_y = max(0, min(left_visible[3], right_visible[3]) - max(left_visible[1], right_visible[1]))
    min_height = max(1, min(left_visible[3] - left_visible[1], right_visible[3] - right_visible[1]))
    if overlap_y / min_height < min_vertical_overlap:
        return False

    left_h = left_visible[3] - left_visible[1]
    right_h = right_visible[3] - right_visible[1]
    height_ratio = max(left_h, right_h) / max(1, min(left_h, right_h))
    if height_ratio > max_height_ratio:
        return False

    merged_width = right_visible[2] - left_visible[0]
    merged_height = max(left_visible[3], right_visible[3]) - min(left_visible[1], right_visible[1])
    if merged_width / max(1, merged_height) > max_merged_width_ratio:
        return False

    return True


def _compose_group(group: list[CutoutRecord], group_index: int, items_dir: Path) -> dict[str, Any]:
    canvas_x1 = min(record.layer_x - record.alpha_x for record in group)
    canvas_y1 = min(record.layer_y - record.alpha_y for record in group)
    canvas_x2 = max((record.layer_x - record.alpha_x) + record.rgba.shape[1] for record in group)
    canvas_y2 = max((record.layer_y - record.alpha_y) + record.rgba.shape[0] for record in group)

    merged_canvas = Image.new("RGBA", (max(1, canvas_x2 - canvas_x1), max(1, canvas_y2 - canvas_y1)), (0, 0, 0, 0))
    for record in group:
        paste_x = (record.layer_x - record.alpha_x) - canvas_x1
        paste_y = (record.layer_y - record.alpha_y) - canvas_y1
        merged_canvas.alpha_composite(Image.fromarray(record.rgba), dest=(paste_x, paste_y))

    merged_rgba = np.array(merged_canvas)
    alpha_bounds = _alpha_bounds(merged_rgba)
    if alpha_bounds is None:
        alpha_x = alpha_y = 0
        alpha_w = merged_rgba.shape[1]
        alpha_h = merged_rgba.shape[0]
    else:
        alpha_x, alpha_y, alpha_w, alpha_h = alpha_bounds

    name = "_".join(record.name for record in group)
    output_path = items_dir / f"{name}_merged_{group_index:03d}.png"
    merged_canvas.save(output_path)

    visible_x1 = canvas_x1 + alpha_x
    visible_y1 = canvas_y1 + alpha_y
    visible_x2 = visible_x1 + alpha_w
    visible_y2 = visible_y1 + alpha_h

    return {
        "id": min(record.id for record in group),
        "name": name,
        "path": output_path.resolve(),
        "box": {
            "id": min(record.id for record in group),
            "x": int(visible_x1),
            "y": int(visible_y1),
            "w": int(visible_x2 - visible_x1),
            "h": int(visible_y2 - visible_y1),
            "x1": int(visible_x1),
            "y1": int(visible_y1),
            "x2": int(visible_x2),
            "y2": int(visible_y2),
            "area": int((visible_x2 - visible_x1) * (visible_y2 - visible_y1)),
        },
        "layer_x": int(visible_x1),
        "layer_y": int(visible_y1),
    }


def _visible_box(record: CutoutRecord) -> tuple[int, int, int, int]:
    x1 = record.layer_x
    y1 = record.layer_y
    x2 = x1 + record.alpha_w
    y2 = y1 + record.alpha_h
    return x1, y1, x2, y2


def _alpha_bounds(rgba: np.ndarray) -> tuple[int, int, int, int] | None:
    alpha = rgba[:, :, 3]
    points = cv2.findNonZero((alpha > 8).astype(np.uint8) * 255)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    return int(x), int(y), int(w), int(h)
