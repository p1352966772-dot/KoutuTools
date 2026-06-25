from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def save_cutouts(image_bgr: np.ndarray, boxes: list[dict[str, int]], items_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    items_dir.mkdir(parents=True, exist_ok=True)
    cutout_config = config.get("cutout", {})
    white_threshold = int(cutout_config.get("white_threshold", 245))
    feather = max(0, int(cutout_config.get("feather", 1)))
    keep_inner_white = bool(cutout_config.get("keep_inner_white", True))

    results: list[dict[str, Any]] = []
    for box in boxes:
        name = f"item_{box['id']:03d}"
        output_path = items_dir / f"{name}.png"
        crop = image_bgr[box["y1"] : box["y2"], box["x1"] : box["x2"]]
        if crop.size == 0:
            print(f"跳过空裁剪区域：{name}")
            continue

        rgba = _make_transparent_cutout(crop, white_threshold, feather, keep_inner_white)
        Image.fromarray(rgba).save(output_path)
        visible_bounds = _alpha_visible_bounds(rgba)
        layer_x = box["x1"]
        layer_y = box["y1"]
        if visible_bounds is not None:
            alpha_x, alpha_y, _, _ = visible_bounds
            layer_x += alpha_x
            layer_y += alpha_y

        results.append(
            {
                "id": box["id"],
                "name": name,
                "path": output_path.resolve(),
                "box": box,
                "layer_x": int(layer_x),
                "layer_y": int(layer_y),
            }
        )

    return results


def save_transparent_map(image_bgr: np.ndarray, output_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    cutout_config = config.get("cutout", {})
    white_threshold = int(cutout_config.get("white_threshold", 245))
    feather = max(0, int(cutout_config.get("feather", 1)))
    keep_inner_white = bool(cutout_config.get("keep_inner_white", True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba = _make_transparent_cutout(image_bgr, white_threshold, feather, keep_inner_white)
    Image.fromarray(rgba).save(output_path)
    visible_bounds = _alpha_visible_bounds(rgba)
    layer_x = 0
    layer_y = 0
    if visible_bounds is not None:
        layer_x, layer_y, _, _ = visible_bounds
    return {"path": output_path.resolve(), "layer_x": int(layer_x), "layer_y": int(layer_y)}


def _make_transparent_cutout(
    crop_bgr: np.ndarray,
    white_threshold: int,
    feather: int,
    keep_inner_white: bool,
) -> np.ndarray:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    white_mask = gray >= white_threshold

    if keep_inner_white:
        background_mask = _edge_connected_mask(white_mask)
    else:
        background_mask = white_mask.astype(np.uint8) * 255

    alpha = np.full(gray.shape, 255, dtype=np.uint8)
    alpha[background_mask > 0] = 0

    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (feather * 2 + 1, feather * 2 + 1), 0)

    return np.dstack([crop_rgb, alpha])


def _alpha_visible_bounds(rgba: np.ndarray) -> tuple[int, int, int, int] | None:
    alpha = rgba[:, :, 3]
    points = cv2.findNonZero((alpha > 8).astype(np.uint8) * 255)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    return int(x), int(y), int(w), int(h)


def _edge_connected_mask(white_mask: np.ndarray) -> np.ndarray:
    height, width = white_mask.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    source = (white_mask.astype(np.uint8) * 255).copy()

    seed_points: list[tuple[int, int]] = []
    for x in range(width):
        if source[0, x] == 255:
            seed_points.append((x, 0))
        if source[height - 1, x] == 255:
            seed_points.append((x, height - 1))
    for y in range(height):
        if source[y, 0] == 255:
            seed_points.append((0, y))
        if source[y, width - 1] == 255:
            seed_points.append((width - 1, y))

    connected = np.zeros_like(source)
    for seed in seed_points:
        x, y = seed
        if source[y, x] != 255 or connected[y, x] == 255:
            continue
        temp = source.copy()
        cv2.floodFill(temp, flood_mask, seed, 128)
        filled = temp == 128
        connected[filled] = 255
        source[filled] = 0
        flood_mask.fill(0)

    return connected
def save_cutouts_from_rembg(
    rembg_rgba: np.ndarray,
    boxes: list[dict[str, int]],
    items_dir: Path,
) -> list[dict[str, Any]]:
    """Cut individual boxes from rembg RGBA result (already transparent)."""
    items_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for box in boxes:
        name = f"item_{box['id']:03d}"
        output_path = items_dir / f"{name}.png"
        crop = rembg_rgba[box["y1"] : box["y2"], box["x1"] : box["x2"]]
        if crop.size == 0:
            print(f"跳过空裁剪区域：{name}")
            continue
        Image.fromarray(crop).save(output_path)
        visible_bounds = _alpha_visible_bounds(crop)
        layer_x = box["x1"]
        layer_y = box["y1"]
        if visible_bounds is not None:
            alpha_x, alpha_y, _, _ = visible_bounds
            layer_x += alpha_x
            layer_y += alpha_y
        results.append({
            "id": box["id"],
            "name": name,
            "path": output_path.resolve(),
            "box": box,
            "layer_x": int(layer_x),
            "layer_y": int(layer_y),
        })
    return results


def save_transparent_map_from_rembg(
    rembg_rgba: np.ndarray,
    output_path: Path,
) -> dict[str, Any]:
    """Save the full rembg RGBA result as transparent map PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rembg_rgba).save(output_path)
    visible_bounds = _alpha_visible_bounds(rembg_rgba)
    layer_x = 0
    layer_y = 0
    if visible_bounds is not None:
        layer_x, layer_y, _, _ = visible_bounds
    return {"path": output_path.resolve(), "layer_x": int(layer_x), "layer_y": int(layer_y)}
