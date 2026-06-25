from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def remove_label_text_by_rules(image_bgr: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    cleanup_config = config.get("label_text_cleanup", {})
    if not bool(cleanup_config.get("enabled", False)):
        return image_bgr

    detect_config = config.get("detect", {})
    full_page = bool(cleanup_config.get("full_page", True))
    region_left_width = int(cleanup_config.get("region_left_width", detect_config.get("ignore_left", 100) + 70))
    region_top_height = int(cleanup_config.get("region_top_height", 45))
    dark_threshold = int(cleanup_config.get("dark_threshold", 120))
    saturation_max = int(cleanup_config.get("saturation_max", 80))
    max_component_area = int(cleanup_config.get("max_component_area", 260))
    max_component_width = int(cleanup_config.get("max_component_width", 36))
    max_component_height = int(cleanup_config.get("max_component_height", 26))
    protect_saturation_min = int(cleanup_config.get("protect_saturation_min", 90))
    protect_color_radius = max(0, int(cleanup_config.get("protect_color_radius", 8)))
    dilate = max(0, int(cleanup_config.get("dilate", 1)))

    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    low_saturation_dark = ((gray <= dark_threshold) & (hsv[:, :, 1] <= saturation_max)).astype(np.uint8) * 255
    cleanup_region = np.full((height, width), 255, dtype=np.uint8) if full_page else np.zeros((height, width), dtype=np.uint8)

    if not full_page and region_left_width > 0:
        cleanup_region[:, : min(width, region_left_width)] = 255
    if not full_page and region_top_height > 0:
        cleanup_region[: min(height, region_top_height), :] = 255

    candidate_mask = cv2.bitwise_and(low_saturation_dark, cleanup_region)
    color_mask = ((hsv[:, :, 1] >= protect_saturation_min) & (gray < 250)).astype(np.uint8) * 255
    if protect_color_radius > 0:
        protect_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (protect_color_radius * 2 + 1, protect_color_radius * 2 + 1),
        )
        color_mask = cv2.dilate(color_mask, protect_kernel, iterations=1)

    remove_mask = np.zeros_like(candidate_mask)
    total_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)

    for label in range(1, total_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        comp_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        comp_width = int(stats[label, cv2.CC_STAT_WIDTH])
        if area > max_component_area:
            continue
        if comp_width > max_component_width:
            continue
        if comp_height > max_component_height:
            continue
        component_mask = labels == label
        if np.any(color_mask[component_mask] > 0):
            continue
        remove_mask[component_mask] = 255

    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate * 2 + 1, dilate * 2 + 1))
        remove_mask = cv2.dilate(remove_mask, kernel, iterations=1)

    cleaned = image_bgr.copy()
    cleaned[remove_mask > 0] = (255, 255, 255)
    return cleaned