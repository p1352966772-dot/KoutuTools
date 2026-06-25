from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_manifest(
    source_image: Path,
    transparent_map: dict[str, Any],
    image_size: tuple[int, int],
    cutouts: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    width, height = image_size
    manifest = {
        "source_image": str(source_image.resolve()),
        "transparent_map": str(Path(transparent_map["path"]).resolve()),
        "transparent_map_layer_x": int(transparent_map.get("layer_x", 0)),
        "transparent_map_layer_y": int(transparent_map.get("layer_y", 0)),
        "canvas_width": width,
        "canvas_height": height,
        "items": [],
    }

    for cutout in cutouts:
        box = cutout["box"]
        manifest["items"].append(
            {
                "id": int(cutout["id"]),
                "name": cutout["name"],
                "png": str(Path(cutout["path"]).resolve()),
                "x": int(box["x"]),
                "y": int(box["y"]),
                "layer_x": int(cutout.get("layer_x", box["x"])),
                "layer_y": int(cutout.get("layer_y", box["y"])),
                "w": int(box["w"]),
                "h": int(box["h"]),
                "x1": int(box["x1"]),
                "y1": int(box["y1"]),
                "x2": int(box["x2"]),
                "y2": int(box["y2"]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest
