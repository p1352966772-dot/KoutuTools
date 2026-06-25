from __future__ import annotations
import json
from pathlib import Path
from typing import Any

_JSX = '#target photoshop\napp.displayDialogs = DialogModes.NO;\napp.preferences.rulerUnits = Units.PIXELS;\n\nvar DATA = __JSON_DATA__;\n\nfunction px(v) {\n    return UnitValue(v, "px");\n}\n\nfunction selectRegion(doc, x1, y1, x2, y2) {\n    doc.selection.deselect();\n    var sel = [\n        [px(x1), px(y1)],\n        [px(x2), px(y1)],\n        [px(x2), px(y2)],\n        [px(x1), px(y2)]\n    ];\n    doc.selection.select(sel, SelectionType.REPLACE, 0, false);\n}\n\n// Open RGBA source\nvar srcFile = new File(DATA.rgba_image);\nif (!srcFile.exists) {\n    throw new Error("RGBA source not found: " + DATA.rgba_image);\n}\nvar doc = app.open(srcFile);\n\n// Name source layer (hidden at end)\nvar srcLayer = doc.artLayers[0];\nsrcLayer.name = "zz_source_hidden";\n\n// Create element layers from source\nfor (var g = 0; g < DATA.groups.length; g++) {\n    var group = DATA.groups[g];\n    var layerSet = doc.layerSets.add();\n    layerSet.name = group.name;\n\n    var groupBoxes = [];\n    for (var b = 0; b < DATA.boxes.length; b++) {\n        if (DATA.boxes[b].group_id == group.id) {\n            groupBoxes.push(DATA.boxes[b]);\n        }\n    }\n    groupBoxes.sort(function(a, b) { return a.x1 - b.x1; });\n\n    for (var i = 0; i < groupBoxes.length; i++) {\n        var item = groupBoxes[i];\n\n        // Select region on source layer\n        app.activeDocument = doc;\n        app.activeDocument.activeLayer = srcLayer;\n        selectRegion(doc, item.x1, item.y1, item.x2, item.y2);\n        doc.selection.copy();\n\n        // Paste into same doc WITH active selection → goes to exact position\n        doc.paste();\n\n        doc.activeLayer.name = item.name;\n        doc.activeLayer.move(layerSet, ElementPlacement.PLACEATEND);\n    }\n}\n\n// Hide source layer\nsrcLayer.visible = false;\n\n// Save PSD\nvar psdFile = new File(DATA.psd_path);\nvar psdOpts = new PhotoshopSaveOptions();\npsdOpts.layers = true;\npsdOpts.alphaChannels = true;\ndoc.saveAs(psdFile, psdOpts, true, Extension.LOWERCASE);\ndoc.close(SaveOptions.DONOTSAVECHANGES);\n\n'


def generate_jsx(
    detect_result: dict[str, Any],
    source_image: Path,
    output_path: Path,
    psd_path: Path,
    config: dict[str, Any],
    rgba_image: Path | None = None,
) -> Path:
    data = {
        "rgba_image": str(rgba_image.resolve()) if rgba_image else "",
        "canvas_width": detect_result.get("canvas_width", 0),
        "canvas_height": detect_result.get("canvas_height", 0),
        "groups": detect_result.get("groups", []),
        "boxes": detect_result.get("boxes", []),
        "psd_path": str(psd_path.resolve()),
    }

    js = json.dumps(data, ensure_ascii=False, indent=2)
    jsx = _JSX.replace("__JSON_DATA__", js)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(jsx, encoding="utf-8-sig")
    return output_path
