import sys, os
sys.stdout.reconfigure(encoding="utf-8")

content = """from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_jsx(
    detect_result: dict[str, Any],
    source_image: Path,
    output_path: Path,
    psd_path: Path,
    config: dict[str, Any],
) -> Path:
    \"\"\"Generate JSX that uses Photoshop selection API (copy from source, no external PNGs).\"\"\"
    ps_config = config.get(\"photoshop\", {})
    data = {
        \"source_image\": str(source_image.resolve()),
        \"canvas_width\": detect_result.get(\"canvas_width\", 0),
        \"canvas_height\": detect_result.get(\"canvas_height\", 0),
        \"groups\": detect_result.get(\"groups\", []),
        \"boxes\": detect_result.get(\"boxes\", []),
        \"psd_path\": str(psd_path.resolve()),
        \"psd_name_suffix\": str(ps_config.get(\"psd_name_suffix\", \"_auto\")),
        \"add_original_layer\": bool(ps_config.get(\"add_original_layer\", False)),
        \"add_box_layers\": bool(ps_config.get(\"add_box_layers\", False)),
        \"box_color\": ps_config.get(\"box_color\", [255, 0, 0]),
        \"box_thickness\": int(ps_config.get(\"box_thickness\", 2)),
    }

    jsx = _build_jsx(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(jsx, encoding=\"utf-8\")
    return output_path


def _build_jsx(data: dict[str, Any]) -> str:
    json_data = json.dumps(data, ensure_ascii=False, indent=2)

    return f\"\"\"#target photoshop
app.displayDialogs = DialogModes.NO;

var DATA = {json_data};

function px(v) {{
    return UnitValue(v, \"px\");
}}

function pt(v) {{
    return UnitValue(v, \"pt\");
}}

function selectRegion(doc, x1, y1, x2, y2) {{
    var sel = [
        [px(x1), px(y1)],
        [px(x2), px(y1)],
        [px(x2), px(y2)],
        [px(x1), px(y2)]
    ];
    doc.selection.select(sel, SelectionType.REPLACE, 0, false);
}}

function makeColor(rgb) {{
    var c = new SolidColor();
    c.rgb.red = rgb[0];
    c.rgb.green = rgb[1];
    c.rgb.blue = rgb[2];
    return c;
}}

function drawBox(doc, item, color, thickness) {{
    selectRegion(doc, item.x1, item.y1, item.x2, item.y2);
    var layer = doc.artLayers.add();
    layer.name = "bbox_" + item.id;
    doc.selection.stroke(color, thickness, StrokeLocation.INSIDE, ColorBlendMode.NORMAL, 100, false);
    doc.selection.deselect();
    return layer;
}}

// Open source image
var srcFile = new File(DATA.source_image);
if (!srcFile.exists) {{
    throw new Error("Source image not found: " + DATA.source_image);
}}
var srcDoc = app.open(srcFile);

// Get source dimensions
var sw = px(srcDoc.width).value;
var sh = px(srcDoc.height).value;
var cw = DATA.canvas_width > 0 ? DATA.canvas_width : sw;
var ch = DATA.canvas_height > 0 ? DATA.canvas_height : sh;

// Create target document
var docName = decodeURI(new File(DATA.psd_path).name).replace(/\\\\.[pP][sS][dD]$/i, \"\");
var targetDoc = app.documents.add(cw, ch, 72, docName, NewDocumentMode.RGB, DocumentFill.TRANSPARENT);
app.activeDocument = targetDoc;

// Track copied layer positions
var boxColor = makeColor(DATA.box_color);

// Process groups
for (var g = 0; g < DATA.groups.length; g++) {{
    var group = DATA.groups[g];
    var groupName = group.name;

    // Create layer set for this row group
    var layerSet = targetDoc.layerSets.add();
    layerSet.name = groupName;

    // Get boxes in this group
    var groupBoxes = [];
    for (var b = 0; b < DATA.boxes.length; b++) {{
        if (DATA.boxes[b].group_id == group.id) {{
            groupBoxes.push(DATA.boxes[b]);
        }}
    }}

    // Sort boxes by x position
    groupBoxes.sort(function(a, b) {{ return a.x1 - b.x1; }});

    for (var i = 0; i < groupBoxes.length; i++) {{
        var item = groupBoxes[i];

        // Switch to source doc, select and copy
        app.activeDocument = srcDoc;
        selectRegion(srcDoc, item.x1, item.y1, item.x2, item.y2);
        srcDoc.selection.copy();

        // Switch to target doc, paste
        app.activeDocument = targetDoc;
        targetDoc.paste();

        var layer = targetDoc.activeLayer;
        layer.name = item.name;

        // Move to correct position (paste positions at original coords by default)
        // Layer may have been pasted at selection position, adjust if needed
        var bounds = layer.bounds;
        var layerX = px(bounds[0].as(\"px\")).value;
        var layerY = px(bounds[1].as(\"px\")).value;
        if (Math.abs(layerX - item.x1) > 1 || Math.abs(layerY - item.y1) > 1) {{
            layer.translate(px(item.x1 - layerX), px(item.y1 - layerY));
        }}

        // Move layer into group
        layer.move(layerSet, ElementPlacement.PLACEATEND);

        // Optional: draw box overlay
        if (DATA.add_box_layers) {{
            app.activeDocument = targetDoc;
            var boxLayer = drawBox(targetDoc, item, boxColor, DATA.box_thickness);
            boxLayer.move(layerSet, ElementPlacement.PLACEATEND);
        }}
    }}
}}

// Optional original reference layer
if (DATA.add_original_layer) {{
    app.activeDocument = srcDoc;
    srcDoc.selection.selectAll();
    srcDoc.selection.copy();
    app.activeDocument = targetDoc;
    targetDoc.paste();
    var refLayer = targetDoc.activeLayer;
    refLayer.name = \"00_original_reference\";
    targetDoc.activeLayer = refLayer;
}}

// Close source doc
srcDoc.close(SaveOptions.DONOTSAVECHANGES);

// Save PSD
var psdFile = new File(DATA.psd_path);
var psdOpts = new PhotoshopSaveOptions();
psdOpts.layers = true;
psdOpts.alphaChannels = true;
targetDoc.saveAs(psdFile, psdOpts, true, Extension.LOWERCASE);
app.activeDocument.close(SaveOptions.DONOTSAVECHANGES);
\"\"\"


"""

dst = os.path.expandvars(r"%USERPROFILE%\Desktop\KoutuTools\auto_psd_cutout\src\photoshop_jsx.py")
with open(dst, "w", encoding="utf-8") as f:
    f.write(content)
print("photoshop_jsx.py written OK")

import ast
with open(dst, "r", encoding="utf-8") as f:
    ast.parse(f.read())
print("Syntax OK")
