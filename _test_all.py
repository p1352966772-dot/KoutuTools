from pathlib import Path
from auto_psd_cutout.src.grid_cutter import SmartGridSplitter
import json

input_dir = Path("auto_psd_cutout/input")
images = sorted(input_dir.glob("*.jpg"))
for img_path in images:
    print(f"\n=== {img_path.name} ===")
    splitter = SmartGridSplitter()
    boxes, clean = splitter.split(str(img_path))
    rows = set(b["row"] for b in boxes)
    print(f"  bg_color: {splitter._bg_color_rgb}")
    print(f"  boxes: {len(boxes)}, rows: {len(rows)}")
    for b in boxes[:6]:
        print(f"    Row{b['row']} Col{b['col']}: ({b['x']},{b['y']}) {b['w']}x{b['h']}")
    if len(boxes) == 1:
        b = boxes[0]
        print(f"  *** ONLY 1 box: ({b['x']},{b['y']}) {b['w']}x{b['h']}")
    elif len(boxes) <= 3:
        print(f"  *** Few boxes!")
