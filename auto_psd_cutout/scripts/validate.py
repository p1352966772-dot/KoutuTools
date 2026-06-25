import sys, os, json, glob, re
sys.stdout.reconfigure(encoding="utf-8")
base = os.path.expandvars(r"%USERPROFILE%\Desktop\KoutuTools\auto_psd_cutout\output")

for jsx_path in sorted(glob.glob(os.path.join(base, "*", "build_psd.jsx"))):
    name = os.path.basename(os.path.dirname(jsx_path))
    with open(jsx_path, encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"var DATA = ({.*?});", content, re.DOTALL)
    if not m:
        print(f"{name}: could not parse JSX")
        continue
    data = json.loads(m.group(1))
    boxes = data["boxes"]
    groups = data["groups"]

    contain = 0
    overlap = 0
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes):
            if i >= j: continue
            if (a["x1"] <= b["x1"] and a["y1"] <= b["y1"] and a["x2"] >= b["x2"] and a["y2"] >= b["y2"]):
                contain += 1
            elif (b["x1"] <= a["x1"] and b["y1"] <= a["y1"] and b["x2"] >= a["x2"] and b["y2"] >= a["y2"]):
                contain += 1
            ox1 = max(a["x1"], b["x1"]); oy1 = max(a["y1"], b["y1"])
            ox2 = min(a["x2"], b["x2"]); oy2 = min(a["y2"], b["y2"])
            oa = max(0, ox2 - ox1) * max(0, oy2 - oy1)
            if oa > 0:
                ma = min((a["x2"] - a["x1"]) * (a["y2"] - a["y1"]), (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
                if oa / ma > 0.20:
                    overlap += 1

    print(f"{name}: {len(groups)} rows, {len(boxes)} boxes, contain={contain}, overlap_gt20pct={overlap}")
    for g in groups:
        gboxes = [b for b in boxes if b["group_id"] == g["id"]]
        print(f"  {g['name']}: {len(gboxes)} elements")
