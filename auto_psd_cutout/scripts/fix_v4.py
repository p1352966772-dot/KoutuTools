import sys, os
sys.stdout.reconfigure(encoding="utf-8")

with open(os.path.expandvars(r"%USERPROFILE%\Desktop\KoutuTools\auto_psd_cutout\src\detector.py"), "r", encoding="utf-8") as f:
    code = f.read()

old_func = """def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    # Morphological close to fill small holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Flood-fill holes not connected to border
    h, w = closed.shape
    flood = closed.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    flood = cv2.bitwise_not(flood)
    hole_filled = cv2.bitwise_or(closed, flood)

    # Optional: convex hull fill for large connected components
    total, labels, stats, _ = cv2.connectedComponentsWithStats(hole_filled, connectivity=8)
    hull_mask = np.zeros_like(hole_filled)
    for label in range(1, total):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 500:
            continue
        comp_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 5:
                hull = cv2.convexHull(cnt)
                cv2.drawContours(hull_mask, [hull], -1, 255, -1)

    result = cv2.bitwise_or(hole_filled, hull_mask)
    return result"""

new_func = """def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
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
    return result"""

code = code.replace(old_func, new_func)

with open(os.path.expandvars(r"%USERPROFILE%\Desktop\KoutuTools\auto_psd_cutout\src\detector.py"), "w", encoding="utf-8") as f:
    f.write(code)

import ast
ast.parse(code)
print("Fixed OK")
