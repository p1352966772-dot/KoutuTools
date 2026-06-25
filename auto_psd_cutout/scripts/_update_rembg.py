path = r'C:\Users\Administrator.SK-20241009ZRBE\Desktop\KoutuTools\auto_psd_cutout\src\rembg_utils.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the get_foreground_probability function
old_func_start = "def get_foreground_probability"
old_func_end = "    return prob"

# Find the function boundaries more carefully
import re
# Match the entire function body
pattern = r"def get_foreground_probability\(image_bgr: np\.ndarray\) -> np\.ndarray:.*?(?=\n\ndef|\Z)"
match = re.search(pattern, content, re.DOTALL)

if match:
    new_func = '''def get_foreground_probability(image_bgr: np.ndarray) -> np.ndarray:
    """Returns float32 foreground probability map (0.0-1.0) for bbox scoring.

    Uses rembg's alpha channel as foreground probability.
    Tries BRIA RMBG-1.4 first (if model available), falls back to U2-Net.
    Used ONLY for bbox scoring (Path B), NEVER for bbox generation.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Try BRIA first (better foreground estimation)
    try:
        session = _load_bria_session()
        rgba = session.predict(pil_img)
        if rgba is not None:
            rgba_np = np.array(rgba, dtype=np.float32)
            prob = rgba_np[:, :, 3] / 255.0
            return np.clip(prob, 0.0, 1.0)
    except Exception:
        pass

    # Fallback to U2-Net (already cached from previous runs)
    try:
        session = _load_u2net_session()
        rgba = session.predict(pil_img)
        if rgba is not None:
            rgba_np = np.array(rgba, dtype=np.float32)
            prob = rgba_np[:, :, 3] / 255.0
            return np.clip(prob, 0.0, 1.0)
    except Exception as exc:
        print(f"RMBG probability map failed: {exc}")

    # Last resort: uniform neutral probability
    h, w = image_bgr.shape[:2]
    return np.ones((h, w), dtype=np.float32) * 0.5
'''
    content = content[:match.start()] + new_func + content[match.end():]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('OK - function replaced')
else:
    print('Pattern not found')
