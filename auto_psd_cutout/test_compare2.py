"""对比：修复后的 BRIA RMBG-1.4 vs 白底阈值法"""
import sys, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

BASE = Path(r"C:\Users\Administrator.SK-20241009ZRBE\Desktop\KoutuTools\auto_psd_cutout")
sys.path.insert(0, str(BASE / "src"))
from rembg_utils import get_bria14_alpha

INPUT_DIR = BASE / "input"
OUT_DIR = BASE / "output_test_matting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

images = sorted(INPUT_DIR.glob("*.jpg"))
if not images:
    images = sorted(INPUT_DIR.glob("*.png"))

print(f"{'图片':<55} {'方法':<12} {'前景>200':<10} {'半透明':<10} {'透明':<10} {'耗时':<8}")
print("=" * 110)

for img_path in images:
    img_bgr = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    name = img_path.stem[:45]

    # === 方法1: 白底阈值法 ===
    t0 = time.time()
    _, bw = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    thresh_alpha = cv2.GaussianBlur(bw, (3, 3), 0)
    tt = time.time() - t0
    ft = (thresh_alpha > 200).sum() / thresh_alpha.size * 100
    ht = ((thresh_alpha > 20) & (thresh_alpha < 200)).sum() / thresh_alpha.size * 100
    zt = (thresh_alpha <= 20).sum() / thresh_alpha.size * 100
    print(f"{name:<55} {'阈值法':<12} {ft:<8.1f}% {ht:<8.1f}% {zt:<8.1f}% {tt:<.3f}s")
    Image.fromarray(np.dstack([rgb, thresh_alpha]), "RGBA").save(str(OUT_DIR / f"{img_path.stem}_threshold.png"))

    # === 方法2: 修复后的 BRIA RMBG-1.4 ===
    t0 = time.time()
    bria_alpha = get_bria14_alpha(img_bgr)
    tb = time.time() - t0
    fb = (bria_alpha > 200).sum() / bria_alpha.size * 100
    hb = ((bria_alpha > 20) & (bria_alpha < 200)).sum() / bria_alpha.size * 100
    zb = (bria_alpha <= 20).sum() / bria_alpha.size * 100
    print(f"{name:<55} {'BRIA修复版':<12} {fb:<8.1f}% {hb:<8.1f}% {zb:<8.1f}% {tb:<.1f}s")
    Image.fromarray(np.dstack([rgb, bria_alpha]), "RGBA").save(str(OUT_DIR / f"{img_path.stem}_bria.png"))

print("=" * 110)
print(f"结果保存: {OUT_DIR}")
print("查看: xxx_threshold.png = 白底阈值法, xxx_bria.png = BRIA修复版")
