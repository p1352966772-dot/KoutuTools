"""测试：白底阈值法 vs BRIA RMBG-1.4 对 UI 图的抠图效果对比"""
import sys
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

BASE = Path(r"C:\Users\Administrator.SK-20241009ZRBE\Desktop\KoutuTools\auto_psd_cutout")
sys.path.insert(0, str(BASE / "src"))

from rembg_utils import get_bria14_alpha

INPUT_DIR = BASE / "input"
OUTPUT_DIR = BASE / "output_test_matting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

images = sorted(INPUT_DIR.glob("*.jpg"))

for img_path in images:
    print(f"\n=== {img_path.name} ===")
    img_bgr = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ---- 方法1: 白底阈值法 (THRESH_BINARY_INV + 轻微模糊边缘) ----
    _, bin_mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    # 闭运算填充小孔
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    # 边缘轻微模糊让过渡自然
    bin_alpha = cv2.GaussianBlur(bin_mask, (3, 3), 0)
    
    print(f"  白底阈值法:")
    print(f"    前景 (alpha>200)：{(bin_alpha>200).sum()/bin_alpha.size*100:.1f}%")
    print(f"    半透明 (20<alpha<200)：{((bin_alpha>20)&(bin_alpha<200)).sum()/bin_alpha.size*100:.1f}%")
    
    rgba_thresh = np.dstack([rgb, bin_alpha])
    Image.fromarray(rgba_thresh, "RGBA").save(str(OUTPUT_DIR / f"{img_path.stem}_threshold.png"))
    Image.fromarray(bin_alpha, "L").save(str(OUTPUT_DIR / f"{img_path.stem}_threshold_alpha.png"))

    # ---- 方法2: BRIA RMBG-1.4 ----
    t0 = time.time()
    bria_alpha = get_bria14_alpha(img_bgr)
    t_bria = time.time() - t0
    
    print(f"  BRIA RMBG-1.4 ({t_bria:.1f}s):")
    print(f"    前景 (alpha>200)：{(bria_alpha>200).sum()/bria_alpha.size*100:.1f}%")
    print(f"    半透明 (20<alpha<200)：{((bria_alpha>20)&(bria_alpha<200)).sum()/bria_alpha.size*100:.1f}%")
    
    rgba_bria = np.dstack([rgb, bria_alpha])
    Image.fromarray(rgba_bria, "RGBA").save(str(OUTPUT_DIR / f"{img_path.stem}_bria.png"))
    Image.fromarray(bria_alpha, "L").save(str(OUTPUT_DIR / f"{img_path.stem}_bria_alpha.png"))

    print(f"  结果已保存：{img_path.stem}_threshold.png + {img_path.stem}_bria.png")
