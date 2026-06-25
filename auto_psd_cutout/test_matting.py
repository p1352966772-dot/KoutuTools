import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Add src to path
BASE = Path(r"C:\Users\Administrator.SK-20241009ZRBE\Desktop\KoutuTools\auto_psd_cutout")
sys.path.insert(0, str(BASE / "src"))

from rembg_utils import get_bria14_alpha

INPUT_DIR = BASE / "input"
OUTPUT_DIR = BASE / "output_test_matting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

images = sorted(INPUT_DIR.glob("*.jpg")) + sorted(INPUT_DIR.glob("*.png")) + sorted(INPUT_DIR.glob("*.jpeg"))
print(f"找到 {len(images)} 张图片")
print("=" * 60)

for img_path in images:
    print(f"\n处理：{img_path.name}")
    
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  读取失败，跳过")
        continue
    h, w = img_bgr.shape[:2]
    print(f"  尺寸：{w}x{h}")
    
    # BRIA RMBG-1.4 全图抠图
    alpha = get_bria14_alpha(img_bgr)
    print(f"  alpha mask：{alpha.shape}, dtype={alpha.dtype}")
    print(f"  值范围：[{alpha.min()}, {alpha.max()}]")
    print(f"  前景 (alpha>200)：{(alpha>200).sum()/alpha.size*100:.1f}%")
    print(f"  半透明 (20<alpha<200)：{((alpha>20)&(alpha<200)).sum()/alpha.size*100:.1f}%")
    print(f"  背景 (alpha<=20)：{(alpha<=20).sum()/alpha.size*100:.1f}%")
    
    # 保存全图RGBA
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, alpha])
    out_path = OUTPUT_DIR / f"{img_path.stem}_matting.png"
    Image.fromarray(rgba, "RGBA").save(str(out_path))
    size_kb = out_path.stat().st_size / 1024
    print(f"  已保存：{out_path.name} ({size_kb:.0f} KB)")
    
    # alpha可视化
    alpha_viz = OUTPUT_DIR / f"{img_path.stem}_alpha.png"
    Image.fromarray(alpha, "L").save(str(alpha_viz))
    print(f"  已保存：{alpha_viz.name}")

print("\n" + "=" * 60)
print(f"全部完成！结果保存在：{OUTPUT_DIR}")
