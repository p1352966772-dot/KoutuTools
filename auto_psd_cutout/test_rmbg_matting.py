"""按照 rmbg.dev 的实际处理流程复现：直接拉伸 + mean=0.5 + 1px blur"""
import sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

BASE = Path(r"C:\Users\Administrator.SK-20241009ZRBE\Desktop\KoutuTools\auto_psd_cutout")
INPUT_DIR = BASE / "input"
OUTPUT_DIR = BASE / "output_test_matting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
model.eval().to(device)
print(f"模型加载到: {device}")

images = sorted(INPUT_DIR.glob("*.jpg")) + sorted(INPUT_DIR.glob("*.png"))
print(f"找到 {len(images)} 张图片\n")

for img_path in images:
    print(f"{'='*60}\n{img_path.name}")
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    t0 = time.time()

    # ===== rmbg.dev 预处理：直接拉伸到 1024x1024 + mean=0.5 =====
    pil_img = Image.fromarray(img_rgb)
    img_1024 = pil_img.resize((1024, 1024), Image.LANCZOS)  # 直接拉伸
    
    # 转 tensor：/255 → -0.5 (mean=0.5, std=1)
    img_tensor = transforms.ToTensor()(img_1024).unsqueeze(0).to(device)  # [0,1]
    img_tensor = img_tensor - 0.5  # mean shift to [-0.5, 0.5]
    
    with torch.no_grad():
        result = model(img_tensor)
    
    # ===== rmbg.dev 后处理 =====
    # result[0][0]: (1, 1, 1024, 1024) — 模型内部已有 sigmoid
    mask_1024 = result[0][0].squeeze().cpu().numpy()  # (1024, 1024) in [0,1]
    
    # 同 rmbg.dev: 确保 [0, 1]
    mask_1024 = np.clip(mask_1024, 0.0, 1.0)
    mask_8u = (mask_1024 * 255).astype(np.uint8)
    
    # ===== rmbg.dev 关键：1px blur + resize 回原图 =====
    # 先转 PIL，用 1px GaussianBlur
    mask_pil = Image.fromarray(mask_8u, "L")
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=1))
    
    # resize 回原图
    mask_pil_orig = mask_pil.resize((w, h), Image.LANCZOS)
    alpha = np.array(mask_pil_orig, dtype=np.uint8)
    
    t_total = time.time() - t0

    # 保存
    stem = img_path.stem
    rgba = np.dstack([img_rgb, alpha])
    Image.fromarray(rgba, "RGBA").save(str(OUTPUT_DIR / f"{stem}_rmbgdev.png"))
    Image.fromarray(alpha, "L").save(str(OUTPUT_DIR / f"{stem}_rmbgdev_alpha.png"))

    # 统计
    fg = (alpha > 200).sum() / alpha.size * 100
    ht = ((alpha > 20) & (alpha < 200)).sum() / alpha.size * 100
    bg = (alpha <= 20).sum() / alpha.size * 100
    print(f"  耗时: {t_total:.1f}s")
    print(f"  前景: {fg:.1f}% | 半透明: {ht:.1f}% | 透明: {bg:.1f}%")
    print(f"  保存: {stem}_rmbgdev.png + {stem}_rmbgdev_alpha.png")
    print()

print("全部完成！")
