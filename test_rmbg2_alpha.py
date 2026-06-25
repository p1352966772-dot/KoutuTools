"""
Standalone test for BRIA RMBG-2.0 (BiRefNet) on existing input images.
Does NOT modify any project files.

Usage:
    pip install transformers torch torchvision accelerate
    huggingface-cli login   # need to accept RMBG-2.0 license first
    python test_rmbg2_alpha.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "auto_psd_cutout" / "input"
OUTPUT_DIR = HERE / "output_rmbg2_test"
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def load_rmbg2():
    print("[load] BRIA RMBG-2.0 (~840MB BiRefNet, gated)...")
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0", trust_remote_code=True
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[load] device = {device}")
    return model, device


def rmbg2_infer(model, device, pil_img: Image.Image) -> np.ndarray:
    """Return uint8 alpha [0, 255] at original size."""
    w, h = pil_img.size
    img_1024 = pil_img.resize((1024, 1024), Image.LANCZOS)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    x = tfm(img_1024).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(x)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    mask = transforms.ToPILImage()(pred).resize((w, h), Image.LANCZOS)
    return np.array(mask, dtype=np.uint8)


def composite_on(rgb: np.ndarray, alpha: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
    """合成到指定纯色背景上."""
    a = alpha.astype(np.float32) / 255.0
    bg_arr = np.array(bg, dtype=np.float32)
    return (rgb * a[..., None] + bg_arr * (1 - a[..., None])).astype(np.uint8)


def composite_checker(rgb: np.ndarray, alpha: np.ndarray, cell: int = 20) -> np.ndarray:
    """合成到透明棋盘格上."""
    h, w = rgb.shape[:2]
    board = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                board[y:y+cell, x:x+cell] = 220
            else:
                board[y:y+cell, x:x+cell] = 245
    a = alpha.astype(np.float32) / 255.0
    return (rgb * a[..., None] + board * (1 - a[..., None])).astype(np.uint8)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.exists():
        print(f"[err] input dir not found: {INPUT_DIR}")
        sys.exit(1)

    images = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in SUPPORTED])
    if not images:
        print(f"[err] no images in {INPUT_DIR}")
        sys.exit(1)

    images = images[:3]  # only first 3 to keep download time minimal
    print(f"[info] will process {len(images)} image(s)")

    try:
        model, device = load_rmbg2()
    except Exception as exc:
        print(f"\n[err] failed to load RMBG-2.0: {exc}")
        print("\nThis model is GATED. You need to:")
        print("  1. Have a Hugging Face account")
        print("  2. Visit https://huggingface.co/briaai/RMBG-2.0 and accept the license")
        print("  3. Run: huggingface-cli login")
        print("  4. Then re-run this script")
        sys.exit(1)

    for img_path in images:
        print(f"\n=== {img_path.name} ===")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[skip] cannot read: {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        t0 = time.time()
        try:
            alpha = rmbg2_infer(model, device, pil_img)
        except Exception as exc:
            print(f"[err] inference failed: {exc}")
            continue
        dt = time.time() - t0

        stem = img_path.stem
        Image.fromarray(alpha, "L").save(OUTPUT_DIR / f"{stem}_rmbg2_alpha.png")
        Image.fromarray(composite_on(rgb, alpha, (255, 255, 255))).save(
            OUTPUT_DIR / f"{stem}_rmbg2_white.png"
        )
        Image.fromarray(composite_checker(rgb, alpha)).save(
            OUTPUT_DIR / f"{stem}_rmbg2_checker.png"
        )

        fg_pix = int(np.count_nonzero(alpha > 8))
        semi_pix = int(np.count_nonzero((alpha > 8) & (alpha < 240)))
        total = alpha.size
        print(f"  time        : {dt:.2f}s")
        print(f"  foreground  : {fg_pix/total*100:.1f}% (alpha>8)")
        print(f"  semi-trans  : {semi_pix/total*100:.1f}% (8<a<240)")
        print(f"  saved to    : {OUTPUT_DIR / (stem + '_rmbg2_*')}")

    print(f"\n[done] results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
