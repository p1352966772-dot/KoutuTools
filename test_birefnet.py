"""
Standalone test for original BiRefNet (Apache 2.0 / MIT).
Tests on existing input images. Does NOT modify any project files.

Usage:
    pip install transformers torch torchvision timm einops kornia accelerate
    python test_birefnet.py
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
OUTPUT_DIR = HERE / "output_birefnet_test"
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def load_birefnet():
    """Load ZhengPeng7/BiRefNet — MIT license, commercially usable."""
    print("[load] BiRefNet (MIT license, no restrictions)...")
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[load] device = {device}")

    # Enable half precision on GPU for speed
    global half_precision
    half_precision = device.type == "cuda"
    if half_precision:
        model.half()
        print("[load] using half precision (FP16)")
    return model, device


half_precision = False


def _estimate_target_size(pil_img: Image.Image, edge_threshold: float = 0.06) -> int:
    """Estimate target long-side size based on edge density.

    Returns 2048 for detail-dense images (piano keys, small devices),
    1024 for simple/broad-subject images (banners, large cells).
    """
    small = pil_img.resize((512, 512), Image.LANCZOS)
    gray = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    density = float(np.count_nonzero(edges)) / (512.0 * 512.0)
    return 2048 if density > edge_threshold else 1024


def _scale32(w: int, h: int, target_long: int) -> tuple[int, int]:
    """Scale (w,h) so long side == target_long, round to 32."""
    if w >= h:
        nw = target_long
        nh = round(h * target_long / w)
    else:
        nh = target_long
        nw = round(w * target_long / h)
    nw = max(((nw + 16) // 32) * 32, 32)
    nh = max(((nh + 16) // 32) * 32, 32)
    return nw, nh


def _auto_protect(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Protect important regions from over-cutting via LAB color analysis.

    - White/light backgrounds (avg L > 200): protect high-saturation and
      edge-bounded white regions (e.g. white text, piano keys, stripes).
    - Dark backgrounds (avg L < 120): protect anything notably brighter.
    - Mid-tone backgrounds: no adjustment.
    Protected areas get alpha raised to at least 0.5.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    avg_l = L.mean()

    out = alpha.astype(np.float32)

    if avg_l > 200.0:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        high_sat = sat > 30.0

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150) > 0
        bright_edge = edges & (L > 240.0)

        protect = high_sat | bright_edge
    elif avg_l < 120.0:
        bright = L > (avg_l + 20.0)
        protect = bright
    else:
        return alpha

    out[protect] = np.maximum(out[protect], 128.0)
    return out.astype(np.uint8)


def birefnet_infer(model, device, pil_img: Image.Image) -> np.ndarray:
    """Return uint8 alpha [0, 255] at original size. Uses dynamic input size."""
    w, h = pil_img.size

    target = _estimate_target_size(pil_img)
    nw, nh = _scale32(w, h, target)
    print(f"  [infer] target={target}, resize=({nw}x{nh})")

    tfm = transforms.Compose([
        transforms.Resize((nh, nw)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    x = tfm(pil_img).unsqueeze(0)
    x = x.half() if half_precision else x
    x = x.to(device)

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

    images = images[:3]
    print(f"[info] will process {len(images)} image(s)")
    print(f"[info] input dir: {INPUT_DIR}")
    print(f"[info] output dir: {OUTPUT_DIR}\n")

    try:
        model, device = load_birefnet()
    except Exception as exc:
        print(f"\n[err] failed to load BiRefNet: {exc}")
        print("\nTip: try `pip install transformers torch torchvision timm einops kornia accelerate`")
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
            alpha = birefnet_infer(model, device, pil_img)
            alpha = _auto_protect(rgb, alpha)
            alpha = _auto_protect(rgb, alpha)
        except Exception as exc:
            print(f"[err] inference failed: {exc}")
            continue
        dt = time.time() - t0

        stem = img_path.stem
        Image.fromarray(alpha, "L").save(OUTPUT_DIR / f"{stem}_birefnet_alpha.png")
        Image.fromarray(composite_on(rgb, alpha, (255, 255, 255))).save(
            OUTPUT_DIR / f"{stem}_birefnet_white.png"
        )
        Image.fromarray(composite_checker(rgb, alpha)).save(
            OUTPUT_DIR / f"{stem}_birefnet_checker.png"
        )

        fg_pix = int(np.count_nonzero(alpha > 8))
        semi_pix = int(np.count_nonzero((alpha > 8) & (alpha < 240)))
        total = alpha.size
        print(f"  time        : {dt:.2f}s")
        print(f"  foreground  : {fg_pix/total*100:.1f}% (alpha>8)")
        print(f"  semi-trans  : {semi_pix/total*100:.1f}% (8<a<240)")
        print(f"  saved to    : {OUTPUT_DIR}")

    print(f"\n{'='*50}")
    print(f"[done] results in: {OUTPUT_DIR}")
    print(f"Open the *_birefnet_checker.png to see transparency quality.")


if __name__ == "__main__":
    main()
