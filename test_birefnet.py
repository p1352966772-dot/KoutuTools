"""
Standalone test for BiRefNet + full RMBG image processing chain (from rembg_utils + cutout).
Tests on existing input images. Does NOT modify any project files.

RMBG processing chain applied:
  1. Model inference
  2. Post: 1px GaussianBlur (BRIA-style from get_bria14_alpha)
  3. Post: Flood-fill edge-connected white removal (_edge_connected_mask from cutout.py)
  4. Post: Morphological close + 3px GaussianBlur (remove_background_hybrid style)
  5. Post: white-bg connectivity refinement (refine_alpha_for_white_bg)

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
from PIL import Image, ImageFilter
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "auto_psd_cutout" / "input"
OUTPUT_DIR = HERE / "output_birefnet_test"
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


# ── import from project for established processing logic ──────────────
try:
    sys.path.insert(0, str(HERE / "auto_psd_cutout" / "src"))
    from rembg_utils import refine_alpha_for_white_bg
    _HAS_REMBG = True
    print("[load] using refine_alpha_for_white_bg from project")
except ImportError:
    _HAS_REMBG = False
    print("[warn] project rembg_utils not importable, using local fallbacks")


# ============================================================
# Model
# ============================================================

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

    global half_precision
    half_precision = device.type == "cuda"
    if half_precision:
        model.half()
        print("[load] using half precision (FP16)")
    return model, device


half_precision = False


# ============================================================
# Pre-processing (BiRefNet native)
# ============================================================

def _estimate_target_size(pil_img: Image.Image, edge_threshold: float = 0.06) -> int:
    """动态分辨率：边缘密集→2048，简单→1024."""
    small = pil_img.resize((512, 512), Image.LANCZOS)
    gray = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    density = float(np.count_nonzero(edges)) / (512.0 * 512.0)
    return 2048 if density > edge_threshold else 1024


def _scale32(w: int, h: int, target_long: int) -> tuple[int, int]:
    """等比缩放后对齐到 32 的倍数."""
    if w >= h:
        nw = target_long
        nh = round(h * target_long / w)
    else:
        nh = target_long
        nw = round(w * target_long / h)
    nw = max(((nw + 16) // 32) * 32, 32)
    nh = max(((nh + 16) // 32) * 32, 32)
    return nw, nh


# ============================================================
# RMBG established post-processing algorithms
# ============================================================

def postproc_gaussian_blur(alpha: np.ndarray, radius: int = 1) -> np.ndarray:
    """BRIA RMBG-1.4 标准后处理：1px GaussianBlur 柔化边缘."""
    return np.array(
        Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(radius=radius)),
        dtype=np.uint8,
    )


def postproc_edge_connected_mask(alpha: np.ndarray, rgb: np.ndarray, white_threshold: int = 230) -> np.ndarray:
    """cutout.py _edge_connected_mask 算法：从边缘白色种子 flood-fill.
    
    原理：
      1. 找出所有触碰图像边缘的白色像素作为种子点
      2. 从每个种子 flood-fill，标记所有连通的白色区域
      3. 这些区域是背景 → alpha=0
      4. 其余像素保持原样
    比 connectedComponents 更彻底，适合形状复杂的白色前景物体。
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    _, white_binary = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
    source = white_binary.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # 收集边缘白色种子
    seeds: list[tuple[int, int]] = []
    for x in range(w):
        if source[0, x] == 255:
            seeds.append((x, 0))
        if source[h - 1, x] == 255:
            seeds.append((x, h - 1))
    for y in range(h):
        if source[y, 0] == 255:
            seeds.append((0, y))
        if source[y, w - 1] == 255:
            seeds.append((w - 1, y))

    # Flood-fill 去重
    connected = np.zeros_like(source)
    for sx, sy in seeds:
        if source[sy, sx] != 255 or connected[sy, sx] == 255:
            continue
        temp = source.copy()
        cv2.floodFill(temp, flood_mask, (sx, sy), 128)
        filled = temp == 128
        connected[filled] = 255
        source[filled] = 0
        flood_mask.fill(0)

    result = alpha.copy()
    result[connected == 255] = 0   # 边缘连通白 → 背景
    return result


def postproc_morphological_refine(alpha: np.ndarray) -> np.ndarray:
    """remove_background_hybrid 风格的形态学后处理.
    
    步骤：
      1. 5x5 椭圆核 MORPH_CLOSE → 填补小洞
      2. 3x3 GaussianBlur → 柔化边缘
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    return cv2.GaussianBlur(closed, (3, 3), 0)


def postproc_white_bg_refine(alpha: np.ndarray, rgb: np.ndarray, threshold: int = 230) -> np.ndarray:
    """refine_alpha_for_white_bg：连通域分析保护内白。
    
    触碰图像边缘的白色连通域 → 背景 (alpha=0)
    不碰边缘的白色连通域 → 前景 (alpha=255)
    非白色 → 保持原 alpha
    """
    if _HAS_REMBG:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return refine_alpha_for_white_bg(alpha, bgr, threshold)

    # Fallback
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    _, white_binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_binary, connectivity=4)
    result = alpha.copy()
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = (x <= 0 or y <= 0 or (x + bw) >= w - 1 or (y + bh) >= h - 1)
        mask = labels == label
        if touches_border:
            result[mask] = 0
        else:
            result[mask] = 255
    return result


# ============================================================
# Full processing pipelines (multiple combos for comparison)
# ============================================================

def pipeline_bria_style(alpha_raw: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """最简 BRIA 后处理：GaussianBlur(1px) + 白底保护.
    对应 rembg_utils.get_bria14_alpha 的核心后处理逻辑.
    """
    return postproc_white_bg_refine(
        postproc_gaussian_blur(alpha_raw, radius=1), rgb
    )


def pipeline_hybrid(alpha_raw: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """完整 RMBG 混合管线：GaussianBlur → flood-fill 去背景 → 形态学 → 内白保护.
    综合了 get_bria14_alpha + _edge_connected_mask + remove_background_hybrid 的处理逻辑.
    """
    a = alpha_raw.copy()
    # 1) BRIA 风格高斯模糊
    a = postproc_gaussian_blur(a, radius=1)
    # 2) flood-fill 边缘连通白 → 背景（cutout.py 方案）
    a = postproc_edge_connected_mask(a, rgb, white_threshold=230)
    # 3) 形态学闭运算 + GaussianBlur（remove_background_hybrid 方案）
    a = postproc_morphological_refine(a)
    # 4) 连通域分析保护内白
    a = postproc_white_bg_refine(a, rgb, threshold=230)
    return a


# ============================================================
# Inference
# ============================================================

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


# ============================================================
# Compositing helpers
# ============================================================

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


# ============================================================
# Main
# ============================================================

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
    print(f"[info] input dir : {INPUT_DIR}")
    print(f"[info] output dir: {OUTPUT_DIR}")
    print(f"[info] pipelines : raw | bria_style | hybrid")
    print()

    try:
        model, device = load_birefnet()
    except Exception as exc:
        print(f"\n[err] failed to load BiRefNet: {exc}")
        print("\nTip: pip install transformers torch torchvision timm einops kornia accelerate")
        sys.exit(1)

    for img_path in images:
        print(f"\n=== {img_path.name} ===")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[skip] cannot read: {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # ── Inference ──
        t0 = time.time()
        try:
            alpha_raw = birefnet_infer(model, device, pil_img)
        except Exception as exc:
            print(f"[err] inference failed: {exc}")
            continue
        dt_infer = time.time() - t0

        # ── Post-processing pipelines ──
        pipelines = {
            "raw": ("BiRefNet raw", lambda a, _r: a),
            "bria": ("+GaussianBlur+whiteRefine", pipeline_bria_style),
            "hybrid": ("+GB+floodFill+morph+whiteRefine", pipeline_hybrid),
        }

        for suffix, (label, proc_fn) in pipelines.items():
            t1 = time.time()
            alpha = proc_fn(alpha_raw, rgb)
            dt_post = time.time() - t1

            stem = img_path.stem
            Image.fromarray(alpha, "L").save(OUTPUT_DIR / f"{stem}_{suffix}_alpha.png")
            Image.fromarray(composite_on(rgb, alpha, (255, 255, 255))).save(
                OUTPUT_DIR / f"{stem}_{suffix}_white.png"
            )
            Image.fromarray(composite_checker(rgb, alpha)).save(
                OUTPUT_DIR / f"{stem}_{suffix}_checker.png"
            )

            total = alpha.size
            fg = int(np.count_nonzero(alpha > 8))
            semi = int(np.count_nonzero((alpha > 8) & (alpha < 240)))
            bg = int(np.count_nonzero(alpha <= 8))
            print(f"  [{suffix:6s}] {label}")
            print(f"           post-proc: {dt_post:.3f}s")
            print(f"           fore:{fg/total*100:5.1f}% semi:{semi/total*100:4.1f}% back:{bg/total*100:5.1f}%")

        # Compare how much each pipeline changed vs raw
        for suffix in ["bria", "hybrid"]:
            a = np.array(Image.open(OUTPUT_DIR / f"{img_path.stem}_{suffix}_alpha.png"))
            if a.shape == alpha_raw.shape:
                diff_pct = float(np.count_nonzero(cv2.absdiff(alpha_raw, a) > 1)) / alpha_raw.size * 100
                print(f"           vs raw    : {diff_pct:.1f}% pixels changed")

    print(f"\n{'='*50}")
    print(f"[done] results in: {OUTPUT_DIR}")
    print()
    print("Per-image output files:")
    print("  *_raw_alpha/white/checker.png  — BiRefNet raw")
    print("  *_bria_alpha/white/checker.png  — +GaussianBlur + whiteRefine")
    print("  *_hybrid_alpha/white/checker.png — +GaussianBlur + floodFill + morph + whiteRefine")


if __name__ == "__main__":
    main()
