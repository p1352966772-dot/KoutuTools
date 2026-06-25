from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


# ============================================================
# ============================================================
# BRIA RMBG-1.4 session (Hugging Face briaai/RMBG-1.4)
# ============================================================

@lru_cache(maxsize=1)
def _load_bria14_model() -> Any:
    """Load briaai/RMBG-1.4 from Hugging Face (~178MB, 44M params)."""
    import warnings
    warnings.filterwarnings("ignore")
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-1.4", trust_remote_code=True
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def _bria14_transform(image: Image.Image) -> torch.Tensor:
    """rmbg.dev 方式：直接拉伸到 1024x1024 + mean=0.5 归一化."""
    img_1024 = image.resize((1024, 1024), Image.LANCZOS)  # 直接拉伸
    tensor = transforms.ToTensor()(img_1024).unsqueeze(0)  # [0, 1]
    tensor = tensor - 0.5  # mean shift to [-0.5, 0.5], std=1
    return tensor


def get_bria14_alpha(image_bgr: np.ndarray) -> np.ndarray:
    """按 rmbg.dev 方式：直接拉伸 + mean=0.5 + 1px blur + resize回原图.

    Returns uint8 alpha mask [0, 255], 255=foreground, 0=background.
    """
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    try:
        model, device = _load_bria14_model()
        input_tensor = _bria14_transform(pil_img)
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            result = model(input_tensor)

        # result[0][0]: model内部已有 sigmoid -> [0, 1]
        mask_1024 = result[0][0].squeeze().cpu().numpy()  # (1024, 1024)
        mask_1024 = np.clip(mask_1024, 0.0, 1.0)
        mask_8u = (mask_1024 * 255).astype(np.uint8)

        # rmbg.dev 后处理：1px GaussianBlur + resize回原图
        mask_pil = Image.fromarray(mask_8u, "L")
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=1))
        mask_orig = mask_pil.resize((w, h), Image.LANCZOS)
        return np.array(mask_orig, dtype=np.uint8)

    except Exception as exc:
        print(f"BRIA RMBG-1.4 alpha failed: {exc}")

    # Fallback to U2-Net
    try:
        session = _load_u2net_session()
        outputs = session.predict(pil_img)
        mask = _extract_foreground_mask(outputs, image_bgr.shape[:2])
        if mask is not None:
            return (mask * 255).astype(np.uint8)
    except Exception as exc:
        print(f"U2-Net fallback failed: {exc}")

    # Last resort: fully opaque
    return np.full((h, w), 255, dtype=np.uint8)


def _load_u2net_session() -> Any:
    from rembg import new_session
    return new_session("u2net")


@lru_cache(maxsize=1)
def _load_bria_session() -> Any:
    from rembg import new_session
    return new_session("bria-rmbg")


# ============================================================
# Foreground probability map (Path B: auxiliary channel)
# ============================================================

def get_foreground_probability(image_bgr: np.ndarray) -> np.ndarray:
    """Returns float32 foreground probability map (0.0-1.0) for bbox scoring.

    Uses rembg U2-Net alpha/foreground mask as foreground probability.
    Used ONLY for bbox scoring (Path B), NEVER for bbox generation.

    Note: rembg >= 2.0 returns session.predict() as list[PIL.Image] with mode='L'.
    """
    import os
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Try BRIA only if already downloaded (no download trigger)
    bria_path = os.path.join(os.path.expanduser("~"), ".u2net", "bria-rmbg.onnx")
    bria_available = os.path.isfile(bria_path) and os.path.getsize(bria_path) > 1000000
    if bria_available:
        try:
            session = _load_bria_session()
            outputs = session.predict(pil_img)
            if outputs is not None:
                mask = _extract_foreground_mask(outputs, image_bgr.shape[:2])
                if mask is not None:
                    return mask
        except Exception:
            pass

    # Primary: U2-Net (cached)
    try:
        session = _load_u2net_session()
        outputs = session.predict(pil_img)
        if outputs is not None:
            mask = _extract_foreground_mask(outputs, image_bgr.shape[:2])
            if mask is not None:
                return mask
    except Exception as exc:
        print(f"RMBG probability map failed: {exc}")

    # Last resort: uniform neutral probability
    h, w = image_bgr.shape[:2]
    return np.ones((h, w), dtype=np.float32) * 0.5


def _extract_foreground_mask(outputs: Any, img_shape: tuple) -> np.ndarray | None:
    """Extract foreground probability mask from rembg predict() output.
    
    Handles both legacy (PIL RGBA) and new (list[PIL 'L']) output formats.
    """
    h, w = img_shape
    # New format: list of PIL images
    if isinstance(outputs, (list, tuple)):
        for out in outputs:
            if hasattr(out, "mode") and out.mode == "L":
                mask = np.array(out, dtype=np.float32)
                if mask.shape == (h, w):
                    return np.clip(mask / 255.0, 0.0, 1.0)
                # Might be transposed
                if mask.shape == (w, h):
                    return np.clip(mask.T / 255.0, 0.0, 1.0)
        return None
    # Legacy format: single PIL RGBA image
    if hasattr(outputs, "mode") and outputs.mode == "RGBA":
        arr = np.array(outputs, dtype=np.float32)
        if arr.shape[:2] == (h, w) and arr.shape[2] >= 4:
            return np.clip(arr[:, :, 3] / 255.0, 0.0, 1.0)
    # Fallback: try direct conversion
    try:
        arr = np.array(outputs, dtype=np.float32)
        if arr.shape == (h, w):
            return np.clip(arr / 255.0, 0.0, 1.0)
        if arr.shape == (w, h):
            return np.clip(arr.T / 255.0, 0.0, 1.0)
    except Exception:
        pass
    return None


def _remove_background_internal(image_bgr: np.ndarray, session: Any) -> np.ndarray:
    """Internal: background removal returning BGRA compatible output."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    outputs = session.predict(pil_img)
    if outputs is None:
        h, w = image_bgr.shape[:2]
        return np.dstack([image_bgr, np.full((h, w), 255, dtype=np.uint8)])
    # New format: list[PIL] -> extract mask
    fg_prob = _extract_foreground_mask(outputs, image_bgr.shape[:2])
    if fg_prob is not None:
        alpha = (fg_prob * 255).astype(np.uint8)
        return np.dstack([image_bgr, alpha])
    # Legacy format
    if hasattr(outputs, "mode") and outputs.mode == "RGBA":
        return np.array(outputs)
    h, w = image_bgr.shape[:2]
    return np.dstack([image_bgr, np.full((h, w), 255, dtype=np.uint8)])



def get_alpha_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Returns uint8 alpha mask (0=background, 255=foreground) using U2-Net.

    Used for generating transparent RGBA crops per bbox.
    Primary: U2-Net (cached). Falls back to BRIA if model file present.
    """
    import os
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Try BRIA only if already downloaded
    bria_path = os.path.join(os.path.expanduser("~"), ".u2net", "bria-rmbg.onnx")
    bria_available = os.path.isfile(bria_path) and os.path.getsize(bria_path) > 1000000
    if bria_available:
        try:
            session = _load_bria_session()
            outputs = session.predict(pil_img)
            mask = _extract_foreground_mask(outputs, image_bgr.shape[:2])
            if mask is not None:
                return (mask * 255).astype(np.uint8)
        except Exception:
            pass

    # Primary: U2-Net
    try:
        session = _load_u2net_session()
        outputs = session.predict(pil_img)
        mask = _extract_foreground_mask(outputs, image_bgr.shape[:2])
        if mask is not None:
            return (mask * 255).astype(np.uint8)
    except Exception as exc:
        print(f"RMBG alpha mask failed: {exc}")

    # Last resort: all-foreground (no transparency)
    h, w = image_bgr.shape[:2]
    return np.full((h, w), 255, dtype=np.uint8)


def remove_background(image_bgr: np.ndarray, session: Any | None = None) -> np.ndarray:
    """Remove background using U2-Net. Returns BGRA image."""
    if session is None:
        session = _load_u2net_session()
    return _remove_background_internal(image_bgr, session)


def remove_background_hybrid(image_bgr: np.ndarray, mask_bgr: np.ndarray | None = None, session: Any | None = None) -> np.ndarray:
    """Hybrid background removal combining rembg mask with OpenCV mask refinement."""
    if session is None:
        session = _load_u2net_session()
    rgba = _remove_background_internal(image_bgr, session)
    rembg_mask = rgba[:, :, 3]

    if mask_bgr is not None:
        gray_mask = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(rembg_mask, binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.GaussianBlur(combined, (3, 3), 0)
    else:
        combined = rembg_mask

    result = np.dstack([image_bgr, combined])
    return result


def alpha_mask_from_rgba(rgba: np.ndarray) -> np.ndarray:
    """Extract alpha channel as binary mask (thresholded at 128)."""
    alpha = rgba[:, :, 3]
    _, binary = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)
    return binary


def refine_alpha_for_white_bg(
    alpha: np.ndarray,
    image_bgr: np.ndarray,
    white_threshold: int = 230,
) -> np.ndarray:
    """针对白底图的后处理：保护元素内的白色不被抠掉。

    标准 RGB 模型（BRIA / U2-Net）会把白色背景和元素内白色一起误删。
    此函数通过连通域分析解决：
      1. 找到图像中所有白色像素
      2. 触碰图像边缘的白色区域 → 强制透明（真正的背景）
      3. 不触碰图像边缘的白色区域 → 强制不透明（元素内的白色内容）
      4. 非白色区域保持原 alpha 不变

    Parameters
    ----------
    alpha : np.ndarray
        原始 alpha 掩码 [0, 255], uint8.
    image_bgr : np.ndarray
        原图 BGR.
    white_threshold : int
        判定为白色的灰度阈值，默认 230.

    Returns
    -------
    np.ndarray
        优化后的 alpha 掩码 [0, 255], uint8.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 二值化：白色像素为 255，其余为 0
    _, white_binary = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)

    # 连通域分析（4 连通，避免对角线误连）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        white_binary, connectivity=4
    )

    result = alpha.copy()

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])

        # 判断是否触碰图像边缘
        touches_border = (
            x <= 0
            or y <= 0
            or (x + bw) >= w - 1
            or (y + bh) >= h - 1
        )

        component_mask = labels == label

        if touches_border:
            # 背景白 → 透明
            result[component_mask] = 0
        else:
            # 前景白（元素内白色内容）→ 不透明
            result[component_mask] = 255

    return result


def get_white_bg_alpha(image_bgr: np.ndarray, white_threshold: int = 230) -> np.ndarray:
    """白底图专用抠图：直接用边缘连通性判断，无需深度学习模型。

    对于纯白背景图，此方法比任何 RGB 模型都更可靠：
      - 100% 保留元素内的白色内容
      - 零模型下载，毫秒级速度
      - 不受模型泛化能力限制

    Returns uint8 alpha mask [0, 255], 255=foreground, 0=background.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 二值化
    _, white_binary = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)

    # 非白色区域直接标记为前景
    not_white = white_binary == 0

    # 白色区域做连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        white_binary, connectivity=4
    )

    # 初始化：非白色 → 255，白色暂设为 0
    result = np.zeros((h, w), dtype=np.uint8)
    result[not_white] = 255

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])

        touches_border = (
            x <= 0
            or y <= 0
            or (x + bw) >= w - 1
            or (y + bh) >= h - 1
        )

        component_mask = labels == label

        if not touches_border:
            # 不与边缘相连的白色区域 → 前景
            result[component_mask] = 255
        # 与边缘相连的白色区域 → 保持 0（背景）

    return result
