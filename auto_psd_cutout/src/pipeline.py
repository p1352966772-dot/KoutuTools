from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .detector import detect_ui_elements
from .grid_cutter import detect_ui_elements_grid
from .rembg_utils import get_foreground_probability, get_bria14_alpha, get_white_bg_alpha, refine_alpha_for_white_bg
from .photoshop_jsx import generate_jsx
from .photoshop_runner import run_photoshop_jsx
from .preview import save_preview


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ProcessResult:
    image: Path
    ok: bool
    item_count: int = 0
    group_count: int = 0
    output_dir: Path | None = None
    psd_created: bool = False
    error: str | None = None


def collect_input_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _get_alpha_mask(image_bgr: np.ndarray, config: dict[str, Any]) -> np.ndarray | None:
    """生成全图 alpha 掩码，自动选择最适合当前配置的方法。"""
    rgba_config = config.get("rgba_crop", {})
    if not bool(rgba_config.get("enabled", True)):
        return None

    # ── 方法 1：白底图连通域抠图（默认）───────────────────────────────
    # 针对纯白背景密集排版图优化。
    # 原理：白色像素连通域分析，触碰边缘 → 背景，不碰边缘 → 前景。
    # 优点：100% 保留元素内白色内容、零模型下载、毫秒级速度。
    if bool(rgba_config.get("white_bg_alpha", True)):
        white_threshold = int(rgba_config.get("white_threshold", 230))
        try:
            alpha = get_white_bg_alpha(image_bgr, white_threshold)
            print(f"白底图抠图完成 ({alpha.shape})")
            return alpha
        except Exception as exc:
            print(f"白底图抠图失败 ({exc})，回退到 BRIA 模型。")

    # ── 方法 2：BRIA RMBG-1.4 深度学习模型 ──────────────────────────
    try:
        alpha = get_bria14_alpha(image_bgr)
        # BRIA 可能误删元素内白色 → 用边缘连通性保护
        if bool(rgba_config.get("protect_inner_white", True)):
            white_threshold = int(rgba_config.get("white_threshold", 230))
            alpha = refine_alpha_for_white_bg(alpha, image_bgr, white_threshold)
        print(f"BRIA alpha mask loaded ({alpha.shape})")
        return alpha
    except Exception as exc:
        print(f"BRIA alpha mask failed: {exc}")

    return None


def process_image(image_path: Path, config: dict[str, Any], run_photoshop: bool = True) -> ProcessResult:
    print(f"开始处理：{image_path.name}")
    try:
        image_bgr = read_image_bgr(image_path)
    except Exception as exc:
        msg = f"图片读取失败：{image_path}，错误：{exc}"
        print(msg)
        return ProcessResult(image=image_path, ok=False, error=msg)

    # Step 1: 生成全图 alpha 掩码（透明底图）
    rmbg_prob_map = None
    scoring_config = config.get("rmbg_scoring", {})
    if bool(scoring_config.get("enabled", False)):
        try:
            rmbg_prob_map = get_foreground_probability(image_bgr)
            print("RMBG foreground probability map loaded")
        except Exception as exc:
            print(f"RMBG probability map failed (proceeding without): {exc}")
            rmbg_prob_map = None

    rmbg_alpha = _get_alpha_mask(image_bgr, config)

    output_root = Path(config["output_dir"])
    image_output_dir = output_root / image_path.stem
    preview_dir = image_output_dir / "preview"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    # Save full RGBA base layer (use OCR-cleaned image if grid mode)
    # Note: full_cutout.png is re-created after detection when ocr_cleaned_bgr exists
    rgba_full_path = None

    # Step 2: Structure channel detection (Path A) + dual-channel scoring (Path B)
    grid_mode = config.get("grid", {}).get("enabled", True)
    if grid_mode:
        detect_result = detect_ui_elements_grid(image_bgr, config)
    else:
        detect_result = detect_ui_elements(image_bgr, config, rmbg_prob_map=rmbg_prob_map)
    boxes = detect_result.get("boxes", [])
    groups = detect_result.get("groups", [])

    # 如果有 OCR 涂白图，用它做抠图底图（中文文字被涂白）
    source_bgr = detect_result.get("ocr_cleaned_bgr")
    if source_bgr is not None:
        image_bgr = source_bgr
        print("使用 OCR 涂白后的图片作为抠图底图")

    # Save full RGBA base layer
    if rmbg_alpha is not None and rgba_full_path is None:
        rgba_full_path = image_output_dir / "full_cutout.png"
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgba_full = np.dstack([rgb, rmbg_alpha])
        Image.fromarray(rgba_full, "RGBA").save(str(rgba_full_path))
        print(f"全图透明底图已保存: {rgba_full_path.name}")

    print(f"检测到 {len(groups)} 行，{len(boxes)} 个元素")
    if not boxes:
        msg = "未检测到UI元素，请检查图片或调整检测参数。"
        print(msg)
        return ProcessResult(image=image_path, ok=False, output_dir=image_output_dir, error=msg)

    for group in groups:
        gboxes = [b for b in boxes if b["group_id"] == group["id"]]
        names = [b["name"] for b in sorted(gboxes, key=lambda x: x["x1"])]
        print(f"  {group['name']}: {', '.join(names)}")

    # Step 3: Debug preview with overlay
    preview_path = preview_dir / f"{image_path.stem}_preview.jpg"
    detect_result["canvas_width"] = image_bgr.shape[1]
    detect_result["canvas_height"] = image_bgr.shape[0]
    save_preview(image_bgr, detect_result, preview_path, config)
    print(f"已生成预览图：{preview_path}")

    # Step 3b: Clear individual rgba_path (now using full_cutout.png as source)
    for box in boxes:
        box["rgba_path"] = None

    # Step 4: Generate JSX (with rgba_path support)
    suffix = config.get("photoshop", {}).get("psd_name_suffix", "_auto")
    psd_path = image_output_dir / f"{image_path.stem}{suffix}.psd"
    jsx_path = image_output_dir / "build_psd.jsx"
    generate_jsx(
        detect_result,
        image_path.resolve(),
        jsx_path,
        psd_path,
        config,
        rgba_image=rgba_full_path,
    )
    print(f"已生成 Photoshop 脚本：{jsx_path}")

    # Step 5: Optional Photoshop execution
    psd_created = False
    if run_photoshop:
        psd_created = run_photoshop_jsx(jsx_path, config)
        if psd_created:
            print(f"已生成 PSD：{psd_path}")
        else:
            print(f"PSD 未自动生成，可手动执行脚本：{jsx_path}")
    else:
        print(f"已按参数跳过 Photoshop，手动脚本位置：{jsx_path}")

    print("处理完成")
    return ProcessResult(
        image=image_path,
        ok=True,
        item_count=len(boxes),
        group_count=len(groups),
        output_dir=image_output_dir,
        psd_created=psd_created,
    )


def read_image_bgr(image_path: Path) -> np.ndarray:
    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            array = np.array(rgb)
    except Exception as exc:
        raise RuntimeError(f"无法打开图片：{exc}") from exc
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
