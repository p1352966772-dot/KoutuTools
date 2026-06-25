from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.config import load_config
from src.pipeline import SUPPORTED_EXTENSIONS, collect_input_images, process_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地半自动抠图拆图层并导出 PSD 工具 V1")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径，默认 config.yaml")
    parser.add_argument("--input", help="输入目录，默认读取 config.yaml 的 input_dir")
    parser.add_argument("--output", help="输出目录，默认读取 config.yaml 的 output_dir")
    parser.add_argument("--file", help="只处理指定图片")
    parser.add_argument("--no-photoshop", action="store_true", help="只生成 PNG、预览、manifest 和 JSX，不自动调用 Photoshop")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"配置加载失败：{exc}")
        return 1

    if args.input:
        config["input_dir"] = _resolve_cli_path(project_root, args.input)
    if args.output:
        config["output_dir"] = _resolve_cli_path(project_root, args.output)
    if args.no_photoshop:
        config.setdefault("photoshop", {})["enabled"] = False

    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        image_path = _resolve_cli_path(project_root, args.file)
        if not image_path.exists():
            print(f"指定图片不存在：{image_path}")
            return 1
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"不支持的图片格式：{image_path.suffix}")
            return 1
        images = [image_path]
    else:
        images = collect_input_images(input_dir)

    if not images:
        print(f"input 目录没有待处理图片：{input_dir}")
        print("支持格式：.jpg .jpeg .png .webp")
        return 0

    results = []
    for image_path in tqdm(images, desc="批量处理", unit="张"):
        results.append(process_image(image_path, config, run_photoshop=not args.no_photoshop))

    ok_count = sum(1 for result in results if result.ok)
    fail_count = len(results) - ok_count
    print("")
    print("处理结果汇总")
    print(f"成功：{ok_count} 张")
    print(f"失败：{fail_count} 张")
    for result in results:
        status = "成功" if result.ok else "失败"
        psd_status = "，PSD 已自动生成" if result.psd_created else ""
        print(f"- {result.image.name}：{status}，{result.group_count} 行 {result.item_count} 个元素{psd_status}")
        if result.error:
            print(f"  原因：{result.error}")

    return 0 if fail_count == 0 else 1


def _resolve_cli_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
