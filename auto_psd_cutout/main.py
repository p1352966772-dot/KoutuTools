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
    parser.add_argument("--watch", action="store_true", help="定时扫描模式，监控 input 目录新增文件自动处理")
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

    if args.watch:
        interval = int(config.get("watch_interval", 10))
        watch_mode(input_dir, config, interval, args.no_photoshop)
        return 0

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
        print("支持格式：.jpg .jpeg .png .webp .ai")
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



def watch_mode(input_dir: Path, config: dict, interval: int, no_ps: bool) -> None:
    """Polling watch loop: scan for new files every N seconds."""
    import time
    print(f"\n[监控模式] 扫描目录: {input_dir}")
    print(f"[监控模式] 扫描间隔: {interval}s, 按 Ctrl+C 停止\n")

    def _already_done(image_path: Path) -> bool:
        """Check if output already exists for this image."""
        out = Path(config["output_dir"]) / image_path.stem
        return out.exists() and any(out.iterdir())  # preview dir exists

    # Mark existing files as already processed (check output dirs)
    existing = collect_input_images(input_dir)
    if existing:
        done = [p for p in existing if _already_done(p)]
        pending = [p for p in existing if not _already_done(p)]
        print(f"[监控] 已有 {len(done)} 个已处理, {len(pending)} 个待处理")
        if pending:
            print(f"[监控] 发现 {len(pending)} 个未处理文件，准备处理...")
            for p in pending:
                print(f"[监控] 处理: {p.name}")
                result = process_image(p, config, run_photoshop=not no_ps)
                status = "OK" if result.ok else f"FAIL: {result.error}"
                print(f"[监控] {p.name}: {status}")
    else:
        print("[监控] 目录为空，等待新增...")

    print()

    while True:
        try:
            current = collect_input_images(input_dir)
            new_files = sorted(
                [p for p in current if not _already_done(p) and p not in existing],
                key=lambda p: p.stat().st_mtime
            )
            if new_files:
                for image_path in new_files:
                    print(f"\n[监控] 发现新文件: {image_path.name}")
                    result = process_image(image_path, config, run_photoshop=not no_ps)
                    if result.ok:
                        existing.add(image_path)
                        print(f"[监控] 处理完成: {image_path.name} -> {result.item_count}个元素")
                    else:
                        print(f"[监控] 处理失败: {image_path.name} - {result.error}")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[监控] 已停止")
            break
        except Exception as exc:
            print(f"[监控] 错误: {exc}，5秒后重试...")
            time.sleep(5)


def _resolve_cli_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
