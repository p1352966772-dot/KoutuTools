from __future__ import annotations

from pathlib import Path
from typing import Any


def run_photoshop_jsx(jsx_path: Path, config: dict[str, Any]) -> bool:
    ps_config = config.get("photoshop", {})
    if not bool(ps_config.get("enabled", True)):
        print("已跳过 Photoshop 自动生成。")
        return False

    try:
        import win32com.client
    except Exception as exc:
        print(f"Photoshop 自动调用失败：无法导入 pywin32。可手动在 Photoshop 中执行 {jsx_path}。错误：{exc}")
        return False

    try:
        ps = win32com.client.Dispatch("Photoshop.Application")
        ps.Visible = bool(ps_config.get("visible", True))
        ps.DoJavaScriptFile(str(jsx_path.resolve()))
        return True
    except Exception as exc:
        print(f"Photoshop 自动调用失败：可手动在 Photoshop 中执行 {jsx_path}。错误：{exc}")
        return False
