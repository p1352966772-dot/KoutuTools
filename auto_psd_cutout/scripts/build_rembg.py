import sys, os
sys.stdout.reconfigure(encoding="utf-8")
content = """from __future__ import annotations
from functools import lru_cache
from typing import Any
import cv2
import numpy as np
from PIL import Image
"""
with open(os.path.expandvars(r"%USERPROFILE%\Desktop\KoutuTools\auto_psd_cutout\src\rembg_utils.py"), "w", encoding="utf-8") as f:
    f.write(content)
print("Part 1 OK")
