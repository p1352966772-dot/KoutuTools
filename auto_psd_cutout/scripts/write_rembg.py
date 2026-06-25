import sys, os
sys.stdout.reconfigure(encoding="utf-8")
content = (
    "from __future__ import annotations\n"
    "\n"
    "from functools import lru_cache\n"
    "from typing import Any\n"
    "\n"
    "import cv2\n"
    "import numpy as np\n"
    "from PIL import Image\n"
)
path = os.path.expandvars(r"%USERPROFILE%\Desktop\KoutuTools\auto_psd_cutout\src\rembg_utils.py")
with open(path, "w", encoding="utf-8") as f:
    f.write(content + "\n")
print("written", len(content), "chars")
