import sys
sys.stdout.reconfigure(encoding="utf-8")

PATH = r"C:\Users\Administrator.SK-20241009ZRBE\Desktop\KoutuTools\auto_psd_cutout\src\pipeline.py"

with open(PATH, "r", encoding="utf-8") as f:
    code = f.read()

old = 'rembg_rgba = remove_background_hybrid(working_image_bgr, white_threshold=240, feather=1)'
new = (
    'ct = config.get("cutout", {})\n'
    '        white_th = int(ct.get("white_threshold", 240))\n'
    '        feather_val = max(0, int(ct.get("feather", 1)))\n'
    '        rembg_rgba = remove_background_hybrid(working_image_bgr, white_threshold=white_th, feather=feather_val)'
)
code = code.replace(old, new)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(code)

import ast
ast.parse(code)
print("Syntax OK")
print("pipeline.py updated")
