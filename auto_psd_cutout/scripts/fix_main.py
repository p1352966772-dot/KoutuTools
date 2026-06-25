import sys
sys.stdout.reconfigure(encoding="utf-8")
with open(r"C:\Users\Administrator.SK-20241009ZRBE\Desktop\KoutuTools\auto_psd_cutout\main.py", "r", encoding="utf-8") as f:
    code = f.read()
code = code.replace(
    '元素 {result.item_count} 个',
    '{result.group_count} 行 {result.item_count} 个元素'
)
with open(r"C:\Users\Administrator.SK-20241009ZRBE\Desktop\KoutuTools\auto_psd_cutout\main.py", "w", encoding="utf-8") as f:
    f.write(code)
import ast; ast.parse(code)
print("main.py updated OK")
