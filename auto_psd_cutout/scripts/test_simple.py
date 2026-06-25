import sys, os
sys.stdout.reconfigure(encoding="utf-8")
src = r"C:\Users\Administrator.SK-20241009ZRBE\Desktop\KoutuTools\auto_psd_cutout\src"
path = os.path.join(src, "rembg_utils.py")
with open(path, "w", encoding="utf-8") as f:
    f.write("x = 1\n")
print("done")
