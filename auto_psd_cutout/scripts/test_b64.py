import sys, os
sys.stdout.reconfigure(encoding='utf-8')
path = r'C:\Users\Administrator.SK-20241009ZRBE\Desktop\KoutuTools\auto_psd_cutout\src\rembg_utils.py'
code = open(path, 'r', encoding='utf-8').read()
print('Current lines:', len(code.split(chr(10))))
