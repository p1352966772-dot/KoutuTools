path = r'C:\Users\Administrator.SK-20241009ZRBE\Desktop\KoutuTools\auto_psd_cutout\src\pipeline.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# After RMBG alpha mask loaded, add saving full RGBA preview
old = '''    if rgba_enabled:
        try:
            rmbg_alpha = get_bria14_alpha(image_bgr)
            print(f"RMBG alpha mask loaded ({rmbg_alpha.shape})")
        except Exception as exc:
            print(f"RMBG alpha mask failed: {exc}")'''

new = '''    if rgba_enabled:
        try:
            rmbg_alpha = get_bria14_alpha(image_bgr)
            print(f"RMBG alpha mask loaded ({rmbg_alpha.shape})")
            # Save full RGBA matting preview
            rgb_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            rgba_full = np.dstack([rgb_full, rmbg_alpha])
            Image.fromarray(rgba_full, "RGBA").save(str(image_output_dir / f"{image_path.stem}_matting.png"))
            print(f"已保存全图抠图：{image_path.stem}_matting.png")
        except Exception as exc:
            print(f"RMBG alpha mask failed: {exc}")'''

content = content.replace(old, new)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('OK')
