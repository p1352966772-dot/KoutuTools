path = r'C:\Users\Administrator.SK-20241009ZRBE\Desktop\KoutuTools\auto_psd_cutout\src\pipeline.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Step 3b with improved version that handles low-alpha bboxes
old_step3b = '''    # Step 3b: Generate RGBA crops for each bbox (real transparency via RMBG alpha)
    if rgba_enabled and rmbg_alpha is not None and boxes:
        crops_dir = image_output_dir / "rgba_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_count = 0
        for box in boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            w = max(1, x2 - x1)
            h_val = max(1, y2 - y1)
            # Crop RGB + alpha
            rgb_crop = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            alpha_crop = rmbg_alpha[y1:y2, x1:x2]
            # Build RGBA PNG
            rgba = np.dstack([rgb_crop, alpha_crop])
            rgba_pil = Image.fromarray(rgba, "RGBA")
            crop_name = f"crop_{box['id']:03d}.png"
            crop_path = crops_dir / crop_name
            rgba_pil.save(str(crop_path))
            box["rgba_path"] = str(crop_path.resolve())
            crop_count += 1
        print(f"已生成 {crop_count} 个 RGBA 透明图层")
    else:
        print("RGBA 透明图层生成已跳过（未启用或无 RMBG alpha）")'''

new_step3b = '''    # Step 3b: Generate RGBA crops for each bbox (real transparency via RMBG alpha)
    if rgba_enabled and rmbg_alpha is not None and boxes:
        crops_dir = image_output_dir / "rgba_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_count = 0
        # Compute a secondary white-background mask for alpha compensation
        gray_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Pixels darker than threshold = foreground in white-background UI images
        _, white_fg = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
        for box in boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            w = max(1, x2 - x1)
            h_val = max(1, y2 - y1)
            # Crop RGB
            rgb_crop = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            # Crop RMBG alpha
            alpha_crop = rmbg_alpha[y1:y2, x1:x2].astype(np.float32)
            # Crop white-background mask
            white_crop = white_fg[y1:y2, x1:x2].astype(np.float32)
            # Quality check: if mean RMBG alpha in bbox is too low, compensate
            mean_alpha = float(np.mean(alpha_crop))
            if mean_alpha < 80:
                # RMBG failed for this UI element — use white-threshold mask as backup
                # Blend: use max of RMBG alpha and white-fg (with RMBG as base)
                compensated = np.maximum(alpha_crop, white_crop * 0.85)
                # If still too low, force full opacity (detector confirmed valid element)
                if float(np.mean(compensated)) < 60:
                    final_alpha = np.full((h_val, w), 255, dtype=np.uint8)
                else:
                    final_alpha = compensated.astype(np.uint8)
            else:
                # RMBG alpha is good enough — use as-is
                final_alpha = alpha_crop.astype(np.uint8)
            # Build RGBA PNG
            rgba = np.dstack([rgb_crop, final_alpha])
            rgba_pil = Image.fromarray(rgba, "RGBA")
            crop_name = f"crop_{box['id']:03d}.png"
            crop_path = crops_dir / crop_name
            rgba_pil.save(str(crop_path))
            box["rgba_path"] = str(crop_path.resolve())
            crop_count += 1
        print(f"已生成 {crop_count} 个 RGBA 透明图层（{sum(1 for b in boxes if float(np.mean(rmbg_alpha[b['y1']:b['y2'], b['x1']:b['x2']])) < 80 for b in boxes)} 个低质量已补偿）")
    else:
        print("RGBA 透明图层生成已跳过（未启用或无 RMBG alpha）")'''

content = content.replace(old_step3b, new_step3b)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('OK')
