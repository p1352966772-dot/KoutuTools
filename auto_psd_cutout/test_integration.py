import sys, time, yaml
from pathlib import Path

BASE = Path('.').resolve()
sys.path.insert(0, str(BASE / 'src'))

import cv2
import numpy as np
from PIL import Image
import rembg_utils
import detector
import photoshop_jsx
from preview import save_preview

with open(BASE / 'config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

img_path = sorted((BASE / 'input').glob('*.jpg'))[4]
print(f'处理: {img_path.name}')

img_bgr = cv2.imread(str(img_path))
h, w = img_bgr.shape[:2]
print(f'尺寸: {w}x{h}')

t0 = time.time()
rmbg_alpha = rembg_utils.get_bria14_alpha(img_bgr)
print(f'BRIA 全图抠图: {time.time()-t0:.1f}s')

t0 = time.time()
detect_result = detector.detect_ui_elements(img_bgr, config)
boxes = detect_result.get('boxes', [])
groups = detect_result.get('groups', [])
print(f'检测: {time.time()-t0:.1f}s, 元素: {len(boxes)}, 行: {len(groups)}')

if boxes:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    output_dir = BASE / 'output' / img_path.stem
    crops_dir = output_dir / 'rgba_crops'
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    for box in boxes[:5]:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        crop_rgb = img_rgb[y1:y2, x1:x2]
        crop_alpha = rmbg_alpha[y1:y2, x1:x2]
        rgba = np.dstack([crop_rgb, crop_alpha])
        bid = box['id']
        crop_path = crops_dir / f'crop_{bid:03d}.png'
        Image.fromarray(rgba, 'RGBA').save(str(crop_path))
        box['rgba_path'] = str(crop_path.resolve())
        print(f'  {box["name"]}: {x1},{y1} {x2},{y2}')
    
    detect_result['canvas_width'] = w
    detect_result['canvas_height'] = h
    psd_path = output_dir / f'{img_path.stem}_auto.psd'
    jsx_path = output_dir / 'build_psd.jsx'
    photoshop_jsx.generate_jsx(detect_result, img_path.resolve(), jsx_path, psd_path, config)
    print(f'JSX: {jsx_path}')
    
    preview_path = output_dir / 'preview' / f'{img_path.stem}_preview.jpg'
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    save_preview(img_bgr, detect_result, preview_path, config)
    print(f'预览: {preview_path}')

print('全部完成!')
