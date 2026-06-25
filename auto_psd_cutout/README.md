# Auto PSD Cutout V1

本工具用于本地批量处理白底美工素材拼版图：自动识别素材矩形框，裁剪透明 PNG，生成预览图和 manifest，并生成可在 Photoshop 中执行的 JSX 脚本来组装 PSD。

V1 只做本地文件夹流程，不做网页、不接 AI、不做手动框选编辑。

## 1. 安装依赖

要求：

- Windows 10 / Windows 11
- Python 3.10+
- 本地安装 Adobe Photoshop

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

## 2. 放入图片

把待处理的大图放入：

```text
input/
```

支持格式：

```text
.jpg .jpeg .png .webp
```

## 3. 运行

默认处理 `input/` 下所有图片：

```bash
python main.py
```

指定输入输出目录：

```bash
python main.py --input input --output output
```

只处理一张图：

```bash
python main.py --file input/sample.jpg
```

只生成 PNG、预览、manifest 和 JSX，不自动调用 Photoshop：

```bash
python main.py --no-photoshop
```

也可以双击：

```text
run.bat
```

## 4. 输出文件

每张输入图片会生成一个独立输出目录：

```text
output/原图文件名/
├── items/
│   ├── item_001.png
│   ├── item_002.png
│   └── item_003.png
├── preview/
│   └── 原图文件名_preview.jpg
├── 原图文件名_transparent.png
├── manifest.json
├── build_psd.jsx
└── 原图文件名_auto.psd
```

如果 Photoshop 自动调用失败，前面的 PNG、预览图、manifest 和 JSX 仍会保留。

PSD 默认图层包含：

- `00_transparent_map`：整张原图扣掉白底后的透明底图，默认隐藏，需要对照时可在 Photoshop 里点眼睛显示。
- `item_001`、`item_002` ...：每个素材的独立透明 PNG 图层，并按原图坐标放回。

红色检测框只会出现在预览图里，不会作为 PSD 图层导入。

## 5. 参数怎么调

编辑 `config.yaml`。

常用参数：

- `detect.merge_gap` 小：元素容易被拆开。
- `detect.merge_gap` 大：相邻元素容易被合并。
- `detect.use_layout_split`：默认开启，但现在不会主动按空白把一个素材切开，而是更偏向“先找连通块，再做保守合并”。
- `detect.split_large_boxes_by_projection`：默认关闭。开启后才会对大框做投影拆分，通常不建议打开。
- `detect.white_threshold` 小：浅色元素更容易保留，但背景可能去不干净。
- `detect.white_threshold` 大：背景更干净，但浅色元素可能丢失。
- `detect.padding`：裁剪元素时额外留边，避免边缘缺失。密集排版图建议 10-20，太大会让预览框互相压住。
- `detect.min_area`：过滤小噪点，值越大过滤越多。
- `detect.component_min_area`：过滤极小碎点，避免彩纸、星星把相邻素材连成一个大框。素材主体特别细碎时再调小。
- `detect.ignore_left`：忽略左侧中文分类标题区域，例如“横幅”“拉花”“贴纸”等。
- `detect.recover_ignored_left`：默认开启。检测时忽略左侧标题，但裁剪前会尝试把伸进左侧区域的素材边缘补回来。
- `detect.left_recover_gap`：左侧恢复的连接距离。第一列素材左边仍缺失时可适当增大，例如 30。
- `detect.post_merge_split_parts`：默认开启。检测阶段会把明显相邻、形态相近的碎框再合并一次，但现在比以前保守得多。
- `detect.post_merge_max_gap`：检测阶段允许合并的最大水平间距。蝴蝶结仍被拆开时调大，误合并相邻素材时调小。
- `detect.post_merge_max_height_ratio` / `post_merge_max_merged_width_ratio` / `post_merge_min_vertical_overlap`：控制后处理合并的保守程度。
- `post_cutout_merge.enabled`：扣图之后再做一轮合并。它是第二道保险，主要兜底那些检测阶段仍然分开的对象。
- `cutout.keep_inner_white`：默认只移除边缘连通白底，保留素材内部白色。
- `ocr_text_cleanup.enabled`：默认开启。使用 PaddleOCR 识别中文位置，只把包含中文的识别框涂白，避免分类文字影响输出。
- `ocr_text_cleanup.fill_entire_box`：默认开启。识别到中文后直接把整块文字框涂白，清理更干净。
- `ocr_text_cleanup.low_saturation_only`：默认开启，只清理 OCR 中文框里的黑灰低饱和像素，减少误删彩色素材。
- `label_text_cleanup.enabled`：规则清理备用方案，默认关闭。会把分类标签里的小号黑灰文字涂白。
- `label_text_cleanup.full_page`：默认开启，全图清理孤立的小号黑灰文字，例如“横幅 印刷”“拉花”“螺旋”“大插排 印刷”“小插排”。
- `label_text_cleanup.region_left_width`：左侧文字清理宽度。左侧中文还残留时增大，误删素材黑色细节时减小。
- `label_text_cleanup.region_top_height`：顶部文字清理高度。顶部说明文字残留时增大。
- `label_text_cleanup.max_component_width/max_component_height/max_component_area`：控制只删除小号文字组件。误删素材黑色线条时调小，文字残留时调大。
- `label_text_cleanup.protect_color_radius`：保护彩色素材附近的黑色线条，避免把素材描边当成文字删除。

## 6. Photoshop 自动生成失败怎么办

如果控制台提示 Photoshop 自动调用失败：

1. 确认本机已安装 Adobe Photoshop。
2. 确认已安装 `pywin32`。
3. 打开 Photoshop。
4. 在 Photoshop 中手动执行输出目录里的 `build_psd.jsx`。

脚本位置示例：

```text
output/sample/build_psd.jsx
```

手动执行脚本后，会按 manifest 坐标导入每个素材 PNG 图层，然后保存 PSD。

## 7. 常见问题

### 识别数量太少

通常是 `merge_gap` 太大，或者 `white_threshold` 太高导致浅色元素丢失。可以先减小 `merge_gap`，再微调 `white_threshold`。

### 一个素材被拆成多个 PNG

通常是 `merge_gap` 太小。适当增大 `merge_gap`，让距离近的碎片合并成一个框。

### 相邻素材被合并

通常是 `merge_gap` 太大。适当减小 `merge_gap`。

### 左侧标题文字也被识别

增大 `detect.ignore_left`。例如左侧标题区域约 120 像素宽，可以设置为 `120` 或更大。

### PNG 背景还有白边

可以适当提高 `cutout.white_threshold`，或略微增大 `detect.padding` 后重新运行。

### 素材内部白色变透明

确认 `cutout.keep_inner_white: true`。V1 的默认逻辑只移除和裁剪图边缘连通的白底。

## 8. 控制台输出示例

```text
开始处理：sample.jpg
识别元素数量：36
已生成预览图：output/sample/preview/sample_preview.jpg
已生成透明底图：output/sample/sample_transparent.png
已输出 PNG：36 个
已生成 manifest：output/sample/manifest.json
已生成 Photoshop 脚本：output/sample/build_psd.jsx
已生成 PSD：output/sample/sample_auto.psd
处理完成
```
