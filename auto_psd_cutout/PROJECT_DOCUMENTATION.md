# Project Documentation: KoutuTools
# KoutuTools

**KoutuTools** 是一个本地批量处理白底美工素材拼版图的抠图工具集。输入包含多行多列小素材的大图，自动识别每个素材的矩形区域，裁剪为独立透明 PNG，生成预览图和 manifest JSON，并可自动调用 Photoshop 组装为 PSD 文件。

---

## 目录结构

```text
KoutuTools/
├── auto_psd_cutout/
│   ├── main.py                  # 入口：CLI 参数解析 + 批量调度
│   ├── config.yaml              # 所有算法参数
│   ├── run.bat                  # Windows 一键启动
│   ├── requirements.txt         # Python 依赖
│   ├── README.md                # 用户使用说明
│   ├── input/                   # 放入待处理的拼版图
│   ├── output/                  # 输出目录（自动生成）
│   ├── src/
│   │   ├── config.py            # YAML 配置加载
│   │   ├── pipeline.py          # 主流程编排
│   │   ├── grid_cutter.py       # [当前默认] 行列分割（OCR + 背景色匹配）
│   │   ├── detector.py          # [备用] 传统 CV 检测（多尺度形态学）
│   │   ├── rembg_utils.py       # 抠图模型封装（BRIA / U2-Net / 白底连通域）
│   │   ├── cutout.py            # 从检测框裁剪为透明 PNG
│   │   ├── postprocess.py       # 扣图后的二次合并（兜底合并）
│   │   ├── preview.py           # 检测结果预览图画布
│   │   ├── manifest.py          # manifest.json 输出
│   │   ├── photoshop_jsx.py     # 生成 Photoshop JSX 脚本
│   │   ├── photoshop_runner.py  # 通过 COM 自动调用 Photoshop
│   │   ├── preprocess.py        # OCR 之外的规则文字清理（备用方案）
│   │   ├── pipeline_new.py      # 实验性新流程（当前仅为占位）
│   │   └── __init__.py
│   ├── scripts/                 # 开发/调试脚本（历史遗留）
│   ├── test_*.py                # 独立测试脚本
│   └── build_psd.jsx            # Photoshop 模板（被 generate_jsx 嵌入）
├── _fix_bg2.py                  # 未完成的 bg 检测修复（工作区临时）
├── _fix_bg3.py                  # 未完成的 bg 检测修复（工作区临时）
├── _test_all.py                 # 工作区临时测试脚本
├── test_birefnet.py             # BiRefNet 模型测试
└── test_rmbg2_alpha.py          # rembg 输出测试
```

---

## 核心流程

### 主流程（`pipeline.py:process_image`）

```
输入图片 → 读取配置
    ↓
[Step 1] 生成全图 alpha 掩码
    ├─ 白底连通域抠图（white_bg_alpha=true，默认）→ get_white_bg_alpha()
    └─ BRIA RMBG-1.4（white_bg_alpha=false）→ get_bria14_alpha()
    ↓
[Step 2] 元素检测
    ├─ grid.enabled=true（默认）→ detect_ui_elements_grid()
    │   ├─ OCR 去中文文字（用背景色填充）
    │   ├─ 行检测：逐行扫描，全部像素匹配背景 → 间隙
    │   └─ 列检测：在行内逐列扫描，同上
    └─ grid.enabled=false → detect_ui_elements()
        └─ 传统 CV 管线（灰度/自适应阈值/Canny/多尺度形态学）
    ↓
[Step 3] 保存全图透明底图（OCR 涂白后的图作为源）
    ↓
[Step 4] 生成预览图（框线 + 编号 + 行色条 + 列间隙线）
    ↓
[Step 5] 生成 Photoshop JSX 脚本
    ↓
[Step 6] 可选：自动调用 Photoshop 生成 PSD
```

### 输出目录结构

```text
output/原图文件名/
├── items/
│   ├── item_001.png              # 每个元素的独立透明PNG
│   ├── item_002.png
│   └── ...
├── preview/
│   └── 原图文件名_preview.jpg     # 检测框预览图
├── full_cutout.png               # 全图透明底图（隐藏层）
├── manifest.json                 # 坐标清单
├── build_psd.jsx                 # Photoshop 脚本
└── 原图文件名_auto.psd           # PSD 文件（自动生成）
```

---

## 两种分割算法

### 方案 A：grid_cutter（当前默认）✅

`grid.enabled: true` → `grid_cutter.py`

专为白底（或浅色底）行列排列密集排版图设计。

**算法步骤：**
1. **OCR 去中文文字** — 用 RapidOCR 检测文字区域，仅识别包含中文的文本框，用检测到的背景色填充（不是白色，避免引入干扰色）
2. **自动背景色检测** — 从图片四角的 50×50 区域采样，取中位数作为背景色。可在 config 中固定 `bg_color: [R, G, B]`
3. **行检测** — 逐行扫描，一行中**所有像素**都与背景色匹配（通道差 ≤ bg_tolerance）→ 间隙行；连续 N 行间隙视为行分割线
4. **列检测** — 在每行区域内逐列扫描，同理
5. **输出** — 每个格子生成 (x, y, w, h) 的 bounding box

**关键参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bg_color` | auto | `auto` 自动检测，或 `[R,G,B]` 固定值 |
| `bg_tolerance` | 30 | 颜色匹配容差，0-255 |
| `min_gap_rows` | 3 | 连续 N 行全背景色 → 行间隙 |
| `min_gap_cols` | 3 | 连续 N 列全背景色 → 列间隙 |
| `min_row_height` | 10 | 过滤太矮的行 |
| `min_crop_area` | 200 | 过滤过小的裁剪区域 |

**已知限制：**
- 白底（或接近白色背景）效果最佳
- 非纯白背景需正确检测 `bg_color`（auto 模式从四角采样）
- 有装饰边框的图片（边框元素延伸到全图高度/宽度）会导致每行都有非背景像素 → 无法检测到间隙。这是当前 `dd553eb8` 图片的问题根因

### 方案 B：detector（传统 CV，备用）

`grid.enabled: false` → `detector.py`

多尺度形态学 + 连通域分析管线。

**算法步骤：**
1. **多掩码融合** — 灰度阈值 + 自适应阈值 + Canny 边缘
2. **多尺度形态学** — 小/中/大三组 kernel 做闭运算
3. **孔洞填充** — 连通域孔洞补偿
4. **行投影分割** — 对填充后掩码做水平投影
5. **逐行检测** — 每行内做连通域分析 + 文字/图标合并
6. **两级检测** — 对大框内部做二次检测
7. **后处理合并** — 基于位置/尺寸做保守合并
8. **文字抑制** — 过滤太小的文字区块
9. **噪声过滤** — 按面积/宽高比/占比
10. **行聚类** — 将检测到的框按 Y 坐标聚成行

**适用场景：** 背景不是纯白/有渐变色/元素排列不规则的图片。

---

## 抠图模块

`rembg_utils.py` 提供三种抠图方法，按优先级尝试：

### 方法 1：白底连通域抠图（默认，推荐）
- `rgba_crop.white_bg_alpha: true`
- 原理：白色像素做 4-连通域分析，不触碰图像边缘的白色区域→前景，触碰的→背景
- 100% 保留元素内的白色内容
- 零模型下载，毫秒级
- 适合纯白背景

### 方法 2：BRIA RMBG-1.4
- `rgba_crop.white_bg_alpha: false` 或白底法失败时自动回退
- Hugging Face `briaai/RMBG-1.4`（178MB，44M 参数）
- 预处理：直接拉伸到 1024×1024 → 归一化 [−0.5, 0.5]
- 后处理：1px GaussianBlur → 缩回原图大小
- 会误删元素内白色 → 自动调用 `refine_alpha_for_white_bg()` 保护

### 方法 3：rembg U2-Net
- BRIA 失败时的最终回退
- 调用 `rembg.new_session("u2net")`

---

## 文字清理

### OCR 文字清理（默认开启）
- `grid_cutter.py` 中 `remove_text()` 方法
- 使用 RapidOCR（`rapidocr_onnxruntime`）
- 仅检测**中文**文字（跳过英文，`\u4e00-\u9fff`）
- 涂白颜色 = 检测到的背景色，不是纯白
- 适用于分类标题、标签文字的清理

### 规则文字清理（备用，默认关闭）
- `preprocess.py` 中 `remove_label_text_by_rules()`
- 不需要 OCR 模型，纯 CV 规则
- 检测低饱和度 + 暗色小连通块
- 保护彩色区域附近的黑色线条不被误删
- 在 config 中 `label_text_cleanup.enabled: true` 启用

---

## 配置详解

完整 `config.yaml` 结构：

```yaml
input_dir: input                # 输入目录
output_dir: output              # 输出目录

grid:                           # grid_cutter 参数
  enabled: true                 # true=grid_cutter, false=传统detector
  bg_color: auto                # 背景色，auto 或 [R,G,B]
  bg_tolerance: 30
  min_gap_rows: 3
  min_gap_cols: 3
  min_row_height: 10
  min_crop_area: 200

detect:                         # 传统 detector 参数
  white_threshold: 230
  adaptive_block: 31
  adaptive_c: 2
  canny_low: 50
  canny_high: 150
  merge_h_gap: 18
  merge_iou: 0.4
  # ... 更多详见 config.yaml

rgba_crop:                      # 抠图参数
  enabled: true
  white_bg_alpha: true          # true=白底连通域，false=BRIA模型
  white_threshold: 230

photoshop:                      # Photoshop 参数
  enabled: true
  psd_name_suffix: _auto

preview:                        # 预览图参数
  draw_box: true
  draw_index: true
  draw_groups: true
  draw_rows: true
  box_thickness: 2

rmbg_scoring:                   # 检测框评分（默认关闭）
  enabled: false
  score_threshold: 0.35

label_text_cleanup:             # 规则文字清理（默认关闭）
  enabled: false
```

---

## Git 分支与提交历史

- `grid-cutter`（当前活动分支）— grid_cutter 替换传统 detector 后的开发分支
- 其他分支按需创建

提交历史：

| Commit | 说明 |
|--------|------|
| `bdaca87` | Initial commit |
| `6ef102a` | 用 grid_cutter 投影法替代分割算法 |
| `85ed049` | 增强预览图：列间隙线和行列编号 |
| `111b1b3` | 重写 grid_cutter 为推荐方案（OCR+非白占比） |
| `3f6c3ff` | 改为全白像素检测（非白占比 → 全白匹配） |
| `6a21b42` | 预览文件强制覆盖 |
| `cfb6c5b` | 用 OCR 涂白图作为抠图源图 |
| `fa43b57` | 自动检测背景色（替代硬编码白色） |

---

## 已知问题

### 1. `dd553eb8` 图片背景色检测失败
- 症状：识别为 1 个区域
- 根因：图片有白色装饰边框 + 浅色非白内背景。当前 bg 检测从四角采样 50×50，若边框元素延伸至全图高度，每行都有非背景像素 → 找不到间隙
- 尝试过的修复方向：
  - 多层边缘采样（5px/25px/60px 深度对比）
  - 更精细的量化（32级代替16级）
  - 自动裁剪边框后再检测
  - 尚未确定最终方案

### 2. 英文文字保留
- OCR 只涂中文（`\u4e00-\u9fff`），英文/POP 标签保持原样
- 若英文文字干扰行列检测，需手动补充规则

### 3. 预览文件覆盖
- 预览图保存前先 `.unlink()` 确保每次都重新生成

---

## 开发指南

### 添加新检测算法
1. 在 `src/` 下新建模块，实现 `detect_ui_elements_xxx(image_bgr, config)` 接口
2. 在 `pipeline.py` 中引用并替换调用
3. 返回格式：
   ```python
   {
       "boxes": [{"id": int, "x1": int, "y1": int, "x2": int, "y2": int,
                   "group_id": int, "col": int, "name": str, "rgba_path": None, "score": 1.0, "area": int}],
       "groups": [{"id": int, "name": str, "row_y": int, "row_h": int, "count": int}],
       "canvas_width": int, "canvas_height": int,
       "ocr_cleaned_bgr": np.ndarray | None,  # 如果有OCR涂白图，传过来做抠图源
   }
   ```

### 测试单个图片
```bash
python main.py --file input/test.jpg
```

### 常用测试代码片段
```python
from pathlib import Path
from src.grid_cutter import SmartGridSplitter

splitter = SmartGridSplitter()
boxes, clean_img = splitter.split("input/test.jpg")
print(f"{len(boxes)} boxes")
```

---

## 环境要求

- Python 3.10+
- Windows 10/11（Photoshop COM 接口需 Windows）
- Adobe Photoshop（可选，非必需）
- 首次运行自动下载模型：
  - RapidOCR（~15MB，自动下载）
  - BRIA RMBG-1.4（~178MB，触发时下载）
  - rembg U2-Net（~176MB，回退时下载）

### 依赖安装

```bash
pip install -r requirements.txt
```

---

## 工作区临时文件说明

项目根目录下有以下工作区临时文件（非项目正式代码）：

| 文件 | 用途 |
|------|------|
| `_fix_bg2.py` | 背景色检测修复尝试 #2 |
| `_fix_bg3.py` | 背景色检测修复尝试 #3 |
| `_test_all.py` | 遍历测试所有 input 图片 |
| `test_birefnet.py` | BiRefNet 模型独立测试 |
| `test_rmbg2_alpha.py` | rembg 输出格式测试 |

