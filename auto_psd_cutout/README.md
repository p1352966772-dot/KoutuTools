# KoutuTools - 自动抠图拆图层工具

批量处理白底/绿幕拼版素材大图：自动识别每个素材区域 → 抠图 → 导出透明PNG → 生成PSD分层文件。

## 环境要求

- Windows 10 / 11
- Python 3.10+
- Adobe Photoshop（可选，不装也能导出PNG+JSX）

## 新电脑配置步骤

```bash
# 1. 安装 Python（勾选 "Add to PATH"）
#    https://www.python.org/downloads/

# 2. 进入项目目录
cd auto_psd_cutout

# 3. 安装依赖（首次会自动下载 AI 模型 ~200MB）
pip install -r requirements.txt
```

## 配置

编辑 `config.yaml`，改输入输出路径：

```yaml
input_dir: input       # 素材目录，可改绝对路径如 D:\workspace\素材
output_dir: output     # 输出目录
```

## 使用

```bash
# 单次批量处理
python main.py

# 只处理指定文件
python main.py --file input/img1.jpg

# 监控模式：每10秒扫描新文件自动处理
python main.py --watch

# 不调用 Photoshop（只生成 PNG + JSX）
python main.py --no-photoshop
```

或双击 `run.bat`，菜单选择模式。

## 支持格式

`.jpg` `.jpeg` `.png` `.webp` `.ai`（AI文件必须有嵌入式PDF）

## 输出结构

```
output/文件名/
├── ocr_cleaned.png       # OCR涂抹中文后的图
├── ocr_cutout.png        # 透明底抠图层（PS分层源）
├── preview/
│   └── xxx_preview.jpg   # 检测框预览图
├── build_psd.jsx          # Photoshop 脚本
└── xxx_auto.psd           # PSD 分层文件
```

## 开机自启

双击 `setup_task.bat` 注册 Windows 计划任务，重启后自动运行监控。

管理命令：
```bash
schtasks /query /tn KoutuTools_Watch    # 查看状态
schtasks /delete /tn KoutuTools_Watch /f # 删除
```

## 参数说明

| 配置 | 说明 | 默认值 |
|------|------|--------|
| `input_dir` | 素材源目录 | `input` |
| `output_dir` | 输出目录 | `output` |
| `watch_interval` | 监控扫描间隔(秒) | `10` |
| `grid.enabled` | 行列分割算法 | `true` |
| `grid.bg_tolerance` | 背景色容差 | `30` |
| `grid.detect_margin` | 边缘忽略像素(防黑边) | `2` |
| `rgba_crop.white_bg_alpha` | true=白底连通域, false=BRIA模型 | `false` |
| `rgba_crop.protect_inner_white` | BRIA后保护内部白色 | `false` |
| `photoshop.enabled` | 自动调用Photoshop | `true` |

## 抠图策略

| 场景 | 方式 |
|------|------|
| 普通图片 | BRIA RMBG-1.4 深度学习模型 |
| AI源文件 | 绿幕渲染 + 色键抠图（自动检测） |
| 白底图（手动切换） | 边缘连通域分析（设置 `white_bg_alpha: true`） |
