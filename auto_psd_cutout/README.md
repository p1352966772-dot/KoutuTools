# KoutuTools - 自动抠图拆图层工具

批量处理白底/绿幕拼版素材大图：自动识别每个素材区域 → 抠图 → 导出PSD分层文件。

## 环境要求

- Windows 10 / 11
- Python 3.10+
- Adobe Photoshop（可选，不装也能导出 JSX）

## 新电脑配置

```bash
git clone https://github.com/p1352966772-dot/KoutuTools.git
cd KoutuTools\auto_psd_cutout
pip install -r requirements.txt
```

首次运行会自动下载 BRIA 模型（~200MB）和 RapidOCR（~15MB）。

## 配置

编辑 `config.yaml`：

```yaml
input_dir: D:\你的素材目录    # 支持子文件夹
output_dir: D:\你的输出目录
```

## 使用

```bash
python main.py              # 单次批量处理
python main.py --watch      # 监控模式：每10秒扫描新文件自动处理
python main.py --debug      # 调试模式：输出预览图等中间文件
python main.py --no-photoshop  # 不调用PS，只生成JSX
```

或双击 `run.bat` 选模式。

## 输入结构

```
input/
├── 素材包A/
│   ├── a.jpg
│   └── b.ai
├── 素材包B/
│   └── c.png
└── d.jpg
```

支持格式：`.jpg` `.jpeg` `.png` `.webp` `.ai`

## 输出结构

```
output/素材包A/
├── a.psd
├── b.psd
└── _work/               # 中间文件
    ├── build_psd.jsx
    └── full_cutout.png
```

`--debug` 模式额外输出 `_work/preview/` 和 `_work/ocr_cleaned.png`。

## 开机自启

双击 `setup_task.bat` 注册 Windows 计划任务，重启后自动运行监控。

## 参数说明

| 配置 | 说明 | 默认值 |
|------|------|--------|
| `input_dir` | 素材目录 | `input` |
| `output_dir` | 输出目录 | `output` |
| `watch_interval` | 监控扫描间隔(秒) | `10` |
| `grid.enabled` | 行列分割算法 | `true` |
| `grid.bg_tolerance` | 背景色容差 | `30` |
| `rgba_crop.white_bg_alpha` | true=白底连通域, false=BRIA模型 | `false` |
| `photoshop.enabled` | 自动调用PS | `true` |

## 抠图策略

| 场景 | 方式 |
|------|------|
| 普通图片 | BRIA RMBG-1.4 深度学习模型 |
| AI源文件 | 绿幕渲染 + 色键抠图（自动检测） |
