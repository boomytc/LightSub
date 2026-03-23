# LightSub
LightSub 利用 FunASR 对音频进行带时间戳的语音识别，并提供零配置可视化界面，轻松完成字幕 分割、合并、删除、快速定位 等可视化编辑。

## 安装与环境配置

本项目使用 [uv](https://github.com/astral-sh/uv) 进行极速依赖管理和环境隔离。推荐使用 Python 3.12（最低支持 3.10）。

### 1. 安装 uv
如果您尚未安装 `uv`，可以通过以下命令进行安装：

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 初始化环境与安装依赖
在项目根目录下运行以下命令，`uv` 将自动创建虚拟环境并同步所有必要依赖：

```bash
# 创建基于 Python 3.12 的虚拟环境
uv venv --python 3.12

# 同步并安装项目依赖
uv sync
```

### 3. 激活虚拟环境
在运行项目之前，请激活虚拟环境：

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\activate
```

## 使用方法

### 项目结构

- `lightsub_webui.py`：主 WebUI 入口。
- `datasets_list_create.py`：音频切分与识别入口。
- `webui/`：WebUI 内部模块，包含切分修正和表格编辑界面逻辑。
- `scripts/demo/`：FunASR 示例脚本与示例音频。
- `scripts/tools/`：辅助转换脚本。
- `finetune/`：微调数据准备脚本。
- `dataset/`、`models/`：本地运行数据与模型目录。

### 1. 提取与切分音频（生成数据集）

将包含原始音频文件（支持 `wav`, `mp3`, `flac`, `m4a`, `aac`）的目录作为参数传入脚本，自动进行 VAD 切片和语音识别：

```bash
# 如果已激活虚拟环境，可直接使用 python
python datasets_list_create.py /path/to/your/audio_dir

# 或者使用 uv run（无需手动激活环境，现代化推荐做法）
uv run datasets_list_create.py /path/to/your/audio_dir
```

> **注意**：脚本依赖 `ffmpeg` 进行音频处理，请确保系统已安装 `ffmpeg` 并添加至环境变量。执行完成后，默认会在 `dataset/audio_list/list.csv` 生成数据列表，切分好的音频会存放在 `dataset/audio_split` 目录中。

### 2. 启动可视化编辑界面 (WebUI)

使用上一步生成的 `list.csv` 启动 WebUI：

```bash
# 如果已激活虚拟环境
python lightsub_webui.py --load_file dataset/audio_list/list.csv

# 或者使用 uv run
uv run lightsub_webui.py --load_file dataset/audio_list/list.csv
```

启动后，在浏览器中打开控制台输出的本地地址（默认为 `http://127.0.0.1:7860`），您将看到以下功能面板：

- **切分与修正 (SubFix Tab)**：可逐条试听切分后的音频，核对、修改对应文本，并支持调整音频合并间隔、按秒重新切分音频或删除不需要的音频片段。
- **编辑 (Edit Tab)**：提供全局的数据表格视图，支持关键字筛选、全局查找替换以及数据保存。

### 3. 运行示例与辅助脚本

```bash
# FunASR 示例
uv run scripts/demo/demo_paraformer.py

# 列表格式转换
uv run scripts/tools/txt2csv.py dataset/audio_list/list.txt
```
