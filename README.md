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
