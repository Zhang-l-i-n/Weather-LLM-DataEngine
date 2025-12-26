# Weather-LLM-DataEngine 🌤️🤖

**Weather-LLM-DataEngine** 是一个气象数据处理与大模型应用流水线。它能够处理 ERA5 气象再分析数据，将其转化为结构化 CSV 格式，并调用大语言模型（如 Qwen, DeepSeek, GPT-4）生成专业的天气预报报告。此外，它还包含构建指令微调（Instruction Fine-Tuning, IFT）数据集的工具，用于训练气象领域的垂直大模型。

## ✨ 主要功能

1.  **多维气象数据提取**: 从 ERA5 GRIB 文件中提取温度、湿度、风向风速、云量、降水等关键要素，并进行复杂的衍生变量计算（如风力等级转换、云量代码判定、降水类型分类）。
2.  **自动化报告生成**: 基于 LangChain 框架，批量读取气象数据并调用 LLM 生成符合气象业务规范的文本报告。
3.  **思维链 (CoT) 支持**: 支持捕获推理模型（Reasoning Models）的思考过程 (`<think>`)，保留专家系统的推理痕迹。
4.  **SFT 数据集构建**: 将“气象数据 + 思考过程 + 最终报告”自动组装为 Alpaca 格式的 JSON 数据集，直接用于模型微调。

## 🛠️ 环境准备

### 1. 安装依赖

建议使用 Python 3.8+ 环境。

```bash
git clone [https://github.com/your-username/Weather-LLM-DataEngine.git](https://github.com/your-username/Weather-LLM-DataEngine.git)
cd Weather-LLM-DataEngine
pip install -r requirements.txt

```

### 2. 准备数据

本项目基于 ECMWF ERA5 数据。请确保你拥有以下格式的 GRIB 文件并放入 `raw_data/` 目录：

* **地面层数据 (`land.grib`)**: 包含 `t2m`, `d2m`, `u10`, `v10`, `tcc`, `lcc`, `tp`, `sf`, `cp` 等变量。
* **高空层数据 (`level.grib`)**: 包含不同气压层的相对湿度 `r` (用于云量判定)。

### 3. 配置环境变量

在项目根目录创建一个 `.env` 文件，填入你的 LLM API 密钥：

```ini
# .env 文件示例
CHAT_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
CHAT_API_BASE_URL=[https://api.your-provider.com/v1](https://api.your-provider.com/v1)
CHAT_MODEL=Qwen3-32B

```

---

## 🚀 使用指南

### 步骤 1: 数据预处理 (`generate_csv.py`)

解析 GRIB 原始数据，计算衍生变量，并按每 3 小时生成 CSV 序列。

```bash
# 基本用法 (默认处理 raw_data 下的数据)
python generate_csv.py

# 指定日期范围和文件路径
python generate_csv.py \
  --start_date 2021-01-01 \
  --end_date 2021-01-31 \
  --land_file ./raw_data/2021_land.grib \
  --level_file ./raw_data/2021_level.grib \
  --output_dir ./forecast_csv

```

**输出**: 在 `forecast_csv/` 目录下生成如 `2021-01-01_050000.csv` 的文件。

### 步骤 2: 生成天气报告 (`generate_report.py`)

读取上一步生成的 CSV，构建 Prompt，调用大模型生成预报文本。支持捕获模型的 CoT (Chain of Thought)。

```bash
# 基本用法
python generate_report.py

# 指定日期范围
python generate_report.py \
  --start_date 2021-01-01 \
  --end_date 2021-01-31 \
  --csv_dir ./forecast_csv \
  --output_dir ./report_by_llm

```

**输出**: 在 `report_by_llm/` 目录下生成：

* `DATE.txt`: 最终的天气预报报告。
* `DATE_think.txt`: 模型的思考推理过程（如果模型支持）。

### 步骤 3: 构建微调数据集 (`build_ift_data.py`)

将输入数据（CSV）和输出数据（思考 + 报告）合并，生成用于微调的 JSON 文件。

```bash
python build_ift_data.py \
  --start_date 2021-01-01 \
  --end_date 2021-01-31 \
  --output_file ./finetune_dataset.json

```

**输出**: 一个标准的 JSON 文件，格式如下：

```json
[
  {
    "instruction": "你是一个专业的气象预报员...",
    "input": "fsttime,max_temp_c,min_temp_c...\n2021-01-01T08:00:00,12.5,5.2...",
    "output": "<think>...\n</think>\n\n【天气预报】今天白天多云..."
  }
]

```

---

## ⚙️ 参数说明

| 脚本 | 参数 | 说明 | 默认值 |
| --- | --- | --- | --- |
| `generate_csv.py` | `--land_file` | 地面层 GRIB 文件路径 | `./raw_data/land.grib` |
|  | `--level_file` | 高空层 GRIB 文件路径 | `./raw_data/level.grib` |
| `generate_report.py` | `--csv_dir` | CSV 输入目录 | `./forecast_csv` |
|  | `--output_dir` | 报告输出目录 | `./report_by_llm` |
| `build_ift_data.py` | `--instruction_file` | 系统指令模板路径 | `./prompt/instruction.txt` |

## ⚠️ 注意事项

1. **数据版权**: ERA5 数据归 ECMWF 所有，请确保你遵守其使用条款。
2. **API 费用**: `generate_report.py` 会批量调用 LLM API，请注意 token 消耗。
3. **时区**: 代码中默认处理逻辑涉及 UTC 到北京时间 (CST) 的转换，请根据需要调整 `data_util` 或主逻辑中的时区设置。



## 📄 许可证

[MIT License](https://www.google.com/search?q=LICENSE)