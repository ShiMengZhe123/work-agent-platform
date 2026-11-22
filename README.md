# WorkMate AI —— 智能办公任务助手

WorkMate AI 是一个可真实运行、可演示、可交付的轻量级智能体平台。项目使用
Python、Streamlit、PyTorch、Pandas 和 SQLite，并默认连接本机 Ollama
`qwen2.5:1.5b` 模型。无需 API Key 即可完成智能聊天、文档分析和 Excel
自动任务。

项目全部文件、虚拟环境、数据库、缓存、模型产物和输出均位于 D 盘：

`D:\PycharmProjects\WorkMateAI`

## 核心功能

### 1. AI 智能对话

- 中文聊天界面，支持多轮上下文。
- 对话写入 SQLite，可在“历史记录”中回看。
- 支持本地 Ollama 和 OpenAI 兼容接口。
- 模型离线、接口报错或 Key 缺失时会明确提示，不使用预设回答冒充模型。

### 2. 智能文档分析

- 支持 PDF、DOCX、XLSX、XLS、TXT。
- 自动提取文本、表格和 Excel 工作表信息。
- 使用大模型生成摘要、关键信息和后续事项。
- 支持针对文件内容提问。
- 文件过大、格式不支持、扫描版 PDF 无文字层时会给出明确错误。

### 3. 智能任务执行

示例任务：

> 分析上传的销售Excel，计算各月份销售额，生成统计图和总结报告。

执行流程：

1. PyTorch 意图分类器识别任务类型。
2. 规划器生成固定、可审计的步骤。
3. 只调用白名单内工具。
4. 自动读取 Excel、识别日期列和销售额列。
5. 按月份汇总销售额并生成 Matplotlib 图表。
6. 生成 Markdown 报告。
7. 页面实时显示各步骤状态。
8. 支持下载 Markdown 报告、PNG 图表和月度 CSV。

系统不会执行大模型生成的系统命令、任意 Python 代码或动态文件路径。

## PyTorch 模型的实际作用

`workmate/intent_model.py` 实现了一个真实训练、保存和推理的 PyTorch
多层感知机：

- 输入：用户任务文本的字、词二元组和英文词元。
- 模型：`Linear → ReLU → Dropout → Linear → ReLU → Linear`。
- 输出四类办公意图：
  - `general_chat`
  - `document_summary`
  - `document_qa`
  - `sales_analysis`
- 用途：为 Agent 规划器选择白名单工具链。
- 低置信度时使用受控关键词回退，保证演示稳定性。

模型文件：

`artifacts/intent_classifier.pt`

重新训练并查看推理结果：

```powershell
.\.venv\Scripts\python.exe scripts\train_intent_model.py
```

## 项目结构

```text
WorkMateAI/
├─ app.py                         # Streamlit 中文界面
├─ workmate/
│  ├─ agent.py                    # 任务规划与白名单工具执行
│  ├─ analysis.py                 # Excel 销售分析、图表和报告
│  ├─ config.py                   # 环境配置
│  ├─ db.py                       # SQLite 会话、任务和步骤日志
│  ├─ document_service.py         # 文档摘要与问答
│  ├─ documents.py                # PDF/Word/Excel/TXT 解析
│  ├─ intent_model.py             # 真实 PyTorch 意图分类器
│  ├─ llm.py                      # Ollama/OpenAI 模型客户端
│  └─ ui_theme.py                 # 界面视觉系统
├─ scripts/
│  ├─ create_demo_data.py
│  ├─ smoke_test_llm.py
│  └─ train_intent_model.py
├─ tests/
├─ data/
├─ outputs/
├─ artifacts/
├─ .streamlit/config.toml
├─ requirements.txt
├─ run.ps1
└─ README.md
```

## 环境状态

- Windows 11
- Python 3.10.6
- Streamlit 1.65.0
- PyTorch 2.14.1
- Pandas 2.3.3
- SQLite 3
- 本地 Ollama `qwen2.5:1.5b`
- Ollama 模型目录：`D:\OllamaModels`

pip 缓存和临时文件也已固定在项目 D 盘目录，不主动写入 C 盘。

## 启动方式

项目虚拟环境已经创建在：

`.venv`

直接启动：

```powershell
cd D:\PycharmProjects\WorkMateAI
.\run.ps1
```

或者：

```powershell
cd D:\PycharmProjects\WorkMateAI
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

浏览器打开：

<http://localhost:8501>

## 验收截图

- [应用首页](docs/screenshots/00_home.png)
- [真实本地模型对话](docs/screenshots/01_chat.png)
- [Excel 文档分析](docs/screenshots/02_document_analysis.png)
- [PyTorch 任务计划](docs/screenshots/03_auto_task_plan.png)
- [自动分析结果](docs/screenshots/04_auto_task_result.png)
- [任务历史](docs/screenshots/05_task_history.png)

## 模型配置

默认无需 API Key。启动前确认 Ollama 正在运行且存在 `qwen2.5:1.5b`：

```powershell
ollama list
ollama serve
```

默认配置见 `.env.example`。如需自定义，在项目根目录创建 `.env.local`：

```text
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:1.5b
LLM_TIMEOUT=180
```

OpenAI 兼容接口：

```text
LLM_PROVIDER=openai
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=你的Key
OPENAI_MODEL=gpt-4.1-mini
```

也可以在网页侧边栏临时切换提供方、地址和模型。界面输入的 API Key 只保存在
当前 Streamlit 会话，不写入代码或数据库。

## 演示数据

生成四类演示文件：

```powershell
.\.venv\Scripts\python.exe scripts\create_demo_data.py
```

输出：

- `data/demo/demo_sales_2026.xlsx`
- `data/demo/demo_project_note.docx`
- `data/demo/demo_contract.pdf`
- `data/demo/demo_meeting_notes.txt`

## 推荐演示流程

### 第一步：智能聊天

打开“智能聊天”，输入：

> 请给我一个适合技术项目周会的汇报提纲，包含进展、风险和下一步。

确认回答来自页面顶部显示的本地模型 `qwen2.5:1.5b`。

### 第二步：文档分析

打开“文件分析”，上传：

`data/demo/demo_sales_2026.xlsx`

观察自动读取的工作表、记录数和列信息。点击“生成智能摘要”，然后提问：

> 数据覆盖哪些月份？销售额字段是什么？

也可以测试：

- `demo_contract.pdf`：提问合同金额和验收负责人。
- `demo_project_note.docx`：提问交付范围和截止日期。
- `demo_meeting_notes.txt`：生成会议摘要。

### 第三步：自动任务

打开“自动任务”，上传同一个 Excel，或点击“使用演示 Excel”。任务输入：

> 分析上传的销售Excel，计算各月份销售额，生成统计图和总结报告。

点击“生成计划并执行”后，页面会显示：

- PyTorch 意图分类和置信度；
- 固定白名单执行计划；
- 五个工具步骤的实时状态；
- 销售额合计、月均、峰值月份；
- 月度汇总表；
- 销售趋势图；
- Markdown 分析报告；
- 报告、图表、CSV 下载按钮。

### 第四步：历史记录

打开“历史记录”，查看：

- 保存的多轮对话；
- 任务意图和置信度；
- 每一步工具执行记录；
- 已生成的报告内容。

## 自动测试

运行全部本地测试：

```powershell
cd D:\PycharmProjects\WorkMateAI
.\.venv\Scripts\python.exe -m pytest -q
```

运行真实模型集成测试：

```powershell
$env:RUN_LLM_TESTS="1"
.\.venv\Scripts\python.exe -m pytest -q -m integration
```

单独验证模型是否真实返回：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_llm.py
```

本次实际测试结果记录在 `test_results.log`，浏览器演示结果记录在
`demo_smoke.log` 和 `outputs/demo_end_to_end.json`。

## 数据与安全

- 上传文件仅保存在本机 `data/uploads/`。
- SQLite 数据库位于 `data/workmate.db`。
- 图表和报告位于 `outputs/`。
- 不执行大模型生成的命令或代码。
- 文档中的指令被视为不可信内容。
- `.env.local` 已被 `.gitignore` 排除。
- 不自动删除或移动用户文件。

## 已知边界

- 扫描版 PDF 需要先 OCR；当前版本会明确提示，而不是返回空摘要。
- 单个上传文件默认限制为 20 MB。
- PyTorch 模块负责意图识别和工具选择，不冒充通用大语言模型。
- 通用自然语言回答由本地 Ollama 或配置的远程模型完成。
- 大模型不可用时，Excel 确定性分析仍可完整运行；文档摘要会明确标记为
  “本地提取式摘要”。
