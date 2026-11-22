# WorkMate AI 交付说明

## 交付目录

`D:\PycharmProjects\WorkMateAI`

所有项目文件、虚拟环境、数据库、缓存、演示数据、PyTorch 模型产物和运行输出
均位于 D 盘。

## 已完成能力

| 模块 | 实现 | 状态 |
|---|---|---|
| 智能聊天 | Ollama `qwen2.5:1.5b` 多轮对话、SQLite 历史 | 已验证 |
| 文件分析 | PDF、DOCX、XLSX、XLS、TXT 文本/表格提取、摘要和问答 | 已验证 |
| 自动任务 | PyTorch 意图识别、白名单计划、五步 Excel 分析 | 已验证 |
| 数据存储 | SQLite 会话、消息、任务、执行步骤 | 已验证 |
| 结果交付 | Markdown 报告、PNG 图表、月度 CSV 下载 | 已验证 |
| 界面 | Streamlit 中文四页签应用 | 浏览器实测 |
| 安全 | 禁止任意代码、系统命令和模型生成代码执行 | 代码审查通过 |

## 真实运行版本

- Python 3.10.6
- PyTorch 2.14.1+cpu
- Streamlit 1.65.0
- Pandas 2.3.3
- SQLite 3
- Ollama `qwen2.5:1.5b`

## 启动命令

```powershell
cd D:\PycharmProjects\WorkMateAI
.\run.ps1
```

浏览器访问：<http://localhost:8501>

## 已执行验收

```text
10 passed
```

测试范围：

- PyTorch 模型训练、保存、加载和推理；
- PDF、Word、Excel、TXT 解析；
- 销售分析、图表、报告和 CSV 生成；
- Agent 白名单计划与任务执行；
- SQLite 会话、消息、任务和步骤日志；
- Streamlit AppTest 四页签启动；
- Ollama 真实模型调用集成测试。

## 真实演示结果

演示 Excel：

`data/demo/demo_sales_2026.xlsx`

实际分析结果：

- 有效记录：96 条；
- 覆盖月份：12 个月；
- 销售额合计：1,607,126.32；
- 月均销售额：133,927.19；
- 峰值月份：2026-06；
- 峰值为：166,859.84；
- PyTorch 意图：`sales_analysis`；
- 分类置信度：100%。

## 交付证据

- 界面截图：`docs/screenshots/`
- 完整测试日志：`test_results.log`
- 端到端演示日志：`demo_smoke.log`
- 端到端结果 JSON：`outputs/demo_end_to_end.json`
- 销售图表：`outputs/*.png`
- 月度数据：`outputs/*.csv`
- 分析报告：`outputs/*_report.md`
- PyTorch 模型：`artifacts/intent_classifier.pt`
- SQLite 数据库：`data/workmate.db`

## PyTorch 的实际用途

PyTorch 模型不是安装后闲置依赖，而是 Agent 的真实任务规划组件：

1. 对自然语言任务进行文本向量化；
2. 通过多层感知机输出四类办公意图；
3. 将意图传给 Agent 规划器；
4. 选择白名单工具链；
5. 使用本地确定性工具完成数据和图表处理。

模型在 D 盘本地训练、保存和加载，推理过程中不依赖外部 API。
