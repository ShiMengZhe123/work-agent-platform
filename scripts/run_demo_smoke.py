"""End-to-end demo without browser: model, document, PyTorch, tools, files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workmate.agent import AgentExecutor, AgentPlanner
from workmate.config import get_settings
from workmate.document_service import DocumentService
from workmate.documents import extract_document
from workmate.intent_model import PyTorchIntentModel
from workmate.llm import LLMClient, config_from_settings


def main() -> None:
    settings = get_settings()
    demo = settings.data_dir / "demo" / "demo_sales_2026.xlsx"
    if not demo.exists():
        raise SystemExit("演示数据不存在，请先运行 scripts/create_demo_data.py")

    client = LLMClient(config_from_settings(settings))
    healthy, health_message = client.health()
    if not healthy:
        raise SystemExit(health_message)

    document = extract_document(demo, settings.max_upload_mb)
    service = DocumentService(client)
    summary = service.summarize(document)
    answer = service.answer(document, "这份数据包含哪些字段？覆盖多少行？")

    model = PyTorchIntentModel(settings.intent_model_path)
    task_text = (
        "分析上传的销售Excel，计算各月份销售额，生成统计图和总结报告。"
    )
    planner = AgentPlanner(model, settings.outputs_dir)
    plan = planner.plan(task_text, has_excel=True, excel_path=demo)
    result = AgentExecutor(settings.outputs_dir, client).execute_sales(demo)

    payload = {
        "model_health": health_message,
        "document": {
            "name": document.name,
            "characters": document.characters,
            "summary": summary.content,
            "summary_used_llm": summary.used_llm,
            "question": "这份数据包含哪些字段？覆盖多少行？",
            "answer": answer.content,
            "answer_used_llm": answer.used_llm,
        },
        "pytorch": {
            "metadata": model.metadata(),
            "intent": plan.intent,
            "confidence": plan.confidence,
            "steps": [step.tool for step in plan.steps],
        },
        "analysis": {
            "metrics": result.metrics,
            "chart_path": str(result.chart_path),
            "csv_path": str(result.csv_path),
        },
    }
    output = settings.outputs_dir / "demo_end_to_end.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"RESULT={output}")


if __name__ == "__main__":
    main()
