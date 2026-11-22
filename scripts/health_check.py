"""Check dependencies, local model, PyTorch artifact, demo data, and SQLite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas
import streamlit
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workmate.config import get_settings
from workmate.db import WorkMateDatabase
from workmate.intent_model import PyTorchIntentModel
from workmate.llm import LLMClient, config_from_settings


def main() -> None:
    settings = get_settings()
    client = LLMClient(config_from_settings(settings))
    model_online, model_message = client.health()
    intent_model = PyTorchIntentModel(settings.intent_model_path)
    prediction = intent_model.predict(
        "分析上传的销售Excel并生成月度统计图和报告"
    )
    demo_dir = settings.data_dir / "demo"
    demo_files = {
        path.name: path.stat().st_size
        for path in sorted(demo_dir.glob("*"))
        if path.is_file()
    }
    database = WorkMateDatabase(settings.database_path)
    tasks = database.list_tasks(limit=5)
    payload = {
        "python": sys.version,
        "d_drive_project": str(ROOT),
        "dependencies": {
            "torch": torch.__version__,
            "streamlit": streamlit.__version__,
            "pandas": pandas.__version__,
        },
        "model": {
            "online": model_online,
            "message": model_message,
            "provider": settings.llm_provider,
            "name": settings.ollama_model,
        },
        "pytorch": {
            **intent_model.metadata(),
            "probe": prediction.label,
            "confidence": prediction.confidence,
        },
        "demo_files": demo_files,
        "database": {
            "path": str(settings.database_path),
            "task_count_sample": len(tasks),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not model_online:
        raise SystemExit(1)
    if prediction.label != "sales_analysis":
        raise SystemExit(2)
    if len(demo_files) < 4:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
