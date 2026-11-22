"""Train and inspect the local PyTorch intent classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workmate.config import get_settings
from workmate.intent_model import PyTorchIntentModel


def main() -> None:
    settings = get_settings()
    model = PyTorchIntentModel(
        settings.intent_model_path,
        force_train=True,
    )
    probes = [
        "分析上传的销售Excel，计算各月份销售额，生成统计图和总结报告。",
        "总结这份PDF的关键结论。",
        "合同里的验收人是谁？",
        "帮我写一封会议通知。",
    ]
    results = {}
    for probe in probes:
        prediction = model.predict(probe)
        results[probe] = {
            "label": prediction.label,
            "confidence": prediction.confidence,
        }
    print(
        json.dumps(
            {
                "metadata": model.metadata(),
                "predictions": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
