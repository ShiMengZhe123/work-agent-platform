"""Verify that the configured model returns a real response."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workmate.config import get_settings
from workmate.llm import LLMClient, config_from_settings


def main() -> None:
    settings = get_settings()
    client = LLMClient(config_from_settings(settings))
    ok, message = client.health()
    if not ok:
        raise SystemExit(message)
    answer = client.chat(
        [
            {
                "role": "user",
                "content": "请只回答：WorkMate 本地模型调用成功。",
            }
        ],
        temperature=0.1,
        max_tokens=80,
    )
    print(json.dumps({"health": message, "answer": answer}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
