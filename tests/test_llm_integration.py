import os

import pytest

from workmate.config import get_settings
from workmate.llm import LLMClient, config_from_settings


@pytest.mark.integration
def test_configured_llm_returns_real_answer():
    if os.getenv("RUN_LLM_TESTS") != "1":
        pytest.skip("Set RUN_LLM_TESTS=1 to call the configured model.")
    client = LLMClient(config_from_settings(get_settings()))
    healthy, message = client.health()
    assert healthy, message
    answer = client.chat(
        [{"role": "user", "content": "只回答四个字：调用成功"}],
        temperature=0.1,
        max_tokens=40,
    )
    assert answer.strip()
