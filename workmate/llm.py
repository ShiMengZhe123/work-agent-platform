"""Real LLM transport for Ollama and OpenAI-compatible APIs."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Iterable

import requests

from workmate.config import SYSTEM_PROMPT, Settings


class LLMError(RuntimeError):
    """Raised when the configured language model cannot answer."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: str = ""
    timeout: int = 180


def config_from_settings(settings: Settings) -> LLMConfig:
    if settings.llm_provider == "openai":
        return LLMConfig(
            provider="openai",
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout,
        )
    return LLMConfig(
        provider="ollama",
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.llm_timeout,
    )


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session = requests.Session()

    @property
    def label(self) -> str:
        return f"{self.config.provider} / {self.config.model}"

    def health(self) -> tuple[bool, str]:
        try:
            if self.config.provider == "ollama":
                response = self.session.get(
                    f"{self.config.base_url}/api/tags", timeout=5
                )
                response.raise_for_status()
                models = {
                    item.get("name", "")
                    for item in response.json().get("models", [])
                }
                if self.config.model in models:
                    return True, f"本地模型可用：{self.config.model}"
                return False, f"Ollama 可用，但未找到模型 {self.config.model}"

            if not self.config.api_key:
                return False, "未配置 OPENAI_API_KEY"
            response = self.session.get(
                f"{self.config.base_url}/models",
                headers=self._headers(),
                timeout=8,
            )
            response.raise_for_status()
            return True, f"模型接口可用：{self.config.model}"
        except Exception as exc:
            return False, f"模型健康检查失败：{exc}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: Iterable[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1200,
    ) -> str:
        tagged_system_prompt = (
            SYSTEM_PROMPT
            + "\n先完成必要的内部思考，但只把最终用户可见内容放在"
            "<final>和</final>之间。不要在 final 标签内展示推理过程。"
        )
        payload_messages = [{"role": "system", "content": tagged_system_prompt}]
        payload_messages.extend(
            {
                "role": item["role"],
                "content": str(item["content"]),
            }
            for item in messages
            if item.get("role") in {"user", "assistant", "system"}
        )

        if self.config.provider == "ollama":
            return self._ollama_chat(
                payload_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._openai_chat(
            payload_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _ollama_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = f"{self.config.base_url}/api/chat"
        predict_limit = max_tokens
        if self.config.model.lower().startswith("qwen3"):
            predict_limit = max(max_tokens, 700)
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "think": self.config.model.lower().startswith("qwen3"),
            "options": {
                "temperature": temperature,
                "num_predict": predict_limit,
                "num_ctx": 8192,
            },
            "keep_alive": "10m",
        }
        last_error = ""
        for attempt in range(2):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = (data.get("message") or {}).get("content", "").strip()
                if content:
                    return self._extract_final_answer(content)
                last_error = "模型返回为空"
            except requests.RequestException as exc:
                last_error = str(exc)
            except (ValueError, KeyError) as exc:
                last_error = f"模型响应格式异常：{exc}"
            if attempt == 0:
                time.sleep(1)
        raise LLMError(f"Ollama 调用失败：{last_error}")

    @staticmethod
    def _extract_final_answer(content: str) -> str:
        match = re.search(
            r"<final>(.*?)</final>",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        stripped = re.sub(
            r"^\s*<final>\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )
        stripped = re.sub(
            r"\s*</final>\s*$",
            "",
            stripped,
            flags=re.IGNORECASE,
        )
        if stripped != content:
            return stripped.strip()
        marker = content.rfind("最终答案：")
        if marker >= 0:
            return content[marker + len("最终答案：") :].strip()
        return content.strip()

    def _openai_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if not self.config.api_key:
            raise LLMError("未配置 OPENAI_API_KEY，无法调用远程模型")
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = self.session.post(
                url,
                headers=self._headers(),
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            if not content:
                raise LLMError("模型返回为空")
            return content
        except requests.RequestException as exc:
            raise LLMError(f"远程模型调用失败：{exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"远程模型响应格式异常：{exc}") from exc
