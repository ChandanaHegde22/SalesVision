"""Thin, provider-agnostic wrapper around an LLM text-completion call.

The Conversational Data Analyst never lets the LLM touch the dataframe.
It only ever asks the LLM to (a) turn a question into a structured JSON
query spec, (b) phrase a short answer from facts we already computed, or
(c) suggest follow-up questions. This module is the single place that
talks to the outside world, which keeps the rest of the feature testable
and provider-independent.

Supported providers (set via the LLM_PROVIDER environment variable):
    - "anthropic" (default): uses ANTHROPIC_API_KEY, Claude models.
    - "gemini": uses GEMINI_API_KEY, Google's Generative AI SDK.

If no API key is configured, `is_available()` returns False and callers
fall back to the deterministic, rule-based path (see query_generator.py)
instead of crashing the app.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


class LLMUnavailableError(RuntimeError):
    """Raised when a caller tries to use the LLM but no provider is configured."""


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: Optional[str]


def _load_config() -> LLMConfig:
    provider = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()
    if provider == "gemini":
        return LLMConfig(
            provider="gemini",
            model=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
    return LLMConfig(
        provider="anthropic",
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


class LLMClient:
    """Provider-agnostic text-completion client used only for planning/phrasing."""

    def __init__(self) -> None:
        self._config = _load_config()

    def is_available(self) -> bool:
        return bool(self._config.api_key)

    @property
    def provider(self) -> str:
        return self._config.provider

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
        """Return the raw text completion for a system/user prompt pair.

        Raises LLMUnavailableError if no provider is configured. Any SDK-level
        error is wrapped and re-raised as LLMUnavailableError so the caller can
        cleanly fall back to the rule-based path instead of crashing the UI.
        """
        if not self.is_available():
            raise LLMUnavailableError(
                f"No API key configured for provider '{self._config.provider}'. "
                "Set ANTHROPIC_API_KEY or GEMINI_API_KEY (and LLM_PROVIDER) to enable "
                "free-form question understanding."
            )

        try:
            if self._config.provider == "gemini":
                return self._complete_gemini(system_prompt, user_prompt, max_tokens)
            return self._complete_anthropic(system_prompt, user_prompt, max_tokens)
        except LLMUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - convert any SDK error to a clean fallback signal
            raise LLMUnavailableError(f"LLM call failed: {exc}") from exc

    def _complete_anthropic(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        import anthropic  # imported lazily so the dependency is optional

        client = anthropic.Anthropic(api_key=self._config.api_key)
        response = client.messages.create(
            model=self._config.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return "".join(parts).strip()

    def _complete_gemini(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        import google.generativeai as genai  # imported lazily so the dependency is optional

        genai.configure(api_key=self._config.api_key)
        model = genai.GenerativeModel(self._config.model, system_instruction=system_prompt)
        response = model.generate_content(
            user_prompt,
            generation_config={"max_output_tokens": max_tokens},
        )
        return (response.text or "").strip()
