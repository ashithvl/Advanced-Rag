"""Chat LLM factory."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from advanced_rag.config import get_settings


def get_chat_llm(temperature: float | None = None, model: str | None = None) -> ChatOpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the chat LLM.")
    return ChatOpenAI(
        model=model or settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        api_key=settings.openai_api_key,
    )
