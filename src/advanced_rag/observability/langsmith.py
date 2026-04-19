"""Enable LangSmith tracing for the LangGraph pipeline if a key is configured."""

from __future__ import annotations

import os

from advanced_rag.config import get_settings


def enable_langsmith() -> bool:
    settings = get_settings()
    if not settings.langsmith_api_key:
        return False
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    return True
