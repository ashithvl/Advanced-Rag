"""Multimodal page-image embeddings via ColPali (late-interaction)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from PIL.Image import Image

from advanced_rag.config import get_settings


@lru_cache(maxsize=1)
def get_colpali_embedder() -> Any:
    """Load ColPali model + processor lazily.

    Returns an object exposing `.embed_images(images)` and `.embed_queries(texts)`.
    """
    import torch
    from colpali_engine.models import ColPali, ColPaliProcessor

    settings = get_settings()
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = ColPali.from_pretrained(
        settings.colpali_model,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device,
    ).eval()
    processor = ColPaliProcessor.from_pretrained(settings.colpali_model)
    return _ColPaliWrapper(model=model, processor=processor, device=device)


class _ColPaliWrapper:
    def __init__(self, model: Any, processor: Any, device: str) -> None:
        self.model = model
        self.processor = processor
        self.device = device

    def embed_images(self, images: list[Image]) -> Any:
        import torch

        batch = self.processor.process_images(images).to(self.device)
        with torch.no_grad():
            return self.model(**batch)

    def embed_queries(self, queries: list[str]) -> Any:
        import torch

        batch = self.processor.process_queries(queries).to(self.device)
        with torch.no_grad():
            return self.model(**batch)
