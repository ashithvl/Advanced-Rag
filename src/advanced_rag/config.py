"""All runtime configuration in one place. Loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # OpenAI: chat (HyDE + final answer), vision extraction, text embeddings.
    openai_api_key: str | None = None
    answer_model: str = "gpt-4o"          # final multimodal answer
    extract_model: str = "gpt-4o-mini"    # per-page vision extraction (Flash equivalent)
    hyde_model: str = "gpt-4o-mini"       # cheap, fast paragraph generation
    embedding_model: str = "text-embedding-3-small"

    # Cohere reranker over the fused candidate list.
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-v3.5"

    # LlamaParse (LlamaIndex's hosted PDF parser). Used as the *primary*
    # structured PDF extractor — markdown + tables + figure captions per
    # page. If LLAMA_CLOUD_API_KEY is unset (or LlamaParse fails), the
    # ingest path falls back to the legacy PyMuPDF + GPT-4o-mini vision JSON
    # flow so the system still works without paying for LlamaParse.
    llama_cloud_api_key: str | None = None
    llama_parse_result_type: str = "markdown"  # "markdown" | "text"
    llama_parse_premium: bool = False           # set true for the premium tier

    # Gemini Embedding 2 — native multimodal model used for img_vec on visual
    # pages (and for embedding the HyDE paragraph on the query side, so the
    # query and the indexed image embeddings live in the same vector space).
    gemini_api_key: str | None = None
    gemini_image_embedding_model: str = "gemini-embedding-2-preview"
    gemini_image_embedding_dim: int = 1536  # 768 / 1536 / 3072 (MRL)

    # Weaviate Cloud (single collection, two named vectors: txt_vec + img_vec).
    weaviate_url: str = Field(default="")
    weaviate_api_key: str | None = None
    weaviate_collection: str = "DocPages"

    # Logging: DEBUG shows longer excerpts (HyDE, super_text previews).
    log_level: str = "INFO"

    # Uploads.
    max_pdf_upload_mb: int = 200

    # Page classifier (cost-aware filtering before any vision call):
    #   text-only  -> chars >= min_chars_text_only AND no drawings AND no raster images
    #   visual     -> any drawing / raster image / very low text
    #   skip       -> empty page (chars < min_chars_keep_page AND no visuals)
    min_chars_text_only: int = 500
    min_chars_keep_page: int = 20
    page_render_dpi: int = 300

    # Retrieval / answer.
    top_k_text: int = 20         # hybrid (BM25 + txt_vec) candidates
    top_k_image: int = 20        # near_vector on img_vec candidates
    top_k_rerank: int = 6        # final pages used to answer
    top_k_vision_in_answer: int = 2  # how many top page PNGs to attach to the answer
    hybrid_alpha: float = 0.5    # 0 = pure BM25, 1 = pure vector
    confidence_threshold: float = 0.05  # cohere relevance is ~[0,1]

    # Minimum Cohere relevance_score required for a page to be SHOWN as a
    # citation in the /query response. The answer LLM still sees every
    # reranked context (top_k_rerank) — this only filters what the user sees.
    # At 0.7, weakly-related pages stop appearing as "citations" next to the
    # answer, so only the actually-relevant page(s) are surfaced.
    citation_min_score: float = 0.7

    # Small-talk gate: short greetings / pleasantries get a fixed reply
    # without any LLM, embedding, vector, or rerank call.
    enable_smalltalk: bool = True
    smalltalk_reply: str = (
        "Hi! I answer questions about the technical PDFs you've ingested. "
        "Try something like: 'What is the torque spec for the M8 bolt?'"
    )

    # Semantic query cache (LangChain BaseCache subclass, OpenAI embeddings,
    # JSON persistence). On hit we skip HyDE / Weaviate / Cohere / GPT-4o
    # entirely and replay the previous answer + citations.
    enable_query_cache: bool = True
    query_cache_threshold: float = 0.95   # cosine on text-embedding-3-small
    query_cache_path: str = "data/query_cache.json"
    query_cache_max_entries: int = 1000   # FIFO trim once exceeded


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
