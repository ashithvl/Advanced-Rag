"""Streamlit UI: documents on the left, chat on the right."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import streamlit as st

from advanced_rag.config import get_settings
from advanced_rag.pretty_log import setup_logging

API_URL = os.getenv("ADVANCED_RAG_API", "http://localhost:8000")

st.set_page_config(page_title="Advanced RAG", layout="wide", initial_sidebar_state="collapsed")
setup_logging(get_settings().log_level)


def _get_json(path: str, **kwargs: object) -> object:
    r = httpx.get(f"{API_URL}{path}", timeout=30.0, **kwargs)
    r.raise_for_status()
    return r.json()


def _delete(path: str) -> None:
    r = httpx.delete(f"{API_URL}{path}", timeout=60.0)
    r.raise_for_status()


def _post_json(path: str, **kwargs: object) -> object:
    r = httpx.post(f"{API_URL}{path}", timeout=1200.0, **kwargs)
    r.raise_for_status()
    return r.json()


if "max_pdf_mb" not in st.session_state:
    # Same limit as the API (from .env / Docker env_file); no extra HTTP round-trip.
    st.session_state["max_pdf_mb"] = int(get_settings().max_pdf_upload_mb)
if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = 0
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Upload a **PDF** on the left, wait for indexing, then ask questions here. "
                "I only answer from your indexed manuals."
            ),
        }
    ]

MAX_PDF_MB = st.session_state["max_pdf_mb"]
MAX_PDF_BYTES = MAX_PDF_MB * 1024 * 1024

st.title("Advanced RAG")
st.caption(
    f"**PDF only**, max {MAX_PDF_MB} MB · per-page classifier + GPT-4o-mini structured "
    "vision extraction · two named vectors (text + image) · Weaviate hybrid + image search "
    "+ RRF + Cohere rerank · GPT-4o multimodal answer."
)

left, right = st.columns([1, 2.1], gap="large")

# ----- Left: upload + library -----
with left:
    _ingest = st.session_state.pop("_ingest_result", None)
    if _ingest is not None:
        st.success("PDF indexed.")
        with st.expander("Ingest details", expanded=False):
            st.json(_ingest)

    st.subheader("Upload PDF")
    uploaded = st.file_uploader(
        f"PDF only · max {MAX_PDF_MB} MB",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key=f"pdf_{st.session_state['upload_key']}",
    )
    if uploaded is not None:
        if uploaded.size > MAX_PDF_BYTES:
            mb = uploaded.size / (1024**2)
            st.error(f"Too large ({mb:.1f} MB). Max {MAX_PDF_MB} MB.")
        elif st.button("Index this PDF", type="primary", key="idx_pdf", use_container_width=True):
            with st.spinner("Indexing… (vision pages take a few seconds each)"):
                try:
                    resp = httpx.post(
                        f"{API_URL}/ingest",
                        files={"file": (uploaded.name, uploaded.getvalue())},
                        timeout=1800.0,
                    )
                    resp.raise_for_status()
                    out = resp.json()
                    st.session_state.pop("sources_cache", None)
                    st.session_state["upload_key"] = int(st.session_state.get("upload_key", 0)) + 1
                    st.toast("Indexed.", icon="✅")
                    st.session_state["_ingest_result"] = out
                    st.rerun()
                except httpx.HTTPStatusError as exc:
                    st.error(f"HTTP {exc.response.status_code}: {exc.response.text}")
                except Exception as exc:
                    st.error(f"Ingest failed: {exc}")

    st.divider()
    st.subheader("Sources")
    if st.button("Refresh", key="refresh_sources", use_container_width=True):
        st.session_state.pop("sources_cache", None)
        st.rerun()

    try:
        sources = st.session_state.get("sources_cache")
        if sources is None:
            sources = _get_json("/documents")
            st.session_state["sources_cache"] = sources
    except Exception as exc:
        st.error(f"Could not load sources: {exc}")
        sources = []

    if not sources:
        st.info("No PDFs yet. Upload one above.")
    else:
        for doc in sources:
            with st.container(border=True):
                st.markdown(f"**{doc['original_filename']}**")
                st.caption(
                    f"`{doc['doc_id'][:8]}…` · {(doc['size_bytes'] / (1024 * 1024)):.1f} MB · "
                    f"{doc['pages_indexed']} pages"
                )
                st.caption(doc["uploaded_at"])
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Reindex", key=f"rex_{doc['doc_id']}", type="primary"):
                        with st.spinner("Reindexing…"):
                            try:
                                _post_json(f"/documents/{doc['doc_id']}/reindex")
                                st.session_state.pop("sources_cache", None)
                                st.toast("Reindexed.", icon="✅")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                with c2:
                    if st.button("Remove", key=f"del_{doc['doc_id']}", type="secondary"):
                        try:
                            _delete(f"/documents/{doc['doc_id']}")
                            st.session_state.pop("sources_cache", None)
                            st.session_state["upload_key"] = (
                                int(st.session_state.get("upload_key", 0)) + 1
                            )
                            st.toast("Removed.", icon="🗑️")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

# ----- Right: chat -----
with right:
    st.subheader("Chat")
    if st.button("Clear conversation", type="secondary"):
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Conversation cleared. Ask a new question when you are ready.",
            }
        ]
        st.rerun()

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta = msg.get("meta")
            if meta:
                m1, m2 = st.columns(2)
                m1.caption(f"Confidence **{meta['confidence']:.2f}**")
                m2.caption(f"**{meta['citations_count']}** citations")
                with st.expander("Citations / pages", expanded=False):
                    for c in meta.get("citations", []):
                        title = (c.get("title") or "").strip()
                        st.markdown(
                            f"**{c.get('source_filename', '?')}** · p.{c.get('page_number', '?')}"
                            f" · score {c.get('score', 0.0):.3f}"
                            + ("  *(visual)*" if c.get("is_visual") else "")
                            + (f"  — {title}" if title else "")
                        )
                        ip = c.get("page_image_path")
                        if ip and Path(ip).is_file():
                            st.image(ip, width=360)

    if prompt := st.chat_input("Ask about your indexed PDFs…"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        try:
            with st.spinner("Thinking…"):
                resp = httpx.post(
                    f"{API_URL}/query",
                    json={"question": prompt},
                    timeout=180.0,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            st.session_state["messages"].append(
                {"role": "assistant", "content": f"Something went wrong: {exc}"}
            )
        else:
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": data.get("answer", ""),
                    "meta": {
                        "confidence": float(data.get("confidence", 0.0)),
                        "citations_count": len(data.get("citations", [])),
                        "citations": data.get("citations", []),
                    },
                }
            )
        st.rerun()
