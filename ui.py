"""Streamlit UI for the Advanced RAG pipeline."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("ADVANCED_RAG_API", "http://localhost:8000")

st.set_page_config(page_title="Advanced RAG", layout="wide")
st.title("Advanced RAG")
st.caption("Multimodal RAG over technical documents — LangGraph + LlamaIndex + Weaviate + Neo4j")

with st.sidebar:
    st.subheader("Ingest a document")
    uploaded = st.file_uploader("PDF / image / office doc", type=None)
    if uploaded and st.button("Ingest", use_container_width=True):
        with st.spinner("Parsing and indexing..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout=600.0,
                )
                resp.raise_for_status()
                st.success(resp.json())
            except Exception as exc:
                st.error(f"Ingest failed: {exc}")

question = st.text_area("Ask a question about your documents", height=120)
ask = st.button("Ask", type="primary", disabled=not question.strip())

if ask:
    with st.spinner("Thinking..."):
        try:
            resp = httpx.post(
                f"{API_URL}/query",
                json={"question": question},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.stop()

    st.markdown("### Answer")
    st.write(data["answer"])

    cols = st.columns(3)
    cols[0].metric("Confidence", f"{data['confidence']:.2f}")
    cols[1].metric("Faithful", "yes" if data.get("faithful") else "no")
    cols[2].metric("Citations", len(data.get("citations", [])))

    if data.get("unsupported"):
        st.warning("Unsupported claims flagged by verifier:")
        for claim in data["unsupported"]:
            st.markdown(f"- {claim}")

    with st.expander("Citations / retrieved chunks"):
        for c in data.get("citations", []):
            st.json(c)
