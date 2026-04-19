"""Smoke tests: importing the package must not require network or external services."""


def test_package_imports() -> None:
    import advanced_rag
    from advanced_rag import config, pipeline
    from advanced_rag.generation import GROUNDED_ANSWER_PROMPT, get_chat_llm  # noqa: F401
    from advanced_rag.ingestion import build_hierarchical_nodes  # noqa: F401

    assert advanced_rag.__version__
    assert config.get_settings().llm_model
    assert callable(pipeline.build_graph)


def test_graph_builds() -> None:
    from advanced_rag.pipeline import build_graph

    assert build_graph() is not None
