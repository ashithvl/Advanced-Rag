"""Strict-grounding prompts for technical-doc QA."""

from langchain_core.prompts import ChatPromptTemplate

GROUNDED_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an assistant for technical documentation. "
                "Answer the user's question using ONLY the provided context. "
                "If the answer is not in the context, reply exactly: "
                "'Not specified in the manual.' "
                "Cite sources inline as [source: <doc>, p.<page>, fig.<figure?>]. "
                "Be precise. Preserve units, part numbers, and procedure steps verbatim."
            ),
        ),
        (
            "human",
            (
                "Question:\n{question}\n\n"
                "Context (ranked, most relevant first):\n{context}\n\n"
                "Answer:"
            ),
        ),
    ]
)


VERIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a strict fact-checker. Decide whether EVERY claim in the draft answer "
                "is fully supported by the provided context. "
                "Return JSON: {{\"faithful\": bool, \"unsupported\": [\"<claim>\", ...]}}."
            ),
        ),
        (
            "human",
            "Context:\n{context}\n\nDraft answer:\n{answer}",
        ),
    ]
)
