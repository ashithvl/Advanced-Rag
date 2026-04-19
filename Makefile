# Loads .env first so PYTHONDONTWRITEBYTECODE=1 applies (no __pycache__ in the repo).
UV_RUN = uv run --env-file .env

.PHONY: api ui lock

api:
	$(UV_RUN) uvicorn api:app --reload --host 127.0.0.1 --port 8000

ui:
	$(UV_RUN) streamlit run ui.py

lock:
	uv lock
