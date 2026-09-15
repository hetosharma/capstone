# Zepto Data & AI Platform Capstone

This repository implements the three-module capstone project:

- `data_pipeline/` — scrape, clean, convert, and load book data into normalized SQLite tables.
- `analytics/` — profile and model the Titanic dataset with a reproducible EDA and ML workflow.
- `support_assistant/` — an offline-first Zepto policy RAG service using local embeddings, ChromaDB, LangGraph, Pydantic, and FastAPI.

## Setup

Use Python 3.10+ and install the consolidated dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Each module also contains its own README with the exact run commands. The support assistant defaults to `MOCK_LLM=1`, so no LLM API key is needed for the graded path.

## Run

```bash
python data_pipeline/pipeline.py
python analytics/pipeline.py
python support_assistant/ingest.py
uvicorn support_assistant.app:app --reload --port 7860
```

Then call the assistant:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "{\"query\":\"How fast is delivery?\"}"
```

The implementation is designed to be repeatable: generated databases, charts, model artifacts, and Chroma persistence are ignored by Git and can be recreated from the checked-in scripts.


