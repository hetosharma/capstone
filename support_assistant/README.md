# Module 3 — Zepto Support Assistant

This module is an offline-first RAG service. Eight policy documents are ingested, chunked one document at a time, embedded locally with `all-MiniLM-L6-v2`, and persisted in the ChromaDB collection `zepto_support_policies`. The LangGraph flow contains three named nodes:

```text
POST /ask
   ↓
classify_intent ── policy_question ──> retrieve_and_answer ──> validated response
       └──────── general_question ──> direct_answer          ──> validated response
```

`classify_intent` uses the required keyword heuristic in mock mode. `retrieve_and_answer` always embeds and queries ChromaDB; only answer generation changes with `MOCK_LLM`. In the default mode (`MOCK_LLM=1`) it returns a deterministic excerpt from the top retrieved chunk. In optional live mode (`MOCK_LLM=0`) the structured prompt in `prompts.py` can call a Groq-compatible LLM when `GROQ_API_KEY` and `GROQ_MODEL` are configured. `direct_answer` is deterministic in mock mode and does not retrieve documents. The final `AnswerResponse` Pydantic model always validates `answer`, `sources`, and `confidence`.

## Run

```bash
pip install -r ../requirements.txt
python support_assistant/ingest.py
MOCK_LLM=1 uvicorn support_assistant.app:app --host 0.0.0.0 --port 7860
```

Windows PowerShell:

```powershell
$env:MOCK_LLM="1"
python -m uvicorn support_assistant.app:app --host 0.0.0.0 --port 7860
```

Example calls:

```bash
curl -X POST http://127.0.0.1:7860/ask -H 'Content-Type: application/json' -d '{"query":"How fast is delivery?"}'
curl -X POST http://127.0.0.1:7860/ask -H 'Content-Type: application/json' -d '{"query":"What is the capital of France?"}'
```

The first routes to retrieval and returns `sources` such as `doc_01`; the second routes to `direct_answer` and returns an empty source list. The Docker baseline is:

```bash
docker build -t zepto-support ./support_assistant
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support
```


