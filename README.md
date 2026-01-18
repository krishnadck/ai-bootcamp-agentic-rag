# ai-agent-bootcamp-prerequisites

RAG-based Amazon electronics product assistant built with FastAPI, Qdrant,
and a Streamlit chat UI. LangSmith tracing is supported.

## Project layout
- `apps/api`: FastAPI RAG backend (`/product_assistant`)
- `apps/chatbot_ui`: Streamlit chat UI
- `data/`: Amazon electronics datasets used in notebooks
- `qdrant_storage/`: local Qdrant persistence

## Environment
Create a `.env` in the repo root (see `env.example`).

Required by the API config:
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `GROQ_API_KEY`

Optional (LangSmith tracing):
- `LANGSMITH_API_KEY`
- `LANGSMITH_TRACING`
- `LANGSMITH_ENDPOINT`
- `LANGSMITH_PROJECT`

## Run all services
Starts Qdrant, the RAG API, and the Streamlit UI:

```
make run-docker-compose
```

Services:
- Front-end: `http://localhost:8501/`
- Backend: `http://localhost:8000/docs`
- Qdrant: `http://localhost:6333/dashboard#/collections`


## Local development (optional)
Install dependencies from the repo root:

```
uv sync
```

Run the API:

```
PYTHONPATH=apps/api/src uvicorn server.app:app --reload --port 8000
```

Run the UI:

```
PYTHONPATH=apps/chatbot_ui/src streamlit run apps/chatbot_ui/src/chatbot_ui/app.py
```

## API usage
`POST /product_assistant` with JSON:

```
{"query": "Tell me about the uni USB 3.0 card reader"}
```

Response includes:
`answer`, `retrieved_context_ids`, `retrieved_context`, `similarity_scores`.

## Evaluation
Run Ragas + LangSmith evaluation:

```
make run-evals-retriever
```