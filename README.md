# RAGForge — Production RAG Assistant

A FastAPI backend + Streamlit UI implementing a RAG-based assistant with
tool calling, structured output, and production reliability features
(retries, rate limiting, provider fallback, caching).

See [ARCHITECTURE.md](ARCHITECTURE.md) for the system diagram and design
rationale.

## Features

- **LLM integration**: Google Gemini (primary, free tier) with automatic
  fallback to Groq, then an optional locally-served open-source model via
  vLLM, then a static degraded message — the API never hard-fails.
- **RAG pipeline**: upload PDF/text docs → chunked with overlap → embedded
  locally (sentence-transformers) → stored in ChromaDB → retrieved top-k
  per query.
- **Tool calling**: `calculator` and `get_current_time` tools, dispatched
  when the model requests them.
- **Structured output**: `/structured` endpoint validates the model's JSON
  response against an expected shape.
- **Reliability**: retry with exponential backoff (`tenacity`), rate
  limiting (`slowapi`), response caching (`diskcache`), graceful
  degradation across providers.
- **Async/concurrent handling**: blocking work runs in a thread pool so the
  event loop can serve concurrent requests.

## Prerequisites

- Python 3.11+ (for local runs) or Docker + Docker Compose
- A free [Gemini API key](https://aistudio.google.com/apikey)
- A free [Groq API key](https://console.groq.com/keys) (used as fallback)

## Quick start (Docker Compose — recommended)

```bash
cp .env.example .env
# edit .env and paste in your GEMINI_API_KEY and GROQ_API_KEY

docker compose up --build
```

- API: http://localhost:8000 (docs at http://localhost:8000/docs)
- UI: http://localhost:8501

To also run the optional local open-source model (requires an NVIDIA GPU),
see `app/local_model/README.md`.

## Quick start (local, no Docker)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your API keys

# Terminal 1 — backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — UI
streamlit run ui/streamlit_app.py
```

## Ingesting documents

Use the UI's sidebar upload, or call the API directly:

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/sample_docs/sample_faq.txt"
```

A sample FAQ doc is included at `data/sample_docs/sample_faq.txt` so you
can test retrieval immediately — try asking "How many vacation days do
employees get?" in the UI.

## API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + which providers are configured |
| `/ingest` | POST | Upload a document to embed and store |
| `/chat` | POST | RAG-grounded chat with tool calling |
| `/agent/chat` | POST | Bounded adaptive-evidence agent with inspectable safe metadata |
| `/structured` | POST | Example structured-JSON-output endpoint |

`/chat` request body:
```json
{ "query": "How many vacation days do I get?", "use_rag": true }
```

## Week 16: Agentic RAG

### Agentic Feature

`POST /agent/chat` runs one bounded agent with typed state: the original
question, candidate answer, capped evidence, compact observations, prior
actions, failures, and token totals. Each iteration validates one action from
`RETRIEVE`, `CALCULATE`, `GET_TIME`, `VERIFY`, `REVISE`, `ASK_USER`, or
`FINALIZE`. The response exposes status, actions, tools, sources, and token
metadata, but never hidden chain-of-thought.

### Why a Fixed Pipeline Is Insufficient

A fixed pipeline is insufficient because the assistant cannot know in advance
whether the first retrieval contains enough evidence; it must inspect
intermediate results and dynamically decide whether to retrieve again, use
another tool, request clarification, revise, or finalize.

### Context Engineering Technique

The old one-shot path could pass every retrieved chunk directly to the model.
The agent uses capped, score-ranked, deduplicated evidence in `app/agent.py`,
preserving source metadata, and keeps only a compact bounded action/observation
history. This prevents repeated retrievals from saturating later prompts while
retaining traceable evidence for verification.

### Agentic Pattern

This is deliberately a single-agent loop. One state owner makes context
updates and stopping decisions easier to inspect and avoids context isolation,
coordination overhead, and skill dilution that would not help this small
sequential RAG task. Retrieval, calculator, and current-time operations remain
bounded tools; they do not need their own reasoning loops.

### Skill vs. Agent

This capability is not merely a Skill: a Skill would primarily provide
instructions or procedural knowledge, while this feature requires runtime
state, repeated observation, conditional tool selection, and a dynamic stopping
decision.

### Evaluation Harness

`evaluation/evaluate_agent.py` runs seven deterministic trajectories covering
first-hit retrieval, adaptive second retrieval, calculator use, insufficient
evidence, clarification, unnecessary-call avoidance, and injected retrieval
failure. It records actions, valid arguments, trajectory length, status,
failure classification, notes, and token fields without using RAGAS or an LLM
judge. Failure labels are defined in the generated report: Hard failure means
no usable decision/response, Soft failure means a bounded action failed but a
transparent degraded response was returned, and Cascading soft failure means
that an earlier bounded failure breaks a dependent later action.

### Token and Cost Accounting

Gemini and Groq usage metadata is normalized and aggregated across every call
in one trajectory; local-provider usage is read when present. Missing usage is
reported as `null` with an explanation, never estimated silently. The
deterministic evaluation fixtures make no provider calls, so their token totals
are correctly unavailable and no monetary cost is fabricated.

### Failure Injection Test

`evaluation/failure_injection.py` raises a controlled retrieval exception only
for the evaluation case. The observed run records the failure, returns
`degraded`, and does not return the fixture's unsupported confident answer.

### Tool vs. Agent Boundary

ChromaDB retrieval, `calculator`, and `get_current_time` are tools controlled
by the agent because each has a narrow input/output contract and no persistent
reasoning loop. The agent owns conditional sequencing, verification, revision,
and the maximum-step guard.

### Evaluation Results

The current generated report in `evaluation/results.md` records a 7/7 (100%)
scenario success rate and a 4/7 (57%) actual task completion rate. The status
distribution is 4 completed, 1 clarification-required, 2 degraded, and 0 failed.
Tool/action correctness is 7/7 (100%), with an average trajectory length of
2.14 steps. Token fields are `null` for deterministic mocked evaluation runs
because those fixtures make no external provider calls; live agent runs report
real provider token usage when available.

## Running tests

```bash
pytest tests/
```

## Deployment

### Docker Compose (single VM)

```bash
docker compose up -d --build
```
Put a reverse proxy (nginx/Caddy) in front for TLS if exposing publicly.

### Cloud deployment (bonus)

The same `docker-compose.yml` services map directly onto managed container
platforms:

- **AWS**: push images to ECR, run via ECS Fargate (one service per
  container) or App Runner for the simplest path; use an ALB in front of
  the API.
- **GCP**: push images to Artifact Registry, deploy each service to Cloud
  Run (`gcloud run deploy`) — Cloud Run's autoscaling suits the API's
  stateless design well (note: mount a persistent volume or switch
  ChromaDB to a managed vector DB if you need data to survive across
  Cloud Run instances).
- **Azure**: push to Azure Container Registry, deploy via Azure Container
  Apps, which supports Compose-like multi-container definitions directly.

In all three cases, set the same environment variables from `.env` as
secrets/environment config in the platform, rather than baking API keys
into the image.

## Project structure

```
app/
  main.py            FastAPI app, endpoints
  config.py          Settings (env-driven)
  schemas.py         Pydantic request/response models
  llm_client.py       Provider orchestration, retries, tool-calling
  agent.py            Bounded adaptive-evidence single-agent loop
  tools.py           Tool declarations + implementations
  cache.py           Response caching
  rate_limiter.py    Rate limiting setup
  rag/
    ingest.py        Document loading + chunking
    embeddings.py    Local embedding generation
    vector_store.py  ChromaDB wrapper
  local_model/
    README.md        vLLM local deployment notes
ui/
  streamlit_app.py   Streamlit front-end
data/sample_docs/    Sample document for testing RAG
tests/               Pytest suite
evaluation/           Deterministic W16 harness, fixtures, and results
Dockerfile           Backend image
Dockerfile.ui        UI image
docker-compose.yml   Full stack + optional vLLM service
```

## Known limitations

- ChromaDB runs embedded on local disk — fine for a single instance, but
  won't share state across horizontally-scaled API replicas; swap in a
  hosted vector DB (e.g. Pinecone, Qdrant Cloud) for that case.
- The local vLLM path requires an NVIDIA GPU and is optional/untested in
  CPU-only environments — see `app/local_model/README.md`.
- Rate limiting is per-process (in-memory); for multi-replica deployments,
  back `slowapi` with Redis instead.
