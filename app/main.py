"""
FastAPI backend for the AI assistant.

Endpoints:
  POST /chat            - RAG-grounded chat with tool calling, caching, and
                           provider fallback
  POST /ingest          - upload a document to be chunked, embedded, and
                           stored in the vector DB
  POST /structured      - example structured-JSON-output endpoint
  GET  /health          - liveness + provider configuration check

Concurrency: ingestion (CPU-bound: chunking/embedding) is offloaded to a
thread pool via FastAPI's `run_in_threadpool` so it doesn't block the event
loop; the LLM calls themselves are also dispatched via threadpool since the
underlying SDKs used here are synchronous. This keeps the server able to
handle multiple concurrent chat requests without one slow call blocking others.
"""
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.cache import get_cached, set_cached
from app.agent import run_agent
from app.config import settings
from app.llm_client import generate_answer, generate_structured
from app.rag.ingest import ingest_file
from app.rag.vector_store import add_chunks, query as rag_query
from app.rate_limiter import limiter, CHAT_RATE_LIMIT
from app.schemas import AgentRequest, AgentResponse, ChatRequest, ChatResponse, HealthResponse, IngestResponse, Source

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="RAGForge API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your UI's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        providers={
            "gemini": bool(settings.gemini_api_key),
            "groq": bool(settings.groq_api_key),
            "local": settings.local_model_enabled,
        },
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = await run_in_threadpool(ingest_file, tmp_path)
        count = await run_in_threadpool(add_chunks, chunks)
        return IngestResponse(filename=file.filename, chunks_created=count, status="success")
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(CHAT_RATE_LIMIT)
async def chat(request: Request, body: ChatRequest):
    context = ""
    sources: list[Source] = []

    if body.use_rag:
        hits = await run_in_threadpool(rag_query, body.query)
        if hits:
            context = "\n\n".join(h["text"] for h in hits)
            sources = [Source(text=h["text"], metadata=h["metadata"], score=h["score"]) for h in hits]

    cached = get_cached(body.query, context)
    if cached is not None:
        cached["cached"] = True
        return ChatResponse(**cached)

    result, provider = await run_in_threadpool(generate_answer, body.query, context)

    response = ChatResponse(
        answer=result["answer"],
        sources=sources,
        tool_calls=result.get("tool_calls", []),
        provider_used=provider,
        cached=False,
    )

    if provider != "fallback_static":
        set_cached(body.query, context, response.model_dump(exclude={"cached"}))

    return response


@app.post("/agent/chat", response_model=AgentResponse)
@limiter.limit(CHAT_RATE_LIMIT)
async def agent_chat(request: Request, body: AgentRequest):
    """Run the bounded, inspectable adaptive-evidence agent."""
    return await run_in_threadpool(run_agent, body.query, body.use_rag)


@app.post("/structured")
async def structured_endpoint(body: ChatRequest):
    """
    Example of forcing structured JSON output: asks the model to summarize
    the query into a fixed shape and validates the result before returning.
    """
    schema_hint = '{"topic": string, "key_points": string[], "sentiment": "positive"|"neutral"|"negative"}'
    try:
        data = await run_in_threadpool(generate_structured, body.query, schema_hint)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model did not return valid JSON: {e}")
    return data
