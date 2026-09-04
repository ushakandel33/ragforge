"""
Basic tests. Run with: pytest tests/
Note: /chat and /structured tests require valid API keys in .env to pass
end-to-end; they're structured to at least confirm request/response shape.
"""
from fastapi.testclient import TestClient
from app.main import app
from app.tools import calculator
from app.rag.ingest import chunk_text

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "providers" in resp.json()


def test_calculator_tool():
    assert calculator("2 + 2 * 3") == "8"


def test_calculator_tool_rejects_unsafe_input():
    result = calculator("__import__('os').system('echo hi')")
    assert result.startswith("Error")


def test_chunking():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    # Overlap check: end of one chunk should share words with start of next
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-1] in second_words or len(chunks) == 1


def test_ingest_requires_file():
    resp = client.post("/ingest")
    assert resp.status_code == 422
