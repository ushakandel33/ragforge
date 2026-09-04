# Backend API image
FROM python:3.11-slim

WORKDIR /app

# System deps for pypdf / sentence-transformers (torch wheel needs libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/sample_docs/ ./data/sample_docs/

RUN mkdir -p /app/data/chroma_db /app/data/response_cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
