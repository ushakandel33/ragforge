# Local Open-Source Model via vLLM

This serves as the assignment's "Local Deployment" requirement and doubles
as the third-tier fallback provider in `app/llm_client.py`.

## Requirements

vLLM requires an NVIDIA GPU (CUDA) to run at usable speed — it is not a
CPU-friendly library. It is included here as an **optional** Docker Compose
service (`vllm` profile) rather than a hard dependency, so the rest of the
stack (Gemini + Groq via API) works on any machine, GPU or not.

## Running it

```bash
# Requires nvidia-container-toolkit installed on the Docker host
docker compose --profile local-model up vllm
```

This uses the official `vllm/vllm-openai` image (see docker-compose.yml) to
serve `mistralai/Mistral-7B-Instruct-v0.2` on an OpenAI-compatible endpoint
at `http://localhost:8001/v1`. Once it's up, set in `.env`:

```
LOCAL_MODEL_ENABLED=true
LOCAL_MODEL_BASE_URL=http://vllm:8001/v1   # 'vllm' hostname when calling from inside docker compose
```

The backend will then use it as a fallback if both Gemini and Groq fail.

## No GPU available?

If you don't have a CUDA GPU (e.g. testing on a laptop or free-tier cloud
VM), skip this service — the rest of the assignment's functionality
(RAG, tool calling, structured output, reliability features) does not
depend on it. Document this limitation in your submission if you can't run
it, per the assignment's "justify why not applicable" allowance for
optimization steps that need hardware you don't have.
