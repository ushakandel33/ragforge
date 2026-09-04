"""
Unified LLM client that:
  1. Tries Gemini (primary, free tier).
  2. Falls back to Groq (secondary, free + fast, OpenAI-compatible) on failure.
  3. Falls back to a locally-served open-source model via vLLM, if enabled.
  4. Falls back to a static degraded response as an absolute last resort,
     so the API never hard-fails on the caller.

Each provider call is wrapped in a retry (exponential backoff) before the
client gives up on that provider and moves to the next one in the chain.
"""
import json
import logging
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google import genai
from google.genai import types as genai_types
from groq import Groq
import httpx

from app.config import settings
from app.tools import TOOL_DECLARATIONS, execute_tool

logger = logging.getLogger("llm_client")

SYSTEM_PROMPT = """You are a precise, helpful AI assistant.
Rules:
- If context passages are provided, ground your answer in them and say so; if the context
  doesn't cover the question, say what's missing rather than guessing.
- Use the provided tools when a question needs live computation (e.g. arithmetic, current time).
- Keep answers concise and directly responsive to the question.
- When asked for structured data, return ONLY valid JSON matching the requested shape -
  no markdown fences, no commentary.
"""


class RetryableProviderError(Exception):
    """Raised for transient provider errors (rate limit, timeout, 5xx) that are worth retrying."""


def _retryable() -> retry:
    return retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(RetryableProviderError),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Gemini (primary)
# ---------------------------------------------------------------------------

def _gemini_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def _to_gemini_schema(schema: dict) -> dict:
    """Recursively uppercase JSON-schema `type` values (google-genai expects
    'OBJECT'/'STRING'/etc., not the lowercase JSON Schema convention)."""
    out = dict(schema)
    if "type" in out:
        out["type"] = out["type"].upper()
    if "properties" in out:
        out["properties"] = {k: _to_gemini_schema(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _to_gemini_schema(out["items"])
    return out


@_retryable()
def _call_gemini(query: str, context: str, allow_tools: bool = True) -> dict:
    if not settings.gemini_api_key:
        raise RetryableProviderError("Gemini API key not configured")

    client = _gemini_client()
    prompt = f"Context:\n{context}\n\nQuestion: {query}" if context else query

    tool_config = None
    if allow_tools:
        gemini_tools = [genai_types.Tool(function_declarations=[
            genai_types.FunctionDeclaration(name=t["name"], description=t["description"],
                                             parameters=_to_gemini_schema(t["parameters"]))
            for t in TOOL_DECLARATIONS
        ])]
    else:
        gemini_tools = None

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=settings.app_temperature,
                top_p=settings.app_top_p,
                tools=gemini_tools,
            ),
        )
    except Exception as e:
        raise RetryableProviderError(str(e)) from e

    tool_calls = []
    answer_text = ""
    for candidate in response.candidates or []:
        for part in candidate.content.parts:
            if getattr(part, "function_call", None):
                fc = part.function_call
                args = dict(fc.args) if fc.args else {}
                result = execute_tool(fc.name, args)
                tool_calls.append({"name": fc.name, "arguments": args, "result": result})
            elif getattr(part, "text", None):
                answer_text += part.text

    # If the model made tool calls, feed results back for a final answer.
    if tool_calls:
        tool_summary = "\n".join(f"{tc['name']}({tc['arguments']}) -> {tc['result']}" for tc in tool_calls)
        followup = client.models.generate_content(
            model=settings.gemini_model,
            contents=f"{prompt}\n\nTool results:\n{tool_summary}\n\nNow answer the question using these results.",
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=settings.app_temperature,
            ),
        )
        answer_text = followup.text or answer_text

    return {"answer": answer_text.strip(), "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# Groq (fallback #1) - OpenAI-compatible tool calling
# ---------------------------------------------------------------------------

@_retryable()
def _call_groq(query: str, context: str, allow_tools: bool = True) -> dict:
    if not settings.groq_api_key:
        raise RetryableProviderError("Groq API key not configured")

    client = Groq(api_key=settings.groq_api_key)
    user_content = f"Context:\n{context}\n\nQuestion: {query}" if context else query

    groq_tools = [
        {"type": "function", "function": {"name": t["name"], "description": t["description"],
                                           "parameters": t["parameters"]}}
        for t in TOOL_DECLARATIONS
    ] if allow_tools else None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=settings.app_temperature,
            top_p=settings.app_top_p,
            tools=groq_tools,
        )
    except Exception as e:
        raise RetryableProviderError(str(e)) from e

    message = response.choices[0].message
    tool_calls = []

    if message.tool_calls:
        messages.append(message)
        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = execute_tool(tc.function.name, args)
            tool_calls.append({"name": tc.function.name, "arguments": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

        followup = client.chat.completions.create(
            model=settings.groq_model, messages=messages, temperature=settings.app_temperature,
        )
        answer = followup.choices[0].message.content or ""
    else:
        answer = message.content or ""

    return {"answer": answer.strip(), "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# Local open-source model via vLLM (fallback #2, optional)
# ---------------------------------------------------------------------------

@_retryable()
def _call_local_model(query: str, context: str) -> dict:
    if not settings.local_model_enabled:
        raise RetryableProviderError("Local model disabled")

    user_content = f"Context:\n{context}\n\nQuestion: {query}" if context else query
    payload = {
        "model": settings.local_model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": settings.app_temperature,
        "top_p": settings.app_top_p,
    }
    try:
        resp = httpx.post(f"{settings.local_model_base_url}/chat/completions", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RetryableProviderError(str(e)) from e

    return {"answer": answer.strip(), "tool_calls": []}


# ---------------------------------------------------------------------------
# Orchestration: try each provider in order
# ---------------------------------------------------------------------------

def generate_answer(query: str, context: str = "", allow_tools: bool = True) -> tuple[dict, str]:
    """
    Returns (result_dict, provider_name). Tries Gemini -> Groq -> local model
    -> static fallback, in that order, so a single provider outage never
    takes the whole assistant down.
    """
    providers = [
        ("gemini", lambda: _call_gemini(query, context, allow_tools)),
        ("groq", lambda: _call_groq(query, context, allow_tools)),
    ]
    if settings.local_model_enabled:
        providers.append(("local", lambda: _call_local_model(query, context)))

    last_error: Optional[Exception] = None
    for name, fn in providers:
        try:
            return fn(), name
        except Exception as e:
            logger.warning("Provider %s failed after retries: %s", name, e)
            last_error = e
            continue

    logger.error("All providers failed. Last error: %s", last_error)
    return (
        {
            "answer": (
                "I'm temporarily unable to reach any language model provider. "
                "Please check API keys/network and try again shortly."
            ),
            "tool_calls": [],
        },
        "fallback_static",
    )


def generate_structured(query: str, json_schema_hint: str) -> dict:
    """
    Ask the model for JSON matching a described shape, then parse and
    validate it. Retries the whole prompt-and-parse cycle on invalid JSON,
    since occasional formatting slips are more common than outright
    provider failure here.
    """
    prompt = (
        f"{query}\n\nRespond with ONLY valid JSON matching this shape "
        f"(no markdown fences, no extra text): {json_schema_hint}"
    )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    def _attempt():
        result, _provider = generate_answer(prompt, allow_tools=False)
        text = result["answer"].strip().strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
        return json.loads(text)

    return _attempt()
