"""A bounded, inspectable single-agent loop for adaptive RAG.

The agent receives compact state, chooses one validated action, observes the
result, and repeats. It never executes a tool name supplied outside the
allow-listed action vocabulary.
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config import settings
from app.llm_client import generate_answer
from app.rag.vector_store import query as rag_query
from app.schemas import AgentDecision, AgentResponse, Source
from app.tools import execute_tool

logger = logging.getLogger("agent")

_ACTIONS = {"RETRIEVE", "CALCULATE", "GET_TIME", "VERIFY", "REVISE", "ASK_USER", "FINALIZE"}


@dataclass
class AgentState:
    original_query: str
    candidate_answer: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    retrieval_attempts: int = 0
    failures: list[str] = field(default_factory=list)
    provider_used: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


def _add_usage(state: AgentState, result: dict[str, Any]) -> None:
    usage = result.get("usage") or {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if value is not None:
            current = getattr(state, key)
            setattr(state, key, (current or 0) + int(value))


def _remember(state: AgentState, action: str, observation: str) -> None:
    state.history.append({"action": action, "observation": observation[:500]})
    state.history = state.history[-settings.agent_history_limit :]


def _merge_evidence(state: AgentState, hits: list[dict[str, Any]]) -> None:
    known = {item.get("text", "") for item in state.evidence}
    for hit in hits:
        if hit.get("text") and hit["text"] not in known:
            state.evidence.append(hit)
            known.add(hit["text"])
    state.evidence.sort(key=lambda item: item.get("score", 0), reverse=True)
    state.evidence = state.evidence[: settings.agent_evidence_limit]


def _compact_context(state: AgentState) -> str:
    evidence = "\n\n".join(
        f"[{i + 1}] {item['text']} (source={item.get('metadata', {}).get('source', 'unknown')})"
        for i, item in enumerate(state.evidence)
    )
    history = json.dumps(state.history, ensure_ascii=True)
    return f"Evidence:\n{evidence or '[none]'}\nRecent observations:\n{history}"


def _decision_prompt(state: AgentState, use_rag: bool) -> str:
    return f"""You are the decision-maker in a bounded self-checking RAG agent.
Return ONLY one JSON object with keys action, reason, query, expression, answer,
evidence_sufficient. action must be one of RETRIEVE, CALCULATE, GET_TIME, VERIFY,
REVISE, ASK_USER, FINALIZE. Never invent facts. RETRIEVE when evidence is absent
or insufficient; use calculator for arithmetic and GET_TIME for current UTC time.
Use ASK_USER only when required information is genuinely missing. FINALIZE only
when the answer is supported by observations or the request is a bounded tool task.
If retrieval is disabled, do not select RETRIEVE. Keep answer concise and do not
include hidden reasoning.

Original query: {state.original_query}
Retrieval allowed: {use_rag}
Candidate answer: {state.candidate_answer or '[none]'}
{_compact_context(state)}"""


def decide_agent_action(state: AgentState, use_rag: bool) -> tuple[AgentDecision | None, str | None]:
    result, provider = generate_answer(_decision_prompt(state, use_rag), allow_tools=False)
    state.provider_used = provider
    _add_usage(state, result)
    try:
        text = result.get("answer", "").strip().strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
        decision = AgentDecision.model_validate_json(text)
        if decision.action not in _ACTIONS:
            raise ValueError("unsupported action")
        return decision, None
    except Exception as exc:
        return None, f"invalid agent decision: {exc}"


def _safe_degraded(state: AgentState) -> str:
    if state.candidate_answer and state.evidence:
        return f"I could not fully verify this response. Available evidence suggests: {state.candidate_answer}"
    if state.failures:
        return "I could not complete this request because the required evidence or tool was unavailable."
    return "I could not complete a sufficiently verified answer within the agent step limit."


def run_agent(
    query: str,
    use_rag: bool = True,
    decider: Callable[[AgentState, bool], tuple[AgentDecision | None, str | None]] | None = None,
    retriever: Callable[[str], list[dict[str, Any]]] = rag_query,
) -> AgentResponse:
    state = AgentState(original_query=query)
    choose = decider or decide_agent_action
    status = "degraded"
    final_answer = ""

    for _ in range(settings.max_agent_steps):
        decision, error = choose(state, use_rag)
        if error or decision is None:
            state.failures.append(error or "missing decision")
            _remember(state, "INVALID", error or "missing decision")
            break

        action = decision.action
        signature = json.dumps({"action": action, "query": decision.query, "expression": decision.expression}, sort_keys=True)
        if any(item.get("signature") == signature for item in state.history):
            state.failures.append("repeated action detected")
            _remember(state, "LOOP_GUARD", "repeated action detected")
            break
        state.history.append({"signature": signature, "action": action, "observation": ""})
        state.actions.append(action)

        if action == "RETRIEVE":
            if not use_rag:
                state.failures.append("retrieval requested while RAG is disabled")
                _remember(state, action, "retrieval disabled")
                continue
            try:
                hits = retriever(decision.query or state.original_query)
                state.retrieval_attempts += 1
                _merge_evidence(state, hits)
                _remember(state, action, f"retrieved {len(hits)} candidate chunks")
            except Exception as exc:
                state.failures.append(f"retrieval failure: {exc}")
                _remember(state, action, "retrieval failed")
        elif action in {"CALCULATE", "GET_TIME"}:
            tool_name = "calculator" if action == "CALCULATE" else "get_current_time"
            arguments = {"expression": decision.expression} if action == "CALCULATE" else {}
            try:
                observation = execute_tool(tool_name, arguments)
                state.tools_used.append(tool_name)
                state.candidate_answer = str(observation)
                _remember(state, action, str(observation))
            except Exception as exc:
                state.failures.append(f"tool failure: {exc}")
                _remember(state, action, "tool failed")
        elif action in {"REVISE", "VERIFY"}:
            if decision.answer:
                state.candidate_answer = decision.answer
            _remember(state, action, decision.answer or "no revised answer")
        elif action == "ASK_USER":
            final_answer = decision.answer or "Could you clarify what information you need?"
            status = "clarification_required"
            break
        elif action == "FINALIZE":
            if decision.answer:
                state.candidate_answer = decision.answer
            verified_claim = decision.evidence_sufficient and not state.failures and (not use_rag or bool(state.evidence))
            if state.candidate_answer and (state.evidence or state.tools_used or verified_claim):
                final_answer = state.candidate_answer
                status = "completed"
            else:
                state.failures.append("finalize without verified evidence or tool observation")
                final_answer = _safe_degraded(state)
            break

    if not final_answer:
        final_answer = _safe_degraded(state)
    if status == "degraded" and state.failures and not state.actions:
        status = "failed"

    sources = [Source(text=h["text"], metadata=h.get("metadata") or {}, score=h.get("score", 0.0)) for h in state.evidence]
    note = None if state.total_tokens is not None else "Provider usage metadata was unavailable for one or more agent calls."
    return AgentResponse(
        answer=final_answer,
        status=status,
        iterations=len(state.actions),
        actions=state.actions,
        tools_used=state.tools_used,
        failures=state.failures,
        sources=sources,
        provider_used=state.provider_used,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        total_tokens=state.total_tokens,
        token_usage_note=note,
    )