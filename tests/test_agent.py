from app.agent import AgentState, run_agent
from app.config import settings
from app.main import app
from app.schemas import AgentDecision, AgentResponse
from fastapi.testclient import TestClient

client = TestClient(app)


def sequence_decider(decisions):
    remaining = iter(decisions)

    def decide(_state, _use_rag):
        return next(remaining), None

    return decide


def test_structured_action_is_validated():
    decision = AgentDecision.model_validate_json(
        '{"action":"RETRIEVE","reason":"Need evidence","query":"policy","evidence_sufficient":false}'
    )
    assert decision.action == "RETRIEVE"


def test_multi_step_trajectory_retrieves_then_finalizes():
    decider = sequence_decider([
        AgentDecision(action="RETRIEVE", query="policy"),
        AgentDecision(action="REVISE", answer="The policy allows 20 days."),
        AgentDecision(action="FINALIZE", answer="The policy allows 20 days.", evidence_sufficient=True),
    ])
    result = run_agent(
        "How many days?",
        decider=decider,
        retriever=lambda _query: [{"text": "Employees receive 20 days.", "metadata": {"source": "faq"}, "score": 0.9}],
    )
    assert result.status == "completed"
    assert result.actions == ["RETRIEVE", "REVISE", "FINALIZE"]
    assert result.iterations == 3


def test_invalid_decision_is_handled_without_answer():
    result = run_agent("unknown", decider=lambda _state, _use_rag: (None, "invalid agent decision"))
    assert result.status == "failed"
    assert "could not complete" in result.answer.lower()


def test_retrieval_failure_never_becomes_confident_answer():
    decider = sequence_decider([
        AgentDecision(action="RETRIEVE", query="missing"),
        AgentDecision(action="FINALIZE", answer="The missing fact is definitely 42.", evidence_sufficient=True),
    ])

    def failing_retriever(_query):
        raise RuntimeError("injected retrieval failure")

    result = run_agent("What is missing?", decider=decider, retriever=failing_retriever)
    assert result.status == "degraded"
    assert "definitely 42" not in result.answer


def test_max_step_guard_stops_trajectory():
    old_limit = settings.max_agent_steps
    settings.max_agent_steps = 2
    try:
        result = run_agent(
            "Need more evidence",
            decider=lambda state, _use_rag: (
                AgentDecision(action="RETRIEVE", query=f"attempt-{len(state.actions)}"), None
            ),
            retriever=lambda _query: [],
        )
    finally:
        settings.max_agent_steps = old_limit
    assert result.status == "degraded"
    assert result.iterations == 2


def test_calculator_action_uses_bounded_tool():
    decider = sequence_decider([
        AgentDecision(action="CALCULATE", expression="2 + 2"),
        AgentDecision(action="FINALIZE", answer="4", evidence_sufficient=True),
    ])
    result = run_agent("What is 2 + 2?", use_rag=False, decider=decider)
    assert result.status == "completed"
    assert result.tools_used == ["calculator"]
    assert result.answer == "4"


def test_agent_endpoint_returns_inspectable_metadata(monkeypatch):
    expected = AgentResponse(
        answer="4", status="completed", iterations=2,
        actions=["CALCULATE", "FINALIZE"], tools_used=["calculator"],
    )
    monkeypatch.setattr("app.main.run_agent", lambda _query, _use_rag: expected)
    response = client.post("/agent/chat", json={"query": "2 + 2", "use_rag": False})
    assert response.status_code == 200
    assert response.json()["actions"] == ["CALCULATE", "FINALIZE"]